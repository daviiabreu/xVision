import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch

from edge.participants.identification import load_tracking, process_identification
from edge.participants.identity import (
    TrackObservation,
    extract_shirt_color,
    stabilize_role,
    teams_from_color_samples,
)
from edge.participants.marking import (
    DEFAULT_RADIUS_M,
    candidates_from_frame,
    pair_markings,
    teams_from_identities,
)
from edge.participants.tracking import PlayerTracker
from edge.participants.types import FramePerception, as_role, ground_point
from edge.participants.video import TEAM_COLORS, UNKNOWN_COLOR, process_video

MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_DIR))

from top_view.homography_view import (  # noqa: E402
    detect_keypoints,
    find_transformation,
    load_model,
)
from top_view.top_view_config import get_homography_model_path  # noqa: E402

OUTPUT_ROOT = Path("data/marcacao_output")
SCHEMA_VERSION = "player-marking.v0"
IMAGE_SIZE = 1280
CONFIDENCE = 0.25
RADIUS_PX = 150.0
MINIMUM_SAMPLES = 2
DEFAULT_WINDOW = 40
LINE_COLOR = (60, 255, 60)
TEXT_COLOR = (20, 20, 20)


def default_weights():
    from edge import artifacts

    manifest = artifacts.read_manifest(Path("models/players/football-player-detection-1.0.0.toml"))
    return artifacts.resolve(manifest)


def build_projection(video_path, frame_index, model):
    if model is None:
        return None, "px", {"reason": "sem peso de homografia"}

    frame = read_frame(video_path, frame_index)
    device = 0 if torch.cuda.is_available() else "cpu"
    points, confidence = detect_keypoints(model, frame, device)
    if points is None:
        return None, "px", {"reason": "nenhum keypoint de campo neste quadro"}

    matrix, valid = find_transformation(points, confidence)
    if matrix is None:
        return None, "px", {"reason": "menos de 4 keypoints confiaveis neste quadro"}

    def project(point):
        source = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        centimeters = cv2.perspectiveTransform(source, matrix).reshape(2)
        return float(centimeters[0]) / 100.0, float(centimeters[1]) / 100.0

    return project, "m", {"keypoints_validos": int(valid.sum()), "keypoints_totais": len(points)}


def frame_count(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"nao foi possivel abrir o video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return total


def read_frame(video_path, frame_index):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"nao foi possivel abrir o video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"nao foi possivel ler o quadro {frame_index} de {video_path}")
    return frame


def track_window(video_path, frame_index, window, tracker):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"nao foi possivel abrir o video: {video_path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    start = max(0, min(frame_index - window // 2, max(0, total - window)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    quadros = []
    amostras = {}
    tracker.start_clip()
    try:
        for offset in range(window):
            ok, frame = capture.read()
            if not ok:
                break
            indice = start + offset
            detections = tracker.track(frame)
            quadros.append(
                FramePerception(
                    frame_index=indice,
                    timestamp_s=indice / fps if fps > 0 else 0.0,
                    players=detections,
                )
            )
            for detection in detections:
                if detection.track_id is None:
                    continue
                cor = extract_shirt_color(frame, detection.box)
                if cor is None:
                    continue
                amostras.setdefault(detection.track_id, []).append(
                    (TrackObservation(frame_index=indice, detection=detection), cor)
                )
    finally:
        capture.release()

    if not quadros:
        raise ValueError(f"nenhum quadro lido a partir de {start} em {video_path}")
    alvo_local = min(max(frame_index - start, 0), len(quadros) - 1)
    return quadros, alvo_local, amostras


def teams_from_window(quadros, amostras):
    observacoes = {}
    for quadro in quadros:
        for detection in quadro.players:
            if detection.track_id is None:
                continue
            observacoes.setdefault(detection.track_id, []).append(
                TrackObservation(frame_index=quadro.frame_index, detection=detection)
            )

    elegiveis = {}
    for track_id, itens in amostras.items():
        papel, _ = stabilize_role(observacoes[track_id])
        if papel == "player":
            elegiveis[track_id] = itens

    return teams_from_color_samples(elegiveis, minimum=MINIMUM_SAMPLES)


def ensure_artifacts(video_path, output_dir, weights, device, reuse, image_size, tracker=None):
    tracking_path = output_dir / "detections.json"
    identities_path = output_dir / "identities.json"
    if reuse and tracking_path.is_file() and identities_path.is_file():
        print(f"reaproveitando artefatos em {output_dir}")
        return tracking_path, identities_path

    output_dir.mkdir(parents=True, exist_ok=True)
    if tracker is None:
        tracker = PlayerTracker(
            weights,
            image_size=image_size,
            confidence_threshold=CONFIDENCE,
            device=device,
        )
    print(f"rastreando {video_path.name} ...")
    process_video(video_path, output_dir, tracker)
    print("identificando equipes ...")
    process_identification(video_path, tracking_path, output_dir)
    return tracking_path, identities_path


def draw_marking(frame, frame_perception, teams, pairs, unit):
    """Desenha caixa por equipe e liga cada par com a distancia."""
    canvas = frame.copy()
    posicoes = {}
    for detection in frame_perception.players:
        if as_role(detection.role) != "player":
            continue
        team = teams.get(detection.track_id)
        color = TEAM_COLORS.get(team, UNKNOWN_COLOR)
        box = detection.box
        corners = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
        cv2.rectangle(canvas, corners[0], corners[1], color, 2)
        if detection.track_id is not None:
            posicoes[detection.track_id] = ground_point(box)
            cv2.putText(
                canvas,
                str(detection.track_id),
                (int(box.x1), int(box.y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    for pair in pairs:
        origem = posicoes.get(pair.team_a_key)
        destino = posicoes.get(pair.team_b_key)
        if origem is None or destino is None:
            continue
        p1 = (int(origem[0]), int(origem[1]))
        p2 = (int(destino[0]), int(destino[1]))
        cv2.line(canvas, p1, p2, LINE_COLOR, 2)
        meio = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        etiqueta = f"{pair.distance:.1f}{unit}"
        (largura, altura), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            canvas,
            (meio[0] - 3, meio[1] - altura - 4),
            (meio[0] + largura + 3, meio[1] + 4),
            LINE_COLOR,
            -1,
        )
        cv2.putText(canvas, etiqueta, meio, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)
    return canvas


class ClipeJanela(NamedTuple):
    """O que a saida precisa saber do recorte."""

    clip_id: str
    first_frame: int
    frame_count: int


def build_document(
    video_path,
    clip,
    frame_perception,
    teams,
    candidates,
    pairs,
    unit,
    radius,
    diagnostico,
    image_size,
    centroides,
):
    marcados = {k for pair in pairs for k in (pair.team_a_key, pair.team_b_key)}
    return {
        "schema_version": SCHEMA_VERSION,
        "clip": {
            "id": clip.clip_id,
            "path": str(video_path),
            "frame_index": frame_perception.frame_index,
            "timestamp_s": frame_perception.timestamp_s,
            "first_frame": clip.first_frame,
            "frame_count": clip.frame_count,
        },
        "config": {
            "radius": radius,
            "distance_unit": unit,
            "confidence_threshold": CONFIDENCE,
            "image_size": image_size,
            "field": diagnostico,
        },
        "teams": {
            nome: {"centroid_lab": [round(v, 1) for v in centroide]}
            for nome, centroide in sorted(centroides.items())
        },
        "players": [
            {
                "track_id": candidate.key,
                "team": candidate.team,
                "position": [round(candidate.position[0], 3), round(candidate.position[1], 3)],
                "marked": candidate.key in marcados,
            }
            for candidate in sorted(candidates, key=lambda c: c.key)
        ],
        "pairs": [
            {
                "team_a_track_id": pair.team_a_key,
                "team_b_track_id": pair.team_b_key,
                "distance": round(pair.distance, 3),
                "unit": unit,
            }
            for pair in pairs
        ],
    }


def write_json(path, document):
    """Escreve por arquivo temporario, para saida parcial nao virar artefato."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def process_clip(video_path, args, tracker, homografia, weights, *, em_lote=False):
    if args.output_dir:
        raiz = Path(args.output_dir)
        output_dir = raiz / video_path.stem if em_lote else raiz
    else:
        output_dir = OUTPUT_ROOT / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.window:
        frame_index = args.frame if args.frame is not None else frame_count(video_path) // 2
        print(f"rastreando {args.window} quadros em volta de {frame_index} ...")
        quadros, alvo, amostras = track_window(video_path, frame_index, args.window, tracker)
        teams, centroides = teams_from_window(quadros, amostras)
        frame_perception = quadros[alvo]
        frame_index = frame_perception.frame_index
        clip = ClipeJanela(video_path.stem, quadros[0].frame_index, len(quadros))
    else:
        tracking_path, identities_path = ensure_artifacts(
            video_path,
            output_dir,
            weights,
            args.device,
            not args.overwrite,
            args.image_size,
            tracker,
        )
        tracking = load_tracking(tracking_path)
        teams = teams_from_identities(json.loads(identities_path.read_text(encoding="utf-8")))
        centroides = {}

        frame_index = args.frame if args.frame is not None else len(tracking.frames) // 2
        if not 0 <= frame_index < len(tracking.frames):
            raise SystemExit(
                f"quadro {frame_index} fora do clipe, que tem {len(tracking.frames)} quadros"
            )
        frame_perception = tracking.frames[frame_index]
        clip = ClipeJanela(tracking.clip_id, 0, len(tracking.frames))

    project, unit, diagnostico = build_projection(video_path, frame_index, homografia)
    radius = (
        args.radius if args.radius is not None else (DEFAULT_RADIUS_M if unit == "m" else RADIUS_PX)
    )

    candidates = candidates_from_frame(frame_perception, teams, project=project)
    pairs = pair_markings(candidates, radius=radius)

    document = build_document(
        video_path,
        clip,
        frame_perception,
        teams,
        candidates,
        pairs,
        unit,
        radius,
        diagnostico,
        args.image_size,
        centroides,
    )
    base = output_dir / f"marcacao_frame{frame_index:05d}"
    write_json(base.with_suffix(".json"), document)

    frame = read_frame(video_path, frame_index)
    cv2.imwrite(
        str(base.with_suffix(".png")), draw_marking(frame, frame_perception, teams, pairs, unit)
    )

    print()
    fim = clip.first_frame + clip.frame_count
    print(f"quadro          : {frame_index} (janela {clip.first_frame}..{fim - 1})")
    print(f"unidade         : {unit}  ({diagnostico})")
    print(f"raio            : {radius} {unit}")
    print(f"jogadores       : {len(candidates)} com equipe resolvida")
    print(f"pares           : {len(pairs)}")
    for pair in pairs:
        print(
            f"  {pair.team_a_key:>3} (a) <-> {pair.team_b_key:>3} (b)   {pair.distance:.2f} {unit}"
        )
    print(f"json            : {base.with_suffix('.json')}")
    print(f"png             : {base.with_suffix('.png')}")
    return {
        "clipe": video_path.stem,
        "quadro": frame_index,
        "unidade": unit,
        "jogadores": len(candidates),
        "pares": len(pairs),
        "distancias": [round(pair.distance, 2) for pair in pairs],
    }


def resolve_inputs(entrada):
    caminho = Path(entrada)
    if caminho.is_dir():
        return sorted(caminho.glob("*.mp4"))
    return [caminho]


def run(args):
    videos = resolve_inputs(args.input)
    if not videos:
        raise SystemExit(f"nenhum .mp4 em {args.input}")
    for video in videos:
        if not video.is_file():
            raise SystemExit(f"clipe nao encontrado: {video}")

    weights = Path(args.weights) if args.weights else default_weights()
    tracker = PlayerTracker(
        weights,
        image_size=args.image_size,
        confidence_threshold=CONFIDENCE,
        device=args.device,
    )
    homography_path = args.homography if args.homography else get_homography_model_path()
    homografia = (
        load_model(Path(homography_path))
        if homography_path and Path(homography_path).exists()
        else None
    )
    if homografia is None:
        print("sem peso de homografia: a distancia sai em pixel")

    resumos = []
    for indice, video in enumerate(videos, start=1):
        if len(videos) > 1:
            print()
            print(f"[{indice}/{len(videos)}] {video.name}")
        try:
            resumos.append(
                process_clip(video, args, tracker, homografia, weights, em_lote=len(videos) > 1)
            )
        except Exception as erro:  # noqa: BLE001 - um clipe ruim nao pode custar o lote
            print(f"  FALHOU: {type(erro).__name__}: {erro}")
            resumos.append({"clipe": video.stem, "erro": f"{type(erro).__name__}: {erro}"})

    if len(videos) > 1:
        report_batch(resumos)
    return resumos


def report_batch(resumos):
    print()
    print("=" * 78)
    print(f"{'clipe':<38} {'quadro':>7} {'jog':>4} {'pares':>6}  distancias")
    print("-" * 78)
    for resumo in resumos:
        if "erro" in resumo:
            print(f"{resumo['clipe'][:38]:<38} {'-':>7} {'-':>4} {'-':>6}  {resumo['erro']}")
            continue
        dist = ", ".join(f"{d:.2f}{resumo['unidade']}" for d in resumo["distancias"]) or "-"
        print(
            f"{resumo['clipe'][:38]:<38} {resumo['quadro']:>7} {resumo['jogadores']:>4} "
            f"{resumo['pares']:>6}  {dist}"
        )
    ok = [r for r in resumos if "erro" not in r]
    falhas = len(resumos) - len(ok)
    print("-" * 78)
    if ok:
        total_pares = sum(r["pares"] for r in ok)
        em_pixel = sum(1 for r in ok if r["unidade"] == "px")
        print(f"{len(ok)} clipes medidos, {total_pares} pares no total", end="")
        if em_pixel:
            print(f", {em_pixel} sem homografia (distancia em pixel)", end="")
        print()
    if falhas:
        print(f"{falhas} clipes falharam")


def build_parser():
    parser = argparse.ArgumentParser(description="Detecta marcacao entre jogadores num quadro")
    parser.add_argument("input", help="clipe de video, ou diretorio com varios .mp4")
    parser.add_argument(
        "--frame", type=int, default=None, help="quadro a medir (padrao: o do meio)"
    )
    parser.add_argument("--weights", default=None, help="peso de deteccao (padrao: o do manifesto)")
    parser.add_argument("--homography", default=None, help="peso de keypoints do campo")
    parser.add_argument("--radius", type=float, default=None, help="raio maximo de marcacao")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--image-size", type=int, default=IMAGE_SIZE, help="lado da entrada do YOLO"
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs="?",
        const=DEFAULT_WINDOW,
        default=None,
        help=(
            "rastreia so N quadros em volta do alvo em vez do clipe inteiro "
            f"(padrao da flag: {DEFAULT_WINDOW}). Nao escreve detections.json nem identities.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="refaz tracking e identificacao")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
