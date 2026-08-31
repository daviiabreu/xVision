"""
Pipeline de detecção de "área do chute lotada" (crowded) — Abordagem A.

1. Frame extractor: pega o frame exato marcado como finalização (via
   scripts/anotar_clips.py) — scripts/cortes/anotacoes.csv.
2. Detector: modelo real de futebol (var/models/players/football-player-detection.pt,
   treinado pelo time — classes bola/goleiro/jogador/juiz), não mais um YOLO
   genérico de COCO. Juiz é descartado pelo próprio papel da detecção, sem
   precisar de heurística de cor.
3. Âncora: a BOLA DETECTADA de verdade, quando o modelo a encontra no frame.
   Só cai no fallback (par de jogadores de times diferentes mais próximo
   entre si, via cor do torso) quando a bola não aparece nessa detecção —
   ainda não temos homografia de campo, então "perto" continua sendo
   pixel, não metro real.
4. Crowdedness scorer: raio em pixels escalado pela altura mediana das
   bounding boxes do frame (compensa zoom da câmera variar entre clipes);
   conta jogadores dentro do raio ao redor da âncora.

Uso:
    python scripts/crowded_pipeline.py                  # calibra + valida contra anotacoes.csv
    python scripts/crowded_pipeline.py --radius-mult 2.5 --threshold 4   # roda com params fixos

Requer o peso do modelo em var/models/players/football-player-detection.pt
(não vem no git — ver models/players/football-player-detection-1.0.0.toml).

Lê:
  - scripts/cortes/anotacoes.csv       (gabarito: shot_frame_index, crowded_manual)
  - scripts/cortes/cortes_metadados.csv (pra achar o file_name/match_id de cada cut_id)
Escreve:
  - scripts/cortes/crowded_validacao.csv (predição x gabarito, por clip)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ultralytics import YOLO

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from models.top_view import homography_view  # noqa: E402 — precisa do sys.path acima

CORTES_DIR = Path(__file__).parent / "cortes"
METADATA_CSV = CORTES_DIR / "cortes_metadados.csv"
ANNOTATIONS_CSV = CORTES_DIR / "anotacoes.csv"
OUTPUT_CSV = CORTES_DIR / "crowded_validacao.csv"

PLAYER_MODEL_PATH = REPO_ROOT / "var" / "models" / "players" / "football-player-detection.pt"
FIELD_MODEL_PATH = REPO_ROOT / "var" / "models" / "field" / "pitch-keypoints-yolo11n-pose.pt"
BALL_MODEL_PATH = REPO_ROOT / "var" / "models" / "ball" / "ball_yolov8n.pt"
PLAYER_ROLES = ("player", "goalkeeper")  # entram na contagem de lotação
BALL_ROLE = "ball"
OUTLIER_DISTANCE_PERCENTILE = 90  # fallback sem bola: acima disso, jogador vira "outlier"

# RADIUS_M_DEFAULT / THRESHOLD_DEFAULT: saída de `--calibrate` contra as 32
# anotações manuais, com a pilha final de modelos (bola dedicada -> bola do
# combinado -> par por cor; sempre metros reais quando a homografia resolve).
# 59.4% de acurácia nessa amostra. Um valor de threshold=1 escolhido à mão
# pareceria pouco discriminativo, mas testar isso empiricamente (a tentação de
# "corrigir" pra um valor mais redondo, tipo radius_m=3/threshold=2) piorou o
# resultado pra 50% — a amostra é pequena, mas ainda assim é o dado que temos,
# não uma intuição substituindo ele. Recalibrar quando a amostra anotada crescer
# (ver seção de próximos passos).
RADIUS_M_DEFAULT = 6.5  # raio real (metros) quando a homografia resolve
THRESHOLD_DEFAULT = 1  # jogadores mínimos no raio pra contar como crowded
RADIUS_MULT_PX_FALLBACK = 3.0  # raio em pixels (× altura mediana) quando NÃO há homografia

# Mediana de (duration_seconds - shot_frame_seconds) nas 32 anotações manuais —
# proxy pro momento do chute nos clipes sem marcação manual de frame. Tem desvio
# de ~0.8s entre clipes, então é impreciso; serve pra rodar em escala, não pra
# recalibrar os thresholds (isso já foi feito com o frame exato anotado).
DEFAULT_SHOT_OFFSET_FROM_END = 3.08
FULL_OUTPUT_CSV = CORTES_DIR / "crowded_metadados.csv"


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def load_frame(mp4_path: Path, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_football(model: YOLO, frame: np.ndarray) -> list[tuple[Detection, str, float]]:
    """Roda o detector real e devolve (caixa, papel, confiança) por objeto —
    papel é um de 'ball' / 'goalkeeper' / 'player' / 'referee'."""
    results = model.predict(frame, verbose=False)[0]
    names = results.names
    return [
        (Detection(*box.xyxy[0].tolist()), names[int(box.cls[0])], float(box.conf[0]))
        for box in results.boxes
    ]


def best_ball(detections: list[tuple[Detection, str, float]]) -> Detection | None:
    balls = [(d, conf) for d, role, conf in detections if role == BALL_ROLE]
    if not balls:
        return None
    return max(balls, key=lambda pair: pair[1])[0]


def detect_ball_dedicated(model: YOLO, frame: np.ndarray) -> Detection | None:
    """Detector de bola dedicado (single-class, 1280px) — mais sensível pra bola
    pequena/borrada do que a classe 'ball' do modelo combinado de jogadores."""
    results = model.predict(frame, verbose=False)[0]
    if len(results.boxes) == 0:
        return None
    best_idx = int(results.boxes.conf.argmax())
    return Detection(*results.boxes.xyxy[best_idx].tolist())


def find_ball(
    detections: list[tuple[Detection, str, float]], ball_model: YOLO | None, frame: np.ndarray,
) -> Detection | None:
    """Prioriza o detector dedicado de bola; cai pra classe 'ball' do modelo
    combinado só se o dedicado (ou a ausência dele) não achar nada."""
    if ball_model is not None:
        dedicated = detect_ball_dedicated(ball_model, frame)
        if dedicated is not None:
            return dedicated
    return best_ball(detections)


def torso_dominant_color(frame: np.ndarray, d: Detection) -> np.ndarray | None:
    x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
    mid_y = y1 + (y2 - y1) // 2
    crop = frame[y1:mid_y, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    return np.median(hsv, axis=0)


def classify_teams(colors: list[np.ndarray]) -> list[str]:
    """Retorna, por jogador: 'A', 'B' ou 'outlier'. Precisa de >= 2 jogadores."""
    x = np.array(colors)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(x)
    dists = np.linalg.norm(x - km.cluster_centers_[km.labels_], axis=1)
    cutoff = np.percentile(dists, OUTLIER_DISTANCE_PERCENTILE)
    labels = []
    for label, dist in zip(km.labels_, dists):
        if dist > cutoff and dist > np.median(dists) * 2:
            labels.append("outlier")
        else:
            labels.append("A" if label == 0 else "B")
    return labels


def nearest_pair_midpoint(points: list[tuple[float, float]], teams: list[str]) -> tuple[float, float] | None:
    """Ponto médio do par de times diferentes mais próximo entre si — funciona em
    qualquer espaço de coordenadas (pixel ou campo real), desde que `points` já
    esteja nesse espaço."""
    team_a = [p for p, t in zip(points, teams) if t == "A"]
    team_b = [p for p, t in zip(points, teams) if t == "B"]
    if not team_a or not team_b:
        return None
    best_pair, best_dist = None, float("inf")
    for a in team_a:
        for b in team_b:
            dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_pair = dist, (a, b)
    a, b = best_pair
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def count_in_radius(dets: list[Detection], anchor: tuple[float, float], radius: float) -> int:
    ax, ay = anchor
    count = 0
    for d in dets:
        cx, cy = d.center
        if ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 <= radius:
            count += 1
    return count


def load_field_model() -> YOLO:
    if not FIELD_MODEL_PATH.exists():
        raise SystemExit(
            f"Peso não encontrado em {FIELD_MODEL_PATH}.\n"
            "Baixe o modelo (ver models/field/pitch-keypoints-yolo11n-pose-0.1.0.toml) "
            "e salve nesse caminho antes de rodar o pipeline."
        )
    return homography_view.load_model(FIELD_MODEL_PATH)


def compute_homography(field_model: YOLO, frame: np.ndarray) -> np.ndarray | None:
    points, confidence = homography_view.detect_keypoints(field_model, frame, device="cpu")
    if points is None:
        return None
    matrix, _valid = homography_view.find_transformation(points, confidence)
    return matrix


def project_to_pitch(matrix: np.ndarray | None, dets: list[Detection]) -> list[tuple[float, float] | None]:
    """Projeta o pé (base) de cada caixa pro campo real, em centímetros —
    None por objeto que não deu pra projetar."""
    if matrix is None or not dets:
        return [None] * len(dets)
    boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
    ids = list(range(len(dets)))
    projected = homography_view.project_players(matrix, boxes, ids)
    result: list[tuple[float, float] | None] = [None] * len(dets)
    for position, idx in projected:
        result[idx] = (float(position[0]), float(position[1]))
    return result


def _classify_by_color(frame: np.ndarray, players: list[Detection]) -> tuple[list[int], list[str]] | None:
    """Cor do torso + K-means, só nos jogadores com cor extraível. Devolve os
    índices (em `players`) que entraram na classificação e o time de cada um —
    None se não sobrou gente suficiente pra separar em times."""
    colors = [torso_dominant_color(frame, d) for d in players]
    valid = [(i, c) for i, c in enumerate(colors) if c is not None]
    if len(valid) < 2:
        return None
    idxs, valid_colors = zip(*valid)
    teams = classify_teams(list(valid_colors))
    return list(idxs), teams


@dataclass
class ClipAnalysis:
    dets: list[Detection]
    num_players: int
    anchor_source: str  # "bola" ou "par_mais_proximo (fallback)"
    anchor_px: tuple[float, float]  # sempre disponível, em pixels do frame — pra desenhar
    median_height: float  # só usado quando NÃO há homografia (raio em pixels)
    has_homography: bool
    player_positions: list[tuple[float, float] | None]  # cm reais de campo, alinhado com `dets`; None sem homografia
    anchor_pitch: tuple[float, float] | None  # cm reais de campo, quando calculável


def analyze_clip(
    player_model: YOLO, field_model: YOLO, mp4_path: Path, frame_index: int,
    prefer_ball: bool = True, ball_model: YOLO | None = None,
) -> ClipAnalysis | str:
    """Roda a detecção real, tenta a homografia de campo, e acha a âncora (bola,
    ou o par mais próximo entre times). `prefer_ball=False` força sempre o par
    mais próximo, mesmo quando a bola foi detectada — é a heurística original
    (a que deu 81% com detector genérico) rodando nos detectores e nas
    coordenadas reais de campo que temos agora. `ball_model`, se passado, é o
    detector dedicado de bola — mais sensível que a classe 'ball' do modelo
    combinado, usado como primeira tentativa. Devolve o motivo (str) se algo
    não deu pra calcular, ou o resultado (ClipAnalysis)."""
    frame = load_frame(mp4_path, frame_index)
    if frame is None:
        return "frame não encontrado"

    detections = detect_football(player_model, frame)
    players = [d for d, role, _ in detections if role in PLAYER_ROLES]
    if len(players) < 2:
        return f"poucos jogadores detectados ({len(players)})"

    matrix = compute_homography(field_model, frame)
    player_positions = project_to_pitch(matrix, players)

    ball = find_ball(detections, ball_model, frame) if prefer_ball else None
    if ball is not None:
        anchor_source = "bola"
        anchor_px = ball.center
        anchor_pitch = project_to_pitch(matrix, [ball])[0] if matrix is not None else None
    else:
        classified = _classify_by_color(frame, players)
        if classified is None:
            return "sem bola detectada, e sem cor suficiente pro fallback"
        idxs, teams = classified
        anchor_px = nearest_pair_midpoint([players[i].center for i in idxs], teams)
        if anchor_px is None:
            return "sem bola detectada, e sem par entre times pro fallback"
        anchor_source = "par_mais_proximo (fallback)"

        anchor_pitch = None
        if matrix is not None:
            pitch_pairs = [(player_positions[i], t) for i, t in zip(idxs, teams) if player_positions[i] is not None]
            if len(pitch_pairs) >= 2:
                pts, tms = zip(*pitch_pairs)
                anchor_pitch = nearest_pair_midpoint(list(pts), list(tms))

    return ClipAnalysis(
        dets=players, num_players=len(players), anchor_source=anchor_source,
        anchor_px=anchor_px, median_height=float(np.median([d.height for d in players])),
        has_homography=matrix is not None, player_positions=player_positions, anchor_pitch=anchor_pitch,
    )


def evaluate_crowded(
    analysis: ClipAnalysis, radius_m: float = RADIUS_M_DEFAULT,
    radius_mult_px: float = RADIUS_MULT_PX_FALLBACK, threshold: int = THRESHOLD_DEFAULT,
) -> tuple[int, bool, str, float]:
    """Conta jogadores no raio e decide crowded. Usa metros reais quando a
    âncora tem posição de campo resolvida; cai pro raio em pixels senão.
    Devolve (contagem, crowded, unidade, raio_usado)."""
    if analysis.anchor_pitch is not None:
        radius_cm = radius_m * 100
        ax, ay = analysis.anchor_pitch
        count = sum(
            1 for pos in analysis.player_positions
            if pos is not None and ((pos[0] - ax) ** 2 + (pos[1] - ay) ** 2) ** 0.5 <= radius_cm
        )
        return count, count >= threshold, "metros_reais", radius_m

    radius_px = analysis.median_height * radius_mult_px
    count = count_in_radius(analysis.dets, analysis.anchor_px, radius_px)
    return count, count >= threshold, "pixels_fallback", radius_px


def goal_center_for(ball_x_cm: float) -> tuple[float, float]:
    """O gol-alvo é o mais próximo da bola — numa finalização, o chute quase
    sempre sai de dentro ou perto do próprio campo de ataque, então o lado do
    campo onde a bola está aponta pro gol que está sendo atacado."""
    x = 0.0 if ball_x_cm < homography_view.HALF_LENGTH else float(homography_view.PITCH_LENGTH)
    return (x, homography_view.PITCH_WIDTH / 2)


def count_in_corridor(
    shooter: tuple[float, float], goal: tuple[float, float],
    positions: list[tuple[float, float] | None], half_width_cm: float,
) -> int:
    """Conta jogadores cuja posição projetada na linha chutador→gol cai ENTRE
    os dois pontos (nem atrás do chutador, nem depois do gol) e dentro da
    faixa lateral — ou seja, quem está no caminho do chute."""
    sx, sy = shooter
    gx, gy = goal
    dx, dy = gx - sx, gy - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return 0
    count = 0
    for pos in positions:
        if pos is None:
            continue
        px, py = pos
        t = ((px - sx) * dx + (py - sy) * dy) / length_sq
        if t <= 0 or t >= 1:
            continue
        closest_x, closest_y = sx + t * dx, sy + t * dy
        dist = ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5
        if dist <= half_width_cm:
            count += 1
    return count


def evaluate_crowded_corridor(
    analysis: ClipAnalysis, half_width_m: float, threshold: int,
) -> tuple[int, bool] | None:
    """Versão 'corredor': só funciona quando a bola foi detectada de verdade
    E a homografia resolveu — sem os dois, não tem como saber a direção do
    chute. Devolve None quando não dá pra calcular."""
    if analysis.anchor_source != "bola" or analysis.anchor_pitch is None:
        return None
    goal = goal_center_for(analysis.anchor_pitch[0])
    count = count_in_corridor(analysis.anchor_pitch, goal, analysis.player_positions, half_width_m * 100)
    return count, count >= threshold


def calibrate_corridor(clips: pd.DataFrame, analyses: dict[str, ClipAnalysis]) -> tuple[float, int, float, pd.DataFrame]:
    best = (None, None, -1.0, None)
    for half_width_m in np.arange(0.2, 4.1, 0.2):
        for threshold in range(1, 6):
            rows = []
            for _, row in clips.iterrows():
                cut_id = row["cut_id"]
                analysis = analyses.get(cut_id)
                if analysis is None:
                    continue
                result = evaluate_crowded_corridor(analysis, half_width_m, threshold)
                if result is None:
                    continue
                count, crowded = result
                rows.append({
                    "cut_id": cut_id, "count": count, "crowded_pred": crowded,
                    "crowded_manual": row["crowded_manual"],
                })
            if not rows:
                continue
            out = pd.DataFrame(rows)
            acc = float((out["crowded_pred"] == out["crowded_manual"]).mean())
            if acc > best[2]:
                out["half_width_m"] = half_width_m
                out["threshold"] = threshold
                best = (half_width_m, threshold, acc, out)
    return best


@dataclass
class BothAnchors:
    """As duas âncoras calculadas lado a lado, pra inspeção visual — analyze_clip()
    só devolve a que valeria pra decisão final; isto aqui é só pra comparar."""
    dets: list[Detection]
    median_height: float
    has_homography: bool
    player_positions: list[tuple[float, float] | None]
    ball_anchor_px: tuple[float, float] | None
    ball_anchor_pitch: tuple[float, float] | None
    fallback_anchor_px: tuple[float, float] | None
    fallback_anchor_pitch: tuple[float, float] | None


def analyze_both_anchors(
    player_model: YOLO, field_model: YOLO, mp4_path: Path, frame_index: int, ball_model: YOLO | None = None,
) -> tuple[np.ndarray, BothAnchors] | str:
    frame = load_frame(mp4_path, frame_index)
    if frame is None:
        return "frame não encontrado"

    detections = detect_football(player_model, frame)
    players = [d for d, role, _ in detections if role in PLAYER_ROLES]
    if len(players) < 2:
        return f"poucos jogadores detectados ({len(players)})"

    matrix = compute_homography(field_model, frame)
    player_positions = project_to_pitch(matrix, players)

    ball = find_ball(detections, ball_model, frame)
    ball_anchor_px = ball.center if ball is not None else None
    ball_anchor_pitch = (
        project_to_pitch(matrix, [ball])[0] if (ball is not None and matrix is not None) else None
    )

    fallback_anchor_px = None
    fallback_anchor_pitch = None
    classified = _classify_by_color(frame, players)
    if classified is not None:
        idxs, teams = classified
        fallback_anchor_px = nearest_pair_midpoint([players[i].center for i in idxs], teams)
        if matrix is not None:
            pitch_pairs = [(player_positions[i], t) for i, t in zip(idxs, teams) if player_positions[i] is not None]
            if len(pitch_pairs) >= 2:
                pts, tms = zip(*pitch_pairs)
                fallback_anchor_pitch = nearest_pair_midpoint(list(pts), list(tms))

    if ball_anchor_px is None and fallback_anchor_px is None:
        return "nenhuma âncora calculável (sem bola e sem par entre times)"

    analysis = BothAnchors(
        dets=players, median_height=float(np.median([d.height for d in players])),
        has_homography=matrix is not None, player_positions=player_positions,
        ball_anchor_px=ball_anchor_px, ball_anchor_pitch=ball_anchor_pitch,
        fallback_anchor_px=fallback_anchor_px, fallback_anchor_pitch=fallback_anchor_pitch,
    )
    return frame, analysis


def load_annotated_clips() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_CSV, dtype={"cut_id": str, "match_id": str})
    ann = pd.read_csv(ANNOTATIONS_CSV, dtype={"cut_id": str, "match_id": str})
    ann = ann[ann["status"] == "done"].copy()
    ann["shot_frame_index"] = ann["shot_frame_index"].astype(float).astype(int)
    ann["crowded_manual"] = ann["crowded_manual"].map({"True": True, "False": False, True: True, False: False})
    meta_cols = ["cut_id", "match_id", "file_name", "home", "away", "minute", "result_label"]
    merged = ann.merge(meta[meta_cols], on=["cut_id", "match_id"])
    return merged


def estimate_shot_frame(mp4_path: Path, offset_from_end: float) -> tuple[int, float]:
    """Sem anotação manual: aproxima o frame do chute como
    `offset_from_end` segundos antes do fim do clip. Devolve (frame_index, fps)."""
    cap = cv2.VideoCapture(str(mp4_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    target = frame_count - round(offset_from_end * fps)
    return max(0, min(target, frame_count - 1)), fps


def load_player_model() -> YOLO:
    if not PLAYER_MODEL_PATH.exists():
        raise SystemExit(
            f"Peso não encontrado em {PLAYER_MODEL_PATH}.\n"
            "Baixe o modelo (ver models/players/football-player-detection-1.0.0.toml) "
            "e salve nesse caminho antes de rodar o pipeline."
        )
    return YOLO(str(PLAYER_MODEL_PATH))


def load_ball_model() -> YOLO | None:
    """None se o peso não estiver presente — find_ball() cai pro modelo
    combinado nesse caso, então isto não é fatal como os outros load_*."""
    if not BALL_MODEL_PATH.exists():
        return None
    return YOLO(str(BALL_MODEL_PATH))


def run_full_dataset(
    player_model: YOLO, field_model: YOLO, radius_m: float, threshold: int,
    ball_model: YOLO | None = None, limit: int | None = None,
) -> None:
    meta = pd.read_csv(METADATA_CSV, dtype={"cut_id": str, "match_id": str})
    meta = meta[meta["clip_url"].notna() & (meta["clip_url"] != "")]
    if limit is not None:
        meta = meta.head(limit)
    print(f"Rodando em {len(meta)} clipes (offset={DEFAULT_SHOT_OFFSET_FROM_END}s, "
          f"radius_m={radius_m}, threshold={threshold})", flush=True)

    rows = []
    for i, (_, row) in enumerate(meta.iterrows(), 1):
        mp4_path = CORTES_DIR / row["match_id"] / row["file_name"]
        out = {"cut_id": row["cut_id"], "match_id": row["match_id"]}
        if not mp4_path.exists():
            out.update(crowded=None, skip_reason="arquivo ausente")
            rows.append(out)
            continue

        frame_index, _ = estimate_shot_frame(mp4_path, DEFAULT_SHOT_OFFSET_FROM_END)
        analysis = analyze_clip(player_model, field_model, mp4_path, frame_index, ball_model=ball_model)
        if isinstance(analysis, str):
            out.update(crowded=None, skip_reason=analysis)
            rows.append(out)
            continue

        count, crowded, unit, radius_used = evaluate_crowded(analysis, radius_m=radius_m, threshold=threshold)
        out.update(
            num_players=analysis.num_players, num_players_in_radius=count, crowded=crowded,
            unit=unit, radius=radius_used, anchor_source=analysis.anchor_source,
            has_homography=analysis.has_homography, skip_reason=None,
        )
        rows.append(out)
        if i % 10 == 0:
            print(f"  {i}/{len(meta)}...", flush=True)
            pd.DataFrame(rows).to_csv(FULL_OUTPUT_CSV, index=False)  # checkpoint parcial, sobrevive a um kill

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FULL_OUTPUT_CSV, index=False)
    n_skipped = out_df["skip_reason"].notna().sum()
    n_crowded = out_df["crowded"].sum()
    n_homography = out_df["has_homography"].sum() if "has_homography" in out_df else 0
    print(f"\n{n_skipped}/{len(out_df)} pulados; dos válidos, {n_crowded} marcados crowded, "
          f"{n_homography} com homografia real")
    print(f"Salvo em {FULL_OUTPUT_CSV}")

    for match_id, group in out_df.groupby("match_id"):
        group.to_csv(CORTES_DIR / match_id / "crowded_metadados.csv", index=False)


def calibrate(clips: pd.DataFrame, analyses: dict[str, ClipAnalysis]) -> tuple[float, int, float, pd.DataFrame]:
    best = (None, None, -1.0, None)
    for radius_m in np.arange(0.5, 12.0, 0.5):
        for threshold in range(1, 10):
            rows = []
            for _, row in clips.iterrows():
                cut_id = row["cut_id"]
                if cut_id not in analyses:
                    continue
                count, crowded, unit, radius_used = evaluate_crowded(
                    analyses[cut_id], radius_m=radius_m, threshold=threshold,
                )
                rows.append({
                    "cut_id": cut_id, "crowded_pred": crowded, "crowded_manual": row["crowded_manual"],
                    "count": count, "unit": unit, "radius": radius_used,
                    "anchor_source": analyses[cut_id].anchor_source,
                })
            out = pd.DataFrame(rows)
            acc = float((out["crowded_pred"] == out["crowded_manual"]).mean())
            if acc > best[2]:
                out["radius_m"] = radius_m
                out["threshold"] = threshold
                best = (radius_m, threshold, acc, out)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radius-m", type=float, default=RADIUS_M_DEFAULT,
                         help=f"Raio real em metros (padrão: {RADIUS_M_DEFAULT}, fixado à mão — ver comentário "
                              "em THRESHOLD_DEFAULT no código)")
    parser.add_argument("--threshold", type=int, default=THRESHOLD_DEFAULT,
                         help=f"Threshold de contagem (padrão: {THRESHOLD_DEFAULT})")
    parser.add_argument("--calibrate", action="store_true",
                         help="Busca em grade radius_m x threshold contra o gabarito, em vez de usar os padrões "
                              "fixos (instável com só 32 anotações — ver comentário no código)")
    parser.add_argument("--full", action="store_true",
                         help="Roda em todos os clipes de cortes_metadados.csv, com --radius-m/--threshold")
    parser.add_argument("--limit", type=int, help="Com --full: processa só os N primeiros clipes (teste rápido)")
    parser.add_argument("--corridor", action="store_true",
                         help="Usa a definição de corredor bola->gol em vez do raio ao redor da âncora "
                              "(só vale nos clipes com bola detectada + homografia)")
    parser.add_argument("--half-width-m", type=float,
                         help="Com --corridor: pula a calibração e usa esta largura lateral (metros)")
    parser.add_argument("--pair-only", action="store_true",
                         help="Ignora a bola detectada e sempre usa o par mais próximo entre times como "
                              "âncora — a heurística original, rodando nos detectores e coordenadas reais")
    args = parser.parse_args()

    if args.full:
        run_full_dataset(
            load_player_model(), load_field_model(), args.radius_m, args.threshold,
            ball_model=load_ball_model(), limit=args.limit,
        )
        return

    clips = load_annotated_clips()
    print(f"{len(clips)} clipes anotados pra validar")

    player_model = load_player_model()
    field_model = load_field_model()
    ball_model = load_ball_model()
    print(f"Modelo dedicado de bola: {'carregado' if ball_model is not None else 'ausente (' + str(BALL_MODEL_PATH) + ')'}")

    analyses: dict[str, ClipAnalysis] = {}
    skips: dict[str, str] = {}
    for _, row in clips.iterrows():
        mp4_path = CORTES_DIR / row["match_id"] / row["file_name"]
        analysis = analyze_clip(
            player_model, field_model, mp4_path, row["shot_frame_index"],
            prefer_ball=not args.pair_only, ball_model=ball_model,
        )
        if isinstance(analysis, str):
            skips[row["cut_id"]] = analysis
            continue
        analyses[row["cut_id"]] = analysis
        print(f"{row['cut_id']}: ok ({analysis.num_players} jogadores, âncora={analysis.anchor_source}, "
              f"homografia={'sim' if analysis.has_homography else 'não'})")

    print(f"\n{len(skips)}/{len(clips)} clipes pulados (ver skip_reason)")

    if args.corridor:
        if args.half_width_m is not None and args.threshold is not None:
            rows = []
            for _, row in clips.iterrows():
                cut_id = row["cut_id"]
                analysis = analyses.get(cut_id)
                if analysis is None:
                    continue
                result = evaluate_crowded_corridor(analysis, args.half_width_m, args.threshold)
                if result is None:
                    continue
                count, crowded = result
                rows.append({
                    "cut_id": cut_id, "crowded_pred": crowded, "crowded_manual": row["crowded_manual"],
                    "count": count, "half_width_m": args.half_width_m, "threshold": args.threshold,
                })
            out = pd.DataFrame(rows)
            acc = float((out["crowded_pred"] == out["crowded_manual"]).mean())
        else:
            half_width_m, threshold, acc, out = calibrate_corridor(clips, analyses)
            print(f"\nMelhor combinação: half_width_m={half_width_m}, threshold={threshold}")

        n_sem_bola = len(analyses) - len(out)
        print(f"Acurácia (corredor): {acc:.1%} ({len(out)} clipes válidos, "
              f"{n_sem_bola} sem bola detectada nesse frame)")
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"Detalhes salvos em {OUTPUT_CSV}")
        return

    if args.calibrate:
        radius_m, threshold, acc, out = calibrate(clips, analyses)
        print(f"\nMelhor combinação: radius_m={radius_m}, threshold={threshold}")
    else:
        rows = []
        for _, row in clips.iterrows():
            cut_id = row["cut_id"]
            if cut_id not in analyses:
                continue
            count, crowded, unit, radius_used = evaluate_crowded(
                analyses[cut_id], radius_m=args.radius_m, threshold=args.threshold,
            )
            rows.append({
                "cut_id": cut_id, "crowded_pred": crowded, "crowded_manual": row["crowded_manual"],
                "count": count, "unit": unit, "radius": radius_used,
                "radius_m": args.radius_m, "threshold": args.threshold,
                "anchor_source": analyses[cut_id].anchor_source,
            })
        out = pd.DataFrame(rows)
        acc = float((out["crowded_pred"] == out["crowded_manual"]).mean())

    print(f"Acurácia: {acc:.1%} ({len(out)} clipes válidos)")
    print(out["unit"].value_counts().to_string())
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Detalhes salvos em {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
