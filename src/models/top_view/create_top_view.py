import argparse
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter

import cv2
import imageio_ffmpeg

# ruff: noqa: E402, I001

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
SRC_DIR = MODELS_DIR.parent
for import_root in (SRC_DIR, MODELS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from edge.modules.device import preferred_device
from edge.participants.tracking import PlayerTracker
from edge.participants.video import TEAM_COLORS, annotate_frame

from field_keypoints import load_model  # noqa: E402
from homography import (  # noqa: E402
    draw_homography,
    draw_keypoints,
    draw_top_view,
)
from top_view.pipeline import log, run_pipeline  # noqa: E402
from top_view.top_view_config import get_keypoint_model_path  # noqa: E402
from track_players.tracking_config import get_model_path  # noqa: E402

PLAYER_DEVICE = preferred_device()
DEVICE = PLAYER_DEVICE
IMGSZ = 960
CONFIDENCE = 0.25
IOU = 0.7
TRAIL_LENGTH = 35
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

PROJECT_DIR = SCRIPT_DIR
OUTPUT_DIR = PROJECT_DIR / "video_tracking"


def color_for_id(track_id):
    return (
        (37 * track_id + 79) % 256,
        (17 * track_id + 149) % 256,
        (29 * track_id + 211) % 256,
    )


def color_for_player(track_id, teams):
    return TEAM_COLORS.get(teams.get(track_id), color_for_id(track_id))


def create_writer(output_folder, video_path, fps, size):
    temporary_path = output_folder / f"{video_path.stem}_temporario.mp4"
    output_path = output_folder / f"{video_path.stem}_tracking_visao_superior.mp4"
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )

    if not writer.isOpened():
        raise RuntimeError("Não foi possível criar o vídeo temporário.")

    return writer, temporary_path, output_path


def convert_to_h264(temporary_path, output_path):
    log("Conversão H.264 iniciada")
    started = perf_counter()
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(temporary_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    temporary_path.unlink()
    elapsed = perf_counter() - started
    log(f"Conversão H.264 concluída — {elapsed:.2f}s")
    return elapsed


def process_video(
    tracker,
    video_path,
    input_dir,
    homography_model,
    *,
    output_dir=None,
):
    output_folder = (output_dir or OUTPUT_DIR) / video_path.relative_to(input_dir).parent
    total_started = perf_counter()
    log(f"Pipeline iniciada — vídeo: {video_path}")
    result = run_pipeline(
        tracker,
        video_path,
        output_folder / video_path.stem / "tracking",
        homography_model,
        DEVICE,
    )
    output_path, render_seconds, conversion_seconds = build_top_view_video(
        video_path, output_folder, result
    )
    total_seconds = perf_counter() - total_started
    log("RESUMO DE TEMPOS")
    log(f"  Modelo de jogadores (acum.) .. {result.timings['players_model']:.2f}s")
    log(f"  Modelo de pontos (acum.) ..... {result.timings['field_keypoints_model']:.2f}s")
    log(f"  Inferência paralela (parede) . {result.timings['parallel_inference']:.2f}s")
    log(f"  Identificação de equipes ..... {result.timings['team_identification']:.2f}s")
    log(f"  Cálculo da homografia ........ {result.timings['homography']:.2f}s")
    log(f"  Projeção e marcação .......... {result.timings['projection_marking']:.2f}s")
    log(f"  Renderização ................. {render_seconds:.2f}s")
    log(f"  Conversão H.264 .............. {conversion_seconds:.2f}s")
    log(f"  TOTAL ........................ {total_seconds:.2f}s")
    log(f"Arquivo gerado: {output_path}")
    return output_path


def build_top_view_video(video_path, output_folder, result):
    log("Renderização do top view iniciada")
    render_started = perf_counter()
    output_folder.mkdir(parents=True, exist_ok=True)
    tracking, teams = result.tracking, result.teams
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Não foi possível abrir: {video_path}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        top_view = draw_top_view(height, [], color_for_id)
        writer, temporary_path, output_path = create_writer(
            output_folder,
            video_path,
            fps,
            (width + top_view.shape[1], height),
        )
    except Exception:
        capture.release()
        raise

    trajectories = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    frame_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_number >= len(result.frames):
                raise ValueError("Video tem mais quadros que os resultados da pipeline")
            prepared = result.frames[frame_number]
            if prepared.frame_index != frame_number:
                raise ValueError("Resultados da pipeline fora de ordem")
            detections = tracking.frames[frame_number].players
            annotated = annotate_frame(frame, detections, trajectories, teams)
            field = prepared.field
            draw_homography(annotated, field.drawing_matrix, field.keypoints, field.confidence)
            draw_keypoints(annotated, field.keypoints, field.confidence, field.inliers)
            top_view = draw_top_view(
                height,
                prepared.positions,
                lambda track_id: color_for_player(track_id, teams),
                prepared.mirror,
            )
            writer.write(cv2.hconcat([annotated, top_view]))
            frame_number += 1

            if frame_number % 30 == 0 or frame_number == total_frames:
                progress = frame_number / total_frames * 100 if total_frames else 0
                log(f"Renderização — {frame_number}/{total_frames} frames ({progress:.1f}%)")
    finally:
        capture.release()
        writer.release()

    if frame_number != len(result.frames):
        raise ValueError("Video terminou antes dos resultados da pipeline")
    render_seconds = perf_counter() - render_started
    log(f"Renderização concluída — {frame_number} frames em {render_seconds:.2f}s")
    conversion_seconds = convert_to_h264(temporary_path, output_path)
    return output_path, render_seconds, conversion_seconds


def main():
    application_started = perf_counter()
    parser = argparse.ArgumentParser(description="Pipeline paralela e visao superior")
    parser.add_argument("input", help="Video ou pasta de videos")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()

    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Arquivo não é um vídeo suportado: {input_path}")
        input_dir = input_path.parent
        videos = [input_path]
    elif input_path.is_dir():
        input_dir = input_path
        videos = [
            path
            for path in input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and not path.is_relative_to(args.output_dir.resolve())
        ]
    else:
        raise FileNotFoundError(f"Vídeo ou pasta não encontrado: {input_path}")

    player_model_path = get_model_path()
    keypoint_model_path = get_keypoint_model_path()

    log(f"Dispositivo: {PLAYER_DEVICE}")
    log(f"Peso esperado de jogadores: {player_model_path}")
    log(f"Peso esperado de pontos do campo: {keypoint_model_path}")

    if not player_model_path.exists():
        log(f"FALHA — peso de jogadores não encontrado: {player_model_path}")
        raise FileNotFoundError(f"Modelo de jogadores não encontrado: {player_model_path}")

    if not keypoint_model_path.exists():
        log(f"FALHA — peso de pontos do campo não encontrado: {keypoint_model_path}")
        raise FileNotFoundError(f"Modelo de pontos do campo não encontrado: {keypoint_model_path}")

    if not videos:
        raise FileNotFoundError("Nenhum vídeo foi encontrado na pasta.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Vídeos encontrados: {len(videos)}")

    started = perf_counter()
    log("Carregando modelo de pontos do campo")
    keypoint_model = load_model(keypoint_model_path)
    log(f"Modelo de pontos carregado — {perf_counter() - started:.2f}s")

    for index, video_path in enumerate(sorted(videos), start=1):
        log(f"VÍDEO {index}/{len(videos)} — {video_path.name}")
        started = perf_counter()
        log("Carregando modelo de jogadores")
        tracker = PlayerTracker(
            player_model_path,
            image_size=IMGSZ,
            confidence_threshold=CONFIDENCE,
            device=PLAYER_DEVICE,
        )
        log(f"Modelo de jogadores carregado — {perf_counter() - started:.2f}s")
        try:
            process_video(
                tracker,
                video_path,
                input_dir,
                keypoint_model,
                output_dir=args.output_dir,
            )
        except Exception as error:
            log(
                f"FALHA no vídeo {video_path.name} após {perf_counter() - started:.2f}s — "
                f"{type(error).__name__}: {error}"
            )
            raise

    log(f"Todos os vídeos foram concluídos em {perf_counter() - application_started:.2f}s")
    log(f"Resultados em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
