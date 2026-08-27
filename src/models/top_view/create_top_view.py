import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import torch

from xvision_players_yolo.tracking import PlayerTracker

MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODELS_DIR))

from top_view.homography_view import (  # noqa: E402
    detect_keypoints,
    draw_homography,
    draw_keypoints,
    draw_top_view,
    find_transformation,
    load_model,
    project_players,
    select_half,
    smooth_positions,
)
from top_view.top_view_config import get_homography_model_path  # noqa: E402
from track_players.tracking_config import get_model_path  # noqa: E402

DEVICE = 0 if torch.cuda.is_available() else "cpu"
PLAYER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = 960
CONFIDENCE = 0.25
IOU = 0.7
TRAIL_LENGTH = 35
MAX_FRAMES_WITHOUT_CALIBRATION = 15
MINIMUM_POINTS_TO_START = 7
MINIMUM_POINTS_TO_UPDATE = 6
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "video_tracking"


def color_for_id(track_id):
    return (
        (37 * track_id + 79) % 256,
        (17 * track_id + 149) % 256,
        (29 * track_id + 211) % 256,
    )


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


def process_video(tracker, video_path, input_dir, homography_model):
    relative_folder = video_path.relative_to(input_dir).parent
    output_folder = OUTPUT_DIR / relative_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {video_path}")

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

    trajectories = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    frame_number = 0
    matrix = None
    frames_without_calibration = 0
    mirror = None
    previous_positions = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated = frame.copy()
            boxes = []
            ids = []

            for detection in tracker.track(frame):
                if detection.track_id is None:
                    continue

                box = detection.box
                boxes.append([box.x1, box.y1, box.x2, box.y2])
                ids.append(detection.track_id)
                x1, y1, x2, y2 = map(int, boxes[-1])
                color = color_for_id(detection.track_id)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                trajectories[detection.track_id].append(center)

                points = list(trajectories[detection.track_id])
                for point_a, point_b in zip(points, points[1:], strict=False):
                    cv2.line(annotated, point_a, point_b, color, 2)

            keypoints, confidence = detect_keypoints(
                homography_model,
                frame,
                DEVICE,
            )
            current_matrix, inliers = find_transformation(keypoints, confidence)
            inlier_count = 0 if inliers is None else int(inliers.sum())

            if current_matrix is not None:
                if (matrix is None and inlier_count >= MINIMUM_POINTS_TO_START) or (
                    matrix is not None and inlier_count >= MINIMUM_POINTS_TO_UPDATE
                ):
                    matrix = current_matrix
                frames_without_calibration = 0
            else:
                frames_without_calibration += 1
                if frames_without_calibration > MAX_FRAMES_WITHOUT_CALIBRATION:
                    matrix = None

            draw_homography(annotated, matrix)
            draw_keypoints(annotated, keypoints, confidence, inliers)
            positions = project_players(matrix, boxes, ids)
            positions = smooth_positions(positions, previous_positions)
            mirror = select_half(positions, mirror)
            top_view = draw_top_view(height, positions, color_for_id, mirror)
            writer.write(cv2.hconcat([annotated, top_view]))
            frame_number += 1

            if frame_number % 30 == 0 or frame_number == total_frames:
                progress = frame_number / total_frames * 100 if total_frames else 0
                print(f"  {video_path.name}: {progress:.1f}%", end="\r", flush=True)
    finally:
        capture.release()
        writer.release()

    convert_to_h264(temporary_path, output_path)

    print(f"  {video_path.name}: concluído em {output_path.name}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Uso: python src/models/top_view/create_top_view.py "
            "caminho/para/video.mp4 ou caminho/para/pasta"
        )

    input_path = Path(sys.argv[1]).expanduser().resolve()

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
            and not path.is_relative_to(OUTPUT_DIR)
        ]
    else:
        raise FileNotFoundError(f"Vídeo ou pasta não encontrado: {input_path}")

    player_model_path = get_model_path()
    homography_model_path = get_homography_model_path()

    if not player_model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {player_model_path}")

    if not homography_model_path.exists():
        raise FileNotFoundError(f"Modelo de homografia não encontrado: {homography_model_path}")

    if not videos:
        raise FileNotFoundError("Nenhum vídeo foi encontrado na pasta.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    else:
        print("Processando na CPU")
    print(f"Vídeos encontrados: {len(videos)}")

    homography_model = load_model(homography_model_path)

    for index, video_path in enumerate(sorted(videos), start=1):
        print(f"\n[{index}/{len(videos)}] Rastreando: {video_path.name}")
        process_video(
            PlayerTracker(
                player_model_path,
                image_size=IMGSZ,
                confidence_threshold=CONFIDENCE,
                device=PLAYER_DEVICE,
            ),
            video_path,
            input_dir,
            homography_model,
        )

    print(f"\nResultados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
