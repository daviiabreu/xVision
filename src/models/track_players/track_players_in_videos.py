import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from tracking_config import MODEL_PATH


DEVICE = 0
IMGSZ = 960
CONFIDENCE = 0.25
IOU = 0.7
TRAIL_LENGTH = 35
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "video_tracking"


def color_for_id(track_id):
    return (
        (37 * track_id + 79) % 256,
        (17 * track_id + 149) % 256,
        (29 * track_id + 211) % 256,
    )


def process_video(model, video_path, input_dir):
    relative_folder = video_path.relative_to(input_dir).parent
    output_folder = OUTPUT_DIR / relative_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{video_path.stem}_tracking.mp4"
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Não foi possível criar: {output_path}")

    trajectories = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    frame_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                imgsz=IMGSZ,
                conf=CONFIDENCE,
                iou=IOU,
                device=DEVICE,
                verbose=False,
            )[0]

            annotated = frame.copy()

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    color = color_for_id(track_id)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    trajectories[track_id].append(center)

                    points = list(trajectories[track_id])
                    for point_a, point_b in zip(points, points[1:]):
                        cv2.line(annotated, point_a, point_b, color, 2)

            writer.write(annotated)
            frame_number += 1

            if frame_number % 30 == 0 or frame_number == total_frames:
                progress = frame_number / total_frames * 100 if total_frames else 0
                print(f"  {video_path.name}: {progress:.1f}%", end="\r", flush=True)
    finally:
        capture.release()
        writer.release()

    print(f"  {video_path.name}: concluído")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Uso: python src/models/track_players/track_players_in_videos.py "
            "caminho/para/videos"
        )

    input_dir = Path(sys.argv[1]).expanduser().resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Pasta não encontrada: {input_dir}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    videos = [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not path.is_relative_to(OUTPUT_DIR)
    ]

    if not videos:
        raise FileNotFoundError("Nenhum vídeo foi encontrado na pasta.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Vídeos encontrados: {len(videos)}")

    for index, video_path in enumerate(sorted(videos), start=1):
        print(f"\n[{index}/{len(videos)}] Rastreando: {video_path.name}")
        process_video(YOLO(str(MODEL_PATH)), video_path, input_dir)

    print(f"\nResultados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
