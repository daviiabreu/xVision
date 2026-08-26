import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from tracking_config import MODEL_PATH


DEVICE = 0
IMGSZ = 960
CONFIDENCE = 0.25
IOU = 0.7
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "video_validation"


def process_video(model, video_path, input_dir):
    relative_folder = video_path.relative_to(input_dir).parent
    output_folder = OUTPUT_DIR / relative_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{video_path.stem}_deteccoes.mp4"
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

    frame_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONFIDENCE,
                iou=IOU,
                device=DEVICE,
                verbose=False,
            )[0]

            writer.write(result.plot(labels=False, conf=True, line_width=2))
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
            "Uso: python src/models/track_players/detect_players_in_videos.py "
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
    model = YOLO(str(MODEL_PATH))

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Vídeos encontrados: {len(videos)}")

    for index, video_path in enumerate(sorted(videos), start=1):
        print(f"\n[{index}/{len(videos)}] Processando: {video_path.name}")
        process_video(model, video_path, input_dir)

    print(f"\nResultados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
