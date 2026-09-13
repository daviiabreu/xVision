import tempfile
import zipfile
from pathlib import Path

import numpy as np


def model_path(path):
    if path.is_file():
        return path

    checkpoint_dir = path / "best"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Modelo de pontos do campo não encontrado: {path}")

    temporary_path = Path(tempfile.gettempdir()) / "field_keypoints_best.pt"
    files = [file for file in checkpoint_dir.rglob("*") if file.is_file()]
    last_change = max(file.stat().st_mtime for file in files)

    if not temporary_path.exists() or temporary_path.stat().st_mtime < last_change:
        with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_STORED) as archive:
            for file in files:
                archive.write(file, file.relative_to(path))

    return temporary_path


def load_model(path):
    from ultralytics import YOLO

    return YOLO(str(model_path(path)))


def detect_keypoints(model, frame, device):
    result = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        device=device,
        verbose=False,
    )[0]

    if result.keypoints is None or len(result.keypoints) == 0 or result.keypoints.conf is None:
        return None, None

    index = 0
    if result.boxes is not None and len(result.boxes) > 1:
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        index = int(np.argmax(areas))

    return (
        result.keypoints.xy[index].cpu().numpy(),
        result.keypoints.conf[index].cpu().numpy(),
    )
