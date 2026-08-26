import os
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from tracking_config import (
    DATASET_DIR,
    MODEL_PATH,
    RUN_DIR,
    RUN_NAME,
    RUNS_DIR,
)
from ultralytics import YOLO

WORKSPACE = "roboflow-jvuqo"
PROJECT = "football-players-detection-3zvbc"
VERSION = 20
KEEP_CLASSES = {"player", "goalkeeper"}
DEVICE = 0
IMGSZ = 960
BATCH = 14
EPOCHS = 150
WORKERS = 8

PREPARED_MARKER = ".players-dataset-prepared"


def iter_label_paths(dataset_dir: Path):
    for split in ("train", "valid", "test"):
        labels_dir = dataset_dir / split / "labels"
        if labels_dir.exists():
            yield from labels_dir.glob("*.txt")


def dataset_is_prepared(dataset_dir: Path) -> bool:
    marker = dataset_dir / PREPARED_MARKER
    if not marker.exists():
        return False

    for label_path in iter_label_paths(dataset_dir):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) == 5 and values[0] != "0":
                return False

    return True


def prepare_dataset(dataset_dir: Path) -> None:
    if dataset_is_prepared(dataset_dir):
        return

    source_yaml = dataset_dir / "data.yaml"

    with source_yaml.open(encoding="utf-8") as file:
        source = yaml.safe_load(file)

    names = source["names"]

    if isinstance(names, dict):
        keep_ids = {
            int(class_id)
            for class_id, class_name in names.items()
            if class_name.lower() in KEEP_CLASSES
        }
    else:
        keep_ids = {
            index for index, class_name in enumerate(names) if class_name.lower() in KEEP_CLASSES
        }

    if not keep_ids:
        raise ValueError("Classes player/goalkeeper não encontradas.")

    for label_path in iter_label_paths(dataset_dir):
        novas_labels = []

        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            if class_id in keep_ids:
                novas_labels.append("0 " + " ".join(values[1:]))

        label_path.write_text(
            "\n".join(novas_labels) + ("\n" if novas_labels else ""),
            encoding="utf-8",
        )

    (dataset_dir / PREPARED_MARKER).write_text(
        "player,goalkeeper -> jogador\n",
        encoding="utf-8",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível. Verifique a instalação do PyTorch com "
            "suporte a CUDA e o driver NVIDIA."
        )

    gpu = torch.cuda.get_device_properties(DEVICE)

    print(f"GPU em uso: {torch.cuda.get_device_name(DEVICE)}")
    print(f"VRAM total: {gpu.total_memory / 1024**3:.1f} GB")
    print(f"CUDA disponível: {torch.version.cuda}")

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError("Crie um arquivo .env com: ROBOFLOW_API_KEY=sua_chave")

    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace(WORKSPACE).project(PROJECT)

    dataset = project.version(VERSION).download(
        "yolov8",
        location=str(DATASET_DIR),
    )

    dataset_dir = Path(dataset.location)

    prepare_dataset(dataset_dir)

    config = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": ["jogador"],
    }

    data_yaml = dataset_dir / "players_data.yaml"

    with data_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    model = YOLO("yolov8s.pt")

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        amp=True,
        cache="disk",
        patience=30,
        optimizer="auto",
        cos_lr=True,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    best_model = MODEL_PATH

    if not best_model.exists():
        raise FileNotFoundError(f"O treinamento terminou sem gerar o peso esperado: {best_model}")

    YOLO(str(best_model)).val(
        data=str(data_yaml),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUN_DIR),
        name="validation",
        exist_ok=True,
    )

    print(f"\nModelo treinado com sucesso: {best_model}")


if __name__ == "__main__":
    main()
