import random

import torch
from ultralytics import YOLO

from tracking_config import DATA_YAML, MODEL_PATH


DEVICE = 0
IMGSZ = 960
BATCH = 14
SAMPLE_COUNT = 12
CONFIDENCE = 0.25


def find_best_model():
    if MODEL_PATH.exists():
        return MODEL_PATH

    raise FileNotFoundError(f"Não encontrei o modelo esperado: {MODEL_PATH}")


def get_metrics(result):
    return {
        "precision": result.box.mp,
        "recall": result.box.mr,
        "map50": result.box.map50,
        "map50_95": result.box.map,
    }


def evaluate(model, split, analysis_dir):
    result = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        plots=True,
        project=str(analysis_dir),
        name=split,
        exist_ok=True,
    )
    return get_metrics(result)


def save_predictions(model, analysis_dir):
    images_dir = DATA_YAML.parent / "valid" / "images"
    images = list(images_dir.glob("*.jpg"))

    if not images:
        raise FileNotFoundError(f"Não encontrei imagens em {images_dir}.")

    samples = random.sample(images, min(SAMPLE_COUNT, len(images)))
    model.predict(
        source=[str(image) for image in samples],
        imgsz=IMGSZ,
        conf=CONFIDENCE,
        device=DEVICE,
        save=True,
        show_labels=False,
        project=str(analysis_dir),
        name="predictions",
        exist_ok=True,
    )


def save_report(validation, test, analysis_dir):
    report = f"""ANÁLISE DO MODELO DE JOGADORES

Validação
---------
Precision: {validation['precision']:.4f}
Recall:    {validation['recall']:.4f}
mAP@50:    {validation['map50']:.4f}
mAP@50-95: {validation['map50_95']:.4f}

Teste
-----
Precision: {test['precision']:.4f}
Recall:    {test['recall']:.4f}
mAP@50:    {test['map50']:.4f}
mAP@50-95: {test['map50_95']:.4f}
"""
    (analysis_dir / "report.txt").write_text(report, encoding="utf-8")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível no PyTorch.")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Não encontrei o dataset em {DATA_YAML}.")

    model_path = find_best_model()
    analysis_dir = model_path.parent.parent / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Modelo: {model_path}")

    model = YOLO(str(model_path))
    validation = evaluate(model, "val", analysis_dir)
    test = evaluate(model, "test", analysis_dir)
    save_predictions(model, analysis_dir)
    save_report(validation, test, analysis_dir)

    print(f"Análise salva em: {analysis_dir}")


if __name__ == "__main__":
    main()
