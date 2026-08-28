"""
Instalação:
    pip install ultralytics

Uso básico:
    python train.py --data ./dataset/dataset.yaml
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Treino YOLO-pose para SoccerNet Keypoints")
    parser.add_argument("--data", type=str, required=True, help="Caminho para dataset.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="Checkpoint pré-treinado (ex: yolo11n-pose.pt, yolo11s-pose.pt, yolov8n-pose.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960, help="Imagens originais são 960x540")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="soccernet_keypoints")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'0' para GPU 0, 'cpu' para CPU, vazio = auto",
    )
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        fliplr=0.0,
    )

    print("\nTreino concluído. Pesos salvos em runs/pose/<name>/weights/best.pt")


if __name__ == "__main__":
    main()
