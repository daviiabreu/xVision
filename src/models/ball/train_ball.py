"""Ajuste fino do YOLOv8n para deteccao da bola, a partir do dataset YOLO gerado do SoccerNet.

Parte do YOLOv8n do COCO (baseline de controle da arquitetura) e treina so a classe bola. Os padroes
miram objeto pequeno: entrada em alta resolucao (`imgsz` 1280) e mosaic desligado perto do
fim, porque juntar quatro imagens encolhe ainda mais uma bola que ja ocupa poucos pixels.

Precisa de GPU na pratica; treinar em CPU e inviavel neste volume. Rode numa maquina com GPU, no
Colab ou no Kaggle. Pesos e resultados (`runs/`, `*.pt`) nao entram no repositorio (ver .gitignore).

Exemplo:

    python src/models/ball/train_ball.py --data data/ball-yolo/data.yaml --epochs 100 --device 0
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine tuning do YOLOv8n para a bola.")
    parser.add_argument("--data", default=None, help="data.yaml gerado pelo soccernet_to_yolo.py")
    parser.add_argument("--weights", default="yolov8n.pt", help="pesos de partida")
    parser.add_argument("--resume", default=None, help="last.pt para retomar o treino")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' para GPU, 'cpu' para CPU")
    parser.add_argument("--patience", type=int, default=25, help="parada antecipada")
    parser.add_argument("--workers", type=int, default=8, help="workers; reduza se faltar RAM")
    parser.add_argument("--project", default="runs/ball", help="pasta de saida (ignorada no git)")
    parser.add_argument("--name", default="yolov8n-ball")
    args = parser.parse_args()

    import torch  # pyright: ignore[reportMissingImports]
    from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"CUDA disponivel: {cuda} | dispositivo: {device_name}")
    if not cuda and str(args.device).lower() != "cpu":
        print("AVISO: PyTorch nao enxerga GPU. Instale a build CUDA (ver README) ou --device cpu.")

    if args.resume:
        # Retoma do last.pt: o Ultralytics le epocas, data e demais args do proprio checkpoint.
        model = YOLO(args.resume)
        model.train(resume=True)
    elif not args.data:
        raise SystemExit("informe --data (data.yaml) para treinar do zero, ou --resume <last.pt>")
    else:
        model = YOLO(args.weights)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            workers=args.workers,
            project=args.project,
            name=args.name,
            cos_lr=True,
            close_mosaic=10,
        )
    metrics = model.val()
    print(f"treino concluido. mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
