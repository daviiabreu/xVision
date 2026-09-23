"""Ajuste fino do estimador de pose sobre os recortes que `dataset.py` monta.

Script de operador: roda a mao, exige GPU e nao entra em CI. Pesos e resultados (`runs/`, `*.pt`)
nao entram no repositorio.

Tres escolhas vem do que a borda faz com o peso. Treina em `IMGSZ` 256, importado do adaptador,
porque treinar em 640 e inferir em 256 resolve outra escala. Liga `fliplr`, o que o treino de campo
nao pode fazer, porque a permutacao do espelhamento dos 17 pontos do COCO esta conferida em
`dataset.FLIP_IDX`. Desliga `mosaic`, que reduz a escala em quatro vezes e desfaz o enquadramento
que o dataset de recortes existe para reproduzir.

O peso de partida e o `yolov8n-pose.pt` por continuidade com o manifesto em producao e pelo
orcamento da Jetson Orin Nano. Trocar de arquitetura pede medir com `validate.py` antes.

Exemplo:

    uv run --extra training python models/pose/train.py \\
        --data ../data/pose-crops/dataset.yaml --device 0 --batch 64
"""

from __future__ import annotations

import argparse

from edge.modules.pose.yolo import IMGSZ

EPOCHS = 150
BATCH = 64
"""Lote da A4000 de 16 GB em 256 px com o nano. Reduza se a memoria acusar."""

PATIENCE = 30
SEED = 20260922
WORKERS = 8


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajuste fino do estimador de pose sobre recortes.")
    parser.add_argument("--data", default=None, help="dataset.yaml gerado por dataset.py")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="peso de partida")
    parser.add_argument("--resume", default=None, help="last.pt para retomar o treino")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="o mesmo que o adaptador usa")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--device", default="0", help="'0' para GPU, 'cpu' para CPU")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="parada antecipada")
    parser.add_argument("--workers", type=int, default=WORKERS, help="reduza se faltar RAM")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--project", default="runs/pose", help="pasta de saida, ignorada no git")
    parser.add_argument("--name", default="xvision-pose")
    args = parser.parse_args()

    import torch  # pyright: ignore[reportMissingImports]
    from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

    cuda = torch.cuda.is_available()
    dispositivo = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"CUDA disponivel: {cuda} | dispositivo: {dispositivo}")
    if not cuda and str(args.device).lower() != "cpu":
        print("AVISO: PyTorch nao enxerga GPU. Instale a build CUDA ou passe --device cpu.")
    if args.imgsz != IMGSZ:
        print(
            f"AVISO: treinando em {args.imgsz} e a borda infere em {IMGSZ}. "
            "A diferenca de escala aparece como acuracia baixa sem causa visivel."
        )

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    elif not args.data:
        raise SystemExit(
            "informe --data (dataset.yaml) para treinar do zero, ou --resume <last.pt>"
        )
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            project=args.project,
            name=args.name,
            cos_lr=True,
            fliplr=0.5,
            mosaic=0.0,
            scale=0.3,
            translate=0.1,
            degrees=5.0,
        )

    metrics = model.val()
    print(
        f"treino concluido. pose mAP50={metrics.pose.map50:.4f} "
        f"mAP50-95={metrics.pose.map:.4f} box mAP50={metrics.box.map50:.4f}"
    )
    print("agora meca o que o produto consome: models/pose/validate.py e depois xvision eval-pose")


if __name__ == "__main__":
    main()
