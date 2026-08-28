"""Pipeline de treino do detector da bola em um comando: download -> conversao -> treino.

Cada etapa continua sendo um script proprio (testavel); este orquestrador so encadeia.
Use --stages para rodar um subconjunto.

Atencao: o **treino precisa de GPU**. No volume do SoccerNet, treinar em CPU e inviavel. O caminho
pratico e rodar `download,convert` localmente e o `train` numa maquina com GPU (Colab, Kaggle ou
servidor). Os dados e os pesos ficam em diretorios ignorados pelo git (`data/`, `runs/`).

Exemplos:

    # local (so baixa e converte)
    python training/run.py --stages download,convert

    # numa GPU (tudo)
    python training/run.py --epochs 100 --device 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _run(script: str, *args: object) -> None:
    cmd = [sys.executable, str(_HERE / script), *[str(a) for a in args]]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de treino do detector da bola.")
    parser.add_argument("--soccernet-dir", default="data/soccernet")
    parser.add_argument("--yolo-dir", default="data/ball-yolo")
    parser.add_argument("--stages", default="download,convert,train", help="separada por virgula")
    parser.add_argument("--split", nargs="+", default=["train", "test"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16, help="suba em GPU com VRAM sobrando")
    parser.add_argument("--workers", type=int, default=8, help="workers; reduza se faltar RAM")
    parser.add_argument("--device", default="0", help="'0' para GPU, 'cpu' para CPU")
    parser.add_argument("--password", default=None, help="senha SoccerNet, se algum split der 403")
    args = parser.parse_args()

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]

    if "download" in stages:
        extra = ["--password", args.password] if args.password else []
        _run("download_soccernet.py", "--out", args.soccernet_dir, "--split", *args.split, *extra)

    if "convert" in stages:
        _run(
            "soccernet_to_yolo.py",
            "--src",
            f"{args.soccernet_dir}/tracking",
            "--out",
            args.yolo_dir,
            "--train",
            "train",
            "--val",
            "test",
        )

    if "train" in stages:
        if str(args.device).lower() == "cpu":
            print("AVISO: treino em CPU e inviavel no volume do SoccerNet; use uma GPU.")
        _run(
            "train_ball.py",
            "--data",
            f"{args.yolo_dir}/data.yaml",
            "--epochs",
            args.epochs,
            "--imgsz",
            args.imgsz,
            "--batch",
            args.batch,
            "--workers",
            args.workers,
            "--device",
            args.device,
        )


if __name__ == "__main__":
    main()
