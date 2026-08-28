"""Baixa o SoccerNet Tracking para treinar o detector da bola.

O tracking (`train`/`test`) baixa **sem senha**: o NDA da pagina do SoccerNet vale para os Broadcast
Videos, nao para esta tarefa. `--password` (ou SOCCERNET_PASSWORD) fica como reserva, caso
algum split retorne 403; preencha o NDA em https://www.soccer-net.org/. Os splits `train`/`test`
tem rotulos; `challenge` nao tem `gt.txt` (e para submissao), nao serve para treino/validacao.

Os dados sao pesados (dezenas de GB) e nao entram no repositorio: baixe para um diretorio ignorado,
como `data/soccernet` (ver .gitignore).

Exemplo:

    python models/ball/download_soccernet.py --out ../data/soccernet --split train test
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def _extract_zips(root: Path) -> None:
    """O tracking baixa em .zip; extrai cada um ao lado, uma vez (marca com .extracted)."""
    for zip_path in sorted(root.rglob("*.zip")):
        marker = zip_path.with_suffix(".extracted")
        if marker.exists():
            continue
        print(f"  extraindo {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(zip_path.parent)
        marker.write_text("ok", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download do SoccerNet Tracking.")
    parser.add_argument("--out", default="data/soccernet", help="diretorio local de destino")
    parser.add_argument("--task", default="tracking", choices=["tracking", "tracking-2023"])
    parser.add_argument("--split", nargs="+", default=["train", "test"])
    parser.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD"))
    args = parser.parse_args()

    from SoccerNet.Downloader import SoccerNetDownloader  # pyright: ignore[reportMissingImports]

    downloader = SoccerNetDownloader(LocalDirectory=args.out)
    if args.password:
        downloader.password = args.password
    downloader.downloadDataTask(task=args.task, split=args.split)
    _extract_zips(Path(args.out))
    print(f"concluido: task={args.task} split={args.split} -> {args.out}")


if __name__ == "__main__":
    main()
