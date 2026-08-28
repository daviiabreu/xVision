"""Converte o SoccerNet Tracking (MOTChallenge) num dataset YOLO de deteccao da bola.

Descobre as sequencias recursivamente (procura `seqinfo.ini` com `gt/gt.txt` e `img1`), entao
nao importa se o download extraiu os zips com uma pasta a mais. Para cada sequencia, descobre
os track_ids da bola no `gameinfo.ini`, converte as caixas do `gt.txt` para rotulos YOLO e monta a
arvore que o Ultralytics espera:

    <out>/images/<split>/<seq>_<frame>.jpg
    <out>/labels/<split>/<seq>_<frame>.txt
    <out>/data.yaml

O prefixo `<seq>_` evita colisao entre sequencias (todas tem `000001.jpg`). Por padrao mantem apenas
quadros com bola; `--keep-background` inclui quadros sem bola (rotulo vazio) como negativos. As
imagens sao ligadas por hardlink quando possivel (sem duplicar GB), com copia como reserva.

A logica de conversao mora em `edge.ball.dataset` (pura e testada); aqui fica so a travessia
dos diretorios e a escrita do dataset.

Exemplo:

    python models/ball/soccernet_to_yolo.py \
        --src ../data/soccernet/tracking --out ../data/ball-yolo --train train --val test
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from edge.ball.dataset import (  # noqa: E402
    build_ball_labels,
    parse_ball_track_ids,
    parse_frame_size,
)


def find_sequences(root: Path) -> list[Path]:
    """Diretorios de sequencia sob `root`: tem `seqinfo.ini`, `gt/gt.txt` e `img1` ao lado."""
    if not root.exists():
        return []
    sequences = [
        seqinfo.parent
        for seqinfo in root.rglob("seqinfo.ini")
        if (seqinfo.parent / "gt" / "gt.txt").exists() and (seqinfo.parent / "img1").is_dir()
    ]
    return sorted(sequences)


def resolve_split_dir(src: Path, split: str) -> Path | None:
    """Acha o diretorio do split, tolerando um nivel extra que o zip possa ter criado."""
    direct = src / split
    if find_sequences(direct):
        return direct
    for candidate in src.rglob(split):
        if candidate.is_dir() and find_sequences(candidate):
            return candidate
    return None


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(split_dir: Path, out: Path, split: str, keep_background: bool) -> tuple[int, int]:
    images_dir = out / "images" / split
    labels_dir = out / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    frames_written = 0
    ball_boxes = 0
    for seq_dir in find_sequences(split_dir):
        gameinfo = seq_dir / "gameinfo.ini"
        if not gameinfo.exists():
            print(f"  pulando {seq_dir.name}: sem gameinfo.ini")
            continue
        frame_w, frame_h = parse_frame_size(seq_dir / "seqinfo.ini")
        ball_ids = parse_ball_track_ids(gameinfo)
        if not ball_ids:
            print(f"  aviso: {seq_dir.name} sem tracklet de bola no gameinfo.ini")
        labels = build_ball_labels(seq_dir / "gt" / "gt.txt", ball_ids, frame_w, frame_h)

        for image_path in sorted((seq_dir / "img1").glob("*.jpg")):
            frame = int(image_path.stem)
            lines = labels.get(frame, [])
            if not lines and not keep_background:
                continue
            stem = f"{seq_dir.name}_{frame:06d}"
            _link(image_path, images_dir / f"{stem}.jpg")
            (labels_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            frames_written += 1
            ball_boxes += len(lines)

    print(f"[{split}] quadros={frames_written} caixas_bola={ball_boxes} <- {split_dir}")
    return frames_written, ball_boxes


def _write_data_yaml(out: Path, has_val: bool) -> None:
    lines = [
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val" if has_val else "val: images/train",
        "names:",
        "  0: ball",
        "",
    ]
    (out / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SoccerNet Tracking -> dataset YOLO da bola.")
    parser.add_argument("--src", required=True, help="raiz das sequencias")
    parser.add_argument("--out", required=True, help="destino do dataset YOLO")
    parser.add_argument("--train", default="train", help="nome do split de treino em --src")
    parser.add_argument("--val", default="test", help="split de validacao (ou vazio)")
    parser.add_argument("--keep-background", action="store_true", help="inclui quadros sem bola")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    train_dir = resolve_split_dir(src, args.train)
    if train_dir is None:
        raise SystemExit(f"nenhuma sequencia de treino sob {src / args.train} (baixou/extraiu?)")
    convert_split(train_dir, out, "train", args.keep_background)

    val_dir = resolve_split_dir(src, args.val) if args.val else None
    if val_dir is not None:
        convert_split(val_dir, out, "val", args.keep_background)
    _write_data_yaml(out, val_dir is not None)
    print(f"data.yaml escrito em {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
