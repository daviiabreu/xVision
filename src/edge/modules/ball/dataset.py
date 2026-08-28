"""Conversao do SoccerNet Tracking (formato MOTChallenge) para rotulos YOLO da bola.

So a bola interessa a esta frente, e o SoccerNet nao guarda a classe no `gt.txt`: as colunas sao
`frame, track_id, left, top, width, height, conf, -1, -1, -1`, e o papel de cada tracklet (jogador,
goleiro, arbitro, bola) fica no `gameinfo.ini`. Entao o caminho e: descobrir no `gameinfo.ini` quais
track_ids sao a bola, ler o tamanho do quadro no `seqinfo.ini` e converter as caixas da bola no
`gt.txt` para o formato YOLO (classe cx cy w h, normalizado).

Estas funcoes sao puras e so usam a biblioteca padrao, para rodar em qualquer lugar e serem testadas
sem baixar o dataset nem depender do YOLO.
"""

from __future__ import annotations

from collections.abc import Iterator
from configparser import ConfigParser
from pathlib import Path


def parse_frame_size(seqinfo_path: Path) -> tuple[int, int]:
    """Le largura e altura do quadro no `seqinfo.ini` (secao [Sequence])."""
    parser = ConfigParser(strict=False)
    parser.read(seqinfo_path, encoding="utf-8")
    width = parser.getint("Sequence", "imWidth")
    height = parser.getint("Sequence", "imHeight")
    if width <= 0 or height <= 0:
        raise ValueError(f"tamanho de quadro invalido em {seqinfo_path}: {width}x{height}")
    return width, height


def parse_ball_track_ids(gameinfo_path: Path) -> set[int]:
    """Descobre no `gameinfo.ini` os track_ids cujo papel e a bola.

    Cada entrada e `trackletID_<id> = <papel>; <extra>; ...` — o **id vem no sufixo da chave** e o
    papel no valor (ex.: `trackletID_21 = Ball;`). Alguns dumps repetem o id no primeiro campo do
    valor; por isso o id sai do sufixo da chave, que existe nos dois casos. Casa 'ball' em qualquer
    campo do valor, sem diferenciar maiuscula, para nao depender da grafia exata.
    """
    parser = ConfigParser(strict=False)
    parser.read(gameinfo_path, encoding="utf-8")
    ball_ids: set[int] = set()
    for section in parser.sections():
        for key, value in parser.items(section):
            if not key.startswith("trackletid_"):
                continue
            suffix = key.rsplit("_", 1)[-1]
            if not suffix.isdigit():
                continue
            if any("ball" in field.lower() for field in value.split(";")):
                ball_ids.add(int(suffix))
    return ball_ids


def mot_box_to_yolo(
    left: float, top: float, width: float, height: float, frame_w: int, frame_h: int
) -> tuple[float, float, float, float] | None:
    """Converte uma caixa MOT (canto + tamanho, px) para YOLO (centro + tamanho, normalizado).

    Devolve None quando a caixa e degenerada ou cai fora do quadro. O centro e recortado para
    [0, 1]; caixas parcialmente fora do quadro tem o tamanho preservado, como o YOLO tolera.
    """
    if width <= 0 or height <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    cx = (left + width / 2.0) / frame_w
    cy = (top + height / 2.0) / frame_h
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    return (cx, cy, width / frame_w, height / frame_h)


def iter_gt_rows(gt_path: Path) -> Iterator[tuple[int, int, float, float, float, float]]:
    """Percorre o `gt.txt` e devolve (frame, track_id, left, top, width, height) por linha."""
    with gt_path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            left, top, width, height = (float(parts[i]) for i in range(2, 6))
            yield frame, track_id, left, top, width, height


def build_ball_labels(
    gt_path: Path,
    ball_ids: set[int],
    frame_w: int,
    frame_h: int,
    class_id: int = 0,
) -> dict[int, list[str]]:
    """Monta as linhas de rotulo YOLO da bola por quadro.

    A chave e o quadro; o valor e a lista de linhas `class cx cy w h`. Quadros sem bola nao
    aparecem no dicionario, entao quem escreve decide se vira imagem de fundo ou e descartado.
    """
    labels: dict[int, list[str]] = {}
    for frame, track_id, left, top, width, height in iter_gt_rows(gt_path):
        if track_id not in ball_ids:
            continue
        box = mot_box_to_yolo(left, top, width, height, frame_w, frame_h)
        if box is None:
            continue
        cx, cy, w, h = box
        labels.setdefault(frame, []).append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return labels
