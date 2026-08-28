"""Conversao SoccerNet -> YOLO: descoberta da bola, normalizacao de caixa e montagem por quadro.

Puro e deterministico, sem baixar o dataset nem depender do YOLO. Monta um mini SoccerNet sintetico
em disco (seqinfo.ini, gameinfo.ini, gt.txt) e verifica cada passo da conversao.
"""

from __future__ import annotations

from pathlib import Path

from edge.modules.ball.dataset import (
    build_ball_labels,
    mot_box_to_yolo,
    parse_ball_track_ids,
    parse_frame_size,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_frame_size(tmp_path: Path) -> None:
    seqinfo = _write(
        tmp_path / "seqinfo.ini",
        "[Sequence]\nname=SNMOT-060\nimWidth=1920\nimHeight=1080\nseqLength=750\n",
    )
    assert parse_frame_size(seqinfo) == (1920, 1080)


def test_parse_ball_track_ids_role_in_value(tmp_path: Path) -> None:
    # Formato real do SoccerNet: id no sufixo da chave, papel no valor.
    gameinfo = _write(
        tmp_path / "gameinfo.ini",
        "[Sequence]\n"
        "num_tracklets=4\n"
        "trackletID_1 = Goalkeeper; team left\n"
        "trackletID_2 = Player; team right; 10\n"
        "trackletID_3 = Ball;\n"
        "trackletID_4 = Referee;\n",
    )
    assert parse_ball_track_ids(gameinfo) == {3}


def test_parse_ball_track_ids_id_repeated_in_value(tmp_path: Path) -> None:
    # Alguns dumps repetem o id no valor; o id sai do sufixo da chave nos dois casos.
    gameinfo = _write(tmp_path / "gameinfo.ini", "[Sequence]\ntrackletID_7 = 7; ball; ;\n")
    assert parse_ball_track_ids(gameinfo) == {7}


def test_mot_box_to_yolo_centers_and_normalizes() -> None:
    # Caixa 100x100 no canto (0,0) de um quadro 1000x500: centro (50,50), normalizado (0.05, 0.1).
    assert mot_box_to_yolo(0, 0, 100, 100, 1000, 500) == (0.05, 0.1, 0.1, 0.2)


def test_mot_box_to_yolo_rejects_degenerate_and_off_frame() -> None:
    assert mot_box_to_yolo(0, 0, 0, 50, 1000, 500) is None
    assert mot_box_to_yolo(2000, 0, 20, 20, 1000, 500) is None


def test_build_ball_labels_keeps_only_ball_frames(tmp_path: Path) -> None:
    gt = _write(
        tmp_path / "gt.txt",
        # frame, id, left, top, w, h, conf, -1, -1, -1
        "1,3,955,535,10,10,1,-1,-1,-1\n"  # bola (id 3)
        "1,2,100,200,40,80,1,-1,-1,-1\n"  # jogador (id 2), ignorado
        "2,2,110,210,40,80,1,-1,-1,-1\n"  # so jogador no quadro 2
        "3,3,950,530,10,10,1,-1,-1,-1\n",  # bola no quadro 3
    )
    labels = build_ball_labels(gt, ball_ids={3}, frame_w=1920, frame_h=1080)

    assert set(labels) == {1, 3}  # quadro 2 (sem bola) fica de fora
    assert labels[1] == ["0 0.500000 0.500000 0.005208 0.009259"]
