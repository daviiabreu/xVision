"""A segunda passada cega da bola: amostra, gravacao, concordancia e o servidor local."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from edge.testing.make_clip import make_synthetic_clip
from mlops.ball_trajectory import BallTrajectoryLabel, read_ball_trajectory_labels
from mlops.bola_cega import (
    BANDEIRAS,
    CONTEXTO,
    BolaCegaError,
    Marca,
    Sessao,
    amostra,
    contatos_da_fila,
    grava,
    marcas_gravadas,
    medir,
    registro,
    servidor,
)

AGORA = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def rotulo(
    clip_id: str,
    frame_index: int,
    *,
    centro: tuple[float, float] | None = (100.0, 100.0),
    diametro: float | None = 10.0,
    visibilidade: str = "visible",
) -> BallTrajectoryLabel:
    visivel = visibilidade == "visible"
    return BallTrajectoryLabel(
        clip_id=clip_id,
        match_id=f"m-{clip_id}",
        video_sha256="a" * 64,
        frame_index=frame_index,
        timestamp_s=frame_index / 60,
        center_xy=centro if visivel else None,
        apparent_diameter_px=diametro if visivel else None,
        visibility=visibilidade,  # type: ignore[arg-type]
        trajectory_segment=0 if visivel else None,
        is_keyframe=True,
        motion_blur=False,
        camera_cut=False,
        annotator="davi",
        reviewed_by=None,
        quality_flags=("human-annotated",),
    )


def piloto(clipes: int = 2, quadros: int = 41) -> list[BallTrajectoryLabel]:
    return [rotulo(f"c{c}", 3 * q) for c in range(clipes) for q in range(quadros)]


def test_a_amostra_leva_todo_contato_e_um_quarto_dos_demais() -> None:
    alvos = amostra(piloto(), {"c0": 60, "c1": 90})

    assert len(alvos) == 2 * (1 + 10)
    assert {(a.clip_id, a.frame_index) for a in alvos if a.contato} == {("c0", 60), ("c1", 90)}
    assert len({(a.clip_id, a.frame_index) for a in alvos}) == len(alvos)


def test_a_amostra_e_reproduzivel_e_mistura_os_clipes() -> None:
    primeira = amostra(piloto(4), {f"c{c}": 60 for c in range(4)})
    segunda = amostra(piloto(4), {f"c{c}": 60 for c in range(4)})

    assert primeira == segunda
    assert [a.clip_id for a in primeira] != sorted(a.clip_id for a in primeira)


def test_contato_fora_da_primeira_passada_e_recusado() -> None:
    with pytest.raises(BolaCegaError, match="contato do clipe c0"):
        amostra(piloto(1), {"c0": 61})


def test_a_fila_da_o_contato_de_cada_clipe(tmp_path: Path) -> None:
    fila = tmp_path / "queue.json"
    entradas = [{"clip_id": "c0", "contact_frame": 60}, {"clip_id": "c1", "contact_frame": 90}]
    fila.write_text(json.dumps({"ball": entradas * 2}))

    assert contatos_da_fila(fila) == {"c0": 60, "c1": 90}


def test_a_fila_com_dois_contatos_no_mesmo_clipe_e_recusada(tmp_path: Path) -> None:
    fila = tmp_path / "queue.json"
    entradas = [{"clip_id": "c0", "contact_frame": 60}, {"clip_id": "c0", "contact_frame": 63}]
    fila.write_text(json.dumps({"ball": entradas}))

    with pytest.raises(BolaCegaError, match="dois contatos"):
        contatos_da_fila(fila)


def test_fila_sem_bola_e_recusada(tmp_path: Path) -> None:
    fila = tmp_path / "queue.json"
    fila.write_text(json.dumps({"field": []}))

    with pytest.raises(BolaCegaError, match="nao tem entrada de bola"):
        contatos_da_fila(fila)


def test_o_registro_leva_a_identidade_da_primeira_e_a_posicao_da_marca(tmp_path: Path) -> None:
    marca = Marca("visible", (120.5, 80.25), 12.0, True, False)
    destino = tmp_path / "cega.jsonl"

    grava(destino, registro(rotulo("c0", 60), marca, "davi", AGORA))
    lidos = read_ball_trajectory_labels(destino)

    assert len(lidos) == 1
    assert lidos[0].center_xy == (120.5, 80.25)
    assert lidos[0].apparent_diameter_px == 12.0
    assert lidos[0].motion_blur is True
    assert lidos[0].quality_flags == BANDEIRAS
    assert lidos[0].video_sha256 == "a" * 64


def test_bola_fora_de_quadro_nao_leva_centro_nem_diametro() -> None:
    marca = Marca("not-visible", (1.0, 1.0), 5.0, False, False)

    documento = registro(rotulo("c0", 60), marca, "davi", AGORA)

    assert documento["center_x"] is None
    assert documento["apparent_diameter_px"] is None
    assert documento["trajectory_segment"] is None


@pytest.mark.parametrize(
    ("marca", "annotator", "erro"),
    [
        (Marca("visible", None, 10.0, False, False), "davi", "centro e diametro"),
        (Marca("visible", (1.0, 1.0), None, False, False), "davi", "centro e diametro"),
        (Marca("occluded", None, None, False, False), "", "quem anotou"),
    ],
)
def test_marca_incompleta_e_recusada(marca: Marca, annotator: str, erro: str) -> None:
    with pytest.raises(BolaCegaError, match=erro):
        registro(rotulo("c0", 60), marca, annotator, AGORA)


def test_regravar_o_mesmo_quadro_troca_a_marca_e_mantem_a_ordem(tmp_path: Path) -> None:
    destino = tmp_path / "cega.jsonl"
    for quadro, x in ((90, 1.0), (60, 2.0), (90, 3.0)):
        marca = Marca("visible", (x, 1.0), 10.0, False, False)
        grava(destino, registro(rotulo("c0", quadro), marca, "davi", AGORA))

    gravadas = marcas_gravadas(destino)

    assert list(gravadas) == [("c0", 60), ("c0", 90)]
    assert gravadas[("c0", 90)]["center_x"] == 3.0


def test_documento_fora_do_schema_nao_chega_ao_arquivo(tmp_path: Path) -> None:
    destino = tmp_path / "cega.jsonl"
    documento = registro(rotulo("c0", 60), Marca("visible", (1, 1), 10, False, False), "d", AGORA)
    documento["frame_index"] = -1

    with pytest.raises(BolaCegaError, match="fora do schema"):
        grava(destino, documento)
    assert not destino.exists()


def test_a_concordancia_mede_o_centro_em_diametros_da_primeira_passada() -> None:
    primeira = [rotulo("c0", 60), rotulo("c0", 63), rotulo("c1", 60)]
    segunda = [
        rotulo("c0", 60, centro=(103.0, 104.0), diametro=12.0),
        rotulo("c0", 63, centro=(100.0, 110.0)),
        rotulo("c1", 60, visibilidade="occluded"),
    ]

    resultado = medir(primeira, segunda, {"c0": 60, "c1": 60})

    assert resultado.n_pares == 3
    assert resultado.n_contatos == 2
    assert resultado.visibilidade == pytest.approx(2 / 3)
    assert resultado.n_centros == 2
    assert resultado.centro_mediana == pytest.approx(0.75)
    assert resultado.centro_max == pytest.approx(1.0)
    assert resultado.contato_mediana == pytest.approx(0.5)
    assert resultado.diametro_mediana == pytest.approx(0.1)
    assert resultado.confusao == {"visible->occluded": 1, "visible->visible": 2}
    assert resultado.aprovada is False


def test_passadas_iguais_passam_no_protocolo() -> None:
    primeira = [rotulo("c0", 60), rotulo("c0", 63, visibilidade="not-visible")]

    resultado = medir(primeira, primeira, {"c0": 60})

    assert resultado.aprovada is True
    assert resultado.kappa == pytest.approx(1.0)
    assert resultado.to_dict()["aprovada"] is True


def test_sem_par_visivel_o_protocolo_fica_indecidido() -> None:
    primeira = [rotulo("c0", 60, visibilidade="occluded")]

    resultado = medir(primeira, primeira, {"c0": 60})

    assert resultado.aprovada is None
    assert resultado.kappa is None
    assert resultado.centro_mediana is None


def test_segunda_passada_sem_quadro_da_primeira_e_recusada() -> None:
    with pytest.raises(BolaCegaError, match="nenhum quadro"):
        medir([rotulo("c0", 60)], [rotulo("c9", 60)], {"c0": 60})


@pytest.fixture
def sessao(tmp_path: Path) -> Iterator[Sessao]:
    video = make_synthetic_clip(tmp_path / "c0.mp4")
    primeira = [rotulo("c0", 3 * q) for q in range(1, 9)]
    alvos = amostra(primeira, {"c0": 12})
    aberta = Sessao(
        alvos,
        {(label.clip_id, label.frame_index): label for label in primeira},
        {"c0": video},
        tmp_path / "cega.jsonl",
        "davi",
        lambda: AGORA,
    )
    yield aberta
    aberta.close()


def test_sessao_sem_video_de_algum_clipe_e_recusada(tmp_path: Path) -> None:
    primeira = [rotulo("c0", 3 * q) for q in range(1, 9)]
    with pytest.raises(BolaCegaError, match="sem video para c0"):
        Sessao(amostra(primeira, {"c0": 12}), {}, {}, tmp_path / "x.jsonl", "davi", lambda: AGORA)


def test_o_estado_nao_carrega_nada_da_primeira_passada(sessao: Sessao) -> None:
    estado = sessao.estado()

    texto = json.dumps(estado)
    assert "center" not in texto.replace('"marca": null', "")
    assert estado["contexto"] == CONTEXTO
    assert len(estado["alvos"]) == len(sessao.alvos)  # type: ignore[arg-type]


def test_o_quadro_sai_em_jpeg_e_o_contexto_tem_limite(sessao: Sessao) -> None:
    assert sessao.quadro(0, 0)[:2] == b"\xff\xd8"
    with pytest.raises(BolaCegaError, match="contexto"):
        sessao.quadro(0, CONTEXTO + 1)
    with pytest.raises(BolaCegaError, match="a fila tem"):
        sessao.quadro(len(sessao.alvos), 0)


def test_a_marca_gravada_volta_no_estado(sessao: Sessao) -> None:
    sessao.marcar(0, {"visibility": "visible", "center": [10, 20], "diameter": 8})

    marca: Any = sessao.estado()["alvos"]
    assert marca[0]["marca"] == {
        "visibility": "visible",
        "center": [10.0, 20.0],
        "diameter": 8.0,
        "motion_blur": False,
        "camera_cut": False,
    }


def test_visibilidade_desconhecida_e_recusada(sessao: Sessao) -> None:
    with pytest.raises(BolaCegaError, match="visibilidade desconhecida"):
        sessao.marcar(0, {"visibility": "talvez"})


def _pede(url: str, corpo: dict[str, Any] | None = None) -> tuple[int, bytes]:
    dados = None if corpo is None else json.dumps(corpo).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=dados)) as resposta:
            return resposta.status, resposta.read()
    except urllib.error.HTTPError as erro:
        return erro.code, erro.read()


def test_o_servidor_serve_pagina_estado_quadro_e_grava(sessao: Sessao) -> None:
    http = servidor(sessao, 0)
    linha = threading.Thread(target=http.serve_forever, daemon=True)
    linha.start()
    base = f"http://127.0.0.1:{http.server_address[1]}"
    try:
        assert b"segunda passada cega" in _pede(f"{base}/")[1]
        assert json.loads(_pede(f"{base}/estado")[1])["annotator"] == "davi"
        assert _pede(f"{base}/quadro?i=0&d=1")[1][:2] == b"\xff\xd8"
        assert _pede(f"{base}/quadro?i=0&d=99")[0] == 400
        assert _pede(f"{base}/nada")[0] == 404
        corpo = {"visibility": "not-visible", "center": None, "diameter": None}
        assert _pede(f"{base}/marca?i=0", corpo) == (200, b'{"ok": true}')
        assert _pede(f"{base}/marca?i=0", {"visibility": "x"})[0] == 400
        assert _pede(f"{base}/outra", corpo)[0] == 404
    finally:
        http.shutdown()
        http.server_close()
