"""A medida do estimador de pose pelo que a feature consome.

O modo de falha que estes testes fecham nao levanta excecao e passa em toda metrica publicada: um
estimador que troque tornozelo esquerdo por direito sai com OKS alta, passa no PCK, e entrega a
`body_part` a resposta invertida.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from contracts import Clip, Keypoint
from edge.features.body_part import MIN_CONFIANCA
from edge.modules.pose import COCO_KEYPOINTS
from mlops.pose_quality import (
    OKS_SIGMAS,
    PoseLabelError,
    measure,
    oks,
    read_pose_labels,
    validate_label_video,
)

CLIPE = "opaque-clip"
PARTIDA = "opaque-match"
QUADRO = 12
CAIXA = (100.0, 200.0, 140.0, 330.0)
"""40 por 130 px: a caixa de jogador de transmissao que o modulo de pose mediu no clipe real."""

TORNOZELO_E = (112.0, 325.0)
TORNOZELO_D = (123.0, 325.0)
"""Onze pixels de separacao, que e a mediana medida nos 38 clipes anotados do parceiro."""


def _posicao(nome: str) -> tuple[float, float]:
    """Um esqueleto plausivel: a cabeca no topo da caixa e os dois tornozelos na base, separados."""
    if nome == "ankle_left":
        return TORNOZELO_E
    if nome == "ankle_right":
        return TORNOZELO_D
    altura = COCO_KEYPOINTS.index(nome) / len(COCO_KEYPOINTS)
    return (120.0, 205.0 + altura * 100.0)


def _documento(**mudancas: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": "1.0",
        "clip_id": CLIPE,
        "match_id": PARTIDA,
        "video_sha256": "a" * 64,
        "frame_index": QUADRO,
        "frame_width": 1920,
        "frame_height": 1080,
        "shooter_box": list(CAIXA),
        "keypoints": [
            {
                "name": nome,
                "x": _posicao(nome)[0],
                "y": _posicao(nome)[1],
                "visibility": "visible",
            }
            for nome in COCO_KEYPOINTS
        ],
        "annotator": "reviewer-a",
        "reviewed_by": None,
        "created_at": "2026-09-22T12:00:00Z",
        "quality_flags": ["human-annotated"],
    }
    base.update(mudancas)
    return base


def _labels(tmp_path: Path, *documentos: dict[str, object]):
    caminho = tmp_path / "pose.jsonl"
    caminho.write_text(
        "\n".join(json.dumps(documento) for documento in documentos) + "\n", encoding="utf-8"
    )
    return read_pose_labels(caminho)


def _previsto(
    *, trocado: bool = False, confianca: float = 0.9, deslocamento: float = 0.0
) -> tuple[Keypoint, ...]:
    pontos: list[Keypoint] = []
    for nome in COCO_KEYPOINTS:
        origem = nome
        if trocado and nome == "ankle_left":
            origem = "ankle_right"
        elif trocado and nome == "ankle_right":
            origem = "ankle_left"
        x, y = _posicao(origem)
        pontos.append(Keypoint(nome, x + deslocamento, y, confianca))
    return tuple(pontos)


def test_estimador_exato_e_utilizavel_por_inteiro(tmp_path: Path) -> None:
    labels = _labels(tmp_path, _documento())

    relatorio = measure(labels, {(CLIPE, QUADRO): _previsto()})

    assert relatorio.n_frames == 1
    assert relatorio.completion == 1.0
    assert relatorio.oks_median == pytest.approx(1.0)
    assert relatorio.ankle_availability == 1.0
    assert relatorio.ankle_side_accuracy == 1.0
    assert relatorio.usable_rate == 1.0
    assert relatorio.score == relatorio.usable_rate


def test_a_troca_de_lado_passa_na_oks_e_zera_o_que_a_feature_consome(tmp_path: Path) -> None:
    """O defeito que este arquivo existe para pegar. Se algum dia a OKS virar o criterio de
    aceite deste modelo, e este teste que tem de quebrar."""
    labels = _labels(tmp_path, _documento())

    relatorio = measure(labels, {(CLIPE, QUADRO): _previsto(trocado=True)})

    assert relatorio.oks_median is not None and relatorio.oks_median > 0.9, (
        "a premissa do teste caiu: a troca precisa ser invisivel para a OKS"
    )
    assert relatorio.pck == 1.0, "a premissa do teste caiu: a troca precisa passar no PCK"
    assert relatorio.ankle_availability == 1.0
    assert relatorio.ankle_side_accuracy == 0.0
    assert relatorio.usable_rate == 0.0
    assert relatorio.score == 0.0


def test_tornozelo_abaixo_do_piso_nao_conta_como_lado_errado(tmp_path: Path) -> None:
    labels = _labels(tmp_path, _documento())

    relatorio = measure(labels, {(CLIPE, QUADRO): _previsto(confianca=MIN_CONFIANCA - 0.01)})

    assert relatorio.n_ankle_frames == 1
    assert relatorio.ankle_available == 0
    assert relatorio.ankle_availability == 0.0
    assert relatorio.ankle_side_accuracy == 0.0
    assert relatorio.usable_rate == 0.0, "o denominador sao os quadros anotados, nao os aceitos"


def test_o_piso_da_disponibilidade_e_o_mesmo_que_a_feature_usa(tmp_path: Path) -> None:
    """Medir com um piso e consumir com outro promete cobertura que `body_part` recusa."""
    labels = _labels(tmp_path, _documento())

    relatorio = measure(labels, {(CLIPE, QUADRO): _previsto()})

    assert relatorio.min_confidence == MIN_CONFIANCA


def test_quadro_sem_resposta_do_estimador_derruba_a_completude(tmp_path: Path) -> None:
    labels = _labels(tmp_path, _documento())

    relatorio = measure(labels, {})

    assert relatorio.n_available == 0
    assert relatorio.n_unavailable == 1
    assert relatorio.completion == 0.0
    assert relatorio.oks_median is None
    assert relatorio.n_ankle_frames == 1, "a anotacao tem os dois pes, e o denominador e dela"
    assert relatorio.usable_rate == 0.0


def test_ponto_que_o_estimador_nao_devolve_derruba_a_oks(tmp_path: Path) -> None:
    """Ignorar o ponto ausente deixaria um estimador que responde pouco com OKS alta."""
    labels = _labels(tmp_path, _documento())
    parcial = tuple(k for k in _previsto() if k.name != "nose")

    relatorio = measure(labels, {(CLIPE, QUADRO): parcial})

    assert relatorio.oks_median is not None
    assert relatorio.oks_median == pytest.approx(16 / 17)


def test_quadro_sem_tornozelo_anotado_sai_do_denominador_do_tornozelo(tmp_path: Path) -> None:
    sem_pes = _documento()
    sem_pes["keypoints"] = [
        {**ponto, "x": None, "y": None, "visibility": "absent"}
        if cast(str, ponto["name"]).startswith("ankle_")
        else ponto
        for ponto in cast(list[dict[str, object]], sem_pes["keypoints"])
    ]
    labels = _labels(tmp_path, sem_pes)

    relatorio = measure(labels, {(CLIPE, QUADRO): _previsto()})

    assert relatorio.n_frames == 1
    assert relatorio.n_ankle_frames == 0
    assert relatorio.ankle_availability == 0.0
    assert relatorio.usable_rate == 0.0
    assert relatorio.oks_median is not None, "os outros quinze pontos continuam medidos"


def test_ponto_ocluso_conta_como_anotado(tmp_path: Path) -> None:
    ocluso = _documento()
    ocluso["keypoints"] = [
        {**ponto, "visibility": "occluded"} if ponto["name"] == "ankle_left" else ponto
        for ponto in ocluso["keypoints"]  # type: ignore[union-attr]
    ]
    labels = _labels(tmp_path, ocluso)

    relatorio = measure(labels, {(CLIPE, QUADRO): _previsto()})

    assert relatorio.n_ankle_frames == 1
    assert relatorio.usable_rate == 1.0


def test_a_proposta_fica_de_fora_por_padrao_e_entra_declarada(tmp_path: Path) -> None:
    labels = _labels(
        tmp_path,
        _documento(),
        _documento(frame_index=QUADRO + 1, quality_flags=["requires-human-review"]),
    )
    previsoes = {(CLIPE, QUADRO): _previsto(), (CLIPE, QUADRO + 1): _previsto()}

    estrito = measure(labels, previsoes)
    exploratorio = measure(labels, previsoes, include_proposals=True)

    assert estrito.n_labels == 2
    assert estrito.n_frames == 1
    assert estrito.n_excluded_proposals == 1
    assert estrito.to_dict()["evaluation_kind"] == "human-confirmed"
    assert exploratorio.n_frames == 2
    assert exploratorio.to_dict()["evaluation_kind"] == "exploratory-with-proposals"


def test_a_anotacao_assistida_e_contada_a_parte(tmp_path: Path) -> None:
    """Quem corrige esqueleto ja desenhado tende a aceitar o lado que o modelo escolheu."""
    labels = _labels(
        tmp_path, _documento(quality_flags=["human-annotated", "model-proposed-reviewed"])
    )

    relatorio = measure(labels, {(CLIPE, QUADRO): _previsto()})

    assert relatorio.n_frames == 1
    assert relatorio.n_assisted_frames == 1


def test_o_relatorio_serializa_com_os_tres_numeros_e_os_intervalos(tmp_path: Path) -> None:
    labels = _labels(tmp_path, _documento())

    documento = measure(labels, {(CLIPE, QUADRO): _previsto()}).to_dict()

    assert documento["score"] == documento["usable_rate"]
    for chave in (
        "oks_median",
        "ankle_availability",
        "ankle_side_accuracy",
        "usable_rate",
        "usable_rate_wilson_95",
        "min_confidence",
        "per_keypoint",
    ):
        assert chave in documento, f"o relatorio nao publica {chave}"
    assert set(documento["per_keypoint"]) == set(COCO_KEYPOINTS)  # type: ignore[arg-type]
    assert json.dumps(documento), "o relatorio precisa ser gravavel em JSON"


def test_o_erro_em_pixel_entra_no_relatorio(tmp_path: Path) -> None:
    labels = _labels(tmp_path, _documento())

    relatorio = measure(labels, {(CLIPE, QUADRO): _previsto(deslocamento=3.0)})

    assert relatorio.error_median_px == pytest.approx(3.0)
    assert relatorio.error_p95_px == pytest.approx(3.0)


def test_a_oks_usa_o_sigma_do_ponto(tmp_path: Path) -> None:
    """Sigma menor pune mais: o mesmo deslocamento vale menos no olho do que no tornozelo."""
    labels = _labels(tmp_path, _documento())
    label = labels[0]

    olho = oks(label, (Keypoint("eye_left", *_posicao("eye_left"), 0.9),))
    tornozelo = oks(label, (Keypoint("ankle_left", *_posicao("ankle_left"), 0.9),))

    assert olho is not None and tornozelo is not None
    assert OKS_SIGMAS["eye_left"] < OKS_SIGMAS["ankle_left"]


def test_o_schema_recusa_keypoint_fora_da_ordem_do_coco(tmp_path: Path) -> None:
    """O contrato e por nome, e a ordem trocada quebraria quem indexa por posicao rio abaixo."""
    fora_de_ordem = _documento()
    pontos = list(fora_de_ordem["keypoints"])  # type: ignore[arg-type]
    pontos[15], pontos[16] = pontos[16], pontos[15]
    fora_de_ordem["keypoints"] = pontos

    with pytest.raises(PoseLabelError, match="ordem do COCO"):
        _labels(tmp_path, fora_de_ordem)


def test_o_schema_recusa_ponto_ausente_com_coordenada(tmp_path: Path) -> None:
    """`absent` com x e y seria ausencia que o avaliador leria como posicao."""
    incoerente = _documento()
    incoerente["keypoints"] = [
        {**ponto, "visibility": "absent"} if ponto["name"] == "nose" else ponto
        for ponto in incoerente["keypoints"]  # type: ignore[union-attr]
    ]

    with pytest.raises(PoseLabelError):
        _labels(tmp_path, incoerente)


def test_o_mesmo_quadro_duas_vezes_levanta(tmp_path: Path) -> None:
    with pytest.raises(PoseLabelError, match="duas vezes"):
        _labels(tmp_path, _documento(), _documento())


def test_caixa_invertida_levanta(tmp_path: Path) -> None:
    with pytest.raises(PoseLabelError, match="shooter_box"):
        _labels(tmp_path, _documento(shooter_box=[140.0, 330.0, 100.0, 200.0]))


def test_arquivo_ausente_levanta(tmp_path: Path) -> None:
    with pytest.raises(PoseLabelError, match="nao existe"):
        read_pose_labels(tmp_path / "nao-existe.jsonl")


def _clip() -> Clip:
    return Clip(
        clip_id=CLIPE,
        path="x.mp4",
        fps=59.94,
        frame_count=300,
        width=1920,
        height=1080,
        match_id=PARTIDA,
    )


def test_a_anotacao_e_conferida_contra_o_video_antes_da_inferencia(tmp_path: Path) -> None:
    """Video rebaixado produz erro em pixel medido contra coordenada de outro video, calado."""
    label = _labels(tmp_path, _documento())[0]

    validate_label_video(label, _clip(), "a" * 64)

    with pytest.raises(PoseLabelError, match="video_sha256"):
        validate_label_video(label, _clip(), "b" * 64)
    with pytest.raises(PoseLabelError, match="dimensoes"):
        validate_label_video(label, replace(_clip(), width=1280), "a" * 64)
    with pytest.raises(PoseLabelError, match="fora do video"):
        validate_label_video(label, replace(_clip(), frame_count=QUADRO), "a" * 64)
    with pytest.raises(PoseLabelError, match="outro clipe"):
        validate_label_video(label, replace(_clip(), clip_id="outro"), "a" * 64)
