"""Nenhuma feature declarada alcança o dado do parceiro enquanto mede.

O defeito que isto fecha reporta acurácia perfeita medindo zero. Três dos sete rótulos já estão
resolvidos por coluna tabular: `cortes_metadados.foot` reproduz `body_part` em 633 de 633,
`play_type` é `origin_of_move` renomeado no mesmo total, e `result` determina `is_goal`. Uma
feature que leia qualquer um desses arquivos acerta tudo, e nem o schema nem o `xvision eval` têm
como distinguir isso de um modelo bom.

A defesa não pode ser disciplina, e o projeto já aprendeu isso uma vez: o catálogo usa
`match_shots.id` como `clip_id` e nunca o nome do arquivo, porque o quarto campo do nome carrega o
desfecho da finalização e ele viajaria até a chave do cache.

O mecanismo é o mesmo que `edge.modules.conformance` usa para provar que módulo não abre vídeo.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import Any, cast

import pytest

from contracts import BallDetection, Box, FramePerception, Keypoint, Perception, PlayerDetection
from edge.features.base import run_features
from edge.features.registry import load_extractors, read_feature_config

CONFIG = Path("config/pipeline.toml")
CONTATO = 60

_lidos: list[list[str]] = []
_instalado = False


def _auditor(evento: str, argumentos: tuple[Any, ...]) -> None:
    if not _lidos or evento not in {"open", "os.open"}:
        return
    alvo = argumentos[0]
    if isinstance(alvo, os.PathLike):
        alvo = cast("os.PathLike[str]", alvo).__fspath__()
    if not isinstance(alvo, str):
        return
    partes = PurePath(alvo).parts
    if "footstats" in partes or "data" in partes:
        for registro in _lidos:
            registro.append(alvo)


@contextmanager
def _sem_acesso_ao_conjunto() -> Generator[list[str]]:
    global _instalado
    if not _instalado:
        sys.addaudithook(_auditor)
        _instalado = True
    registro: list[str] = []
    _lidos.append(registro)
    try:
        yield registro
    finally:
        _lidos.remove(registro)


class _Shot:
    frame_index = CONTATO
    shooter_track_id = None


def _percepcao() -> Perception:
    corpo = (
        Keypoint("nose", 110.0, 95.0, 0.9),
        Keypoint("ankle_left", 100.0, 215.0, 0.9),
        Keypoint("ankle_right", 120.0, 215.0, 0.9),
    )
    jogador = PlayerDetection(
        box=Box(90.0, 90.0, 130.0, 220.0), confidence=0.9, track_id=1, keypoints=corpo
    )
    quadros = tuple(
        FramePerception(
            frame_index=CONTATO - i * 12,
            timestamp_s=(CONTATO - i * 12) / 59.94,
            players=(jogador,) if i == 0 else (),
            ball=BallDetection(
                box=Box(115.0 + i * 30, 211.0, 125.0 + i * 30, 221.0), confidence=0.8
            ),
        )
        for i in range(5)
    )
    return Perception(clip_id="c", frames=tuple(reversed(quadros)))


@pytest.fixture(scope="module")
def selecao() -> tuple[Any, ...]:
    return load_extractors(read_feature_config(CONFIG))


def test_a_selecao_declarada_nao_esta_vazia(selecao: tuple[Any, ...]) -> None:
    """Sem esta guarda, o teste abaixo passaria sem exercitar feature nenhuma."""
    assert selecao


def test_nenhuma_feature_toca_o_conjunto_do_parceiro_enquanto_mede(
    selecao: tuple[Any, ...],
) -> None:
    produzidas = frozenset({"players", "ball", "field", "tracking", "pose", "identity", "shot"})

    with _sem_acesso_ao_conjunto() as lidos:
        valores = run_features(selecao, _percepcao(), _Shot(), produzidas)

    assert valores
    assert lidos == [], f"feature leu o conjunto do parceiro: {lidos}"


def test_o_hook_pega_a_leitura_que_ele_diz_pegar(tmp_path: Path) -> None:
    """Prova vermelho-verde do próprio guarda: sem isto ele passaria por não observar nada.

    O arquivo é criado aqui e não lido do dump real, porque o `conftest` aponta
    `XVISION_DATA_DIR` para lugar nenhum de propósito: a suíte inteira nunca toca o material
    restrito. O que se prova é o mecanismo, e ele reage ao nome do diretório.
    """
    parecido = tmp_path / "footstats" / "db" / "0005_match_shots_1390.csv"
    parecido.parent.mkdir(parents=True)
    parecido.write_text("id,is_goal\n1,true\n", encoding="utf-8")

    with _sem_acesso_ao_conjunto() as lidos:
        parecido.open("rb").close()

    assert lidos
