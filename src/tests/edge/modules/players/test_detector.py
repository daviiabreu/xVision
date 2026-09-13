"""A especie `players`: caixa por jogador, em ordem estavel, no espaco do quadro original.

A ordem importa mais do que parece: `detection_index` e a posicao nesta lista, e e por ela que
`tracking` e `pose` dizem de quem e o `track_id` e de quem e o corpo. Ordem instavel entre duas
execucoes poe o numero e a pose no jogador errado, sem levantar nada.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from contracts import Box
from edge.modules.base import FrameBatch, ModuleNotLoaded, PerceptionModule
from edge.modules.conformance import ModuleConformance
from edge.modules.players import COCO_PERSON, FUTEBOL, PREPROCESS, PlayersConfig, YoloPlayers
from edge.modules.players.detector import PlayersConfigError
from edge.modules.records import check_records

FONTE = (1920, 1080)


class _BackendFalso:
    def __init__(self, deteccoes: tuple[tuple[Box, float, int], ...]) -> None:
        self._deteccoes = deteccoes

    def detect(self, frame: Any) -> tuple[tuple[Box, float, int], ...]:
        return self._deteccoes


def _lote(indices: tuple[int, ...] = (0,), fonte: tuple[int, int] | None = FONTE) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS, source_size=fonte)


class TestYoloPlayers(ModuleConformance):
    @pytest.fixture
    def module(self) -> PerceptionModule:
        return YoloPlayers()


def test_sem_peso_a_especie_sai_sem_deteccao() -> None:
    modulo = YoloPlayers()
    modulo.load()

    assert modulo.infer(_lote((0, 1)), {}).records == ((), ())


def test_a_caixa_volta_para_o_espaco_do_quadro_original() -> None:
    meio_x, meio_y = PREPROCESS.width / 2.0, PREPROCESS.height / 2.0
    modulo = YoloPlayers(
        _BackendFalso(((Box(meio_x - 20, meio_y - 60, meio_x + 20, meio_y + 60), 0.9, 0),))
    )
    modulo.load()

    (registro,) = modulo.infer(_lote(), {}).records[0]

    x1, _y1, x2, _y2 = registro["box"]  # type: ignore[misc]
    assert (x1 + x2) / 2 == pytest.approx(FONTE[0] / 2.0, abs=4.0)


def test_classe_fora_do_mapa_nao_vira_jogador() -> None:
    """Com o peso de futebol, a classe 0 e a bola: contar bola como gente estraga toda contagem."""
    modulo = YoloPlayers(
        _BackendFalso(((Box(100.0, 100.0, 140.0, 220.0), 0.9, 0),)),
        config=PlayersConfig(classes=FUTEBOL),
    )
    modulo.load()

    assert modulo.infer(_lote(), {}).records == ((),)


def test_o_papel_sai_so_quando_o_peso_sabe_distinguir() -> None:
    """COCO so tem `person`, e chamar todo mundo de `player` poria o arbitro na conta."""
    caixa = Box(100.0, 100.0, 140.0, 220.0)
    coco = YoloPlayers(_BackendFalso(((caixa, 0.9, 0),)), config=PlayersConfig(classes=COCO_PERSON))
    futebol = YoloPlayers(_BackendFalso(((caixa, 0.9, 3),)), config=PlayersConfig(classes=FUTEBOL))
    coco.load()
    futebol.load()

    (sem_papel,) = coco.infer(_lote(), {}).records[0]
    (com_papel,) = futebol.infer(_lote(), {}).records[0]

    assert "role" not in sem_papel
    assert com_papel["role"] == "referee"


def test_deteccao_abaixo_do_piso_de_confianca_e_descartada() -> None:
    modulo = YoloPlayers(
        _BackendFalso(((Box(100.0, 100.0, 140.0, 220.0), 0.05, 0),)),
        config=PlayersConfig(conf=0.25),
    )
    modulo.load()

    assert modulo.infer(_lote(), {}).records == ((),)


def test_a_ordem_e_estavel_e_por_confianca() -> None:
    """`detection_index` e a posicao nesta lista, e ordem instavel troca o dono do `track_id`."""
    modulo = YoloPlayers(
        _BackendFalso(
            (
                (Box(100.0, 100.0, 140.0, 220.0), 0.5, 0),
                (Box(300.0, 100.0, 340.0, 220.0), 0.9, 0),
                (Box(500.0, 100.0, 540.0, 220.0), 0.7, 0),
            )
        )
    )
    modulo.load()

    registros = modulo.infer(_lote(), {}).records[0]

    assert [r["confidence"] for r in registros] == [0.9, 0.7, 0.5]


def test_duas_classes_sobre_a_mesma_pessoa_mantem_a_maior_confianca() -> None:
    modulo = YoloPlayers(
        _BackendFalso(
            (
                (Box(100.0, 100.0, 140.0, 220.0), 0.92, 3),
                (Box(101.0, 102.0, 141.0, 222.0), 0.70, 2),
                (Box(400.0, 100.0, 440.0, 220.0), 0.88, 2),
            )
        ),
        config=PlayersConfig(classes=FUTEBOL),
    )
    modulo.load()

    registros = modulo.infer(_lote(fonte=None), {}).records[0]

    assert [(r["role"], r["confidence"]) for r in registros] == [
        ("referee", 0.92),
        ("player", 0.88),
    ]


def test_jogadores_aglomerados_nao_sao_removidos_por_deslocamento_parcial() -> None:
    modulo = YoloPlayers(
        _BackendFalso(
            (
                (Box(10.0, 10.0, 30.0, 50.0), 0.90, 2),
                (Box(11.0, 25.0, 31.0, 70.0), 0.85, 2),
            )
        ),
        config=PlayersConfig(classes=FUTEBOL),
    )
    modulo.load()

    registros = modulo.infer(_lote(fonte=None), {}).records[0]

    assert len(registros) == 2


def test_empate_de_confianca_desempata_pela_caixa() -> None:
    """Sem desempate, duas caixas de confianca igual poderiam trocar de posicao entre execucoes."""
    modulo = YoloPlayers(
        _BackendFalso(
            (
                (Box(500.0, 100.0, 540.0, 220.0), 0.8, 0),
                (Box(100.0, 100.0, 140.0, 220.0), 0.8, 0),
            )
        )
    )
    modulo.load()

    registros = modulo.infer(_lote(), {}).records[0]

    # ja no espaco do quadro original: o letterbox 1280x736 sobre 1920x1080 escala por 1,5
    assert [r["box"][0] for r in registros] == [150.0, 750.0]  # type: ignore[index]


def test_o_registro_cumpre_o_que_a_especie_promete() -> None:
    modulo = YoloPlayers(_BackendFalso(((Box(100.0, 100.0, 140.0, 220.0), 0.9, 0),)))
    modulo.load()

    check_records(modulo.infer(_lote((0, 1)), {}))


def test_infer_sem_load_recusa() -> None:
    with pytest.raises(ModuleNotLoaded):
        YoloPlayers().infer(_lote(), {})


def test_close_solta_o_backend() -> None:
    modulo = YoloPlayers(_BackendFalso(((Box(100.0, 100.0, 140.0, 220.0), 0.9, 0),)))
    modulo.load()
    modulo.close()

    with pytest.raises(ModuleNotLoaded):
        modulo.infer(_lote(), {})


def test_a_precisao_declarada_no_toml_chega_ao_adaptador(monkeypatch: pytest.MonkeyPatch) -> None:
    """A via e a mesma dos outros ajustes: o registro entrega `options` ao `configure`."""
    from pathlib import Path

    from edge.modules.players.yolo import UltralyticsPlayers

    chamada: dict[str, object] = {}

    def montar(weights: str, **kwargs: object) -> object:
        chamada.update(weights=weights, **kwargs)
        return _BackendFalso(())

    def entrypoint(artifact: str | None) -> Path:
        return Path("p.pt")

    monkeypatch.setattr("edge.modules.players.detector.entrypoint_of", entrypoint)
    monkeypatch.setattr(UltralyticsPlayers, "from_weights", montar)
    modulo = YoloPlayers()
    modulo.configure({"quantize": "fp16"})

    modulo.load()

    assert chamada["quantize"] == "fp16"


def test_sem_declaracao_o_adaptador_recebe_precisao_cheia(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from edge.modules.players.yolo import UltralyticsPlayers

    chamada: dict[str, object] = {}

    def montar(weights: str, **kwargs: object) -> object:
        chamada.update(weights=weights, **kwargs)
        return _BackendFalso(())

    def entrypoint(artifact: str | None) -> Path:
        return Path("p.pt")

    monkeypatch.setattr("edge.modules.players.detector.entrypoint_of", entrypoint)
    monkeypatch.setattr(UltralyticsPlayers, "from_weights", montar)

    YoloPlayers().load()

    assert chamada["quantize"] == "fp32"


@pytest.mark.parametrize("valor", ["fp8", True, 16, "half"])
def test_precisao_invalida_na_secao_de_players_levanta(valor: object) -> None:
    with pytest.raises(PlayersConfigError, match="fp32.*fp16"):
        YoloPlayers().configure({"quantize": valor})


def test_ajuste_desconhecido_na_secao_de_players_levanta_em_vez_de_ficar_calado() -> None:
    """`quantise` no lugar de `quantize` deixaria a especie em fp32 sem ninguem saber."""
    with pytest.raises(PlayersConfigError, match="quantise"):
        YoloPlayers().configure({"quantise": "fp16"})


def test_a_precisao_declarada_muda_a_chave_do_cache_de_percepcao() -> None:
    """Sem isto a alavanca fica muda: o cache responde com a percepcao da outra precisao.

    `run_module` monta a chave de disco com `tuning_of`, que le `fingerprint` por `getattr` e
    devolve `None` para quem nao tem o metodo. Sem ele, rodar o clipe no padrao e depois declarar
    `quantize = "fp16"` cai na mesma chave e o modelo nem e chamado.
    """
    from edge.cache.perception import perception_key, tuning_of

    def chave(modulo: YoloPlayers) -> str:
        return perception_key("clipe", modulo.spec, (0, 1), {}, "impl", tuning=tuning_of(modulo))

    cheia = YoloPlayers()
    meia = YoloPlayers()
    meia.configure({"quantize": "fp16"})

    assert tuning_of(cheia) is None, "quem nao declarou precisao fica no endereco de sempre"
    assert tuning_of(meia) == "quantize=fp16"
    assert chave(cheia) != chave(meia)
