"""A especie `pose`: o corpo do finalizador, e o silencio quando nao da para saber quem ele e.

O modo de falha que estes testes fecham nao levanta excecao: keypoint do jogador errado chega a
feature como um corpo plausivel virado para o lado errado, e `looked_before` mede o marcador em vez
do finalizador. Nada no schema nem no relatorio distingue isso de uma medida.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from edge.modules.base import FrameBatch, ModuleNotLoaded, ModuleOutput, PerceptionModule
from edge.modules.conformance import ModuleConformance
from edge.modules.pose import COCO_KEYPOINTS, PREPROCESS, Keypoints, YoloPose, keypoints_de
from edge.modules.pose.estimator import PoseConfigError
from edge.modules.records import check_records

FONTE = (1920, 1080)
CONTATO = 4
ATIRADOR = 7
OUTRO = 9


class _EstimadorFalso:
    """Marca cada keypoint com o canto do recorte, para o teste seguir a volta de coordenada."""

    def __init__(self, *, responde: bool = True) -> None:
        self.recortes: list[Any] = []
        self._responde = responde

    def estimate_many(self, crops: Sequence[Any]) -> tuple[Keypoints | None, ...]:
        self.recortes.extend(crops)
        if not self._responde:
            return tuple(None for _ in crops)
        return tuple(tuple((nome, 1.0, 2.0, 0.9) for nome in COCO_KEYPOINTS) for _ in crops)


def _lote(indices: tuple[int, ...], fonte: tuple[int, int] | None = FONTE) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS, source_size=fonte)


def _players(indices: tuple[int, ...], caixas: list[tuple[float, float, float, float]]):
    return ModuleOutput(
        kind="players",
        frames=indices,
        records=tuple(
            tuple({"box": caixa, "confidence": 0.9} for caixa in caixas) for _ in indices
        ),
    )


def _tracking(indices: tuple[int, ...], ids: list[int]) -> ModuleOutput:
    return ModuleOutput(
        kind="tracking",
        frames=indices,
        records=tuple(
            tuple({"detection_index": i, "track_id": t} for i, t in enumerate(ids)) for _ in indices
        ),
    )


def _shot(frame_index: int = CONTATO, shooter_track_id: int | None = None) -> ModuleOutput:
    registro: dict[str, object] = {"frame_index": frame_index}
    if shooter_track_id is not None:
        registro["shooter_track_id"] = shooter_track_id
    return ModuleOutput(kind="shot", frames=(0,), records=((registro,),))


def _ball(centro: tuple[float, float], quadro: int = CONTATO) -> ModuleOutput:
    x, y = centro
    return ModuleOutput(
        kind="ball",
        frames=(quadro,),
        records=(({"box": (x - 5, y - 5, x + 5, y + 5), "confidence": 0.9},),),
    )


PERTO = (900.0, 400.0, 960.0, 600.0)
LONGE = (100.0, 400.0, 160.0, 600.0)


def _ctx(estimador_ball: ModuleOutput | None = None, **troca: Any) -> dict[str, ModuleOutput]:
    indices = (CONTATO,)
    base = {
        "players": _players(indices, [PERTO, LONGE]),
        "tracking": _tracking(indices, [ATIRADOR, OUTRO]),
        "shot": _shot(),
        "ball": estimador_ball or _ball((930.0, 500.0)),
    }
    base.update(troca)
    return base


class TestYoloPose(ModuleConformance):
    @pytest.fixture
    def module(self) -> PerceptionModule:
        return YoloPose()


def test_mede_o_jogador_mais_perto_da_bola_no_contato() -> None:
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()

    saida = modulo.infer(_lote((CONTATO,)), _ctx())

    (registro,) = saida.records[0]
    assert registro["track_id"] == ATIRADOR
    assert len(registro["keypoints"]) == len(COCO_KEYPOINTS)  # type: ignore[arg-type]


def test_a_bola_do_outro_lado_muda_quem_e_o_finalizador() -> None:
    """Prova que a escolha e geometrica, e nao a primeira deteccao da lista."""
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()

    saida = modulo.infer(_lote((CONTATO,)), _ctx(_ball((130.0, 500.0))))

    assert saida.records[0][0]["track_id"] == OUTRO


def test_prefere_o_finalizador_declarado_pelo_shot() -> None:
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()

    saida = modulo.infer(
        _lote((CONTATO,)),
        _ctx(_ball((130.0, 500.0)), shot=_shot(shooter_track_id=ATIRADOR)),
    )

    assert saida.records[0][0]["track_id"] == ATIRADOR


def test_recorta_so_o_finalizador_e_nao_os_vinte_e_dois() -> None:
    """Recortar todo mundo custaria uma inferencia por jogador por quadro."""
    estimador = _EstimadorFalso()
    modulo = YoloPose(estimador)
    modulo.load()
    indices = (CONTATO, CONTATO + 1, CONTATO + 2)

    modulo.infer(
        _lote(indices),
        _ctx(
            players=_players(indices, [PERTO, LONGE]),
            tracking=_tracking(indices, [ATIRADOR, OUTRO]),
        ),
    )

    assert len(estimador.recortes) == len(indices)


def test_sem_instante_nao_ha_de_quem_medir() -> None:
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()

    saida = modulo.infer(_lote((CONTATO,)), _ctx(shot=ModuleOutput("shot", (0,), ((),))))

    assert saida.records == ((),)


def test_sem_bola_nao_ha_como_dizer_quem_chutou() -> None:
    """Chutar o mais central poria a pose no corpo errado, com a mesma cara de uma medida."""
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()
    ctx = _ctx()
    del ctx["ball"]

    assert modulo.infer(_lote((CONTATO,)), ctx).records == ((),)


def test_sem_tracking_nao_ha_dono_para_o_corpo() -> None:
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()
    ctx = _ctx()
    del ctx["tracking"]

    assert modulo.infer(_lote((CONTATO,)), ctx).records == ((),)


def test_sem_estimador_a_especie_sai_sem_registro() -> None:
    modulo = YoloPose()
    modulo.load()

    assert modulo.infer(_lote((CONTATO,)), _ctx()).records == ((),)


def test_recorte_sem_corpo_nao_vira_keypoint_inventado() -> None:
    modulo = YoloPose(_EstimadorFalso(responde=False))
    modulo.load()

    assert modulo.infer(_lote((CONTATO,)), _ctx()).records == ((),)


def test_o_keypoint_volta_para_o_espaco_do_quadro_original() -> None:
    """O estimador ve o recorte, e a feature mede em pixel do quadro original."""
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()

    saida = modulo.infer(_lote((CONTATO,)), _ctx())

    _nome, x, y, _c = saida.records[0][0]["keypoints"][0]  # type: ignore[index]
    assert PERTO[0] - 100 < x < PERTO[2] + 100
    assert PERTO[1] - 100 < y < PERTO[3] + 100


def test_o_recorte_cresce_alem_da_caixa() -> None:
    """Sem margem o estimador perde pe e cabeca, que sao os dois pontos que as features consomem."""
    estimador = _EstimadorFalso()
    modulo = YoloPose(estimador)
    modulo.load()

    modulo.infer(_lote((CONTATO,)), _ctx())

    altura_da_caixa = (PERTO[3] - PERTO[1]) * PREPROCESS.width / FONTE[0]
    assert estimador.recortes[0].shape[0] > altura_da_caixa


def test_o_registro_cumpre_o_que_a_especie_promete() -> None:
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()

    check_records(modulo.infer(_lote((CONTATO,)), _ctx()))


def test_quadro_em_que_o_finalizador_sumiu_sai_sem_registro() -> None:
    """Oclusao e ausencia declarada, e nunca a caixa do quadro anterior repetida."""
    estimador = _EstimadorFalso()
    modulo = YoloPose(estimador)
    modulo.load()
    indices = (CONTATO, CONTATO + 1)
    players = ModuleOutput(
        kind="players",
        frames=indices,
        records=(
            ({"box": PERTO, "confidence": 0.9}, {"box": LONGE, "confidence": 0.9}),
            (),
        ),
    )
    tracking = ModuleOutput(
        kind="tracking",
        frames=indices,
        records=(
            (
                {"detection_index": 0, "track_id": ATIRADOR},
                {"detection_index": 1, "track_id": OUTRO},
            ),
            (),
        ),
    )

    saida = modulo.infer(_lote(indices), _ctx(players=players, tracking=tracking))

    assert len(saida.records[0]) == 1
    assert saida.records[1] == ()


def test_infer_sem_load_recusa() -> None:
    with pytest.raises(ModuleNotLoaded):
        YoloPose().infer(_lote((0,)), {})


def test_keypoints_de_nomeia_a_matriz_na_ordem_do_coco() -> None:
    linha = [[float(i), float(i), 0.5] for i in range(len(COCO_KEYPOINTS))]

    nomeados = keypoints_de(linha)

    assert [n for n, _, _, _ in nomeados] == list(COCO_KEYPOINTS)
    assert nomeados[15][0] == "ankle_left"
    assert nomeados[16][0] == "ankle_right"


def test_close_solta_o_estimador() -> None:
    modulo = YoloPose(_EstimadorFalso())
    modulo.load()
    modulo.close()

    with pytest.raises(ModuleNotLoaded):
        modulo.infer(_lote((0,)), {})


def test_a_precisao_declarada_no_toml_chega_ao_estimador(monkeypatch: pytest.MonkeyPatch) -> None:
    """A via e a mesma das outras especies: o registro entrega `options` ao `configure`."""
    from edge.modules.pose.yolo import YoloPoseEstimator

    chamada: dict[str, object] = {}

    def montar(weights: str, **kwargs: object) -> object:
        chamada.update(weights=weights, **kwargs)
        return _EstimadorFalso()

    monkeypatch.setattr(YoloPoseEstimator, "from_weights", montar)
    modulo = YoloPose(weights="pose.pt")
    modulo.configure({"quantize": "fp16"})

    modulo.load()

    assert chamada == {"weights": "pose.pt", "quantize": "fp16"}


def test_sem_declaracao_o_estimador_recebe_precisao_cheia(monkeypatch: pytest.MonkeyPatch) -> None:
    from edge.modules.pose.yolo import YoloPoseEstimator

    chamada: dict[str, object] = {}

    def montar(weights: str, **kwargs: object) -> object:
        chamada.update(weights=weights, **kwargs)
        return _EstimadorFalso()

    monkeypatch.setattr(YoloPoseEstimator, "from_weights", montar)

    YoloPose(weights="pose.pt").load()

    assert chamada == {"weights": "pose.pt", "quantize": "fp32"}


@pytest.mark.parametrize("valor", ["fp8", True, 16, "half"])
def test_precisao_invalida_na_secao_de_pose_levanta(valor: object) -> None:
    with pytest.raises(PoseConfigError, match="fp32.*fp16"):
        YoloPose().configure({"quantize": valor})


def test_ajuste_desconhecido_na_secao_de_pose_levanta_em_vez_de_ficar_calado() -> None:
    with pytest.raises(PoseConfigError, match="quantise"):
        YoloPose().configure({"quantise": "fp16"})


def test_a_precisao_declarada_muda_a_chave_do_cache_de_percepcao() -> None:
    """Mesmo motivo de `players`: keypoint de fp32 guardado nao responde por um pedido de fp16."""
    from edge.cache.perception import perception_key, tuning_of

    def chave(modulo: YoloPose) -> str:
        return perception_key("clipe", modulo.spec, (0, 1), {}, "impl", tuning=tuning_of(modulo))

    cheia = YoloPose()
    meia = YoloPose()
    meia.configure({"quantize": "fp16"})

    assert tuning_of(cheia) is None, "quem nao declarou precisao fica no endereco de sempre"
    assert tuning_of(meia) == "quantize=fp16"
    assert chave(cheia) != chave(meia)
