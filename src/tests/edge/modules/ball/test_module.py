"""A casca `PerceptionModule` em volta do detector e do rastreador do Gabriel.

O nucleo dele ja tem teste proprio, portado junto em `test_ball_tracker.py` e
`test_ball_shape_filter.py`. Estes aqui cobram so o que a passagem para a fundacao acrescentou: o
espaco da caixa, o ponto interpolado que derrubaria o `assemble`, e a ausencia declarada.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from contracts import BallDetection, Box
from edge.modules.ball import DetectorConfig
from edge.modules.ball.detector import AjusteDesconhecido
from edge.modules.ball.module import PREPROCESS, YoloBall, chave_do_cache
from edge.modules.base import FrameBatch, ModuleNotLoaded, PerceptionModule
from edge.modules.conformance import ModuleConformance
from edge.modules.records import check_records

FONTE = (1920, 1080)


class _DetectorFalso:
    """Devolve sempre a mesma caixa, no espaco do tensor pre-processado."""

    def __init__(self, caixa: Box | None, confianca: float = 0.8) -> None:
        self._caixa = caixa
        self._confianca = confianca

    def detect_raw(self, frame: Any) -> tuple[tuple[Box, float, int], ...]:
        if self._caixa is None:
            return ()
        return ((self._caixa, self._confianca, 32),)

    def candidates_from(
        self, cruas: Sequence[tuple[Box, float, int]], frame_height: float
    ) -> tuple[BallDetection, ...]:
        return tuple(BallDetection(box=c, confidence=k) for c, k, _ in cruas)

    def detect_candidates(self, frame: Any) -> tuple[BallDetection, ...]:
        return self.candidates_from(self.detect_raw(frame), 0.0)


def _lote(indices: tuple[int, ...], fonte: tuple[int, int] | None = FONTE) -> FrameBatch:
    tensor = np.zeros((len(indices), PREPROCESS.height, PREPROCESS.width, 3), dtype=np.uint8)
    return FrameBatch(indices=indices, tensor=tensor, preprocess=PREPROCESS, source_size=fonte)


def _centro(caixa: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = caixa
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class TestYoloBall(ModuleConformance):
    @pytest.fixture
    def module(self) -> PerceptionModule:
        return YoloBall()


def test_sem_detector_a_especie_sai_sem_posicao() -> None:
    """Peso ausente nao pode virar bola no canto do campo: sai sem registro, e o cliente ve."""
    modulo = YoloBall()
    modulo.load()

    saida = modulo.infer(_lote((0, 12, 24)), {})

    assert saida.frames == (0, 12, 24)
    assert saida.records == ((), (), ())


def test_a_caixa_volta_para_o_espaco_do_quadro_original() -> None:
    """O detector ve 1280x736 e o `records.py` cobra pixel do quadro original.

    Sem a volta, a bola no centro do tensor viraria bola em (640, 368) para a feature, que num
    quadro de 1920x1080 e o quadrante superior esquerdo.
    """
    meio = PREPROCESS.width / 2.0
    altura_conteudo = PREPROCESS.width * FONTE[1] / FONTE[0]
    meio_y = PREPROCESS.height / 2.0
    modulo = YoloBall(_DetectorFalso(Box(meio - 6, meio_y - 6, meio + 6, meio_y + 6)))
    modulo.load()

    saida = modulo.infer(_lote((0,)), {})

    (registro,) = saida.records[0]
    x, y = _centro(registro["box"])  # type: ignore[arg-type]
    assert altura_conteudo < PREPROCESS.height
    assert x == pytest.approx(FONTE[0] / 2.0, abs=4.0)
    assert y == pytest.approx(FONTE[1] / 2.0, abs=4.0)


def test_caixa_grande_demais_e_reprovada_pelo_filtro_de_forma() -> None:
    """O filtro do Gabriel roda depois da volta, senao a fracao de altura mede outra coisa."""
    modulo = YoloBall(_DetectorFalso(Box(100.0, 100.0, 700.0, 700.0)))
    modulo.load()

    saida = modulo.infer(_lote((0,)), {})

    assert saida.records == ((),)


def test_candidate_with_center_outside_source_frame_is_rejected() -> None:
    modulo = YoloBall(_DetectorFalso(Box(-30.0, 100.0, -10.0, 120.0)))
    modulo.load()

    saida = modulo.infer(_lote((0,)), {})

    assert saida.records == ((),)


def test_oregistro_de_cumpre_o_que_a_especie_promete() -> None:
    meio = PREPROCESS.width / 2.0
    modulo = YoloBall(_DetectorFalso(Box(meio - 6, meio - 6, meio + 6, meio + 6)))
    modulo.load()

    check_records(modulo.infer(_lote((0, 12)), {}))


def test_ponto_interpolado_sai_com_caixa_e_nao_com_none() -> None:
    """`assemble` faz `Box(*registro["box"])` fora do try da etapa: um None derruba o clipe.

    O `TrackPoint` interpolado da branch de origem sai com `box=None`, entao a casca precisa
    sintetizar a caixa em volta do centro e declarar que ela e interpolada.
    """
    from edge.modules.ball.module import registro_de
    from edge.modules.ball.tracking import TrackPoint

    ponto = TrackPoint(
        frame_index=4,
        timestamp_s=4.0,
        source="interpolated",
        center=(100.0, 200.0),
        box=None,
        confidence=None,
        trajectory_segment=0,
        gap_frames=1,
        velocity_xy_px_s=(30.0, -15.0),
        velocity_source="trajectory-model",
    )

    (registro,) = registro_de(ponto)

    assert registro["source"] == "interpolated"
    assert registro["trajectory_segment"] == 0
    assert registro["gap_frames"] == 1
    assert registro["velocity_xy_px_s"] == (30.0, -15.0)
    assert registro["velocity_source"] == "trajectory-model"
    assert _centro(registro["box"]) == (100.0, 200.0)  # type: ignore[arg-type]
    assert isinstance(registro["confidence"], float)


def test_lacuna_longa_nao_vira_caixa_de_zeros() -> None:
    from edge.modules.ball.module import registro_de
    from edge.modules.ball.tracking import TrackPoint

    perdido = TrackPoint(
        frame_index=4, timestamp_s=4.0, source="missing", center=None, box=None, confidence=None
    )

    assert registro_de(perdido) == ()
    assert registro_de(None) == ()


def test_infer_sem_load_recusa() -> None:
    with pytest.raises(ModuleNotLoaded):
        YoloBall().infer(_lote((0,)), {})


def test_lote_sem_tamanho_de_origem_nao_estoura() -> None:
    """Lote montado a mao nao carrega `source_size`, e o modulo nao pode morrer por isso."""
    meio = PREPROCESS.width / 2.0
    modulo = YoloBall(_DetectorFalso(Box(meio - 6, meio - 6, meio + 6, meio + 6)))
    modulo.load()

    saida = modulo.infer(_lote((0,), fonte=None), {})

    assert len(saida.records[0]) == 1


def test_close_solta_o_detector() -> None:
    modulo = YoloBall(_DetectorFalso(Box(0.0, 0.0, 10.0, 10.0)))
    modulo.load()
    modulo.close()

    with pytest.raises(ModuleNotLoaded):
        modulo.infer(_lote((0,)), {})


def test_o_detector_yolo_so_e_construido_quando_ha_configuracao() -> None:
    """Sem `DetectorConfig` o `load` nao importa ultralytics, e a conformidade roda sem o extra."""
    modulo = YoloBall()
    modulo.load()

    assert modulo.infer(_lote((0,)), {}).records == ((),)
    assert DetectorConfig(weights="x.pt").ball_class_id == 32


ASPECTO_BORRADO = 2.05
"""Aspecto de bola borrada pelo chute, dentro da faixa medida no clipe 4996030 (1,75 a 2,09)."""


def _caixa_com_aspecto(aspecto: float, lado_maior: float) -> Box:
    """Caixa no espaco do tensor.

    O letterbox escala igual nos dois eixos, entao o aspecto sobrevive a volta para o quadro
    original e o teste pode escolher exatamente o numero que quer medir.
    """
    meio_x, meio_y = PREPROCESS.width / 2.0, PREPROCESS.height / 2.0
    altura = lado_maior / aspecto
    return Box(
        meio_x - lado_maior / 2, meio_y - altura / 2, meio_x + lado_maior / 2, meio_y + altura / 2
    )


def test_o_limite_de_aspecto_do_toml_alcanca_o_filtro_no_espaco_do_quadro() -> None:
    """O filtro de forma roda duas vezes, e o ajuste precisa valer nas duas.

    A segunda passada, ja em pixel do quadro original, montava um `DetectorConfig` do zero quando o
    modulo nao tinha recebido configuracao por parametro. Com isso um `max_aspect` declarado no
    TOML valia so na primeira passada, a caixa borrada era descartada de novo pelo padrao 2,0, e
    nada no arquivo de configuracao indicava o motivo.
    """
    modulo = YoloBall(_DetectorFalso(_caixa_com_aspecto(ASPECTO_BORRADO, 27.0)))
    modulo.configure({"max_aspect": 2.2})
    modulo.load()

    assert len(modulo.infer(_lote((0,)), {}).records[0]) == 1


def test_sem_ajuste_o_padrao_de_hoje_continua_valendo() -> None:
    """O padrao 2,0 recusa a mesma caixa: e o que prova que o teste acima mede o ajuste."""
    modulo = YoloBall(_DetectorFalso(_caixa_com_aspecto(ASPECTO_BORRADO, 27.0)))
    modulo.load()

    assert modulo.infer(_lote((0,)), {}).records[0] == ()


def test_o_limite_de_velocidade_chega_ao_rastreador_e_muda_a_chave_do_cache() -> None:
    """Sem entrar na impressao digital, mudar o limite pelo TOML reusaria a saida do valor
    antigo, e a medicao registraria a mudanca que nao aconteceu."""
    module = YoloBall(_DetectorFalso(None))
    baseline_fingerprint = module.fingerprint()

    module.configure({"max_ball_speed_px_per_s": 1800})

    assert module.fingerprint() != baseline_fingerprint


@pytest.mark.parametrize(
    "options",
    [
        {"max_ball_speed_px_per_s": True},
        {"max_ball_speed_px_per_s": -1},
        {"max_ball_speed_px_per_s": float("nan")},
    ],
)
def test_invalid_temporal_configuration_is_not_silently_accepted(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="ball"):
        YoloBall().configure(options)


def test_ajuste_desconhecido_no_toml_levanta_em_vez_de_ficar_calado() -> None:
    """`confidence` no lugar de `conf` deixaria o detector no padrao e a medicao mentiria."""
    modulo = YoloBall()
    modulo.configure({"confidence": 0.10})

    with pytest.raises(AjusteDesconhecido, match="confidence"):
        modulo.load()


def test_a_chave_do_cache_compartilhado_separa_conf_e_imgsz() -> None:
    """O `PreprocessSpec` nao carrega `conf`, e `ball` e `players` usam o mesmo peso.

    Com a chave sendo so o caminho do peso, a passada de `players` a 0,25 servia `ball` a 0,10 e o
    limiar novo nao mudava um candidato sequer.
    """
    peso = "var/models/abc/yolov8n.pt"
    baixa = chave_do_cache(DetectorConfig(weights=peso, conf=0.10))
    alta = chave_do_cache(DetectorConfig(weights=peso, conf=0.25))
    outro_tamanho = chave_do_cache(DetectorConfig(weights=peso, conf=0.10, imgsz=1920))

    assert baixa != alta
    assert baixa != outro_tamanho
    assert peso not in (baixa, alta, outro_tamanho)


def test_com_ajustes_recusa_valor_que_nao_e_numero() -> None:
    with pytest.raises(AjusteDesconhecido, match="conf"):
        DetectorConfig(weights="x.pt").com_ajustes({"conf": "0.10"})


def test_com_ajustes_aplica_conf_e_preserva_o_resto() -> None:
    ajustada = DetectorConfig(weights="x.pt", device="mps").com_ajustes({"conf": 0.10})

    assert ajustada.conf == pytest.approx(0.10)
    assert ajustada.weights == "x.pt"
    assert ajustada.device == "mps"


def test_infer_exposes_ball_telemetry_without_changing_records() -> None:
    meio = PREPROCESS.width / 2.0
    modulo = YoloBall(_DetectorFalso(Box(meio - 6, meio - 6, meio + 6, meio + 6)))
    modulo.load()

    saida = modulo.infer(_lote((0, 12, 24)), {})

    assert len(saida.records) == 3
    assert modulo.telemetry["frames_requested"] == 3
    assert modulo.telemetry["tiles_executed"] == 3
    assert modulo.telemetry["raw_candidates"] == 3
    assert modulo.telemetry["candidates_after_filter"] == 3
    assert modulo.telemetry["detected_points"] == 3
    assert modulo.telemetry["interpolated_points"] == 0
    assert modulo.telemetry["missing_points"] == 0
    assert modulo.telemetry["segments"] == 1
    assert float(modulo.telemetry["detector_ms"]) >= 0.0
    assert float(modulo.telemetry["resolver_ms"]) >= 0.0


def test_tracker_state_does_not_cross_infer_calls() -> None:
    meio = PREPROCESS.width / 2.0
    modulo = YoloBall(_DetectorFalso(Box(meio - 6, meio - 6, meio + 6, meio + 6)))
    modulo.load()

    first = modulo.infer(_lote((0,)), {})
    second = modulo.infer(_lote((30,)), {})

    assert first.records[0][0]["trajectory_segment"] == 0
    assert second.records[0][0]["trajectory_segment"] == 0
    assert "velocity_xy_px_s" not in second.records[0][0]


class _DetectorDeDois:
    """Duas caixas por quadro, para separar o que o filtro deixa passar do que o ranking escolhe."""

    def __init__(self, primeira: Box, segunda: Box) -> None:
        self.caixas = (primeira, segunda)

    def detect_raw(self, frame: Any) -> tuple[tuple[Box, float, int], ...]:
        return tuple((caixa, 0.5, 32) for caixa in self.caixas)

    def candidates_from(
        self, cruas: Sequence[tuple[Box, float, int]], frame_height: float
    ) -> tuple[BallDetection, ...]:
        return tuple(BallDetection(box=c, confidence=k) for c, k, _ in cruas)

    def detect_candidates(self, frame: Any) -> tuple[BallDetection, ...]:
        return self.candidates_from(self.detect_raw(frame), 0.0)


def _duas_caixas() -> _DetectorDeDois:
    meio = PREPROCESS.width / 2.0
    return _DetectorDeDois(
        Box(meio - 6, meio - 6, meio + 6, meio + 6),
        Box(meio + 200, meio + 100, meio + 212, meio + 112),
    )


def test_a_caixa_emitida_e_a_que_o_ranking_escolhe() -> None:
    """O modulo tem de emitir exatamente o candidato que `CandidateRanker` escolheria sozinho.

    Sem esta amarra, um erro de espaco de coordenada ou de ordem das medidas na ligacao entre o
    modulo e o ranking passaria calado: as duas caixas sao validas, e qualquer uma das duas produz
    registro plausivel.
    """
    from edge.frames.preprocess import to_source_box, to_source_image
    from edge.modules.ball.ranking import CandidateRanker, load_ranker

    modulo = YoloBall(_duas_caixas(), ranking=True)
    modulo.load()
    lote = _lote((0,))

    saida = modulo.infer(lote, {})

    original = to_source_image(lote.tensor[0], PREPROCESS, FONTE)
    candidatos = tuple(
        BallDetection(Box(*to_source_box(_as_caixa(caixa), PREPROCESS, FONTE)), 0.5)
        for caixa in _duas_caixas().caixas
    )
    (esperado,) = CandidateRanker(load_ranker()).rank(original, 0, candidatos)
    (registro,) = saida.records[0]
    assert registro["box"] == (esperado.box.x1, esperado.box.y1, esperado.box.x2, esperado.box.y2)
    assert modulo.telemetry["ranked_frames"] == 1
    assert 0.0 <= modulo.telemetry["ranking_score_sum"] <= modulo.telemetry["ranked_frames"]


def _as_caixa(caixa: Box) -> tuple[float, float, float, float]:
    return (caixa.x1, caixa.y1, caixa.x2, caixa.y2)


def test_sem_ranking_o_rastreador_recebe_todos_os_candidatos() -> None:
    sem = YoloBall(_duas_caixas(), ranking=False)
    sem.load()

    saida = sem.infer(_lote((0,)), {})

    assert len(saida.records[0]) == 1
    assert sem.telemetry.get("ranked_frames", 0) == 0
    assert sem.telemetry["candidates_after_filter"] == 2


def test_sem_source_size_o_modulo_nao_pontua() -> None:
    """Sem `source_size` a caixa fica no espaco do tensor, e pontuar ali mediria cor e posicao no
    lugar errado. Nao pontuar e declarar isso e melhor do que um numero plausivel e sem sentido."""
    modulo = YoloBall(_duas_caixas(), ranking=True)
    modulo.load()

    modulo.infer(_lote((0,), fonte=None), {})

    assert modulo.telemetry["ranked_frames"] == 0


def test_ranking_desligado_nao_carrega_artefato(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com o ranking desligado, `load` nao pode nem tocar no artefato.

    A assercao e sobre o carregamento e nao sobre a impressao digital: carregar o artefato sempre
    produziria a mesma impressao e passaria por uma checagem de texto.
    """
    from edge.modules.ball import module as modulo_da_bola

    def recusa(*_: object, **__: object) -> object:
        raise AssertionError("load_ranker nao devia ser chamado com o ranking desligado")

    monkeypatch.setattr(modulo_da_bola, "load_ranker", recusa)
    modulo = YoloBall(ranking=False)

    modulo.load()

    assert ";ranking=False," in modulo.fingerprint()


def test_configure_recusa_ranking_que_nao_e_booleano() -> None:
    modulo = YoloBall()

    with pytest.raises(ValueError, match="ball.ranking espera booleano"):
        modulo.configure({"ranking": "sim"})


def test_configure_desliga_o_ranking_pelo_toml() -> None:
    modulo = YoloBall(_duas_caixas())
    modulo.configure({"ranking": False})
    modulo.load()

    modulo.infer(_lote((0,)), {})

    assert modulo.telemetry["ranked_frames"] == 0


def test_o_ranking_liga_a_ordem_no_rastreador() -> None:
    """A saida do modulo tem de bater com ranking mais rastreador que respeita a ordem.

    Sem `candidate_order`, o rastreador reescolheria por proximidade da posicao prevista e desfaria
    a decisao do ranking. Medido nas 328 posicoes anotadas, isso custa 16 acertos com dois
    candidatos, entao a amarra precisa estar em teste e nao em leitura do codigo.
    """
    from edge.frames.preprocess import to_source_box, to_source_image
    from edge.modules.ball.ranking import CandidateRanker, load_ranker
    from edge.modules.ball.tracking import BallObservation, BallTracker, TrackerConfig

    modulo = YoloBall(_duas_caixas(), ranking=True, ranking_candidates=2)
    modulo.load()
    lote = _lote((0,))

    saida = modulo.infer(lote, {})

    candidatos = tuple(
        BallDetection(Box(*to_source_box(_as_caixa(caixa), PREPROCESS, FONTE)), 0.5)
        for caixa in _duas_caixas().caixas
    )
    ordenados = CandidateRanker(load_ranker(), keep=2).rank(
        to_source_image(lote.tensor[0], PREPROCESS, FONTE), 0, candidatos
    )
    esperado = list(
        BallTracker(TrackerConfig(candidate_order=True)).track([BallObservation(0, 0.0, ordenados)])
    )
    (registro,) = saida.records[0]
    assert esperado[0].center is not None
    assert _centro(registro["box"]) == pytest.approx(esperado[0].center)  # type: ignore[arg-type]


def test_ranking_candidates_chega_ao_ranker() -> None:
    modulo = YoloBall(_duas_caixas(), ranking=True, ranking_candidates=2)
    modulo.load()

    saida = modulo.infer(_lote((0,)), {})

    assert modulo.telemetry["ranked_frames"] == 1
    assert len(saida.records[0]) == 1
    assert ";ranking=True,2" in modulo.fingerprint()


@pytest.mark.parametrize("valor", [0, -1, True, 1.5, "dois"])
def test_ranking_candidates_invalido_nao_passa_calado(valor: object) -> None:
    with pytest.raises(ValueError, match="ball.ranking_candidates"):
        YoloBall().configure({"ranking_candidates": valor})


def test_ranking_artifact_escolhe_o_arquivo_carregado(monkeypatch: pytest.MonkeyPatch) -> None:
    from edge.modules.ball import module as modulo_da_bola
    from edge.modules.ball.ranking import ARTIFACTS

    carregados: list[object] = []
    original = modulo_da_bola.load_ranker

    def espia(caminho: object = None) -> object:
        carregados.append(caminho)
        return original(caminho)  # type: ignore[arg-type]

    monkeypatch.setattr(modulo_da_bola, "load_ranker", espia)
    modulo = YoloBall(_duas_caixas(), ranking=True)
    modulo.configure({"ranking_artifact": "aparencia"})

    modulo.load()
    modulo.infer(_lote((0,)), {})

    assert carregados == [ARTIFACTS["aparencia"]]
    assert modulo.telemetry["ranked_frames"] == 1


def test_ranking_artifact_desconhecido_nao_passa_calado() -> None:
    with pytest.raises(ValueError, match="ball.ranking_artifact"):
        YoloBall().configure({"ranking_artifact": "camera"})


def test_o_artefato_entra_na_impressao_so_quando_nao_e_o_padrao() -> None:
    """A chave do cache de percepcao nao pode servir a bola de um artefato para o outro."""
    padrao = YoloBall(ranking=True, ranking_candidates=2)
    aparencia = YoloBall(ranking=True, ranking_candidates=2, ranking_artifact="aparencia")

    assert padrao.fingerprint().endswith(";ranking=True,2")
    assert aparencia.fingerprint().endswith(";ranking=True,2,aparencia")


def test_a_telemetria_do_ranking_soma_entre_clipes() -> None:
    """A pontuacao sai como soma, e nao como mediana, porque `aggregate_telemetry` soma tudo.

    Publicar uma mediana ali produzia um numero proporcional ao numero de clipes, sem teto em um, e
    a pagina do modelo vendia isso como sinal de deriva de dominio. Soma e contagem agregam certo, e
    a media sai da divisao das duas.
    """
    from mlops.ball_trajectory import aggregate_telemetry

    modulo = YoloBall(_duas_caixas(), ranking=True)
    modulo.load()

    por_clipe: list[dict[str, int | float]] = []
    for _ in (0, 1):
        modulo.infer(_lote((0,)), {})
        por_clipe.append(dict(modulo.telemetry))

    somado = aggregate_telemetry(por_clipe)

    assert somado["ranked_frames"] == 2
    assert somado["ranking_score_sum"] <= somado["ranked_frames"]


def test_o_padrao_do_modulo_esta_preso_por_teste() -> None:
    """Virar o padrao sem passar por aqui era possivel, e o padrao decide o que roda na Jetson.

    Desligado no codigo e ligado em `config/pipeline.models.toml` de proposito: o valor escolhido
    por medicao pertence ao arquivo que descreve a execucao, e nao a um argumento nomeado.
    """
    padrao = YoloBall()

    assert ";ranking=False,1" in padrao.fingerprint()


def test_production_config_ranks_by_appearance_only() -> None:
    from pathlib import Path

    from edge.modules.registry import read_config

    entrada = {c.kind: c for c in read_config(Path("config/pipeline.models.toml"))}["ball"]

    assert entrada.options["ranking"] is True
    assert entrada.options["ranking_candidates"] == 2
    assert entrada.options["ranking_artifact"] == "aparencia"


def test_a_chave_do_cache_compartilhado_separa_a_precisao() -> None:
    """Meia precisao devolve outros numeros, entao a passada de uma nao pode servir a outra.

    E o mesmo defeito que `conf` e `imgsz` ja fecham nesta chave: o `PreprocessSpec` nao carrega
    precisao, entao duas especies sobre o mesmo peso casariam na chave e a segunda consumiria em
    silencio a passada da primeira, medindo a precisao que nao pediu.
    """
    peso = "var/models/abc/yolov8n.pt"
    cheia = chave_do_cache(DetectorConfig(weights=peso))
    meia = chave_do_cache(DetectorConfig(weights=peso, quantize="fp16"))

    assert cheia != meia


def test_candidatos_saem_no_quadro_original_e_antes_do_ranking() -> None:
    """O TB-04 compara o detector: o ranking corta a lista, e o cortado tambem tem de parear."""
    from edge.frames.preprocess import to_source_box

    modulo = YoloBall(_duas_caixas(), ranking=True, ranking_candidates=1)
    modulo.load()

    (candidatos,) = modulo.candidatos(_lote((0,)))

    esperados = [to_source_box(_as_caixa(c), PREPROCESS, FONTE) for c in _duas_caixas().caixas]
    assert [_as_caixa(c.box) for c in candidatos] == esperados
    assert [c.confidence for c in candidatos] == [0.5, 0.5]
    assert len(modulo.infer(_lote((0,)), {}).records[0]) == 1


def test_candidatos_sem_detector_levantam_em_vez_de_devolver_vazio() -> None:
    """Vazio dos dois lados de uma comparacao passaria por paridade perfeita."""
    modulo = YoloBall()
    with pytest.raises(ModuleNotLoaded):
        modulo.candidatos(_lote((0,)))
    modulo.load()

    with pytest.raises(ModuleNotLoaded, match="sem detector"):
        modulo.candidatos(_lote((0,)))


def test_runtime_info_e_o_do_detector_e_some_sem_ele() -> None:
    from edge.runtime import RuntimeInfo

    info = RuntimeInfo(
        backend="tensorrt",
        provider="tensorrt",
        device="cuda",
        precision="float32",
        runtime_versions=(("tensorrt", "11.3"),),
        artifact_sha256="0" * 64,
        parent_onnx_sha256="1" * 64,
        inputs=(),
        outputs=(),
        inference_count=0,
        fallback_reason=None,
    )

    class _ComRuntime(_DetectorFalso):
        runtime_info = info

    modulo = YoloBall(_ComRuntime(None))
    modulo.load()

    assert modulo.runtime_info is info
    assert YoloBall(_DetectorFalso(None)).runtime_info is None
    modulo.close()
    assert modulo.runtime_info is None
