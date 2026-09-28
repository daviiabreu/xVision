"""O contrato do extractor: o que uma feature declara e como ela e recusada na construcao."""

from __future__ import annotations

import json

import pytest

from edge.features.base import (
    FeatureError,
    FeatureSpec,
    ShotActions,
    parse_shot_actions,
    shot_actions,
    shot_provenance,
    unavailable,
)


def test_unknown_requirement_is_refused_at_construction() -> None:
    with pytest.raises(FeatureError, match="fieldd"):
        FeatureSpec(name="shot_distance_m", version="0.1.0", requires=("fieldd",))


def test_every_module_kind_is_accepted_as_a_requirement() -> None:
    spec = FeatureSpec(name="tudo", version="0.1.0", requires=("field", "players", "shot"))

    assert spec.requires == ("field", "players", "shot")


def test_a_feature_can_require_nothing() -> None:
    assert FeatureSpec(name="constante", version="0.1.0", requires=()).requires == ()


def test_the_spec_carries_only_what_the_feature_knows_about_itself() -> None:
    """`unit` entrou porque a fronteira HTTP precisa dela e adivinhar seria pior.

    Derivar metro de um sufixo `_m` no nome quebra na primeira feature que nao seguir a
    convencao, e uma tabela na fronteira ficaria longe de quem sabe a resposta.
    """
    assert set(FeatureSpec.__dataclass_fields__) == {"name", "version", "requires", "unit"}


def test_a_feature_without_a_declared_unit_is_dimensionless() -> None:
    """Booleano e classe nao tem unidade, e a maioria das metricas do projeto e booleana."""
    assert FeatureSpec(name="marcado", version="0.1.0", requires=()).unit == ""


def test_unavailable_carries_the_declared_name_version_and_the_reason() -> None:
    spec = FeatureSpec(name="shot_angle_deg", version="0.2.0", requires=("field",))

    valor = unavailable(spec, "modulo field nao rodou")

    assert (valor.name, valor.version, valor.status) == ("shot_angle_deg", "0.2.0", "unavailable")
    assert valor.value is None
    assert valor.confidence is None
    assert valor.reason == "modulo field nao rodou"
    assert dict(valor.evidence) == {}


class TestShotProvenance:
    """Ler procedencia de um `ShotOutput` que talvez nao a declare, sem inventar valor."""

    class _Mudo:
        frame_index = 10
        shooter_track_id = None

    class _Declara:
        frame_index = 10
        shooter_track_id = 7
        source = "ball-break"
        confidence = 0.9
        ball_support = "detected"

    class _Mente:
        """Campos com tipo errado, que e o que um double desatento ou um json torto produz."""

        frame_index = 10
        shooter_track_id = None
        source = 3
        confidence = "alta"
        ball_support = ["detected"]

    def test_o_duble_sem_procedencia_devolve_ausente_em_tudo(self) -> None:
        p = shot_provenance(self._Mudo())  # pyright: ignore[reportArgumentType]

        assert (p.source, p.confidence, p.ball_support) == (None, None, None)

    def test_o_duble_que_declara_devolve_o_que_declarou(self) -> None:
        p = shot_provenance(self._Declara())  # pyright: ignore[reportArgumentType]

        assert (p.source, p.confidence, p.ball_support) == ("ball-break", 0.9, "detected")

    def test_tipo_errado_vira_ausente_e_nao_texto_convertido(self) -> None:
        """`str(3)` daria a procedencia `"3"`, plausivel e inventada, e nenhum agregado
        desconfiaria dela."""
        p = shot_provenance(self._Mente())  # pyright: ignore[reportArgumentType]

        assert (p.source, p.confidence, p.ball_support) == (None, None, None)

    def test_confianca_inteira_e_aceita_como_float(self) -> None:
        class Inteira:
            frame_index = 10
            shooter_track_id = None
            confidence = 1

        p = shot_provenance(Inteira())  # pyright: ignore[reportArgumentType]

        assert p.confidence == 1.0 and isinstance(p.confidence, float)

    def test_nan_e_infinito_nao_sao_confianca(self) -> None:
        """`json.loads` aceita o token `NaN` por padrao, e `valor is None` nao pega `nan`.

        Sem esta guarda o `nan` atravessa para `PlayEvidence.shot_confidence`, e um `nan` num
        logit zera o softmax inteiro sem levantar excecao nenhuma. O modulo vizinho `touch.py` ja
        tinha exatamente esta guarda, com o mesmo motivo.
        """
        for valor in (float("nan"), float("inf"), float("-inf")):

            class Nao:
                frame_index = 10
                shooter_track_id = None
                confidence = valor

            assert shot_provenance(Nao()).confidence is None, valor  # pyright: ignore[reportArgumentType]

    def test_booleano_nao_e_confianca(self) -> None:
        """`bool` e subclasse de `int`, entao `True` passaria por confianca 1,0."""

        class Booleana:
            frame_index = 10
            shooter_track_id = None
            confidence = True

        assert shot_provenance(Booleana()).confidence is None  # pyright: ignore[reportArgumentType]


def _acoes(**trocas: object) -> dict[str, object]:
    """O campo `actions` como o T-DEED o grava, com o que o teste quiser estragar."""
    return {
        "classes": ["BACKGROUND", "CROSS", "SHOT"],
        "frame_indices": [100, 105, 110],
        "probabilities": [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.0, 0.9]],
        **trocas,
    }


class TestShotActions:
    """A matriz de passo por classe do detector de instante, sem aproveitar pedaco estragado."""

    def test_o_registro_bem_formado_vira_matriz_e_a_serie_sai_por_classe(self) -> None:
        acoes = parse_shot_actions(_acoes())

        assert acoes is not None
        assert acoes.classes == ("BACKGROUND", "CROSS", "SHOT")
        assert acoes.serie("CROSS") == ((100, 0.1), (105, 0.7), (110, 0.0))

    def test_a_matriz_sobrevive_ao_caminho_do_cache(self) -> None:
        """O registro volta do cache por JSON, que devolve lista onde o modulo gravou lista."""
        assert parse_shot_actions(json.loads(json.dumps(_acoes()))) == parse_shot_actions(_acoes())

    def test_classe_que_o_detector_nao_pontua_devolve_serie_vazia(self) -> None:
        acoes = parse_shot_actions(_acoes())

        assert acoes is not None
        assert acoes.serie("CORNER") == ()

    @pytest.mark.parametrize(
        "estragado",
        [
            None,
            "actions",
            _acoes(classes=[]),
            _acoes(classes=["BACKGROUND", "CROSS", "CROSS"]),
            _acoes(classes=["BACKGROUND", "CROSS", 3]),
            _acoes(frame_indices=[100, True, 110]),
            _acoes(frame_indices=[100, 110, 105]),
            _acoes(frame_indices=[100, 105]),
            _acoes(probabilities=[[0.9, 0.1, 0.0], [0.2, 0.7], [0.1, 0.0, 0.9]]),
            _acoes(probabilities=[[0.9, 0.1, 0.0], [0.2, float("nan"), 0.1], [0.1, 0.0, 0.9]]),
            _acoes(probabilities=[[0.9, 0.1, 0.0], [0.2, 1.5, 0.1], [0.1, 0.0, 0.9]]),
            _acoes(probabilities=[[0.9, 0.1, 0.0], [0.2, True, 0.1], [0.1, 0.0, 0.9]]),
            _acoes(probabilities=[[0.9, 0.1, 0.0], [0.2, "0.7", 0.1], [0.1, 0.0, 0.9]]),
            {"classes": ["SHOT"], "frame_indices": [100]},
        ],
    )
    def test_qualquer_defeito_recusa_a_matriz_inteira(self, estragado: object) -> None:
        assert parse_shot_actions(estragado) is None

    def test_a_construcao_direta_recusa_linha_sem_as_classes_todas(self) -> None:
        with pytest.raises(ValueError, match="classes"):
            ShotActions(classes=("SHOT", "PASS"), frame_indices=(1,), probabilities=((0.5,),))

    def test_shot_actions_so_devolve_o_tipo_certo(self) -> None:
        acoes = parse_shot_actions(_acoes())

        class Tdeed:
            frame_index = 110
            shooter_track_id = None
            actions = acoes

        class Cru:
            """O dicionario do registro nunca chega a feature sem passar pela leitura."""

            frame_index = 110
            shooter_track_id = None
            actions = _acoes()

        class Regra:
            frame_index = 110
            shooter_track_id = None

        assert shot_actions(Tdeed()) is acoes  # pyright: ignore[reportArgumentType]
        assert shot_actions(Cru()) is None  # pyright: ignore[reportArgumentType]
        assert shot_actions(Regra()) is None  # pyright: ignore[reportArgumentType]
