"""Testes dos tipos congelados e das invariantes que eles garantem na construcao."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, MutableMapping, MutableSequence, MutableSet
from dataclasses import is_dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import pytest

import contracts as contratos
from contracts import (
    BallDetection,
    Box,
    Clip,
    ContractError,
    FeatureStatus,
    FeatureValue,
    FieldGeometry,
    FramePerception,
    PlayerDetection,
    SchemaValidationError,
    ShotEvent,
    make_shot_id,
)
from contracts.demo import build_demo_event
from contracts.validation import validate_shot_event


def _tipos_concretos(anotacao: object) -> Iterator[type]:
    origem = get_origin(anotacao)
    if origem is UnionType or origem is Union:
        for parte in get_args(anotacao):
            yield from _tipos_concretos(parte)
    elif isinstance(origem, type):
        yield origem
    elif isinstance(anotacao, type):
        yield anotacao


@pytest.fixture
def event() -> ShotEvent:
    """Definida aqui, e nao num `conftest.py` da pasta.

    Medido: com os arquivos passados na ordem `contracts/...`, algo fora de `contracts/`, e
    `contracts/...` de novo, o conftest da pasta deixa de ser registrado no reentrar e quatro
    testes morrem com `fixture 'event' not found`. A suite so e verde na ordem em que ela e
    normalmente coletada, e qualquer fatiamento de CI ou selecao manual de arquivos derruba
    testes por motivo que nao tem nada a ver com o codigo.

    Uma linha de import custa menos que depender dessa ordem.
    """
    return build_demo_event()


def test_shot_event_roundtrip(event: ShotEvent) -> None:
    restored = ShotEvent.from_dict(json.loads(json.dumps(event.to_dict())))

    assert restored == event

    assert restored.to_dict()["features"][0]["evidence"]["intermediate"] == {
        "homography_error_px": 2.1,
        "used_keypoints": ["big_box_tl"],
    }

    intermediario = cast("Mapping[str, object]", restored.features[0].evidence["intermediate"])
    assert intermediario["used_keypoints"] == ("big_box_tl",)


def test_feature_value_rejects_missing_reason_when_status_not_ok() -> None:
    with pytest.raises(ContractError):
        FeatureValue(
            name="shot_angle_deg",
            version="1.0.0",
            status="unavailable",
            value=None,
            confidence=None,
            reason=None,
            evidence={},
        )


def test_shot_id_is_deterministic() -> None:
    assert make_shot_id("clip-001", 187) == make_shot_id("clip-001", 187)
    assert make_shot_id("clip-001", 187) != make_shot_id("clip-001", 188)
    assert make_shot_id("clip-001", 187) != make_shot_id("clip-002", 187)


def test_pitch_xy_is_none_when_homography_is_none() -> None:
    box = Box(x1=10.0, y1=20.0, x2=40.0, y2=90.0)
    sem_homografia = FieldGeometry(
        homography=None,
        reprojection_error_px=None,
        reprojection_error_p95_px=None,
        keypoints=(),
        observed_keypoint_count=0,
        inlier_count=0,
        source="unavailable",
        age_frames=0,
        template_id="pitch-105x68-center-v1",
        reason="insufficient-keypoints",
    )

    with pytest.raises(ContractError):
        FramePerception(
            frame_index=187,
            timestamp_s=7.48,
            players=(PlayerDetection(box=box, confidence=0.88, track_id=9, pitch_xy=(-3.4, 15.8)),),
            field=sem_homografia,
        )

    with pytest.raises(ContractError):
        FramePerception(
            frame_index=187,
            timestamp_s=7.48,
            ball=BallDetection(box=box, confidence=0.7, pitch_xy=(1.0, 2.0)),
            field=sem_homografia,
        )

    sem_coordenada = FramePerception(
        frame_index=187,
        timestamp_s=7.48,
        players=(PlayerDetection(box=box, confidence=0.88, track_id=9),),
        field=sem_homografia,
    )
    assert sem_coordenada.players[0].pitch_xy is None


def test_detected_field_requires_a_valid_homography() -> None:
    with pytest.raises(ContractError, match="detected"):
        FieldGeometry(
            homography=None,
            reprojection_error_px=None,
            reprojection_error_p95_px=None,
            keypoints=(),
            observed_keypoint_count=0,
            inlier_count=0,
            source="detected",
            age_frames=0,
            template_id="pitch-105x68-center-v1",
            reason=None,
        )


def test_detected_field_rejects_a_singular_homography() -> None:
    with pytest.raises(ContractError, match="degenerada"):
        FieldGeometry(
            homography=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            reprojection_error_px=1.0,
            reprojection_error_p95_px=2.0,
            keypoints=(),
            observed_keypoint_count=6,
            inlier_count=6,
            source="detected",
            age_frames=0,
            template_id="pitch-105x68-center-v1",
            reason=None,
        )


def test_unavailable_field_rejects_an_unknown_reason() -> None:
    with pytest.raises(ContractError, match="reason"):
        FieldGeometry(
            homography=None,
            reprojection_error_px=None,
            reprojection_error_p95_px=None,
            keypoints=(),
            observed_keypoint_count=0,
            inlier_count=0,
            source="unavailable",
            age_frames=0,
            template_id="pitch-105x68-center-v1",
            reason=cast("Any", "guess"),
        )


def test_confidence_fora_do_intervalo_e_recusada() -> None:
    with pytest.raises(ContractError, match="fora de 0..1"):
        FeatureValue(
            name="shot_distance_m",
            version="1.0.0",
            status="ok",
            value=16.2,
            confidence=5.0,
            reason=None,
            evidence={},
        )


def test_confidence_nos_extremos_e_aceita() -> None:
    for limite in (0.0, 1.0):
        assert (
            FeatureValue(
                name="x",
                version="1",
                status="ok",
                value=1,
                confidence=limite,
                reason=None,
                evidence={},
            ).confidence
            == limite
        )


def test_shot_id_precisa_derivar_do_clip_id(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="nao deriva do clip_id"):
        replace(event, shot_id="outro-clipe:999")


def test_shot_id_precisa_terminar_em_indice_inteiro(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="indice de quadro inteiro"):
        replace(event, shot_id=f"{event.clip_id}:abc")


def test_shot_id_com_digito_nao_ascii_e_recusado(event: ShotEvent) -> None:
    for sufixo in ("١٨٧", "²"):
        with pytest.raises(ContractError, match="indice de quadro inteiro"):
            replace(event, shot_id=f"{event.clip_id}:{sufixo}")


def test_shot_id_com_quebra_de_linha_final_e_recusado(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="indice de quadro inteiro"):
        replace(event, shot_id=f"{event.shot_id}\n")


def test_status_fora_do_enum_e_recusado() -> None:
    with pytest.raises(ContractError, match="fora do enum"):
        FeatureValue(
            name="shot_angle_deg",
            version="1.0.0",
            status=cast("FeatureStatus", "talvez"),
            value=None,
            confidence=None,
            reason="camera perdeu o lance",
            evidence={},
        )


def test_schema_version_desconhecida_e_recusada(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="desconhecida"):
        replace(event, schema_version="2.0")


def test_evidence_nao_pode_ser_mutada_depois_de_construida(event: ShotEvent) -> None:
    with pytest.raises(TypeError):
        event.features[0].evidence["injetado"] = True  # pyright: ignore[reportIndexIssue]


def test_nenhum_campo_do_contrato_promete_escrita() -> None:
    publicos = [getattr(contratos, nome) for nome in contratos.__all__]
    tipos = [alvo for alvo in publicos if isinstance(alvo, type) and is_dataclass(alvo)]
    visitados: set[tuple[str, str]] = set()

    for tipo in tipos:
        for campo, anotacao in get_type_hints(tipo).items():
            visitados.add((tipo.__name__, campo))
            for concreto in _tipos_concretos(anotacao):
                assert not issubclass(concreto, MutableMapping | MutableSequence | MutableSet), (
                    f"{tipo.__name__}.{campo} anuncia {concreto.__name__}, que promete escrita"
                )

    assert visitados >= {
        ("FeatureValue", "evidence"),
        ("PipelineMetadata", "versions"),
        ("PipelineMetadata", "timings_ms"),
    }


def test_o_valor_carregado_e_somente_leitura(event: ShotEvent) -> None:
    for valor in (event.features[0].evidence, event.pipeline.versions, event.pipeline.timings_ms):
        assert isinstance(valor, Mapping)
        assert not isinstance(valor, MutableMapping)


def test_evidence_congela_em_profundidade() -> None:
    feature = FeatureValue(
        name="x",
        version="1",
        status="ok",
        value=1,
        confidence=None,
        reason=None,
        evidence={"aninhado": {"lista": [1, 2]}},
    )
    with pytest.raises(TypeError):
        feature.evidence["aninhado"]["lista"] = []  # pyright: ignore[reportIndexIssue]


def test_evidence_volta_serializavel_no_to_dict() -> None:
    feature = FeatureValue(
        name="x",
        version="1",
        status="ok",
        value=1,
        confidence=None,
        reason=None,
        evidence={"frames": [1, 2], "meta": {"a": 1}},
    )
    assert json.dumps(feature.to_dict()["evidence"]) == '{"frames": [1, 2], "meta": {"a": 1}}'


@pytest.mark.parametrize(
    "valor",
    [
        pytest.param(Decimal("16.24"), id="decimal"),
        pytest.param(datetime(2026, 8, 18, tzinfo=timezone.utc), id="datetime"),
        pytest.param({185, 186, 187}, id="set"),
        pytest.param(b"\x00", id="bytes"),
    ],
)
def test_evidence_recusa_o_que_o_json_nao_serializa(valor: object) -> None:
    with pytest.raises(ContractError, match="json nao serializa"):
        FeatureValue(
            name="x",
            version="1",
            status="ok",
            value=1,
            confidence=None,
            reason=None,
            evidence={"v": valor},
        )


def test_bool_em_campo_numerico_e_recusado_pelo_tipo(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="espera number"):
        replace(event, shot_timestamp_s=cast("float", True))
    with pytest.raises(ContractError, match="espera number"):
        replace(event.features[0], confidence=cast("float", True))
    with pytest.raises(ContractError, match="espera number"):
        replace(event.pipeline, timings_ms={"decode": cast("float", True)})


def test_o_schema_tambem_recusa_bool_em_campo_numerico(event: ShotEvent) -> None:
    payload = event.to_dict()
    payload["shot_timestamp_s"] = True

    with pytest.raises(SchemaValidationError, match="number"):
        validate_shot_event(payload)


def test_escalar_que_o_json_nao_serializa_e_recusado(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="confidence"):
        replace(event.features[0], confidence=cast("float", Decimal("0.9")))
    with pytest.raises(ContractError, match="value"):
        replace(event.features[0], value=cast("float", Decimal("16.24")))


def test_versions_com_valor_que_nao_e_texto_e_recusado(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="espera string"):
        replace(event.pipeline, versions={"players": cast("str", 1)})


def test_evidence_com_chave_que_nao_e_texto_e_recusada() -> None:
    with pytest.raises(ContractError, match="chave"):
        FeatureValue(
            name="x",
            version="1",
            status="ok",
            value=1,
            confidence=None,
            reason=None,
            evidence={"por_quadro": cast("dict[str, object]", {185: "pre"})},
        )


def test_mutar_o_dicionario_de_origem_nao_afeta_a_feature() -> None:
    origem: dict[str, object] = {"frames": [1]}
    feature = FeatureValue(
        name="x", version="1", status="ok", value=1, confidence=None, reason=None, evidence=origem
    )
    origem["frames"] = [99]
    assert feature.to_dict()["evidence"] == {"frames": [1]}


def test_shot_timestamp_negativo_e_recusado(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="negativo"):
        replace(event, shot_timestamp_s=-0.1)


def test_clip_id_vazio_e_recusado(event: ShotEvent) -> None:
    with pytest.raises(ContractError):
        replace(event, clip_id="", shot_id=":187")


def test_host_vazio_e_recusado(event: ShotEvent) -> None:
    with pytest.raises(ContractError, match="host vazio"):
        replace(event, pipeline=replace(event.pipeline, host=""))


def test_feature_com_name_vazio_e_recusada() -> None:
    with pytest.raises(ContractError, match="name vazio"):
        FeatureValue(
            name="", version="1", status="ok", value=1, confidence=None, reason=None, evidence={}
        )


def test_o_tipo_nunca_e_mais_permissivo_que_o_schema(event: ShotEvent) -> None:
    payload = event.to_dict()
    payload["shot_id"] = "outro-clipe:999"

    validate_shot_event(payload)
    with pytest.raises(ContractError, match="nao deriva do clip_id"):
        ShotEvent.from_dict(payload)


CASOS_STARTED_AT = (
    pytest.param("2026-08-18T13:00:00Z", True, True, id="utc-com-z"),
    pytest.param("2026-08-18T13:00:00.123456Z", True, True, id="com-fracao-de-segundo"),
    pytest.param("2026-08-18T13:00:00+00:00", True, True, id="utc-com-offset-zero"),
    pytest.param("2026-08-18T10:00:00-03:00", True, True, id="com-offset-negativo"),
    pytest.param("isto nao e uma data", False, False, id="texto-qualquer"),
    pytest.param("", False, False, id="vazio"),
    pytest.param("2026-08-18 13:00:00Z", False, False, id="espaco-no-lugar-do-t"),
    pytest.param("2026-08-18T13:00:00", False, False, id="sem-fuso"),
    pytest.param("2026-08-18T25:00:00Z", False, False, id="hora-fora-de-0-a-23"),
    pytest.param("2026-08-18t13:00:00z", False, False, id="t-e-z-minusculos"),
    pytest.param("2026-12-31T23:59:60Z", False, False, id="segundo-bissexto"),
    pytest.param("2026-02-31T13:00:00Z", True, False, id="dia-que-nao-existe-no-mes"),
    pytest.param("2026-08-18T13:00:00Z\n", True, False, id="quebra-de-linha-no-fim"),
)


@pytest.mark.parametrize(("valor", "schema_aceita", "tipo_aceita"), CASOS_STARTED_AT)
def test_started_at_e_cobrado_pelos_dois_lados(
    event: ShotEvent, valor: str, schema_aceita: bool, tipo_aceita: bool
) -> None:
    payload = event.to_dict()
    payload["pipeline"]["started_at"] = valor

    if schema_aceita:
        validate_shot_event(payload)
    else:
        with pytest.raises(SchemaValidationError, match="started_at"):
            validate_shot_event(payload)

    if tipo_aceita:
        assert ShotEvent.from_dict(payload).pipeline.started_at == valor
    else:
        with pytest.raises(ContractError, match="started_at .* fora do formato do contrato"):
            ShotEvent.from_dict(payload)


def test_started_at_nao_string_vira_contract_error(event: ShotEvent) -> None:
    payload = event.to_dict()
    payload["pipeline"]["started_at"] = None

    with pytest.raises(SchemaValidationError, match="started_at"):
        validate_shot_event(payload)
    with pytest.raises(ContractError, match="started_at None fora do formato do contrato"):
        ShotEvent.from_dict(payload)


@pytest.mark.parametrize(
    ("campo", "valor"),
    [("fps", 0.0), ("fps", -1.0), ("frame_count", 0), ("width", 0), ("height", -2)],
)
def test_a_clip_refuses_a_dimension_that_makes_no_sense(campo: str, valor: float) -> None:
    """`fps` zero virava ZeroDivisionError la dentro do pipeline, longe de quem montou o Clip."""
    medidas: dict[str, float] = {"fps": 30.0, "frame_count": 10, "width": 320, "height": 240}
    medidas[campo] = valor

    with pytest.raises(ContractError, match=campo):
        Clip(
            clip_id="c",
            path="x.mp4",
            fps=medidas["fps"],
            frame_count=int(medidas["frame_count"]),
            width=int(medidas["width"]),
            height=int(medidas["height"]),
        )


def test_a_clip_refuses_an_empty_identifier() -> None:
    with pytest.raises(ContractError, match="clip_id"):
        Clip(clip_id="", path="x.mp4", fps=30.0, frame_count=10, width=320, height=240)
