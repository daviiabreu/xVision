"""Qualidade do estimador de pose medida pelo que a feature consome.

A media de OKS nao serve de criterio aqui. O finalizador tem 93,4 px de altura na imagem (mediana
sobre os 38 clipes anotados; 23,5 minimo) e os dois tornozelos ficam a 10,7 px um do outro (2,0
minimo). Um estimador que acerte os quinze pontos do tronco e da cabeca e troque os dois tornozelos
de lado sai com OKS perto de 0,95, passa no PCK, e entrega a `body_part` a resposta invertida.

Por isso o `score` e `usable_rate`: a fracao dos quadros anotados em que os dois tornozelos se
sustentam acima do piso de confianca E estao no lado certo. E o unico numero que o treino precisa
subir.

Nao mede quem e o finalizador. A anotacao ja traz a caixa dele, pela mesma razao de `eval-field`
medir homografia em linhas independentes dos keypoints que ajustaram a matriz: misturar escolha de
corpo com qualidade de keypoint produz um numero que nao diz qual dos dois mudou.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from contracts import Clip, Keypoint
from edge.features.body_part import MIN_CONFIANCA, TORNOZELO_DIREITO, TORNOZELO_ESQUERDO
from edge.modules.pose import COCO_KEYPOINTS

SCHEMA = Path(__file__).with_name("pose-keypoints-v1.schema.json")

OKS_SIGMAS: Mapping[str, float] = {
    "nose": 0.026,
    "eye_left": 0.025,
    "eye_right": 0.025,
    "ear_left": 0.035,
    "ear_right": 0.035,
    "shoulder_left": 0.079,
    "shoulder_right": 0.079,
    "elbow_left": 0.072,
    "elbow_right": 0.072,
    "wrist_left": 0.062,
    "wrist_right": 0.062,
    "hip_left": 0.107,
    "hip_right": 0.107,
    "knee_left": 0.087,
    "knee_right": 0.087,
    "ankle_left": 0.089,
    "ankle_right": 0.089,
}
"""As tolerancias por ponto do protocolo COCO, traduzidas para a nomenclatura do contrato."""

PCK_FRACAO = 0.2
"""Folga do PCK em fracao da altura da caixa, como a literatura publica.

Esta no relatorio para o numero ser comparavel, e nao como criterio: na altura mediana de 93,4 px
ela vale 18,7 px, e os dois tornozelos ficam a 10,7 px. Ponto no pe errado passa neste limiar.
"""

VISIVEL = "visible"
OCLUSO = "occluded"
AUSENTE = "absent"


class PoseLabelError(ValueError): ...


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float | None
    y: float | None
    visibility: str

    @property
    def anotado(self) -> bool:
        """Ocluso entra, pela convencao do COCO: descartar mediria so onde o estimador tem vida
        facil, e o quadro do contato numa area cheia e onde ele nao tem."""
        return self.visibility in (VISIVEL, OCLUSO) and self.x is not None and self.y is not None

    @property
    def ponto(self) -> tuple[float, float]:
        if self.x is None or self.y is None:
            raise PoseLabelError(f"{self.name} nao tem posicao anotada")
        return (self.x, self.y)


@dataclass(frozen=True)
class PoseLabel:
    clip_id: str
    match_id: str
    frame_index: int
    frame_width: int
    frame_height: int
    shooter_box: tuple[float, float, float, float]
    keypoints: tuple[PoseKeypoint, ...]
    video_sha256: str
    annotator: str
    reviewed_by: str | None
    quality_flags: tuple[str, ...]
    shooter_track_id: int | None = None

    @property
    def human_confirmed(self) -> bool:
        return (
            "human-annotated" in self.quality_flags
            and "requires-human-review" not in self.quality_flags
        )

    @property
    def assisted(self) -> bool:
        """Anotacao corrigida a partir de palpite do modelo nao equivale a revisao cega: quem
        corrige esqueleto ja desenhado tende a aceitar o lado que o modelo escolheu."""
        return bool(
            {"model-proposed-reviewed", "temporal-propagation-reviewed"} & set(self.quality_flags)
        )

    @property
    def altura_da_caixa(self) -> float:
        return abs(self.shooter_box[3] - self.shooter_box[1])

    @property
    def area_da_caixa(self) -> float:
        return abs(self.shooter_box[2] - self.shooter_box[0]) * self.altura_da_caixa

    def ponto(self, nome: str) -> PoseKeypoint | None:
        return next((k for k in self.keypoints if k.name == nome and k.anotado), None)


@dataclass(frozen=True)
class PoseReport:
    n_labels: int
    n_proposal_labels: int
    n_excluded_proposals: int
    n_assisted_frames: int
    includes_proposals: bool
    n_frames: int
    n_matches: int
    n_clips: int
    n_available: int
    n_unavailable: int
    completion: float
    completion_wilson_95: tuple[float, float]
    oks_median: float | None
    oks_at_50: float
    oks_at_75: float
    pck: float
    error_median_px: float | None
    error_p95_px: float | None
    n_ankle_frames: int
    ankle_available: int
    ankle_side_correct: int
    min_confidence: float
    per_keypoint: Mapping[str, Mapping[str, float | int]]

    @property
    def ankle_availability(self) -> float:
        """Sobre os quadros cuja anotacao tem os dois pes: quadro em que o anotador nao viu pe
        nenhum mede o enquadramento, e nao o estimador."""
        return self.ankle_available / self.n_ankle_frames if self.n_ankle_frames else 0.0

    @property
    def ankle_side_accuracy(self) -> float:
        return self.ankle_side_correct / self.ankle_available if self.ankle_available else 0.0

    @property
    def usable(self) -> int:
        return self.ankle_side_correct

    @property
    def usable_rate(self) -> float:
        """Sobre `n_ankle_frames` e nao sobre `ankle_available`: tornozelo que o estimador recusa
        falha do mesmo jeito que tornozelo no pe errado, so que com motivo declarado. Dividir pelos
        disponiveis premiaria um estimador que aprendesse a nao responder."""
        return self.usable / self.n_ankle_frames if self.n_ankle_frames else 0.0

    @property
    def score(self) -> float:
        return self.usable_rate

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "n_labels": self.n_labels,
            "n_proposal_labels": self.n_proposal_labels,
            "n_excluded_proposals": self.n_excluded_proposals,
            "n_assisted_frames": self.n_assisted_frames,
            "evaluation_kind": (
                "exploratory-with-proposals" if self.includes_proposals else "human-confirmed"
            ),
            "n_frames": self.n_frames,
            "n_matches": self.n_matches,
            "n_clips": self.n_clips,
            "n_available": self.n_available,
            "n_unavailable": self.n_unavailable,
            "completion": round(self.completion, 4),
            "completion_wilson_95": [round(v, 4) for v in self.completion_wilson_95],
            "oks_median": None if self.oks_median is None else round(self.oks_median, 4),
            "oks_at_50": round(self.oks_at_50, 4),
            "oks_at_75": round(self.oks_at_75, 4),
            "pck": round(self.pck, 4),
            "pck_fracao_da_altura": PCK_FRACAO,
            "error_median_px": self.error_median_px,
            "error_p95_px": self.error_p95_px,
            "min_confidence": self.min_confidence,
            "n_ankle_frames": self.n_ankle_frames,
            "ankle_available": self.ankle_available,
            "ankle_availability": round(self.ankle_availability, 4),
            "ankle_availability_wilson_95": [
                round(v, 4) for v in _wilson(self.ankle_available, self.n_ankle_frames)
            ],
            "ankle_side_correct": self.ankle_side_correct,
            "ankle_side_accuracy": round(self.ankle_side_accuracy, 4),
            "ankle_side_wilson_95": [
                round(v, 4) for v in _wilson(self.ankle_side_correct, self.ankle_available)
            ],
            "usable": self.usable,
            "usable_rate": round(self.usable_rate, 4),
            "usable_rate_wilson_95": [
                round(v, 4) for v in _wilson(self.usable, self.n_ankle_frames)
            ],
            "score": round(self.score, 4),
            "per_keypoint": {
                nome: dict(valores) for nome, valores in sorted(self.per_keypoint.items())
            },
        }


def read_pose_labels(path: Path) -> tuple[PoseLabel, ...]:
    if not path.is_file():
        raise PoseLabelError(f"{path} nao existe")
    schema = cast("dict[str, Any]", json.loads(SCHEMA.read_text(encoding="utf-8")))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    labels: list[PoseLabel] = []
    seen: set[tuple[str, int]] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            document = cast("dict[str, Any]", json.loads(raw))
        except json.JSONDecodeError as error:
            raise PoseLabelError(f"{path.name}:{number}: JSON invalido: {error.msg}") from error
        errors = sorted(
            cast("Any", validator).iter_errors(document), key=lambda error: list(error.path)
        )
        if errors:
            raise PoseLabelError(f"{path.name}:{number}: {errors[0].message}")
        label = _label(path.name, number, document)
        key = (label.clip_id, label.frame_index)
        if key in seen:
            raise PoseLabelError(
                f"{path.name}:{number}: quadro {label.frame_index} do clipe aparece duas vezes"
            )
        seen.add(key)
        labels.append(label)
    return tuple(labels)


def evaluation_labels(
    labels: Sequence[PoseLabel], *, include_proposals: bool = False
) -> tuple[PoseLabel, ...]:
    if include_proposals:
        return tuple(labels)
    return tuple(label for label in labels if label.human_confirmed)


def validate_label_video(label: PoseLabel, clip: Clip, video_sha256: str) -> None:
    """Sem esta conferencia, um clipe rebaixado produz erro em pixel medido contra coordenada de
    outro video, e o relatorio sai plausivel."""
    if label.clip_id != clip.clip_id or label.match_id != clip.match_id:
        raise PoseLabelError("anotacao pertence a outro clipe ou partida")
    if label.video_sha256 != video_sha256:
        raise PoseLabelError("video_sha256 da anotacao difere do video medido")
    if (label.frame_width, label.frame_height) != (clip.width, clip.height):
        raise PoseLabelError("dimensoes da anotacao diferem do video medido")
    if label.frame_index >= clip.frame_count:
        raise PoseLabelError("frame_index da anotacao esta fora do video medido")


def measure(
    labels: Sequence[PoseLabel],
    predictions: Mapping[tuple[str, int], Sequence[Keypoint]],
    *,
    include_proposals: bool = False,
    min_confidence: float = MIN_CONFIANCA,
) -> PoseReport:
    selecionadas = evaluation_labels(labels, include_proposals=include_proposals)
    okss: list[float] = []
    erros: list[float] = []
    acertos_pck = 0
    pontos_pck = 0
    por_ponto: dict[str, list[int]] = {nome: [0, 0] for nome in COCO_KEYPOINTS}
    disponiveis = 0
    quadros_com_tornozelo = 0
    tornozelo_disponivel = 0
    lado_certo = 0
    for label in selecionadas:
        previsto = predictions.get((label.clip_id, label.frame_index))
        tem_tornozelos = _tem_os_dois_tornozelos(label)
        quadros_com_tornozelo += tem_tornozelos
        if previsto is None or not previsto:
            continue
        disponiveis += 1
        valor = oks(label, previsto)
        if valor is not None:
            okss.append(valor)
        indexado = {k.name: k for k in previsto}
        folga = PCK_FRACAO * label.altura_da_caixa
        for anotado in label.keypoints:
            if not anotado.anotado:
                continue
            estimado = indexado.get(anotado.name)
            if estimado is None:
                por_ponto[anotado.name][1] += 1
                pontos_pck += 1
                continue
            distancia = math.dist(anotado.ponto, (estimado.x, estimado.y))
            erros.append(distancia)
            pontos_pck += 1
            dentro = distancia <= folga
            acertos_pck += dentro
            por_ponto[anotado.name][0] += dentro
            por_ponto[anotado.name][1] += 1
        if not tem_tornozelos:
            continue
        sustentado = _tornozelos_sustentados(indexado, min_confidence)
        if sustentado is None:
            continue
        tornozelo_disponivel += 1
        lado_certo += _lado_correto(label, sustentado)
    total = len(selecionadas)
    return PoseReport(
        n_labels=len(labels),
        n_proposal_labels=sum(not label.human_confirmed for label in labels),
        n_excluded_proposals=len(labels) - total,
        n_assisted_frames=sum(label.assisted for label in selecionadas),
        includes_proposals=include_proposals,
        n_frames=total,
        n_matches=len({label.match_id for label in selecionadas}),
        n_clips=len({label.clip_id for label in selecionadas}),
        n_available=disponiveis,
        n_unavailable=total - disponiveis,
        completion=disponiveis / total if total else 0.0,
        completion_wilson_95=_wilson(disponiveis, total),
        oks_median=float(np.median(okss)) if okss else None,
        oks_at_50=sum(v >= 0.50 for v in okss) / total if total else 0.0,
        oks_at_75=sum(v >= 0.75 for v in okss) / total if total else 0.0,
        pck=acertos_pck / pontos_pck if pontos_pck else 0.0,
        error_median_px=float(np.median(erros)) if erros else None,
        error_p95_px=float(np.percentile(erros, 95)) if erros else None,
        n_ankle_frames=quadros_com_tornozelo,
        ankle_available=tornozelo_disponivel,
        ankle_side_correct=lado_certo,
        min_confidence=min_confidence,
        per_keypoint={
            nome: {"pck": round(certos / vistos, 4) if vistos else 0.0, "n": vistos}
            for nome, (certos, vistos) in por_ponto.items()
        },
    )


def oks(label: PoseLabel, predicted: Sequence[Keypoint]) -> float | None:
    """A OKS do COCO com a area da caixa no lugar da area do poligono, porque a anotacao deste
    projeto e de caixa e keypoint. A troca e generosa por cima, entao a mediana daqui serve para
    comparar dois pesos medidos deste mesmo jeito, e nao com o leaderboard deles.

    Ponto que a anotacao traz e o estimador nao devolve soma zero: ignora-lo deixaria um estimador
    que responde pouco com OKS alta.
    """
    if label.area_da_caixa <= 0:
        return None
    anotados = [k for k in label.keypoints if k.anotado]
    if not anotados:
        return None
    indexado = {k.name: k for k in predicted}
    soma = 0.0
    for anotado in anotados:
        estimado = indexado.get(anotado.name)
        if estimado is None:
            continue
        sigma = OKS_SIGMAS[anotado.name]
        distancia2 = _distancia2(anotado.ponto, (estimado.x, estimado.y))
        soma += math.exp(-distancia2 / (2 * label.area_da_caixa * (2 * sigma) ** 2))
    return soma / len(anotados)


def _tem_os_dois_tornozelos(label: PoseLabel) -> bool:
    esquerdo = label.ponto(TORNOZELO_ESQUERDO)
    direito = label.ponto(TORNOZELO_DIREITO)
    return esquerdo is not None and direito is not None


def _tornozelos_sustentados(
    indexado: Mapping[str, Keypoint], min_confidence: float
) -> tuple[Keypoint, Keypoint] | None:
    """Um lado so nao serve: `body_part` compara as velocidades dos dois, entao quadro com um pe
    sustentado nao entra na conta dela e nao pode entrar nesta."""
    esquerdo = indexado.get(TORNOZELO_ESQUERDO)
    direito = indexado.get(TORNOZELO_DIREITO)
    if esquerdo is None or direito is None:
        return None
    for ponto in (esquerdo, direito):
        if ponto.confidence is not None and ponto.confidence < min_confidence:
            return None
    return (esquerdo, direito)


def _lado_correto(label: PoseLabel, sustentado: tuple[Keypoint, Keypoint]) -> bool:
    """Criterio relativo e nao limiar em pixel: a 23,5 px de altura de jogador, que e o minimo
    medido, qualquer limiar absoluto que aceite o caso facil aceita tambem a troca. Empate exato
    conta como errado."""
    esquerdo_anotado = label.ponto(TORNOZELO_ESQUERDO)
    direito_anotado = label.ponto(TORNOZELO_DIREITO)
    if esquerdo_anotado is None or direito_anotado is None:
        return False
    esquerdo, direito = sustentado
    direto = _distancia2(esquerdo_anotado.ponto, (esquerdo.x, esquerdo.y)) + _distancia2(
        direito_anotado.ponto, (direito.x, direito.y)
    )
    trocado = _distancia2(esquerdo_anotado.ponto, (direito.x, direito.y)) + _distancia2(
        direito_anotado.ponto, (esquerdo.x, esquerdo.y)
    )
    return direto < trocado


def _label(arquivo: str, numero: int, document: dict[str, Any]) -> PoseLabel:
    pontos = tuple(
        PoseKeypoint(
            name=str(raw["name"]),
            x=None if raw["x"] is None else float(raw["x"]),
            y=None if raw["y"] is None else float(raw["y"]),
            visibility=str(raw["visibility"]),
        )
        for raw in cast("list[dict[str, Any]]", document["keypoints"])
    )
    nomes = tuple(ponto.name for ponto in pontos)
    if nomes != COCO_KEYPOINTS:
        faltando = sorted(set(COCO_KEYPOINTS) - set(nomes))
        raise PoseLabelError(
            f"{arquivo}:{numero}: os 17 keypoints tem de vir na ordem do COCO"
            + (f", e faltam {', '.join(faltando)}" if faltando else ", e a ordem esta trocada")
        )
    caixa = cast("list[float]", document["shooter_box"])
    x1, y1, x2, y2 = (float(valor) for valor in caixa[:4])
    if x2 <= x1 or y2 <= y1:
        raise PoseLabelError(
            f"{arquivo}:{numero}: shooter_box tem de ser x1 y1 x2 y2 com x2 > x1 e y2 > y1"
        )
    return PoseLabel(
        clip_id=str(document["clip_id"]),
        match_id=str(document["match_id"]),
        frame_index=int(document["frame_index"]),
        frame_width=int(document["frame_width"]),
        frame_height=int(document["frame_height"]),
        shooter_box=(x1, y1, x2, y2),
        keypoints=pontos,
        video_sha256=str(document["video_sha256"]),
        annotator=str(document["annotator"]),
        reviewed_by=cast("str | None", document["reviewed_by"]),
        quality_flags=tuple(cast("list[str]", document["quality_flags"])),
        shooter_track_id=cast("int | None", document.get("shooter_track_id")),
    )


def _distancia2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))
