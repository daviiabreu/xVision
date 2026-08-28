"""Rastreador da bola: escolhe o candidato certo entre quadros e mantem a trajetoria.

Nucleo escrito por Gabriel Alves na branch `feat/ball-tracking` (card #51), trazido para a
fundacao do develop. A logica e dele e nao mudou: a casca de execucao, que abria video por conta
propria, foi trocada pelo contrato `PerceptionModule`, que a suite de conformidade cobra.

O detector diz onde a bola pode estar em cada quadro, ja filtrado por forma e tamanho; o
rastreador cuida do tempo. Entre os candidatos de um quadro, ele escolhe o mais coerente com o
movimento previsto, rejeita saltos improvaveis, interpola lacunas curtas e marca lacunas longas
como perda, sem inventar posicao. A escolha por movimento e o que tira a marca do penalti do
caminho: ela fica parada, fora da trajetoria extrapolada, entao perde para a bola que se move.

Trabalha em pixels de proposito: `pitch_xy` depende da homografia, que e outra frente.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from contracts import BallDetection, Box

TrackPointSource = Literal["detected", "interpolated", "missing"]


@dataclass(frozen=True)
class BallObservation:
    """Candidatos a bola em um quadro, com o indice e o instante de tempo.

    `candidates` vazio significa que a bola nao foi observada no quadro (oclusao, saturacao, borrao,
    quadro nao amostrado ou todos os candidatos reprovados no filtro de forma), caso que o
    rastreador registra como lacuna, sem inventar posicao.
    """

    frame_index: int
    timestamp_s: float
    candidates: tuple[BallDetection, ...]


@dataclass(frozen=True)
class TrackPoint:
    """Um ponto da trajetoria ja resolvido pelo rastreador.

    `source` diz de onde a posicao veio: `detected` (candidato escolhido), `interpolated` (estimada
    numa lacuna curta entre duas deteccoes) ou `missing` (lacuna longa, sem posicao). `center` e
    None apenas quando `missing`. `suspect_static` sinaliza uma bola parada ha muitos quadros; e
    apenas telemetria, so preenchida quando a imobilidade esta ligada na configuracao.
    """

    frame_index: int
    timestamp_s: float
    source: TrackPointSource
    center: tuple[float, float] | None
    box: Box | None
    confidence: float | None
    suspect_static: bool = False


@dataclass(frozen=True)
class TrackerConfig:
    """Limites que separam uma deteccao plausivel de um salto, e uma lacuna curta de uma perda.

    O gate de salto e em pixels por segundo porque nao ha homografia aqui; ele depende da
    resolucao e do zoom do video, entao e um alvo preliminar a calibrar com clipes reais. A
    imobilidade fica **desligada por padrao**: bola parada (falta, escanteio, penalti real)
    tambem e estatica, entao o sinal serve de telemetria, nao de rejeicao automatica.
    """

    max_ball_speed_px_per_s: float = 3000.0
    max_gap_frames: int = 5
    gate_margin: float = 1.5
    flag_stationary: bool = False
    stationary_radius_px: float = 6.0
    stationary_frames: int = 45


def _box_center(box: Box) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _missing(observation: BallObservation) -> TrackPoint:
    return TrackPoint(
        frame_index=observation.frame_index,
        timestamp_s=observation.timestamp_s,
        source="missing",
        center=None,
        box=None,
        confidence=None,
    )


class BallTracker:
    """Mantem a trajetoria da bola ao longo de uma sequencia de quadros.

    Emite os pontos confirmados na hora; durante uma lacuna, segura os quadros pendentes ate a
    proxima deteccao os fechar por interpolacao, ou ate a lacuna passar de `max_gap_frames`, quando
    eles viram perda e o rastreador reancora na deteccao seguinte. A latencia de emissao fica,
    assim, limitada ao tamanho da lacuna tolerada.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        self._last: TrackPoint | None = None
        self._prev: TrackPoint | None = None
        self._pending: list[BallObservation] = []
        self._static_run = 0

    def update(self, observation: BallObservation) -> list[TrackPoint]:
        """Incorpora um quadro e devolve os pontos que este quadro conseguiu finalizar."""
        chosen = self._select(observation.candidates, observation.timestamp_s)
        if chosen is not None:
            center = _box_center(chosen.box)
            confirmed = TrackPoint(
                frame_index=observation.frame_index,
                timestamp_s=observation.timestamp_s,
                source="detected",
                center=center,
                box=chosen.box,
                confidence=chosen.confidence,
                suspect_static=self._update_static_run(center),
            )
            resolved = self._resolve_pending(confirmed)
            self._prev = self._last
            self._last = confirmed
            self._pending = []
            return [*resolved, confirmed]

        self._pending.append(observation)
        if len(self._pending) > self._config.max_gap_frames:
            lost = [_missing(pending) for pending in self._pending]
            self._pending = []
            self._reset_track()
            return lost
        return []

    def flush(self) -> list[TrackPoint]:
        """Fecha o stream: uma lacuna aberta no fim nunca sera resolvida, entao vira perda."""
        lost = [_missing(pending) for pending in self._pending]
        self._pending = []
        return lost

    def track(self, observations: Iterable[BallObservation]) -> Iterator[TrackPoint]:
        """Processa uma sequencia inteira e produz a trajetoria em ordem de quadro."""
        for observation in observations:
            yield from self.update(observation)
        yield from self.flush()

    def _select(
        self, candidates: tuple[BallDetection, ...], timestamp_s: float
    ) -> BallDetection | None:
        """Entre os candidatos plausiveis, o mais proximo da posicao prevista pelo movimento."""
        if not candidates:
            return None
        last = self._last
        if last is None or last.center is None:
            return max(candidates, key=lambda d: d.confidence)

        dt = timestamp_s - last.timestamp_s
        allowed = self._config.max_ball_speed_px_per_s * dt * self._config.gate_margin
        lx, ly = last.center
        gated = [
            candidate
            for candidate in candidates
            if dt <= 0 or math.hypot(*_delta(_box_center(candidate.box), (lx, ly))) <= allowed
        ]
        if not gated:
            return None

        predicted = self._predict(timestamp_s)
        return min(
            gated,
            key=lambda d: (
                math.hypot(*_delta(_box_center(d.box), predicted)),
                -d.confidence,
            ),
        )

    def _predict(self, timestamp_s: float) -> tuple[float, float]:
        """Extrapola a proxima posicao pela velocidade dos dois ultimos pontos confirmados."""
        last = self._last
        prev = self._prev
        if last is None or last.center is None:
            return (0.0, 0.0)
        if prev is None or prev.center is None:
            return last.center
        span = last.timestamp_s - prev.timestamp_s
        if span <= 0:
            return last.center
        vx = (last.center[0] - prev.center[0]) / span
        vy = (last.center[1] - prev.center[1]) / span
        ahead = timestamp_s - last.timestamp_s
        return (last.center[0] + vx * ahead, last.center[1] + vy * ahead)

    def _update_static_run(self, center: tuple[float, float]) -> bool:
        if not self._config.flag_stationary:
            return False
        last = self._last
        if last is not None and last.center is not None:
            moved = math.hypot(*_delta(center, last.center))
            self._static_run = (
                self._static_run + 1 if moved <= self._config.stationary_radius_px else 0
            )
        return self._static_run >= self._config.stationary_frames

    def _reset_track(self) -> None:
        self._last = None
        self._prev = None
        self._static_run = 0

    def _resolve_pending(self, end: TrackPoint) -> list[TrackPoint]:
        if not self._pending:
            return []
        start = self._last
        if start is None or start.center is None or end.center is None:
            return [_missing(pending) for pending in self._pending]
        if len(self._pending) > self._config.max_gap_frames:
            return [_missing(pending) for pending in self._pending]
        return [self._interpolate(start, end, pending) for pending in self._pending]

    def _interpolate(
        self, start: TrackPoint, end: TrackPoint, observation: BallObservation
    ) -> TrackPoint:
        if start.center is None or end.center is None:
            return _missing(observation)
        span = end.timestamp_s - start.timestamp_s
        fraction = 0.0 if span <= 0 else (observation.timestamp_s - start.timestamp_s) / span
        sx, sy = start.center
        ex, ey = end.center
        return TrackPoint(
            frame_index=observation.frame_index,
            timestamp_s=observation.timestamp_s,
            source="interpolated",
            center=(sx + (ex - sx) * fraction, sy + (ey - sy) * fraction),
            box=None,
            confidence=None,
        )


def _delta(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])
