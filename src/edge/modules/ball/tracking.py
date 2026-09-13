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
from dataclasses import dataclass, replace
from typing import Literal

from contracts import BallDetection, Box
from edge.modules.base import ModuleError

TrackPointSource = Literal["detected", "interpolated", "missing"]
VelocitySource = Literal["finite-difference", "trajectory-model"]


class TrajectoryOrderError(ModuleError): ...


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
    camera_cut: bool = False


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
    trajectory_segment: int | None = None
    gap_frames: int | None = None
    velocity_xy_px_s: tuple[float, float] | None = None
    velocity_source: VelocitySource | None = None
    suspect_static: bool = False


@dataclass(frozen=True)
class TrackerTelemetry:
    detected_points: int = 0
    interpolated_points: int = 0
    missing_points: int = 0
    segments: int = 0
    resets: int = 0
    rejected_jumps: int = 0
    camera_cuts: int = 0
    max_gap_frames: int = 0


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
    candidate_order: bool = False
    """Aceitar o primeiro candidato que sobrevive ao gate, na ordem em que a lista chegou.

    Existe para quem entrega a lista ja ordenada por outro criterio, hoje o ranking de
    `edge.modules.ball.ranking`. Sem isso, o desempate por proximidade da posicao prevista reescolhe
    e desfaz a ordem: medido nas 328 posicoes anotadas, entregar os dois melhores e desempatar por
    proximidade cai para 161 acertos, e respeitando a ordem fica em 177.
    """


def _box_center(box: Box) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _missing(observation: BallObservation, segment: int | None, gap_frames: int) -> TrackPoint:
    return TrackPoint(
        frame_index=observation.frame_index,
        timestamp_s=observation.timestamp_s,
        source="missing",
        center=None,
        box=None,
        confidence=None,
        trajectory_segment=segment,
        gap_frames=gap_frames,
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
        self._segment = -1
        self._last_frame_index: int | None = None
        self._last_timestamp_s: float | None = None
        self._perdido: tuple[tuple[float, float], float] | None = None
        self._telemetry = TrackerTelemetry()

    @property
    def telemetry(self) -> TrackerTelemetry:
        return self._telemetry

    def update(self, observation: BallObservation) -> list[TrackPoint]:
        """Incorpora um quadro e devolve os pontos que este quadro conseguiu finalizar."""
        self._check_order(observation)
        prefix: list[TrackPoint] = []
        if observation.camera_cut:
            self._telemetry = replace(self._telemetry, camera_cuts=self._telemetry.camera_cuts + 1)
            prefix = self._close_pending()
            self._reset_track(count=bool(self._last is not None or prefix), lembrar=False)

        chosen, rejected_jump = self._select(observation.candidates, observation.timestamp_s)
        if rejected_jump:
            self._telemetry = replace(
                self._telemetry, rejected_jumps=self._telemetry.rejected_jumps + 1
            )
        if chosen is not None:
            if self._last is None:
                self._segment += 1
                self._telemetry = replace(self._telemetry, segments=self._telemetry.segments + 1)
            center = _box_center(chosen.box)
            velocity = self._velocity_from_last(center, observation.timestamp_s)
            confirmed = TrackPoint(
                frame_index=observation.frame_index,
                timestamp_s=observation.timestamp_s,
                source="detected",
                center=center,
                box=chosen.box,
                confidence=chosen.confidence,
                trajectory_segment=self._segment,
                gap_frames=0,
                velocity_xy_px_s=velocity,
                velocity_source="finite-difference" if velocity is not None else None,
                suspect_static=self._update_static_run(center),
            )
            resolved = self._resolve_pending(confirmed)
            self._prev = self._last
            self._last = confirmed
            self._pending = []
            self._telemetry = replace(
                self._telemetry,
                detected_points=self._telemetry.detected_points + 1,
                interpolated_points=self._telemetry.interpolated_points
                + sum(point.source == "interpolated" for point in resolved),
                missing_points=self._telemetry.missing_points
                + sum(point.source == "missing" for point in resolved),
            )
            return [*prefix, *resolved, confirmed]

        self._pending.append(observation)
        self._telemetry = replace(
            self._telemetry,
            max_gap_frames=max(self._telemetry.max_gap_frames, len(self._pending)),
        )
        if len(self._pending) > self._config.max_gap_frames:
            lost = self._close_pending()
            self._pending = []
            self._reset_track(count=True)
            return [*prefix, *lost]
        return prefix

    def flush(self) -> list[TrackPoint]:
        """Fecha o stream: uma lacuna aberta no fim nunca sera resolvida, entao vira perda."""
        return self._close_pending()

    def track(self, observations: Iterable[BallObservation]) -> Iterator[TrackPoint]:
        """Processa uma sequencia inteira e produz a trajetoria em ordem de quadro."""
        for observation in observations:
            yield from self.update(observation)
        yield from self.flush()

    def _select(
        self, candidates: tuple[BallDetection, ...], timestamp_s: float
    ) -> tuple[BallDetection | None, bool]:
        """Entre os candidatos plausiveis, o mais proximo da posicao prevista pelo movimento."""
        if not candidates:
            return None, False
        last = self._last
        if last is None or last.center is None:
            return self._reancorar(candidates, timestamp_s)

        dt = timestamp_s - last.timestamp_s
        allowed = self._config.max_ball_speed_px_per_s * dt * self._config.gate_margin
        lx, ly = last.center
        gated = [
            candidate
            for candidate in candidates
            if dt <= 0 or math.hypot(*_delta(_box_center(candidate.box), (lx, ly))) <= allowed
        ]
        if not gated:
            return None, True

        if self._config.candidate_order:
            return gated[0], False
        predicted = self._predict(timestamp_s)
        return (
            min(
                gated,
                key=lambda d: (
                    math.hypot(*_delta(_box_center(d.box), predicted)),
                    -d.confidence,
                ),
            ),
            False,
        )

    def _reancorar(
        self, candidates: tuple[BallDetection, ...], timestamp_s: float
    ) -> tuple[BallDetection | None, bool]:
        """A primeira ancora de um segmento, restrita ao que a fisica admite desde a ultima perda.

        Sem essa restricao a reancora era a maior confianca em qualquer lugar do quadro, e uma
        deteccao marginal perdida no corte de confianca levava a trajetoria para outro objeto:
        medido num clipe, o fp16 divergia em 61 dos 360 quadros e o pior caso ficou a 1641 px, com
        a bola do fim do clipe sendo outro objeto.

        O gate e o MESMO que protege a continuacao, `max_ball_speed_px_per_s * dt * gate_margin`,
        so aplicado atraves da lacuna em vez de descartado nela. Ele nao precisa de prazo: com os
        padroes de 3000 px/s e margem 1,5, passado cerca de meio segundo o raio permitido excede a
        diagonal de um quadro 1920x1080 e a memoria deixa de restringir sozinha.

        Sem memoria, que e o comeco do clipe, o criterio segue sendo a maior confianca.
        """
        perdido = self._perdido
        if perdido is not None:
            (px, py), quando = perdido
            dt = timestamp_s - quando
            allowed = self._config.max_ball_speed_px_per_s * dt * self._config.gate_margin
            alcancaveis = tuple(
                candidate
                for candidate in candidates
                if dt <= 0 or math.hypot(*_delta(_box_center(candidate.box), (px, py))) <= allowed
            )
            if not alcancaveis:
                return None, True
            candidates = alcancaveis
        if self._config.candidate_order:
            return candidates[0], False
        return max(candidates, key=lambda d: d.confidence), False

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

    def _reset_track(self, *, count: bool, lembrar: bool = True) -> None:
        """Esquece a trajetoria, mas lembra ONDE ela morreu, para a reancora nao trocar de objeto.

        `lembrar` fica falso quando a posicao anterior deixou de significar algo: corte de camera
        muda o plano inteiro, e erro de ordem quebra a identidade do stream. Nos dois casos guardar
        a ultima posicao faria a reancora respeitar uma coordenada de outro enquadramento.
        """
        if lembrar and self._last is not None and self._last.center is not None:
            self._perdido = (self._last.center, self._last.timestamp_s)
        elif not lembrar:
            self._perdido = None
        self._last = None
        self._prev = None
        self._static_run = 0
        if count:
            self._telemetry = replace(self._telemetry, resets=self._telemetry.resets + 1)

    def _check_order(self, observation: BallObservation) -> None:
        if not math.isfinite(observation.timestamp_s):
            self._reset_track(count=self._last is not None, lembrar=False)
            raise TrajectoryOrderError("timestamp da bola precisa ser finito")
        if self._last_frame_index is not None and observation.frame_index <= self._last_frame_index:
            self._reset_track(count=self._last is not None, lembrar=False)
            raise TrajectoryOrderError(
                f"frame da bola {observation.frame_index} nao sucede {self._last_frame_index}"
            )
        if self._last_timestamp_s is not None and observation.timestamp_s <= self._last_timestamp_s:
            self._reset_track(count=self._last is not None, lembrar=False)
            raise TrajectoryOrderError(
                f"timestamp da bola {observation.timestamp_s} nao sucede {self._last_timestamp_s}"
            )
        self._last_frame_index = observation.frame_index
        self._last_timestamp_s = observation.timestamp_s

    def _velocity_from_last(
        self, center: tuple[float, float], timestamp_s: float
    ) -> tuple[float, float] | None:
        last = self._last
        if last is None or last.center is None:
            return None
        span = timestamp_s - last.timestamp_s
        if span <= 0:
            return None
        dx, dy = _delta(center, last.center)
        return (dx / span, dy / span)

    def _close_pending(self) -> list[TrackPoint]:
        gap = len(self._pending)
        lost = [
            _missing(pending, self._segment if self._last is not None else None, gap)
            for pending in self._pending
        ]
        self._pending = []
        if lost:
            self._telemetry = replace(
                self._telemetry, missing_points=self._telemetry.missing_points + len(lost)
            )
        return lost

    def _resolve_pending(self, end: TrackPoint) -> list[TrackPoint]:
        if not self._pending:
            return []
        start = self._last
        if start is None or start.center is None or end.center is None:
            return [
                _missing(pending, self._segment if start is not None else None, len(self._pending))
                for pending in self._pending
            ]
        if len(self._pending) > self._config.max_gap_frames:
            return [
                _missing(pending, self._segment, len(self._pending)) for pending in self._pending
            ]
        return [
            self._interpolate(start, end, pending, len(self._pending)) for pending in self._pending
        ]

    def _interpolate(
        self,
        start: TrackPoint,
        end: TrackPoint,
        observation: BallObservation,
        gap_frames: int,
    ) -> TrackPoint:
        if start.center is None or end.center is None:
            return _missing(observation, self._segment, gap_frames)
        span = end.timestamp_s - start.timestamp_s
        fraction = 0.0 if span <= 0 else (observation.timestamp_s - start.timestamp_s) / span
        sx, sy = start.center
        ex, ey = end.center
        velocity = None if span <= 0 else ((ex - sx) / span, (ey - sy) / span)
        return TrackPoint(
            frame_index=observation.frame_index,
            timestamp_s=observation.timestamp_s,
            source="interpolated",
            center=(sx + (ex - sx) * fraction, sy + (ey - sy) * fraction),
            box=None,
            confidence=None,
            trajectory_segment=self._segment,
            gap_frames=gap_frames,
            velocity_xy_px_s=velocity,
            velocity_source="trajectory-model" if velocity is not None else None,
        )


def _delta(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])
