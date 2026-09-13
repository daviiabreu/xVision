"""A especie `identity`: equipes por cor de camisa acumulada na trajetória.

O detector de jogadores enxerga uma caixa isolada; a cor de uma camisa fica confiável quando
amostras de uma mesma trajetória são acumuladas durante o clipe. Este módulo só projeta a equipe,
nunca infere número de camisa.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from contracts import Box, PlayerDetection
from edge.frames.preprocess import to_tensor_box
from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    check_preprocess,
)
from edge.modules.players import PREPROCESS as PLAYERS_PREPROCESS
from edge.participants.identity import (
    DEFAULT_SAMPLE_LIMIT,
    LabColor,
    TrackObservation,
    extract_shirt_color,
    stabilize_role,
    teams_from_color_samples,
)

PREPROCESS = PLAYERS_PREPROCESS


class TeamIdentity:
    """Agrupa trajetórias em duas equipes a partir de cores extraídas dos frames BGR."""

    spec = ModuleSpec(
        kind="identity",
        name="team-color",
        version="0.1.0",
        frames="window",
        preprocess=PREPROCESS,
        needs=("players", "tracking"),
    )

    def __init__(self, *, limit: int = DEFAULT_SAMPLE_LIMIT, minimum: int = 2) -> None:
        self._limit = limit
        self._minimum = minimum
        self._samples: dict[int, list[tuple[TrackObservation, LabColor]]] = {}
        self._pronto = False

    def load(self) -> None:
        self._samples = {}
        self._pronto = True

    def close(self) -> None:
        self._samples = {}
        self._pronto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)

        players = _by_frame(ctx.get("players"))
        tracks = _tracks_by_frame(ctx.get("tracking"))
        # Uma lista por quadro do lote, na ordem de `batch.indices`, porque `ModuleOutput` cobra um
        # conjunto de registros por quadro e o mesmo `track_id` reaparece de um quadro para o outro.
        vistos: list[list[int]] = []
        for position, frame_index in enumerate(batch.indices):
            do_quadro: list[int] = []
            detections = players.get(frame_index, ())
            for detection_index, track_id in tracks.get(frame_index, {}).items():
                if detection_index >= len(detections):
                    continue
                observation = _observation(frame_index, detections[detection_index], track_id)
                if observation is None:
                    continue
                do_quadro.append(track_id)
                color = _shirt_color(batch, position, observation.detection.box)
                if color is not None:
                    self._samples.setdefault(track_id, []).append((observation, color))
            vistos.append(do_quadro)

        # Só trajetória de jogador entra no agrupamento: árbitro e goleiro vestem cor própria e
        # puxariam um dos dois grupos para a cor errada, contaminando o time de quem está em campo.
        jogadores = {
            track_id: amostras
            for track_id, amostras in self._samples.items()
            if stabilize_role(observation for observation, _ in amostras)[0] == "player"
        }
        # Reclassifica o clipe inteiro a cada lote: a equipe de uma trajetória antiga pode mudar
        # quando amostras novas deslocam os dois grupos, e o registro emitido agora é o que vale.
        teams, _ = teams_from_color_samples(jogadores, limit=self._limit, minimum=self._minimum)
        records = tuple(
            tuple(_identity_record(track_id, teams.get(track_id)) for track_id in do_quadro)
            for do_quadro in vistos
        )
        return ModuleOutput(kind=self.spec.kind, frames=batch.indices, records=records)


def _by_frame(output: ModuleOutput | None) -> dict[int, tuple[Mapping[str, object], ...]]:
    if output is None:
        return {}
    return dict(zip(output.frames, output.records, strict=True))


def _tracks_by_frame(output: ModuleOutput | None) -> dict[int, dict[int, int]]:
    if output is None:
        return {}
    return {
        frame: {
            int(cast(Any, record["detection_index"])): int(cast(Any, record["track_id"]))
            for record in records
            if "detection_index" in record and "track_id" in record
        }
        for frame, records in zip(output.frames, output.records, strict=True)
    }


def _observation(
    frame_index: int, record: Mapping[str, object], track_id: int
) -> TrackObservation | None:
    if "box" not in record or "confidence" not in record:
        return None
    box = Box(*cast(tuple[float, float, float, float], record["box"]))
    detection = PlayerDetection(
        box=box,
        confidence=float(cast(Any, record["confidence"])),
        track_id=track_id,
        role=cast(str | None, record.get("role")),
    )
    return TrackObservation(frame_index=frame_index, detection=detection)


def _shirt_color(batch: FrameBatch, position: int, box: Box) -> LabColor | None:
    raw_box = (box.x1, box.y1, box.x2, box.y2)
    # `players` emite caixa em pixel do quadro original, e o extrator lê o tensor já redimensionado
    # e letterboxed: sem esta conversão o recorte cai no lugar errado e ainda parece uma cor válida.
    tensor_box = (
        raw_box
        if batch.source_size is None
        else to_tensor_box(raw_box, batch.preprocess, batch.source_size)
    )
    return extract_shirt_color(batch.tensor[position], Box(*tensor_box))


def _identity_record(track_id: int, team: str | None) -> dict[str, object]:
    record: dict[str, object] = {"track_id": track_id}
    if team is not None:
        record["team"] = team
    return record
