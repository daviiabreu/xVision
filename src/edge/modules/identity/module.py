"""A especie `identity`: equipe e papel por cor de camisa acumulada na trajetória.

O detector de jogadores enxerga uma caixa isolada; a cor de uma camisa fica confiável quando
amostras de uma mesma trajetória são acumuladas durante o clipe. Este módulo projeta a equipe e
corrige o papel da arbitragem pela cor, nunca infere número de camisa.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import median
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
from edge.modules.field.geometry import FieldProjectionError, image_to_pitch
from edge.modules.players import PREPROCESS as PLAYERS_PREPROCESS
from edge.participants.identity import (
    DEFAULT_SAMPLE_LIMIT,
    OFF_PITCH_ROLE,
    LabColor,
    TrackObservation,
    ambiguous_team,
    cluster_teams,
    enough_team_contrast,
    extract_shirt_color,
    off_pitch,
    representative_colors,
    resolve_goalkeepers,
    resolve_roles_and_teams,
    selected_colors,
    stabilize_role,
)
from edge.participants.types import Role, as_role

PREPROCESS = PLAYERS_PREPROCESS


class TeamIdentity:
    """Agrupa trajetórias em duas equipes e separa a arbitragem como terceira cor."""

    spec = ModuleSpec(
        kind="identity",
        name="team-color",
        version="0.2.0",
        frames="window",
        preprocess=PREPROCESS,
        needs=("players", "tracking"),
        optional_needs=("field",),
    )

    def __init__(self, *, limit: int = DEFAULT_SAMPLE_LIMIT, minimum: int = 2) -> None:
        self._limit = limit
        self._minimum = minimum
        self._pronto = False

    def load(self) -> None:
        self._pronto = True

    def close(self) -> None:
        self._pronto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)

        players = _by_frame(ctx.get("players"))
        tracks = _tracks_by_frame(ctx.get("tracking"))
        homografias = _homographies(ctx.get("field"))
        # Sem estado entre chamadas: o lote é o clipe inteiro, porque a espécie roda com
        # `PerFrame`, e o cache guarda a saída pela chave do lote, o que só vale se ela depender
        # só dele. Acumular entre lotes deixava a mesma chave com respostas diferentes.
        amostras_por_track: dict[int, list[tuple[TrackObservation, LabColor]]] = {}
        observacoes: dict[int, list[TrackObservation]] = {}
        pontos_no_campo: dict[int, list[tuple[float, float]]] = {}
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
                observacoes.setdefault(track_id, []).append(observation)
                ponto = _pitch(observation.detection.box, homografias.get(frame_index))
                if ponto is not None:
                    pontos_no_campo.setdefault(track_id, []).append(ponto)
                color = _shirt_color(batch, position, observation.detection.box)
                if color is not None:
                    amostras_por_track.setdefault(track_id, []).append((observation, color))
            vistos.append(do_quadro)

        # O papel vem do `players` quadro a quadro, e não do papel já estável do tracking: o
        # árbitro que o detector marca como jogador na maior parte dos quadros ainda deixa algum
        # quadro de `referee`, e é essa evidência que a resolução pela cor usa.
        evidencia: dict[int, set[Role]] = {
            track_id: {
                papel for observacao in itens if (papel := as_role(observacao.detection.role))
            }
            for track_id, itens in observacoes.items()
        }
        posicoes = {
            track_id: (median(x for x, _ in pontos), median(y for _, y in pontos))
            for track_id, pontos in pontos_no_campo.items()
        }
        # Quem está fora das linhas, staff, técnico e fotógrafo, sai antes de tudo: senão entra na
        # semente dos times e na do árbitro, e o fotógrafo atrás do gol ganha o lugar do goleiro.
        fora = {track_id for track_id, posicao in posicoes.items() if off_pitch(posicao)}
        estaveis: dict[int, Role] = {
            track_id: stabilize_role(observacoes[track_id])[0]
            for track_id, papeis in evidencia.items()
            if papeis and track_id not in fora
        }
        amostras = selected_colors(amostras_por_track, limit=self._limit)
        cores = {
            track_id: cor
            for track_id, cor in representative_colors(amostras, minimum=self._minimum).items()
            if track_id not in fora
        }
        # Só trajetória de jogador semeia o agrupamento: árbitro e goleiro vestem cor própria e
        # puxariam um dos dois grupos para a cor errada, contaminando o time de quem está em campo.
        # Quem teve quadro de `referee` também fica fora da semente, mesmo com voto de jogador, e a
        # resolução devolve o time a quem for jogador de fato.
        agrupado = cluster_teams(
            {
                track_id: cor
                for track_id, cor in cores.items()
                if estaveis.get(track_id) == "player" and "referee" not in evidencia[track_id]
            }
        )
        sementes, centroides = agrupado if agrupado is not None else ({}, {})
        if not enough_team_contrast(centroides):
            sementes, centroides = {}, {}
        papeis, teams, _ = resolve_roles_and_teams(
            {track_id: papel for track_id, papel in estaveis.items() if track_id in cores},
            cores,
            sementes,
            centroides,
            role_evidence=evidencia,
            color_samples=amostras,
            referee_votes={
                track_id: sum(
                    o.detection.confidence for o in itens if o.detection.role == "referee"
                )
                for track_id, itens in observacoes.items()
            },
        )
        # O goleiro não entra na resolução acima, que o deixa de fora, e sai depois dela porque
        # precisa saber quem é árbitro e quem já tem time para achar o time do gol que defende.
        papeis, teams = resolve_goalkeepers(
            papeis,
            teams,
            cores,
            centroides,
            goalkeeper_votes={
                track_id: sum(
                    o.detection.confidence for o in itens if o.detection.role == "goalkeeper"
                )
                for track_id, itens in observacoes.items()
            },
            positions=posicoes,
            frames={
                track_id: {o.frame_index for o in itens} for track_id, itens in observacoes.items()
            },
        )
        teams = {
            track_id: team
            for track_id, team in teams.items()
            if papeis.get(track_id) == "goalkeeper"
            or not ambiguous_team(cores[track_id], centroides)
        }
        emitidos: dict[int, str] = {**papeis, **dict.fromkeys(fora, OFF_PITCH_ROLE)}
        records = tuple(
            tuple(
                _identity_record(track_id, teams.get(track_id), emitidos.get(track_id))
                for track_id in do_quadro
            )
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


def _homographies(output: ModuleOutput | None) -> dict[int, tuple[float, ...]]:
    if output is None:
        return {}
    return {
        frame: tuple(float(v) for v in cast(Any, records[0]["homography"]))
        for frame, records in zip(output.frames, output.records, strict=True)
        if records and records[0].get("homography") is not None
    }


def _pitch(box: Box, homography: tuple[float, ...] | None) -> tuple[float, float] | None:
    """Pé da caixa no campo, sem cortar o que cai fora das linhas, que é o que o staff tem."""
    if homography is None:
        return None
    try:
        return image_to_pitch(homography, ((box.x1 + box.x2) / 2.0, box.y2))
    except FieldProjectionError:
        return None


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


def _identity_record(track_id: int, team: str | None, role: str | None) -> dict[str, object]:
    # `role` só sai para trajetória com cor representativa, a mesma condição de `team`, ou fora do
    # campo pela posição: antes disso a identidade não julgou nada, e o papel do tracking vale.
    record: dict[str, object] = {"track_id": track_id}
    if team is not None:
        record["team"] = team
    if role is not None:
        record["role"] = role
    return record
