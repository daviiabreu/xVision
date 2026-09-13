"""Identifica se o finalizador estava marcado por um adversario no contato."""

from __future__ import annotations

from contracts import FeatureValue, Perception
from edge.features.base import FeatureExtractor, FeatureSpec, ShotOutput, unavailable
from edge.participants.identity import Team
from edge.participants.marking import candidates_from_frame, marked_keys, pair_markings


class Marked(FeatureExtractor):
    spec = FeatureSpec(name="marked", version="0.1.0", requires=("players", "ball"))

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        frame = next((f for f in perception.frames if f.frame_index == shot.frame_index), None)
        if frame is None:
            return unavailable(self.spec, f"sem percepcao pro frame {shot.frame_index}")
        if shot.shooter_track_id is None:
            return unavailable(self.spec, "shot sem shooter_track_id")

        teams: dict[int, Team] = {
            player.track_id: player.team
            for player in frame.players
            if player.track_id is not None and player.team in ("team_a", "team_b")
        }
        if any(
            player.track_id is not None and player.track_id in teams and player.pitch_xy is None
            for player in frame.players
        ):
            return unavailable(self.spec, "jogador com equipe sem pitch_xy nesse frame")
        candidates = candidates_from_frame(frame, teams)
        if not candidates:
            return unavailable(self.spec, "nenhum jogador com equipe e posicao no frame")
        pairs = pair_markings(candidates)
        marked = marked_keys(pairs)
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=shot.shooter_track_id in marked,
            confidence=None,
            reason=None,
            evidence={
                "frame_index": frame.frame_index,
                "shooter_track_id": shot.shooter_track_id,
                "marked_keys": sorted(marked),
                "pairs": [
                    {"team_a": p.team_a_key, "team_b": p.team_b_key, "distance": p.distance}
                    for p in pairs
                ],
            },
        )
