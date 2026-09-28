"""Identifica se houve passe de um companheiro para o finalizador.

Assistencia exige duas coisas juntas: que o toque anterior seja de outro jogador, e que esse
jogador seja da mesma equipe. Bola tomada do adversario nao e assistencia — ninguem assistiu, o
finalizador roubou a bola.

Ate a versao 0.1.0 a feature media apenas troca de pe, e deixava a distincao de equipe para o
`same_team` da evidencia. Isso publicava `value is True` para bola tomada, e quem consumisse o
valor sem ler a evidencia contava roubo como passe. A decisao de setembro de 2026 mudou a
semantica: a equipe entra no valor, e `same_team` continua na evidencia para explicar a resposta.

Equipe desconhecida nao vira `False`: sem saber de quem era a bola nao da para afirmar que nao
houve passe, entao a resposta e `unavailable`. Preferir o `False` silencioso publicaria negativa
onde ha ignorancia, que e o defeito que as outras features deste pacote evitam.
"""

from __future__ import annotations

from contracts import FeatureValue, Perception
from edge.features.base import FeatureExtractor, FeatureSpec, ShotOutput, unavailable
from edge.features.touch import previous_touch, shooter_at_contact, telemetry


class Assist(FeatureExtractor):
    spec = FeatureSpec(name="assist", version="0.1.0", requires=("players", "ball"))

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        shooter = shooter_at_contact(perception, shot.frame_index)
        if shooter is None:
            details = telemetry(perception, shot.frame_index)
            reason = (
                "shooter-at-contact-ambiguous"
                if details.motivo_do_contato == "ambiguous"
                else "shooter-at-contact-unavailable"
            )
            return unavailable(self.spec, reason)

        previous = previous_touch(perception, shot.frame_index)
        if previous is None:
            details = telemetry(perception, shot.frame_index)
            reason = (
                "previous-touch-ambiguous"
                if details.motivo_do_toque_anterior == "ambiguous"
                else "previous-touch-unavailable"
            )
            return unavailable(self.spec, reason)

        equipe_do_finalizador = _team_at_contact(
            perception, shot.frame_index, shooter.player_track_id
        )
        if previous.team is None or equipe_do_finalizador is None:
            return unavailable(self.spec, "team-unavailable")
        mesma_equipe = previous.team == equipe_do_finalizador

        assisted = previous.player_track_id != shooter.player_track_id and mesma_equipe
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=assisted,
            confidence=min(previous.confidence, shooter.confidence),
            reason=None,
            evidence={
                "shooter_track_id": shooter.player_track_id,
                "previous_touch_track_id": previous.player_track_id,
                "previous_touch_frame_index": previous.frame_index,
                "same_team": mesma_equipe,
                "previous_touch_source": previous.source,
                "shooter_source": shooter.source,
            },
        )


def _team_at_contact(perception: Perception, frame_index: int, track_id: int) -> str | None:
    frame = next((f for f in perception.frames if f.frame_index == frame_index), None)
    if frame is None:
        return None
    return next((p.team for p in frame.players if p.track_id == track_id), None)
