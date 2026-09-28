"""Identifica se o finalizador estava sob pressao de um adversario no contato.

Marcacao aqui e uma **afirmacao existencial**: havia adversario dentro do raio no instante do
chute. Nao e o pareamento reciproco de `pair_markings`, que responde outra pergunta ("quem esta
marcando quem no quadro inteiro") e continua servindo os consumidores de producao que a usam. Um
zagueiro cobrindo dois atacantes marca os dois, mas o pareamento mutuo elegia um so e publicava
`False` para o outro.

Porque existencial, dado faltante so pode produzir falso negativo, nunca falso positivo. Um
adversario sem `pitch_xy` nao vira distancia inventada: se ninguem foi achado no raio e ele podia
ser o marcador, a resposta e `unavailable` em vez de um `False` que a percepcao nao sustenta.

O quadro e o finalizador vem da cena do chute, `edge.features.scene`: o quadro com homografia mais
perto do contato, e o finalizador declarado pelo `shot` ou, na falta dele, o dono do episodio de
posse do contato. Ate a 0.2.0 a feature exigia `shooter_track_id`, que nenhum modulo de instante
emite, e saia `unavailable` em todo clipe real.
"""

from __future__ import annotations

from contracts import FeatureValue, Perception
from edge.features.base import FeatureExtractor, FeatureSpec, ShotOutput, unavailable
from edge.features.scene import SceneRefusal, shot_scene
from edge.participants.identity import Team
from edge.participants.marking import DEFAULT_RADIUS_M, opponent_pressure, pressure_roster


class Marked(FeatureExtractor):
    spec = FeatureSpec(
        name="marked", version="0.3.0", requires=("players", "ball", "field", "tracking")
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        cena = shot_scene(perception, shot)
        if isinstance(cena, SceneRefusal):
            return unavailable(self.spec, cena.detail)
        frame = cena.frame
        finalizador = cena.shooter.track_id
        if finalizador is None:
            return unavailable(self.spec, "finalizador sem track_id no quadro de referencia")

        teams: dict[int, Team] = {
            player.track_id: player.team
            for player in frame.players
            if player.track_id is not None and player.team in ("team_a", "team_b")
        }
        if finalizador not in teams:
            return unavailable(self.spec, "finalizador sem equipe nesse frame")

        roster = pressure_roster(frame, teams)
        pressao = opponent_pressure(roster, finalizador, radius=DEFAULT_RADIUS_M)
        if pressao.marcado is None:
            return unavailable(self.spec, pressao.motivo or "marcacao indeterminada")
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=pressao.marcado,
            confidence=None,
            reason=None,
            evidence={
                "frame_index": frame.frame_index,
                "shot_frame_index": shot.frame_index,
                "defasagem_s": round(cena.offset_s, 4),
                "origem_do_finalizador": cena.shooter_source,
                "shooter_track_id": finalizador,
                "radius_m": DEFAULT_RADIUS_M,
                "markers": [
                    {"track_id": chave, "distance": distancia}
                    for chave, distancia in pressao.marcadores
                ],
                "opponents_without_position": pressao.adversarios_sem_posicao,
            },
        )
