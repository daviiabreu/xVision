"""Identifica se o finalizador estava sob pressao de um adversario no contato.

Marcacao aqui e uma **afirmacao existencial**: havia adversario dentro do raio no instante do
chute. Nao e o pareamento reciproco de `pair_markings`, que responde outra pergunta ("quem esta
marcando quem no quadro inteiro") e continua servindo os consumidores de producao que a usam. Um
zagueiro cobrindo dois atacantes marca os dois, mas o pareamento mutuo elegia um so e publicava
`False` para o outro.

Porque existencial, dado faltante so pode produzir falso negativo, nunca falso positivo. Um
adversario sem `pitch_xy` nao vira distancia inventada: se ninguem foi achado no raio e ele podia
ser o marcador, a resposta e `unavailable` em vez de um `False` que a percepcao nao sustenta.
"""

from __future__ import annotations

from contracts import FeatureValue, Perception
from contracts.perception import FramePerception
from edge.features.base import FeatureExtractor, FeatureSpec, ShotOutput, unavailable
from edge.participants.identity import Team
from edge.participants.marking import DEFAULT_RADIUS_M, opponent_pressure, pressure_roster

TOLERANCIA_EM_PASSOS = 2.0
"""Quantos passos de amostragem separam o contato do quadro que ainda pode descrever a marcacao.

Mesma razao do `touch.py`: o instante do contato e refinado entre amostras, entao exigir quadro
exato transformava em `unavailable` justamente os chutes cuja deteccao foi mais precisa. Um passo
cobre o refinamento, o segundo cobre a defasagem de uma amostra.
"""


class Marked(FeatureExtractor):
    spec = FeatureSpec(name="marked", version="0.2.0", requires=("players", "ball"))

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        if shot.shooter_track_id is None:
            return unavailable(self.spec, "shot sem shooter_track_id")
        frame = _quadro_do_contato(perception, shot.frame_index)
        if frame is None:
            return unavailable(self.spec, f"sem percepcao pro frame {shot.frame_index}")

        teams: dict[int, Team] = {
            player.track_id: player.team
            for player in frame.players
            if player.track_id is not None and player.team in ("team_a", "team_b")
        }
        if shot.shooter_track_id not in teams:
            return unavailable(self.spec, "finalizador sem equipe nesse frame")

        roster = pressure_roster(frame, teams)
        pressao = opponent_pressure(roster, shot.shooter_track_id, radius=DEFAULT_RADIUS_M)
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
                "shooter_track_id": shot.shooter_track_id,
                "radius_m": DEFAULT_RADIUS_M,
                "markers": [
                    {"track_id": chave, "distance": distancia}
                    for chave, distancia in pressao.marcadores
                ],
                "opponents_without_position": pressao.adversarios_sem_posicao,
            },
        )


def _quadro_do_contato(perception: Perception, frame_index: int) -> FramePerception | None:
    """Quadro com jogadores mais proximo do contato, dentro da tolerancia.

    Empate entre um quadro antes e um depois fica com o anterior, que e o que ja aconteceu quando a
    bola saiu do pe.
    """
    com_jogadores = [q for q in perception.frames if q.players]
    if not com_jogadores:
        return None
    tolerancia = _passo(com_jogadores) * TOLERANCIA_EM_PASSOS
    melhor = min(com_jogadores, key=lambda q: (abs(q.frame_index - frame_index), q.frame_index))
    return melhor if abs(melhor.frame_index - frame_index) <= tolerancia else None


def _passo(quadros: list[FramePerception]) -> float:
    """Passo de amostragem observado sobre os quadros com jogadores.

    Medido sobre os quadros que esta feature consome, e nao sobre todos nem sobre os com bola como
    em `touch.py`: `assemble` monta `perception.frames` como a uniao dos indices de cada especie, e
    uma configuracao que amostre pose mais fino que jogadores daria um passo menor do que a cadencia
    real dos jogadores, encolhendo a tolerancia.

    O menor salto positivo, e nao a mediana, porque a amostragem e de passo constante e qualquer
    salto maior que o menor e buraco. Com menos de dois quadros a cadencia e desconhecida, e
    cadencia desconhecida vira tolerancia minima.
    """
    indices = sorted(q.frame_index for q in quadros)
    saltos = [b - a for a, b in zip(indices, indices[1:], strict=False) if b > a]
    return float(min(saltos)) if saltos else 1.0
