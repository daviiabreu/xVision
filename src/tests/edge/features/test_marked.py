"""A feature `marked`: a cena acha o quadro e o finalizador, e a regra de pressao decide.

Marcacao e afirmacao existencial, entao os casos cobrem as tres respostas: `True` com marcador
achado, `False` so quando todos os adversarios tinham posicao, e `unavailable` quando o buraco na
percepcao poderia esconder justamente o marcador.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from contracts import (
    BallDetection,
    Box,
    FieldGeometry,
    FramePerception,
    Perception,
    PlayerDetection,
)
from edge.features.marked import Marked

FPS = 59.94
CAMPO = FieldGeometry(
    homography=(10.0, 0.0, 960.0, 0.0, -10.0, 540.0, 0.0, 0.0, 1.0),
    reprojection_error_px=0.2,
    reprojection_error_p95_px=0.4,
    keypoints=(),
    observed_keypoint_count=8,
    inlier_count=8,
    source="detected",
    age_frames=0,
    template_id="pitch-105x68-center-v1",
    reason=None,
)


@dataclass(frozen=True)
class _Chute:
    frame_index: int
    shooter_track_id: int | None = None


def _jogador(x: float, track: int, team: str, metro: tuple[float, float] | None) -> PlayerDetection:
    return PlayerDetection(
        Box(x - 20, 500, x + 20, 600), 0.8, track_id=track, team=team, role="player", pitch_xy=metro
    )


def _percepcao(*, defensor_a: float, campo_em: set[int] | None = None) -> Perception:
    """Finalizador 7 com a bola no pe e o defensor 9 a `defensor_a` metros dele."""
    quadros: list[FramePerception] = []
    for i in range(0, 44, 4):
        campo = campo_em is None or i in campo_em
        jogadores = (
            _jogador(500, 7, "team_a", (-46.0, -6.0) if campo else None),
            _jogador(1100, 9, "team_b", (-46.0 + defensor_a, -6.0) if campo else None),
        )
        bola = BallDetection(Box(495, 585, 505, 595), 0.9, source="detected")
        quadros.append(
            FramePerception(
                frame_index=i,
                timestamp_s=i / FPS,
                players=jogadores,
                ball=bola,
                field=CAMPO if campo else None,
            )
        )
    return Perception(clip_id="c", frames=tuple(quadros))


def test_defensor_colado_no_finalizador_e_marcacao() -> None:
    valor = Marked().extract(_percepcao(defensor_a=1.5), _Chute(28))

    assert valor.status == "ok"
    assert valor.value is True
    assert valor.evidence["origem_do_finalizador"] == "touch"


def test_defensor_longe_nao_marca() -> None:
    valor = Marked().extract(_percepcao(defensor_a=8.0), _Chute(28))

    assert (valor.status, valor.value) == ("ok", False)


def test_o_quadro_de_referencia_pode_ser_vizinho_do_contato() -> None:
    valor = Marked().extract(_percepcao(defensor_a=1.5, campo_em={24}), _Chute(28))

    assert valor.status == "ok"
    assert valor.evidence["frame_index"] == 24


def test_sem_homografia_perto_do_contato_e_indisponivel() -> None:
    valor = Marked().extract(_percepcao(defensor_a=1.5, campo_em=set()), _Chute(28))

    assert valor.status == "unavailable"
    assert "homografia" in (valor.reason or "")


def test_sem_jogador_com_equipe_e_indisponivel() -> None:
    cena = _percepcao(defensor_a=1.5)
    sem_time = Perception(
        clip_id="c",
        frames=tuple(
            replace(q, players=tuple(replace(j, team=None) for j in q.players)) for q in cena.frames
        ),
    )

    assert Marked().extract(sem_time, _Chute(28)).status == "unavailable"


def test_jogador_com_equipe_fora_do_campo_e_indisponivel() -> None:
    cena = _percepcao(defensor_a=1.5)
    quadros = list(cena.frames)
    q = quadros[7]
    quadros[7] = replace(q, players=(q.players[0], replace(q.players[1], pitch_xy=None)))

    valor = Marked().extract(Perception(clip_id="c", frames=tuple(quadros)), _Chute(28))

    assert valor.status == "unavailable"
    assert valor.reason == "adversario-sem-pitch-xy"


# Casos da regra de pressao, com o finalizador declarado pelo `shot`. A cena precisa de dois quadros
# para situar o contato no tempo, entao cada percepcao repete o quadro do contato um quadro antes.

SHOT_FRAME = 18


@dataclass(frozen=True)
class _Shot:
    frame_index: int = SHOT_FRAME
    shooter_track_id: int | None = 7


def _box() -> Box:
    return Box(x1=100.0, y1=100.0, x2=140.0, y2=220.0)


def _field() -> FieldGeometry:
    return FieldGeometry(
        homography=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        reprojection_error_px=0.5,
        reprojection_error_p95_px=0.6,
        keypoints=(),
        observed_keypoint_count=6,
        inlier_count=6,
        source="detected",
        age_frames=0,
        template_id="pitch-105x68-center-v1",
        reason=None,
    )


def _player(
    track_id: int,
    team: str | None,
    pitch_xy: tuple[float, float] | None,
    *,
    role: str = "player",
) -> PlayerDetection:
    return PlayerDetection(
        box=_box(),
        confidence=0.9,
        track_id=track_id,
        team=team,
        role=role,
        pitch_xy=pitch_xy,
    )


def _frame(frame_index: int, *players: PlayerDetection) -> FramePerception:
    return FramePerception(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        players=players,
        ball=BallDetection(box=_box(), confidence=0.9, pitch_xy=(0.0, 0.0), source="detected"),
        field=_field(),
    )


def _perception(*players: PlayerDetection) -> Perception:
    return Perception(
        clip_id="clipe-1",
        frames=(_frame(SHOT_FRAME - 1, *players), _frame(SHOT_FRAME, *players)),
    )


def test_finalizador_com_adversario_perto_sai_marcado() -> None:
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(4, "team_b", (42.0, -6.0)),  # 2m do finalizador, dentro do raio
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is True
    assert valor.evidence["markers"] == ({"track_id": 4, "distance": 2.0},)
    assert valor.evidence["opponents_without_position"] == 0


def test_adversario_longe_deixa_o_finalizador_livre() -> None:
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(4, "team_b", (50.0, -6.0)),  # 10m: fora do raio de 5m
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is False
    assert valor.evidence["markers"] == ()


def test_companheiro_perto_nao_marca() -> None:
    """Marcacao e do adversario: colega colado nao e pressao."""
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(2, "team_a", (41.0, -6.0)),
        _player(4, "team_b", (50.0, -6.0)),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is False


def test_zagueiro_que_cobre_dois_marca_os_dois() -> None:
    """O pareamento reciproco elegia um atacante so e publicava `False` para o outro."""
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),  # finalizador, a 1m do zagueiro
        _player(2, "team_a", (41.5, -6.0)),  # colega, a 0,5m do mesmo zagueiro
        _player(4, "team_b", (41.0, -6.0)),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is True
    assert valor.evidence["markers"] == ({"track_id": 4, "distance": 1.0},)


def test_goleiro_adversario_conta_como_marcador() -> None:
    """Goleiro que sai do gol e a maior pressao possivel, e ficava de fora da conta."""
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(1, "team_b", (42.0, -6.0), role="goalkeeper"),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is True
    assert valor.evidence["markers"] == ({"track_id": 1, "distance": 2.0},)


def test_arbitro_colado_nao_marca() -> None:
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(9, "team_b", (40.5, -6.0), role="referee"),
        _player(4, "team_b", (50.0, -6.0)),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is False


def test_marcador_achado_vale_mesmo_com_outro_adversario_sem_posicao() -> None:
    """Mais dado nao desfaz um marcador ja encontrado, entao a resposta continua `True`."""
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(4, "team_b", (42.0, -6.0)),
        _player(5, "team_b", None),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is True
    assert valor.evidence["opponents_without_position"] == 1


def test_adversario_sem_posicao_sem_marcador_achado_e_indeterminado() -> None:
    """O adversario sem `pitch_xy` podia ser o marcador, entao `False` seria invencao."""
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(4, "team_b", None),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.value is None
    assert valor.reason == "adversario-sem-pitch-xy"


def test_companheiro_sem_posicao_nao_atrapalha() -> None:
    """So adversario sem posicao gera indeterminacao: colega nao marcaria de todo jeito."""
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(2, "team_a", None),
        _player(4, "team_b", (50.0, -6.0)),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is False


def test_finalizador_sem_posicao_e_indisponivel() -> None:
    perception = _perception(
        _player(7, "team_a", None),
        _player(4, "team_b", (42.0, -6.0)),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert "metro" in (valor.reason or "")


def test_finalizador_sem_equipe_e_indisponivel() -> None:
    perception = _perception(
        _player(7, None, (40.0, -6.0)),
        _player(4, "team_b", (42.0, -6.0)),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.reason == "finalizador sem equipe nesse frame"
