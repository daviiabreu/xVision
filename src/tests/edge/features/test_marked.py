"""A feature `marked`: diz se havia adversario no raio do finalizador no contato.

Marcacao e afirmacao existencial, entao os testes cobrem as tres respostas separadamente: `True`
com marcador achado, `False` so quando todos os adversarios tinham posicao, e `unavailable` quando
o buraco na percepcao poderia esconder justamente o marcador.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts import (
    BallDetection,
    Box,
    FieldGeometry,
    FramePerception,
    Perception,
    PlayerDetection,
)
from edge.features.marked import Marked

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


def _perception(*players: PlayerDetection, frame_index: int = SHOT_FRAME) -> Perception:
    return Perception(clip_id="clipe-1", frames=(_frame(frame_index, *players),))


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
    assert valor.reason == "observado-sem-pitch-xy"


def test_finalizador_sem_equipe_e_indisponivel() -> None:
    perception = _perception(
        _player(7, None, (40.0, -6.0)),
        _player(4, "team_b", (42.0, -6.0)),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.reason == "finalizador sem equipe nesse frame"


def test_shot_sem_finalizador_e_indisponivel() -> None:
    perception = _perception(_player(7, "team_a", (40.0, -6.0)))

    valor = Marked().extract(perception, _Shot(shooter_track_id=None))

    assert valor.status == "unavailable"
    assert valor.reason == "shot sem shooter_track_id"


def test_quadro_vizinho_dentro_da_tolerancia_responde_pelo_contato() -> None:
    """Contato refinado entre amostras nao cai mais em `unavailable` por um quadro de diferenca."""
    perception = _perception(
        _player(7, "team_a", (40.0, -6.0)),
        _player(4, "team_b", (42.0, -6.0)),
        frame_index=SHOT_FRAME + 1,
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is True
    assert valor.evidence["frame_index"] == SHOT_FRAME + 1
    assert valor.evidence["shot_frame_index"] == SHOT_FRAME


def test_empate_de_distancia_fica_com_o_quadro_anterior() -> None:
    perception = Perception(
        clip_id="clipe-1",
        frames=(
            _frame(
                SHOT_FRAME - 1,
                _player(7, "team_a", (40.0, -6.0)),
                _player(4, "team_b", (42.0, -6.0)),
            ),
            _frame(
                SHOT_FRAME + 1,
                _player(7, "team_a", (40.0, -6.0)),
                _player(4, "team_b", (60.0, -6.0)),
            ),
        ),
    )

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.evidence["frame_index"] == SHOT_FRAME - 1
    assert valor.value is True


def test_quadro_longe_demais_e_indisponivel() -> None:
    perception = _perception(_player(7, "team_a", (40.0, -6.0)), frame_index=SHOT_FRAME + 50)

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.value is None
    assert valor.reason == f"sem percepcao pro frame {SHOT_FRAME}"


def test_percepcao_sem_jogador_nenhum_e_indisponivel() -> None:
    perception = Perception(clip_id="clipe-1", frames=(_frame(SHOT_FRAME),))

    valor = Marked().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.reason == f"sem percepcao pro frame {SHOT_FRAME}"
