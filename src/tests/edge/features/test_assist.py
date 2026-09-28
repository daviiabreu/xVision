"""A feature `assist`: compara o toque anterior com o finalizador, e separa os modos de ausencia.

As percepcoes aqui sao montadas em pixel, e nao em metro, porque e assim que `touch` mede: a
distancia vai da borda da caixa do jogador ao centro da caixa da bola, normalizada pela altura
mediana das caixas da janela. Com caixas de 120px de altura o raio de toque vale 144px e a margem
de empate vale 6px, e as posicoes abaixo foram escolhidas contra esses dois numeros.
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
from edge.features.assist import Assist

SHOT_FRAME = 8
ALTURA = 120.0

CAIXAS = {
    7: Box(x1=100.0, y1=100.0, x2=140.0, y2=220.0),
    9: Box(x1=600.0, y1=100.0, x2=640.0, y2=220.0),
    3: Box(x1=141.0, y1=100.0, x2=181.0, y2=220.0),
}

# Centros de bola por situacao. `7` e `9` caem dentro da caixa do dono (distancia zero); `livre`
# fica a milhares de pixels de todos; `disputa` fica a meio pixel das caixas de 7 e de 3, dentro
# da margem de empate.
CENTROS = {
    "7": (120.0, 160.0),
    "9": (620.0, 160.0),
    "livre": (2000.0, 2000.0),
    "disputa": (140.5, 160.0),
}


@dataclass(frozen=True)
class _Shot:
    frame_index: int = SHOT_FRAME
    shooter_track_id: int | None = 7


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


def _ball(situacao: str, *, source: str = "detected") -> BallDetection:
    """Bola no centro pedido. Interpolada carrega o que o contrato exige dela: confianca zero e
    lacuna positiva, porque e uma reta que o rastreador desenhou e nao uma medida."""
    x, y = CENTROS[situacao]
    interpolada = source == "interpolated"
    return BallDetection(
        box=Box(x1=x - 5.0, y1=y - 5.0, x2=x + 5.0, y2=y + 5.0),
        confidence=0.0 if interpolada else 0.9,
        source=source,
        gap_frames=1 if interpolada else 0,
    )


def _player(track_id: int, team: str | None) -> PlayerDetection:
    return PlayerDetection(
        box=CAIXAS[track_id],
        confidence=0.9,
        track_id=track_id,
        team=team,
        role="player",
    )


def _perception(
    situacoes: list[str],
    *,
    elenco: dict[int, str | None],
    source: str = "detected",
) -> Perception:
    jogadores = tuple(_player(track_id, team) for track_id, team in elenco.items())
    frames = tuple(
        FramePerception(
            frame_index=indice,
            timestamp_s=indice / 30.0,
            players=jogadores,
            ball=_ball(situacao, source=source),
            field=_field(),
        )
        for indice, situacao in enumerate(situacoes)
    )
    return Perception(clip_id="clipe-1", frames=frames)


def test_passe_de_companheiro_conta_como_assistencia() -> None:
    """Posse do 9, bola solta, posse do 7: o toque anterior e de outro jogador."""
    perception = _perception(
        ["9", "9", "9", "9", "livre", "livre", "7", "7", "7", "7"],
        elenco={7: "team_a", 9: "team_a"},
    )

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is True
    assert valor.evidence["shooter_track_id"] == 7
    assert valor.evidence["previous_touch_track_id"] == 9
    assert valor.evidence["previous_touch_frame_index"] == 3
    assert valor.evidence["same_team"] is True
    assert valor.evidence["previous_touch_source"] == "detected"


def test_bola_tomada_do_adversario_nao_e_assistencia() -> None:
    """Troca de pe entre equipes diferentes e roubo: ninguem assistiu o finalizador."""
    perception = _perception(
        ["9", "9", "9", "9", "livre", "livre", "7", "7", "7", "7"],
        elenco={7: "team_a", 9: "team_b"},
    )

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is False
    assert valor.evidence["same_team"] is False
    assert valor.evidence["previous_touch_track_id"] == 9


def test_equipe_desconhecida_no_toque_anterior_nao_decide_assistencia() -> None:
    """Sem saber de quem era a bola nao da para afirmar que nao houve passe."""
    perception = _perception(
        ["9", "9", "9", "9", "livre", "livre", "7", "7", "7", "7"],
        elenco={7: "team_a", 9: None},
    )

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.value is None
    assert valor.reason == "team-unavailable"


def test_conducao_do_proprio_finalizador_nao_e_assistencia() -> None:
    """Dois episodios do mesmo jogador: a bola saiu e voltou ao pe dele, ninguem passou."""
    perception = _perception(
        ["7", "7", "7", "7", "livre", "livre", "7", "7", "7", "7"],
        elenco={7: "team_a", 9: "team_a"},
    )

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "ok"
    assert valor.value is False
    assert valor.evidence["previous_touch_track_id"] == 7
    assert valor.evidence["previous_touch_frame_index"] == 3


def test_clipe_que_comeca_com_a_bola_no_pe_do_finalizador_nao_tem_toque_anterior() -> None:
    """Um unico episodio cobre a janela inteira: nao ha nada antes dele para comparar."""
    perception = _perception(["7"] * 10, elenco={7: "team_a", 9: "team_a"})

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.value is None
    assert valor.reason == "previous-touch-unavailable"


def test_disputa_antes_do_contato_e_toque_anterior_ambiguo() -> None:
    """Disputa nao vira episodio, e a ausencia por disputa se distingue da ausencia por nada."""
    perception = _perception(
        ["disputa", "disputa", "7", "7", "7", "7", "7", "7", "7", "7"],
        elenco={7: "team_a", 3: "team_b"},
    )

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.reason == "previous-touch-ambiguous"


def test_disputa_no_contato_deixa_o_finalizador_ambiguo() -> None:
    perception = _perception(["disputa"] * 10, elenco={7: "team_a", 3: "team_b"})

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.reason == "shooter-at-contact-ambiguous"


def test_bola_so_interpolada_nao_sustenta_o_contato() -> None:
    """A bola que o rastreador desenhou nao e medida, entao nenhum episodio se forma."""
    perception = _perception(["7"] * 10, elenco={7: "team_a", 9: "team_a"}, source="interpolated")

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.reason == "shooter-at-contact-unavailable"


def test_bola_longe_de_todos_na_janela_inteira_nao_sustenta_o_contato() -> None:
    perception = _perception(["livre"] * 10, elenco={7: "team_a", 9: "team_a"})

    valor = Assist().extract(perception, _Shot())

    assert valor.status == "unavailable"
    assert valor.reason == "shooter-at-contact-unavailable"
