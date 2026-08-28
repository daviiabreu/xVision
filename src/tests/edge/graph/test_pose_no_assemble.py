"""A pose chega ao `PlayerDetection`, e nao morre no `assemble`.

Ate a mudanca que estes testes cobrem, a saida da especie `pose` era produzida, contava como
`produced` para o portao de `requires` e era descartada no `assemble`. Uma feature que declarasse
`requires=("pose",)` passava no portao, entrava no `extract` e nao tinha de onde ler keypoint, ou
seja, so conseguia sair `unavailable` para sempre.

A associacao e por `track_id`, igual a de `identity`, e nao por `detection_index`, igual a de
`tracking`. O motivo e que o estimador de pose roda sobre o recorte de um jogador ja rastreado, e
nao sobre a lista de deteccoes do quadro.
"""

from __future__ import annotations

from contracts import Box, Clip, Keypoint
from edge.graph.runner import assemble
from edge.modules.base import ModuleOutput

TRACK = 7
OUTRO_TRACK = 9
CLIP = Clip(clip_id="c", path="c.mp4", fps=30.0, frame_count=10, width=1920, height=1080)


def _saidas(
    *, pose: tuple[tuple[str, float, float] | tuple[str, float, float, float], ...] | None = None
) -> dict[str, ModuleOutput]:
    """Um quadro, um jogador rastreado, e a pose que o teste quiser pendurar nele."""
    quadro = (0,)
    saidas = {
        "players": ModuleOutput(
            kind="players",
            frames=quadro,
            records=(({"box": (10.0, 20.0, 30.0, 60.0), "confidence": 0.9},),),
        ),
        "tracking": ModuleOutput(
            kind="tracking",
            frames=quadro,
            records=(({"detection_index": 0, "track_id": TRACK},),),
        ),
    }
    if pose is not None:
        saidas["pose"] = ModuleOutput(
            kind="pose",
            frames=quadro,
            records=(({"track_id": TRACK, "keypoints": pose},),),
        )
    return saidas


def test_keypoint_do_track_certo_chega_no_jogador() -> None:
    perception = assemble(CLIP, _saidas(pose=(("head", 20.0, 22.0, 0.8),)))

    jogadores = perception.frames[0].players
    assert len(jogadores) == 1
    assert jogadores[0].keypoints == (Keypoint("head", 20.0, 22.0, 0.8),)


def test_sem_saida_de_pose_o_jogador_fica_sem_keypoint() -> None:
    perception = assemble(CLIP, _saidas())

    jogadores = perception.frames[0].players
    assert len(jogadores) == 1
    assert jogadores[0].keypoints == ()


def test_tripla_sem_confianca_continua_valida() -> None:
    """O stub e a geometria de campo emitem tripla, e recusa-la quebraria os dois por nada."""
    perception = assemble(CLIP, _saidas(pose=(("foot", 20.0, 60.0),)))

    assert perception.frames[0].players[0].keypoints == (Keypoint("foot", 20.0, 60.0, None),)


def test_pose_de_outro_track_nao_vaza_para_o_jogador() -> None:
    """Sem esta guarda, o keypoint do marcador viraria o keypoint do finalizador em silencio."""
    saidas = _saidas()
    saidas["pose"] = ModuleOutput(
        kind="pose",
        frames=(0,),
        records=(({"track_id": OUTRO_TRACK, "keypoints": (("head", 99.0, 99.0, 0.9),)},),),
    )

    perception = assemble(CLIP, saidas)

    assert perception.frames[0].players[0].keypoints == ()


def test_jogador_sem_track_id_nao_recebe_keypoint() -> None:
    """Sem `tracking`, nao ha chave para associar, e associar pela ordem seria chute."""
    saidas = _saidas(pose=(("head", 20.0, 22.0, 0.8),))
    del saidas["tracking"]

    perception = assemble(CLIP, saidas)

    jogador = perception.frames[0].players[0]
    assert jogador.track_id is None
    assert jogador.keypoints == ()


def test_a_caixa_do_jogador_continua_inteira() -> None:
    """Guarda de regressao: a mudanca nao pode ter mexido no que o `_player` ja montava."""
    perception = assemble(CLIP, _saidas(pose=(("head", 20.0, 22.0, 0.8),)))

    jogador = perception.frames[0].players[0]
    assert jogador.box == Box(10.0, 20.0, 30.0, 60.0)
    assert jogador.track_id == TRACK
