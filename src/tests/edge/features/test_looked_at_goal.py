"""A pose de cada teste e construida a partir do rumo DESEJADO no campo, e nao em pixel.

O teste escolhe para onde o corpo aponta em metro, projeta esse passo pela homografia real da
fixture e monta a linha dos ombros que produz aquele rumo. Assim o que o teste afirma e o que a
feature promete, em vez de um angulo em pixel que ninguem sabe a quanto corresponde no gramado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from contracts import (
    BallDetection,
    Box,
    FieldGeometry,
    FramePerception,
    Keypoint,
    Perception,
    PlayerDetection,
)
from edge.features.gol import CRITERIO_PELA_BOLA, CRITERIO_PELA_DISTANCIA
from edge.features.looked_at_goal import LookedAtGoal
from edge.testing.pitch import clip_truth, project

TRUTH = clip_truth()
HOMOGRAPHY = TRUTH.homography
FPS = 30.0
SHOT_FRAME = 30
TRACK_ID = 7
GOL_DIREITO = (52.5, 0.0)
GOL_ESQUERDO = (-52.5, 0.0)
JANELA = (27, 28, 29, SHOT_FRAME, 31)


@dataclass(frozen=True)
class _Shot:
    frame_index: int = SHOT_FRAME
    shooter_track_id: int | None = TRACK_ID


def _field(homography: tuple[float, ...] | None = HOMOGRAPHY) -> FieldGeometry:
    if homography is None:
        return FieldGeometry(
            homography=None,
            reprojection_error_px=None,
            reprojection_error_p95_px=None,
            keypoints=(),
            observed_keypoint_count=2,
            inlier_count=0,
            source="unavailable",
            age_frames=0,
            template_id="pitch-105x68-center-v1",
            reason="insufficient-keypoints",
        )
    return FieldGeometry(
        homography=homography,
        reprojection_error_px=2.0,
        reprojection_error_p95_px=3.0,
        keypoints=(),
        observed_keypoint_count=6,
        inlier_count=6,
        source="detected",
        age_frames=0,
        template_id="pitch-105x68-center-v1",
        reason=None,
    )


def _rumo_em_pixel(posicao: tuple[float, float], alvo: tuple[float, float]) -> tuple[float, float]:
    """A direcao em pixel de um passo de 1 m dado em `posicao` na direcao de `alvo`."""
    para_alvo = (alvo[0] - posicao[0], alvo[1] - posicao[1])
    norma = math.hypot(*para_alvo)
    unitario = (para_alvo[0] / norma, para_alvo[1] / norma)
    origem = project(HOMOGRAPHY, posicao[0], posicao[1])
    adiante = project(HOMOGRAPHY, posicao[0] + unitario[0], posicao[1] + unitario[1])
    return (adiante[0] - origem[0], adiante[1] - origem[1])


def _pose(
    posicao: tuple[float, float],
    alvo: tuple[float, float],
    *,
    largura_px: float = 20.0,
    confidence: float = 0.9,
    olhos: int = 2,
    orelhas: int = 2,
    com_nariz: bool = False,
) -> tuple[Keypoint, ...]:
    """Ombros, olhos e orelhas que fazem a feature ler o corpo apontado de `posicao` para `alvo`.

    O sentido sai da visibilidade: rumo que cresce em y da imagem e corpo voltado para o lado da
    camera, e ai valem os dois olhos; rumo que decresce e nuca, e ai valem as orelhas.
    """
    rumo = _rumo_em_pixel(posicao, alvo)
    norma = math.hypot(*rumo)
    f = (rumo[0] / norma, rumo[1] / norma)
    de_frente = f[1] > 0
    para_camera = f if de_frente else (-f[0], -f[1])
    eixo = (para_camera[1], -para_camera[0])

    ancora = project(HOMOGRAPHY, posicao[0], posicao[1])
    meia = largura_px / 2.0
    ombro_y = ancora[1] - 14.0
    cabeca_y = ancora[1] - 20.0
    esquerdo = (ancora[0] - eixo[0] * meia, ombro_y - eixo[1] * meia)
    direito = (ancora[0] + eixo[0] * meia, ombro_y + eixo[1] * meia)
    pontos = [
        Keypoint("shoulder_left", esquerdo[0], esquerdo[1], confidence),
        Keypoint("shoulder_right", direito[0], direito[1], confidence),
        Keypoint("ankle_left", ancora[0] - 1.5, ancora[1], confidence),
        Keypoint("ankle_right", ancora[0] + 1.5, ancora[1], confidence),
    ]
    visiveis_olhos = ("eye_left", "eye_right")[:olhos] if de_frente else ()
    visiveis_orelhas = ("ear_left", "ear_right")[:orelhas] if not de_frente else ()
    for indice, nome in enumerate(visiveis_olhos):
        pontos.append(Keypoint(nome, ancora[0] + 2.0 * indice - 1.0, cabeca_y, confidence))
    for indice, nome in enumerate(visiveis_orelhas):
        pontos.append(Keypoint(nome, ancora[0] + 2.0 * indice - 1.0, cabeca_y, confidence))
    if com_nariz:
        pontos.append(Keypoint("nose", ancora[0], cabeca_y - 1.0, confidence))
    return tuple(pontos)


def _perception(
    *,
    posicao: tuple[float, float] = (41.5, -6.0),
    alvo: tuple[float, float] = GOL_DIREITO,
    bola_para: tuple[float, float] | None = GOL_DIREITO,
    bola_detectada: bool = True,
    homography: tuple[float, ...] | None = HOMOGRAPHY,
    quadros: tuple[int, ...] = JANELA,
    **pose_kwargs: object,
) -> Perception:
    frames: list[FramePerception] = []
    for indice in quadros:
        pose = _pose(posicao, alvo, **pose_kwargs)  # type: ignore[arg-type]
        jogador = PlayerDetection(
            box=Box(
                project(HOMOGRAPHY, posicao[0], posicao[1])[0] - 6.0,
                project(HOMOGRAPHY, posicao[0], posicao[1])[1] - 22.0,
                project(HOMOGRAPHY, posicao[0], posicao[1])[0] + 6.0,
                project(HOMOGRAPHY, posicao[0], posicao[1])[1],
            ),
            confidence=0.9,
            track_id=TRACK_ID,
            role="player",
            pitch_xy=posicao if homography is not None else None,
            keypoints=pose,
        )
        frames.append(
            FramePerception(
                frame_index=indice,
                timestamp_s=indice / FPS,
                players=(jogador,),
                ball=_bola(indice, posicao, bola_para, bola_detectada, homography),
                field=_field(homography),
            )
        )
    return Perception(clip_id="clipe-olhou", frames=tuple(frames))


def _bola(
    indice: int,
    posicao: tuple[float, float],
    bola_para: tuple[float, float] | None,
    detectada: bool,
    homography: tuple[float, ...] | None,
) -> BallDetection | None:
    if bola_para is None or indice < SHOT_FRAME:
        return None
    para = (bola_para[0] - posicao[0], bola_para[1] - posicao[1])
    norma = math.hypot(*para)
    passo = (indice - SHOT_FRAME) * 4.0
    pitch = (
        posicao[0] + para[0] / norma * passo,
        posicao[1] + para[1] / norma * passo,
    )
    u, v = project(HOMOGRAPHY, pitch[0], pitch[1])
    return BallDetection(
        box=Box(u - 3.0, v - 3.0, u + 3.0, v + 3.0),
        confidence=0.8 if detectada else 0.0,
        pitch_xy=pitch if homography is not None else None,
        source="detected" if detectada else "interpolated",
        gap_frames=0 if detectada else 2,
    )


def test_corpo_apontado_para_o_gol_sai_verdadeiro() -> None:
    valor = LookedAtGoal().extract(_perception(), _Shot())
    assert valor.status == "ok"
    assert valor.value is True
    assert valor.evidence["angulo_mediano_deg"] < 15.0
    assert valor.evidence["gol"] == "direito"


def test_corpo_de_costas_para_o_gol_sai_falso() -> None:
    posicao = (41.5, -6.0)
    oposto = (2 * posicao[0] - GOL_DIREITO[0], 2 * posicao[1] - GOL_DIREITO[1])
    valor = LookedAtGoal().extract(_perception(posicao=posicao, alvo=oposto), _Shot())
    assert valor.status == "ok"
    assert valor.value is False
    assert valor.evidence["angulo_mediano_deg"] > 170.0


def test_o_alvo_vem_do_rumo_da_bola_e_nao_da_distancia() -> None:
    """O finalizador esta mais perto do gol esquerdo e chuta no direito, apontado para o direito.

    E o caso que separa esta feature de `looked_before`: pelo criterio da distancia o alvo seria a
    baliza errada, e a mesma pose sairia como nao tendo olhado para o gol.
    """
    posicao = (-20.0, 0.0)
    valor = LookedAtGoal().extract(
        _perception(posicao=posicao, alvo=GOL_DIREITO, bola_para=GOL_DIREITO), _Shot()
    )
    assert valor.status == "ok"
    assert valor.evidence["criterio_do_gol"] == CRITERIO_PELA_BOLA
    assert valor.evidence["gol"] == "direito"
    assert valor.value is True


def test_sem_bola_detectada_o_alvo_cai_para_a_distancia_e_declara() -> None:
    valor = LookedAtGoal().extract(_perception(bola_detectada=False), _Shot())
    assert valor.status == "ok"
    assert valor.evidence["criterio_do_gol"] == CRITERIO_PELA_DISTANCIA
    assert valor.evidence["gol"] == "direito"


def test_bola_interpolada_nao_escolhe_o_alvo() -> None:
    """Interpolacao e reta por cima da virada da trajetoria, e a virada e o contato."""
    pela_bola = LookedAtGoal().extract(_perception(), _Shot())
    interpolada = LookedAtGoal().extract(_perception(bola_detectada=False), _Shot())
    assert pela_bola.evidence["criterio_do_gol"] == CRITERIO_PELA_BOLA
    assert interpolada.evidence["criterio_do_gol"] == CRITERIO_PELA_DISTANCIA


def test_um_olho_so_recusa_em_vez_de_arbitrar_o_sentido() -> None:
    """Perfil nao separa de frente de de costas, e sentido errado inverte a resposta."""
    valor = LookedAtGoal().extract(_perception(olhos=1), _Shot())
    assert valor.status == "unavailable"
    assert valor.value is None
    assert "sentido do corpo indeterminado" in (valor.reason or "")


def test_ombro_a_ombro_pequeno_demais_recusa() -> None:
    valor = LookedAtGoal().extract(_perception(largura_px=4.0), _Shot())
    assert valor.status == "unavailable"
    assert "px" in (valor.reason or "")


def test_ombros_abaixo_do_piso_de_confianca_recusam() -> None:
    valor = LookedAtGoal().extract(_perception(confidence=0.2), _Shot())
    assert valor.status == "unavailable"
    assert "ombros" in (valor.reason or "")


def test_poucos_quadros_na_janela_recusam() -> None:
    valor = LookedAtGoal().extract(_perception(quadros=(SHOT_FRAME,)), _Shot())
    assert valor.status == "unavailable"
    assert "quadros validos" in (valor.reason or "")


def test_sem_homografia_recusa() -> None:
    valor = LookedAtGoal().extract(_perception(homography=None), _Shot())
    assert valor.status == "unavailable"


def test_a_evidencia_publica_o_que_decidiu() -> None:
    valor = LookedAtGoal().extract(_perception(com_nariz=True), _Shot())
    assert valor.status == "ok"
    esperadas = {
        "score",
        "threshold",
        "decision",
        "track_id",
        "frame_do_chute",
        "gol",
        "criterio_do_gol",
        "angulo_mediano_deg",
        "quadros_validos",
        "quadros_na_janela",
        "janela_s",
        "frames",
        "lado_da_camera",
        "largura_dos_ombros_px",
        "desvio_lateral_mediano",
        "recusas_por_quadro",
    }
    assert esperadas <= set(valor.evidence)
    assert tuple(valor.evidence["frames"]) == JANELA
    assert valor.evidence["quadros_validos"] == len(JANELA)


def test_o_angulo_nao_depende_de_onde_o_jogador_esta_no_quadro() -> None:
    """O mesmo rumo em metro mede o mesmo angulo no fundo e na frente da imagem.

    E o que a jacobiana compra. Em pixel o mesmo giro de corpo sairia com angulos diferentes nas
    duas profundidades, porque a perspectiva comprime o fundo do quadro.
    """
    perto = LookedAtGoal().extract(_perception(posicao=(50.0, -2.0), alvo=GOL_DIREITO), _Shot())
    longe = LookedAtGoal().extract(_perception(posicao=(33.0, -18.0), alvo=GOL_DIREITO), _Shot())
    assert perto.status == longe.status == "ok"
    assert abs(perto.evidence["angulo_mediano_deg"] - longe.evidence["angulo_mediano_deg"]) < 5.0


def test_a_versao_e_o_nome_saem_como_a_spec_declara() -> None:
    valor = LookedAtGoal().extract(_perception(), _Shot())
    assert (valor.name, valor.version) == ("looked_at_goal", "0.1.0")
