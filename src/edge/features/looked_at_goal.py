"""Decide se o finalizador estava apontado para o gol no instante do contato.

Nao se confunda com `looked_before`, que ja existe e mede intencao parecida. As duas diferem em
tres pontos, e e por isso que esta e feature nova em vez de versao daquela:

| | `looked_before` | `looked_at_goal` |
|---|---|---|
| janela | de 0,5 s a 0,1 s ANTES do contato | em volta do contato |
| alvo | gol mais proximo | gol que a bola procurou depois do contato |
| angulo | entre dois vetores em PIXEL | entre dois vetores em METRO, no plano do campo |

A terceira diferenca e a que mais muda o numero. Em pixel o angulo esta deformado pela
perspectiva: o mesmo giro de corpo mede diferente no fundo e na frente do quadro. Aqui a direcao
do corpo e levada ao plano do campo pela jacobiana local da homografia, e o angulo sai entre dois
rumos em metro, onde ele tem significado geometrico.

Como a direcao do corpo e construida, e por que assim:

1. A linha dos ombros da o eixo lateral do tronco em pixel. A direcao de frente e perpendicular a
   ela, e perpendicular tem dois sentidos.
2. O sentido entre os dois (de frente ou de costas para a camera) vem da visibilidade do rosto:
   dois olhos acima do piso de confianca e alguem olhando para o lado da camera, nenhum olho com
   orelha visivel e alguem de costas. Sem nenhum dos dois casos a medida RECUSA o quadro, em vez
   de escolher um sentido: errar o sentido inverte a resposta, e inverter calado e pior que recusar.
3. O desvio lateral do nariz em relacao ao meio dos ombros, medido em larguras de ombro, gira esse
   rumo. E a parte que capta cabeca virada com tronco parado.

O que esta medida NAO e: ela nao mede olhar. Com o finalizador a 93,4 px de altura mediana nos
clipes do parceiro, pupila e direcao de iris nao existem no pixel disponivel. O que sai daqui e
rumo de tronco corrigido por desvio de cabeca, que e uma aproximacao de para onde a pessoa estava
voltada, e nao de para onde ela estava olhando. `GANHO_LATERAL` tambem nao e calibrado: nao ha
gabarito que ligue desvio de nariz a angulo medido, e o valor escolhido esta registrado abaixo
como pendencia.

A feature NAO declara `ball` em `requires` de proposito. A bola serve para escolher a baliza, e
quando ela falta sobra o criterio da distancia, declarado em `criterio_do_gol`. Exigir `ball`
faria a feature recusar clipe que ela ainda consegue medir, so com alvo escolhido por criterio
mais fraco.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from contracts import FeatureValue, FramePerception, Keypoint, Perception, PlayerDetection
from edge.features.base import FeatureSpec, ShotOutput, unavailable
from edge.features.gol import (
    CRITERIO_PELA_BOLA,
    CRITERIO_PELA_DISTANCIA,
    gol_mais_proximo,
    gol_pelo_rumo_da_bola,
    nome_do_gol,
)
from edge.modules.field.geometry import FieldProjectionError, image_to_pitch, pitch_to_image

MIN_KEYPOINT_CONFIDENCE = 0.5
"""O mesmo piso de `looked_before`, e nao um proprio: duas features que medem o mesmo corpo com
pisos diferentes publicam cobertura que nao se compara."""

JANELA_ANTES_S = 0.10
JANELA_DEPOIS_S = 0.04
"""A janela do contato, e nao a anterior. Depois do pe a cabeca cai para acompanhar a bola, entao
a parte de depois e curta so para o quadro do contato entrar quando o instante vier arredondado."""

MIN_QUADROS_VALIDOS = 2
"""A 30 fps a janela inteira tem cerca de 4 quadros, e exigir 3 como `looked_before` recusaria
quase todo clipe. Dois ja distingue medida de quadro unico, e a cobertura vai na evidencia."""

MIN_LARGURA_DOS_OMBROS_PX = 8.0
"""Abaixo disso a linha dos ombros nao tem direcao: o finalizador tem cerca de 32 px de largura de
caixa nos clipes do parceiro, e ombro a ombro e parte disso."""

MIN_COMPONENTE_VERTICAL = 0.2
"""Piso do |y| da perpendicular aos ombros, em unidade. Perto de zero a perpendicular corre na
horizontal da imagem e nao separa de frente de de costas, que e justamente o sentido que o passo 2
precisa resolver."""

GANHO_LATERAL = 2.0
"""Quantas vezes o desvio do nariz, em larguras de ombro, entra no rumo. Com 2,0 meia largura de
ombro de desvio vira 45 graus de giro. NAO E CALIBRADO: nenhuma anotacao do projeto liga desvio de
nariz a angulo de cabeca medido, e trocar este numero move a feature inteira."""

LOOKED_AT_GOAL_THRESHOLD = 0.5
"""Limiar semantico, como em `looked_before`: 0,5 e o escore de quem esta a 90 graus do gol. Nao e
calibracao estatistica, e nao vai ser enquanto nao houver gabarito proprio."""

EPS_JACOBIANO_M = 0.25
"""Passo da diferenca central que estima a jacobiana. Grande demais mistura a curvatura da
perspectiva, pequeno demais perde digito na subtracao de pixels proximos."""

_OLHOS = ("eye_left", "eye_right")
_ORELHAS = ("ear_left", "ear_right")


class LookedAtGoal:
    spec = FeatureSpec(
        name="looked_at_goal",
        version="0.1.0",
        requires=("players", "tracking", "field", "shot", "pose"),
        unit="bool",
    )

    def extract(self, perception: Perception, shot: ShotOutput) -> FeatureValue:
        contato = _frame_mais_proximo(perception, shot.frame_index)
        if contato is None:
            return unavailable(self.spec, f"sem percepcao pro frame {shot.frame_index}")

        track_id = _finalizador(perception, shot)
        if track_id is None:
            return unavailable(self.spec, "nao foi possivel resolver um unico finalizador")

        gol, criterio = _alvo(perception, shot, contato, track_id)
        if gol is None:
            return unavailable(
                self.spec,
                "nem a bola nem a posicao do finalizador em metros dizem qual baliza era o alvo",
            )

        inicio = contato.timestamp_s - JANELA_ANTES_S
        fim = contato.timestamp_s + JANELA_DEPOIS_S
        candidatos = tuple(
            quadro for quadro in perception.frames if inicio <= quadro.timestamp_s <= fim
        )
        amostras: list[_Amostra] = []
        recusas: dict[str, int] = {}
        for quadro in candidatos:
            jogador = _jogador(quadro, track_id)
            if jogador is None or quadro.field is None or quadro.field.homography is None:
                _conta(recusas, "sem jogador com pose ou sem homografia")
                continue
            resultado = _amostra(quadro.frame_index, quadro.field.homography, jogador, gol)
            if isinstance(resultado, str):
                _conta(recusas, resultado)
                continue
            amostras.append(resultado)

        if len(amostras) < MIN_QUADROS_VALIDOS:
            motivo = ", ".join(f"{chave}: {valor}" for chave, valor in sorted(recusas.items()))
            return unavailable(
                self.spec,
                f"somente {len(amostras)} quadros validos na janela do contato, "
                f"e a medida exige {MIN_QUADROS_VALIDOS} ({motivo or 'sem quadro na janela'})",
            )

        angulo_mediano = median(amostra.angulo_deg for amostra in amostras)
        escore = round((math.cos(math.radians(angulo_mediano)) + 1.0) / 2.0, 6)
        cobertura = len(amostras) / max(1, len(candidatos))
        confianca = median(amostra.confianca for amostra in amostras) * cobertura
        decisao = escore >= LOOKED_AT_GOAL_THRESHOLD
        return FeatureValue(
            name=self.spec.name,
            version=self.spec.version,
            status="ok",
            value=decisao,
            confidence=round(min(1.0, confianca), 6),
            reason=None,
            evidence={
                "score": escore,
                "threshold": LOOKED_AT_GOAL_THRESHOLD,
                "decision": decisao,
                "track_id": track_id,
                "frame_do_chute": shot.frame_index,
                "gol": nome_do_gol(gol),
                "criterio_do_gol": criterio,
                "angulo_mediano_deg": round(angulo_mediano, 3),
                "quadros_validos": len(amostras),
                "quadros_na_janela": len(candidatos),
                "janela_s": [-JANELA_ANTES_S, JANELA_DEPOIS_S],
                "frames": [amostra.frame_index for amostra in amostras],
                "lado_da_camera": sorted({amostra.lado for amostra in amostras}),
                "largura_dos_ombros_px": round(
                    median(amostra.largura_dos_ombros_px for amostra in amostras), 2
                ),
                "desvio_lateral_mediano": round(median(amostra.lateral for amostra in amostras), 4),
                "recusas_por_quadro": dict(sorted(recusas.items())),
            },
        )


@dataclass(frozen=True)
class _Amostra:
    """Um quadro que entrou na mediana, com o que a evidencia precisa publicar sobre ele."""

    frame_index: int
    angulo_deg: float
    confianca: float
    lado: str
    lateral: float
    largura_dos_ombros_px: float


def _conta(recusas: dict[str, int], motivo: str) -> None:
    recusas[motivo] = recusas.get(motivo, 0) + 1


def _alvo(
    perception: Perception,
    shot: ShotOutput,
    contato: FramePerception,
    track_id: int,
) -> tuple[tuple[float, float] | None, str]:
    """A baliza alvo, pela bola quando a trajetoria sustenta e pela distancia quando nao."""
    pela_bola = gol_pelo_rumo_da_bola(perception, shot.frame_index, contato.timestamp_s)
    if pela_bola is not None:
        return pela_bola, CRITERIO_PELA_BOLA
    posicao = _posicao_em_metros(perception, shot.frame_index, track_id)
    if posicao is None:
        return None, CRITERIO_PELA_DISTANCIA
    return gol_mais_proximo(posicao), CRITERIO_PELA_DISTANCIA


def _posicao_em_metros(
    perception: Perception, frame_index: int, track_id: int
) -> tuple[float, float] | None:
    candidatos = tuple(
        (quadro, jogador)
        for quadro in perception.frames
        for jogador in quadro.players
        if jogador.track_id == track_id and jogador.pitch_xy is not None
    )
    mais_proximo = min(
        candidatos, key=lambda par: abs(par[0].frame_index - frame_index), default=None
    )
    return None if mais_proximo is None else mais_proximo[1].pitch_xy


def _amostra(
    frame_index: int,
    homography: tuple[float, ...],
    jogador: PlayerDetection,
    gol: tuple[float, float],
) -> _Amostra | str:
    """A amostra do quadro, ou o motivo da recusa como texto, para a evidencia contar por motivo."""
    pontos = {ponto.name: ponto for ponto in jogador.keypoints}
    ombros = _ombros(pontos)
    if ombros is None:
        return "ombros ausentes ou abaixo do piso de confianca"
    esquerdo, direito = ombros
    eixo = (direito.x - esquerdo.x, direito.y - esquerdo.y)
    largura = math.hypot(*eixo)
    if largura < MIN_LARGURA_DOS_OMBROS_PX:
        return f"ombro a ombro abaixo de {MIN_LARGURA_DOS_OMBROS_PX:.0f} px"
    u = (eixo[0] / largura, eixo[1] / largura)
    perpendicular = (-u[1], u[0])
    if abs(perpendicular[1]) < MIN_COMPONENTE_VERTICAL:
        return "perpendicular aos ombros na horizontal da imagem, sem separar frente de costas"
    para_camera = perpendicular if perpendicular[1] > 0 else (-perpendicular[0], -perpendicular[1])

    lado = _lado_da_camera(pontos)
    if lado is None:
        return "nem dois olhos nem orelha acima do piso, sentido do corpo indeterminado"
    sentido = 1.0 if lado == "de frente" else -1.0

    lateral = _desvio_lateral(pontos, esquerdo, direito, u, largura)
    rumo_imagem = (
        sentido * para_camera[0] + GANHO_LATERAL * lateral * u[0],
        sentido * para_camera[1] + GANHO_LATERAL * lateral * u[1],
    )
    if math.hypot(*rumo_imagem) < 1e-9:
        return "rumo do corpo degenerado"

    ancora = _ancora_no_chao(pontos, jogador)
    try:
        posicao = image_to_pitch(homography, ancora)
        jacobiana = _jacobiana(homography, posicao)
    except FieldProjectionError:
        return "homografia sem inversa finita no ponto do jogador"
    if jacobiana is None:
        return "jacobiana da homografia degenerada no ponto do jogador"

    rumo_campo = _resolve(jacobiana, rumo_imagem)
    if rumo_campo is None:
        return "rumo do corpo sem volta ao plano do campo"
    para_gol = (gol[0] - posicao[0], gol[1] - posicao[1])
    cosseno = _cosseno(rumo_campo, para_gol)
    if cosseno is None:
        return "finalizador sobre o centro do gol"

    confiancas = _confiancas(pontos, ombros)
    return _Amostra(
        frame_index=frame_index,
        angulo_deg=math.degrees(math.acos(cosseno)),
        confianca=min(confiancas),
        lado=lado,
        lateral=lateral,
        largura_dos_ombros_px=largura,
    )


def _ombros(pontos: dict[str, Keypoint]) -> tuple[Keypoint, Keypoint] | None:
    esquerdo, direito = pontos.get("shoulder_left"), pontos.get("shoulder_right")
    if esquerdo is None or direito is None:
        return None
    if not _confiavel(esquerdo) or not _confiavel(direito):
        return None
    return esquerdo, direito


def _confiavel(ponto: Keypoint | None) -> bool:
    return (
        ponto is not None
        and ponto.confidence is not None
        and ponto.confidence >= MIN_KEYPOINT_CONFIDENCE
    )


def _lado_da_camera(pontos: dict[str, Keypoint]) -> str | None:
    """De frente, de costas, ou indeterminado, pela visibilidade de olhos e orelhas.

    Dois olhos e rosto voltado para o lado da camera. Nenhum olho com orelha visivel e nuca. Um
    olho so e perfil, e perfil nao separa os dois sentidos: a recusa e declarada em vez de
    arbitrada, porque sentido errado inverte a resposta da feature.
    """
    olhos = sum(1 for nome in _OLHOS if _confiavel(pontos.get(nome)))
    orelhas = sum(1 for nome in _ORELHAS if _confiavel(pontos.get(nome)))
    if olhos == 2:
        return "de frente"
    if olhos == 0 and orelhas >= 1:
        return "de costas"
    return None


def _desvio_lateral(
    pontos: dict[str, Keypoint],
    esquerdo: Keypoint,
    direito: Keypoint,
    u: tuple[float, float],
    largura: float,
) -> float:
    """Quanto o nariz saiu do meio dos ombros, ao longo da linha dos ombros, em larguras de ombro.

    Sem nariz confiavel o desvio e zero e nao ausencia: a perpendicular aos ombros ja da um rumo,
    e o desvio e correcao em cima dele. Zero significa cabeca alinhada com o tronco, que e a
    hipotese mais fraca disponivel e nao uma invencao.
    """
    nariz = pontos.get("nose")
    if not _confiavel(nariz) or nariz is None:
        return 0.0
    meio = ((esquerdo.x + direito.x) / 2.0, (esquerdo.y + direito.y) / 2.0)
    desvio = (nariz.x - meio[0], nariz.y - meio[1])
    projecao = (desvio[0] * u[0] + desvio[1] * u[1]) / largura
    return max(-1.0, min(1.0, projecao))


def _ancora_no_chao(pontos: dict[str, Keypoint], jogador: PlayerDetection) -> tuple[float, float]:
    """O ponto do jogador que esta no plano do campo, e so ele pode passar pela homografia.

    Tornozelo quando ha, base da caixa quando nao. Ombro nao serve: a homografia mapeia o plano do
    gramado, e ponto a 1,4 m do chao projetado por ela cai metros adiante de onde a pessoa esta.
    """
    tornozelos = tuple(
        pontos[nome]
        for nome in ("ankle_left", "ankle_right")
        if nome in pontos and _confiavel(pontos[nome])
    )
    if tornozelos:
        return (
            sum(ponto.x for ponto in tornozelos) / len(tornozelos),
            sum(ponto.y for ponto in tornozelos) / len(tornozelos),
        )
    return ((jogador.box.x1 + jogador.box.x2) / 2.0, jogador.box.y2)


def _jacobiana(
    homography: tuple[float, ...], posicao: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Como um passo em metro no campo vira passo em pixel, aqui, por diferenca central.

    E o que permite comparar rumo de corpo com rumo de gol: o rumo do corpo e medido em pixel, o
    do gol em metro, e sem a jacobiana a comparacao sairia num dos dois espacos, deformada pela
    perspectiva no primeiro e impossivel no segundo.
    """
    passos: list[tuple[float, float]] = []
    for eixo in (0, 1):
        adiante = list(posicao)
        atras = list(posicao)
        adiante[eixo] += EPS_JACOBIANO_M
        atras[eixo] -= EPS_JACOBIANO_M
        try:
            frente = pitch_to_image(homography, (adiante[0], adiante[1]))
            tras = pitch_to_image(homography, (atras[0], atras[1]))
        except FieldProjectionError:
            return None
        passos.append(
            (
                (frente[0] - tras[0]) / (2.0 * EPS_JACOBIANO_M),
                (frente[1] - tras[1]) / (2.0 * EPS_JACOBIANO_M),
            )
        )
    coluna_x, coluna_y = passos
    if not all(math.isfinite(valor) for valor in (*coluna_x, *coluna_y)):
        return None
    return coluna_x, coluna_y


def _resolve(
    jacobiana: tuple[tuple[float, float], tuple[float, float]], pixel: tuple[float, float]
) -> tuple[float, float] | None:
    """O passo em metro cujo efeito em pixel e `pixel`, resolvendo o sistema 2 por 2."""
    (ax, ay), (bx, by) = jacobiana
    determinante = ax * by - ay * bx
    if abs(determinante) < 1e-12:
        return None
    metros = (
        (by * pixel[0] - bx * pixel[1]) / determinante,
        (ax * pixel[1] - ay * pixel[0]) / determinante,
    )
    return metros if all(math.isfinite(valor) for valor in metros) else None


def _cosseno(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    norma = math.hypot(*a) * math.hypot(*b)
    if norma < 1e-9:
        return None
    return max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / norma))


def _confiancas(
    pontos: dict[str, Keypoint], ombros: tuple[Keypoint, Keypoint]
) -> tuple[float, ...]:
    usados = [*ombros]
    usados += [
        pontos[nome]
        for nome in (*_OLHOS, *_ORELHAS, "nose")
        if nome in pontos and _confiavel(pontos[nome])
    ]
    return tuple(ponto.confidence for ponto in usados if ponto.confidence is not None) or (0.0,)


def _frame_mais_proximo(perception: Perception, frame_index: int) -> FramePerception | None:
    if not perception.frames:
        return None
    return min(perception.frames, key=lambda quadro: abs(quadro.frame_index - frame_index))


def _finalizador(perception: Perception, shot: ShotOutput) -> int | None:
    """O mesmo criterio de `looked_before`: track declarado, unico com pose, ou o mais perto da
    bola no quadro do contato."""
    com_pose = {
        jogador.track_id
        for quadro in perception.frames
        for jogador in quadro.players
        if jogador.track_id is not None and jogador.keypoints
    }
    if shot.shooter_track_id is not None and shot.shooter_track_id in com_pose:
        return shot.shooter_track_id
    if len(com_pose) == 1:
        return next(iter(com_pose))

    quadro = _frame_mais_proximo(perception, shot.frame_index)
    if quadro is None or quadro.ball is None or quadro.ball.source == "interpolated":
        return None
    centro_bola = (
        (quadro.ball.box.x1 + quadro.ball.box.x2) / 2.0,
        (quadro.ball.box.y1 + quadro.ball.box.y2) / 2.0,
    )
    candidatos = tuple(
        j for j in quadro.players if j.track_id is not None and j.role != "goalkeeper"
    )
    mais_proximo = min(
        candidatos,
        key=lambda j: math.dist(((j.box.x1 + j.box.x2) / 2.0, j.box.y2), centro_bola),
        default=None,
    )
    return None if mais_proximo is None else mais_proximo.track_id


def _jogador(quadro: FramePerception, track_id: int) -> PlayerDetection | None:
    return next((j for j in quadro.players if j.track_id == track_id and j.keypoints), None)
