"""Um navegador de quadros para anotar o instante do contato, quadro a quadro.

A folha de contato existe porque o `opencv-python-headless` do lock sai com `GUI: NONE` no wheel de
Linux, e nenhuma janela abre para quem nao esta num Mac. Onde a janela abre, ela e muito melhor: a
folha grossa localiza o segundo, e so; achar o quadro em que o pe encosta na bola pede taxa nativa,
e vinte celulas de 480 por 270 nao cabem vinte quadros consecutivos.

Por isso os dois caminhos convivem. Este exige janela e por isso nao entra em teste de CI nem roda
no servidor. O outro nao exige nada e por isso e o que o time usa.

Teclas: seta para navegar, `espaco` para marcar o quadro atual como o contato, `s` para pular o
clipe sem marcar, `q` para sair. `Shift` com as setas anda de dez em dez.

Nao decide nada e nao mede nada: escreve a anotacao pelo mesmo caminho que a folha usa, entao os
dois anotadores caem no mesmo arquivo e a concordancia entre eles sai do mesmo lugar.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import cv2

from contracts import Clip
from mlops.instant import Anotacao, write

CONTEXTO = (
    "origin_of_move",
    "meters",
    "penalty_area",
    "ball_at_post_detail",
    "is_goal",
    "is_blocked",
    "savedGoalkeeper",
)
"""O que a marcacao do parceiro diz do lance, para achar qual contato e o evento dela.

`body_part` e `used` ficam de fora de proposito. O clipe com rebote tem chute, defesa e segundo
chute, e sem contexto nao da para saber qual dos dois a Footstats marcou: origem, distancia e
desfecho resolvem isso. Dizer se foi cabeca ou pe resolveria melhor ainda e diria a resposta do
que estamos medindo, entao nao entra.
"""

VERDADEIROS = {"is_goal": "GOL", "is_blocked": "bloqueado", "savedGoalkeeper": "defendido"}

MARCAR = 32
PULAR = ord("s")
VOLTAR = ord("v")
SAIR = ord("q")
ANDAR = {81: -1, 2: -1, 63234: -1, 83: 1, 3: 1, 63235: 1}
SALTO = {ord("a"): -10, ord("d"): 10, ord("A"): -60, ord("D"): 60}
"""Tres passos, porque o clipe tem ate 540 quadros e o contato dura menos de um.

Atravessar 540 quadros de dez em dez sao 54 toques, e sao 40 clipes. Com shift o passo vai a um
segundo a 60 fps, que acha o lance; sem shift acha o quadro. As setas ficam para o quadro exato.
"""
FAIXA = 34


class SemJanela(RuntimeError):
    """O wheel instalado nao tem backend de janela, e `cv2.imshow` derrubaria a sessao.

    Verificado antes de abrir o video em vez de deixar `cv2.error` subir de dentro do laco, porque
    a mensagem dele nao diz o que fazer e a alternativa existe: `xvision label sheet`.
    """


def tem_janela() -> bool:
    """Le do proprio wheel se ele foi compilado com backend de janela.

    O `opencv-python-headless` sai com `GUI: NONE`, o `opencv-python` de Mac sai com `COCOA` e o de
    Linux com `GTK` ou `QT`. Perguntar ao build e mais barato que abrir uma janela para descobrir.
    """
    achado = re.search(r"^\s*GUI:\s*(\S+)", cv2.getBuildInformation(), re.MULTILINE)
    return achado is not None and achado.group(1).upper() != "NONE"


Acao = Literal["marcar", "pular", "voltar", "sair"]


@dataclass(frozen=True)
class Decisao:
    clip_id: str
    acao: Acao
    frame_index: int | None = None
    """So preenchido quando a acao e `marcar`: clipe pulado nao vira palpite de quadro zero."""


def contexto(labels: Mapping[str, str]) -> str:
    """Uma linha com o que o parceiro marcou do lance, sem dizer com que parte do corpo foi.

    Mantem o vocabulario dele em vez de traduzir: `DENTRO_AREA` e o que esta no CSV, e quem anota
    consegue conferir a linha depois. Traduzir criaria um mapa para manter em sincronia com a
    marcacao, e o ganho seria estetico.
    """
    partes: list[str] = []
    for campo in CONTEXTO:
        valor = str(labels.get(campo, "")).strip()
        if not valor or valor == "false":
            continue
        if campo in VERDADEIROS:
            partes.append(VERDADEIROS[campo] if valor == "true" else valor)
        elif campo == "meters":
            partes.append(f"{valor} m")
        else:
            partes.append(valor.replace("_", " ").lower())
    return "  ".join(partes)


def scrub(
    clip: Clip,
    *,
    sugestao: int | None = None,
    titulo: str = "xvision",
    marcacao: str = "",
    ja_marcado: bool = False,
) -> Decisao:
    """Abre o clipe e devolve o quadro que a pessoa marcou, ou `None` se ela pulou."""
    if not tem_janela():
        raise SemJanela("este cv2 nao tem janela; use `xvision label sheet`")
    captura = cv2.VideoCapture(str(clip.path))
    if not captura.isOpened():
        raise RuntimeError(f"{clip.path} nao abriu")
    atual = sugestao if sugestao is not None else clip.frame_count // 2
    try:
        while True:
            captura.set(cv2.CAP_PROP_POS_FRAMES, atual)
            lido, quadro = captura.read()
            if not lido:
                atual = max(0, min(clip.frame_count - 1, atual))
                continue
            cv2.imshow(titulo, _com_regua(quadro, clip, atual, sugestao, marcacao, ja_marcado))
            tecla = cv2.waitKeyEx(0) & 0xFFFFFF
            if tecla == SAIR:
                return Decisao(clip.clip_id, "sair")
            if tecla == PULAR:
                return Decisao(clip.clip_id, "pular")
            if tecla == VOLTAR:
                return Decisao(clip.clip_id, "voltar")
            if tecla == MARCAR:
                return Decisao(clip.clip_id, "marcar", atual)
            passo = ANDAR.get(tecla) or SALTO.get(tecla) or 0
            atual = max(0, min(clip.frame_count - 1, atual + passo))
    finally:
        captura.release()
        cv2.destroyWindow(titulo)


def _com_regua(
    quadro: Any,
    clip: Clip,
    atual: int,
    sugestao: int | None,
    marcacao: str = "",
    ja_marcado: bool = False,
) -> Any:
    """Uma faixa com o numero do quadro, o total, e o que o parceiro marcou do lance."""
    altura, largura = quadro.shape[0], quadro.shape[1]
    alto = FAIXA * 2 if marcacao else FAIXA
    cv2.rectangle(quadro, (0, altura - alto), (largura, altura), (0, 0, 0), -1)
    if marcacao:
        cv2.putText(
            quadro,
            marcacao,
            (12, altura - FAIXA - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (120, 220, 255),
            1,
            cv2.LINE_AA,
        )
    detectado = "" if sugestao is None else f"   modelo: {sugestao}"
    if ja_marcado:
        detectado = "   ja marcado, marcar de novo substitui"
    cv2.putText(
        quadro,
        f"quadro {atual} de {clip.frame_count - 1}   "
        f"{atual / clip.fps:.2f}s{detectado}   "
        "[setas] 1  [a/d] 10  [A/D] 60  [espaco] marca  [s] pula  [v] volta  [q] sai",
        (12, altura - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if sugestao is not None:
        x = int(largura * sugestao / max(1, clip.frame_count - 1))
        cv2.line(quadro, (x, altura - FAIXA), (x, altura), (0, 220, 0), 3)
    x = int(largura * atual / max(1, clip.frame_count - 1))
    cv2.line(quadro, (x, altura - FAIXA), (x, altura), (0, 160, 255), 2)
    return quadro


def anotar(
    clipes: Sequence[tuple[Clip, int | None]],
    annotator: str,
    marcacoes: Mapping[str, str] | None = None,
) -> int:
    """Percorre os clipes e grava cada marcacao. Devolve quantos clipes ficaram com quadro.

    Anda por indice e nao por iteracao porque `voltar` precisa existir: pular e uma tecla ao lado
    de marcar, e sem volta um toque errado tira o clipe da amostra em silencio.

    Voltar num clipe ja marcado abre no quadro que a pessoa escolheu, e nao no meio do clipe nem no
    palpite do modelo: quem volta esta corrigindo, e corrigir a partir do proprio valor anterior e
    mais curto que reencontrar o lance.
    """
    posicao = 0
    escolhidos: dict[str, int] = {}
    while 0 <= posicao < len(clipes):
        clip, sugestao = clipes[posicao]
        anterior = escolhidos.get(clip.clip_id)
        decisao = scrub(
            clip,
            sugestao=anterior if anterior is not None else sugestao,
            titulo=f"xvision  {posicao + 1} de {len(clipes)}",
            marcacao=(marcacoes or {}).get(clip.clip_id, ""),
            ja_marcado=anterior is not None,
        )
        if decisao.acao == "sair":
            break
        if decisao.acao == "voltar":
            posicao = max(0, posicao - 1)
            continue
        if decisao.acao == "marcar" and decisao.frame_index is not None:
            write(Anotacao(clip.clip_id, decisao.frame_index, annotator))
            escolhidos[clip.clip_id] = decisao.frame_index
        posicao += 1
    return len(escolhidos)
