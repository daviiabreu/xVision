"""Keypoints do finalizador ao longo da janela, a partir do recorte dele e nao do quadro inteiro.

Este modulo existe porque `pose` era o unico dos sete buracos que travava duas entregas ao mesmo
tempo: a feature `looked_before`, que precisa saber se o finalizador teve nocao de onde ia chutar,
e a `body_part`, cujo teto sem keypoint e 0,6705 medido sobre os 604 registros do conjunto
(acertar toda cabeca e responder sempre `PERNA_DIREITA`). Todo o ganho acima disso mora em separar
perna esquerda de direita, e tornozelo distingue as duas.

O desenho saiu de uma medicao, e nao de preferencia. A primeira versao rodava o estimador uma vez
por quadro e casava as pessoas com as caixas de `players` por sobreposicao, que e mais barato.
Medido num clipe do parceiro, o estimador achou 2 pessoas onde o detector de jogadores achou 14:
jogador de transmissao tem caixa de cerca de 32 por 102 pixels num quadro de 1920 por 1080, e
modelo de pose e muito mais fraco que modelo de deteccao nesse tamanho. Sobre o recorte do mesmo
jogador, ampliado, foram 14 de 14. Quadro inteiro simplesmente nao serve.

Recortar todo mundo custaria uma inferencia por jogador por quadro, algo como 1900 num clipe. Como
as duas features que precisam de pose falam do finalizador, o modulo recorta so ele: cerca de 121
recortes por clipe, em lotes, medidos em 14,8 ms cada contra 39,6 ms um a um.

Quem e o finalizador sai da geometria e nao de um campo do contrato: e o jogador cuja caixa esta
mais perto da bola no instante que a especie `shot` apontou. O `shooter_track_id` do registro de
`shot` nao serve porque ele nao existe: o modulo de instante decide sobre a trajetoria da bola e
nao sobre caixas de jogador, e por isso nunca preenche o campo. O ciclo no grafo que existia quando
`tracking` rodava na janela ancorada no `shot` foi removido pela uniformizacao dos conjuntos.

Nao detecta jogador, nao rastreia e nao mede angulo. Quem transforma keypoint em "olhou antes" ou
em "pe direito" e a feature, sobre a `Perception` ja montada.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from contracts import Box
from edge.artifacts import entrypoint_of
from edge.frames.preprocess import to_source_box, to_tensor_box
from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)
from edge.modules.precision import PADRAO, precisao_valida

PREPROCESS = PreprocessSpec(width=1280, height=736, color="bgr", letterbox=True, normalize="none")

COCO_KEYPOINTS = (
    "nose",
    "eye_left",
    "eye_right",
    "ear_left",
    "ear_right",
    "shoulder_left",
    "shoulder_right",
    "elbow_left",
    "elbow_right",
    "wrist_left",
    "wrist_right",
    "hip_left",
    "hip_right",
    "knee_left",
    "knee_right",
    "ankle_left",
    "ankle_right",
)
"""A ordem que todo estimador treinado em COCO devolve, e que da nome ao ponto no contrato.

O nome importa mais que o indice: a feature que separa perna esquerda de direita pergunta por
`ankle_left`, e um estimador com outra ordem quebraria calado se o contrato fosse posicional.
"""

MARGEM = 0.25
"""Quanto o recorte cresce alem da caixa, em fracao do lado.

A caixa de `players` fecha no corpo, e pe e cabeca costumam encostar na borda. Sem margem o
estimador perde justamente os dois pontos que as duas features consomem.
"""

LOTE = 32
"""Recortes por chamada do estimador. Medido: 14,8 ms cada em lote de 32 contra 39,6 ms um a um."""

Keypoints = tuple[tuple[str, float, float, float], ...]


class PoseConfigError(ModuleError): ...


class PoseEstimator(Protocol):
    """Um conjunto de keypoints por recorte, no espaco do recorte, ou `None` quando nao ha corpo."""

    def estimate_many(self, crops: Sequence[Any]) -> tuple[Keypoints | None, ...]: ...


@dataclass(frozen=True)
class _Recorte:
    frame_index: int
    origem: tuple[int, int]
    imagem: Any


class YoloPose:
    spec = ModuleSpec(
        kind="pose",
        name="yolo-pose",
        version="0.2.0",
        frames="window",
        preprocess=PREPROCESS,
        needs=("players", "tracking", "shot", "ball"),
    )

    def __init__(
        self, estimator: PoseEstimator | None = None, *, weights: str | None = None
    ) -> None:
        self._estimator = estimator
        self._weights = weights
        self._quantize = PADRAO
        self._pronto = False

    def fingerprint(self) -> str | None:
        """O ajuste em vigor, pelo mesmo motivo de `players`: o cache guarda numero, nao intencao.

        `run_module` le isto por `getattr` e, sem o metodo, a chave de disco saia igual nas duas
        precisoes. Declarar `quantize = "fp16"` depois de um clipe ja rodado devolvia o keypoint
        de fp32 do cache, e a comparacao entre as duas reportaria "identico" por acerto de chave.

        O padrao devolve `None` pela mesma razao de `players`: quem nao declarou precisao continua
        no endereco de sempre, e o cache que ja existe nesta maquina segue valido.
        """
        return None if self._quantize == PADRAO else f"quantize={self._quantize}"

    def configure(self, options: Mapping[str, object]) -> None:
        """A especie ajusta a precisao da inferencia, e so ela.

        `IMGSZ` e `CONF` continuam constantes medidas do adaptador: 256 e o que amplia o recorte de
        cerca de 40 por 130 px sem inventar, e abri-los ao TOML agora seria oferecer numero que
        ninguem calibrou. Nome desconhecido levanta para `quantise` no lugar de `quantize` nao
        virar meia precisao imaginaria lida no arquivo.
        """
        desconhecidas = sorted(set(options) - {"quantize"})
        if desconhecidas:
            raise PoseConfigError(f"{', '.join(desconhecidas)}: a especie pose ajusta quantize")
        if "quantize" in options:
            self._quantize = precisao_valida(
                "modules.pose.quantize", options["quantize"], PoseConfigError
            )

    def load(self) -> None:
        if self._estimator is None:
            peso = self._weights or _peso_declarado(self.spec.artifact)
            if peso is not None:
                from edge.modules.pose.yolo import YoloPoseEstimator

                self._estimator = YoloPoseEstimator.from_weights(peso, quantize=self._quantize)
        self._pronto = True

    def close(self) -> None:
        self._estimator = None
        self._pronto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        vazio = ModuleOutput(
            kind=self.spec.kind, frames=batch.indices, records=((),) * len(batch.indices)
        )
        if not batch.indices or self._estimator is None:
            return vazio

        jogadores = _por_quadro(ctx.get("players"))
        trajetorias = _track_ids(ctx.get("tracking"))
        atirador = _atirador(jogadores, trajetorias, ctx.get("shot"), ctx.get("ball"))
        if atirador is None:
            return vazio

        recortes = _recortes(batch, jogadores, trajetorias, atirador)
        if not recortes:
            return vazio
        medidos = self._medir(recortes)
        por_quadro = {
            recorte.frame_index: _no_quadro_original(pontos, recorte, batch)
            for recorte, pontos in zip(recortes, medidos, strict=True)
            if pontos is not None
        }
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=tuple(
                ({"track_id": atirador, "keypoints": por_quadro[indice]},)
                if indice in por_quadro
                else ()
                for indice in batch.indices
            ),
        )

    def _medir(self, recortes: Sequence[_Recorte]) -> tuple[Keypoints | None, ...]:
        estimador = cast("PoseEstimator", self._estimator)
        medidos: list[Keypoints | None] = []
        for inicio in range(0, len(recortes), LOTE):
            fatia = recortes[inicio : inicio + LOTE]
            medidos.extend(estimador.estimate_many([r.imagem for r in fatia]))
        return tuple(medidos)


def _atirador(
    jogadores: Mapping[int, tuple[Mapping[str, object], ...]],
    trajetorias: Mapping[int, Mapping[int, int]],
    shot: ModuleOutput | None,
    ball: ModuleOutput | None,
) -> int | None:
    """O `track_id` de quem estava mais perto da bola no instante do contato.

    Duas ausencias legitimas devolvem `None` em vez de um palpite: sem instante nao ha em torno de
    que quadro procurar, e sem bola nao ha de quem medir a distancia. Chutar o jogador mais central
    poria a pose no corpo errado, e nada rio abaixo distinguiria isso de uma medida.
    """
    contato = _instante(shot)
    if contato is None:
        return None
    bola = _bola_perto_de(ball, contato)
    if bola is None:
        return None
    quadro = _quadro_mais_perto(jogadores, contato)
    if quadro is None:
        return None
    caixas = jogadores.get(quadro, ())
    distancias = [
        (_distancia(_centro(Box(*cast("Any", c["box"]))), bola), posicao)
        for posicao, c in enumerate(caixas)
        if "box" in c
    ]
    if not distancias:
        return None
    _, posicao = min(distancias)
    return trajetorias.get(quadro, {}).get(posicao)


def _instante(shot: ModuleOutput | None) -> int | None:
    if shot is None:
        return None
    for registros in shot.records:
        for registro in registros:
            valor = registro.get("frame_index")
            if isinstance(valor, int) and not isinstance(valor, bool):
                return valor
    return None


def _bola_perto_de(ball: ModuleOutput | None, contato: int) -> tuple[float, float] | None:
    """A especie `ball` roda no conjunto grosso, entao o quadro do contato pode nao ter amostra."""
    if ball is None:
        return None
    candidatos = [
        (abs(indice - contato), _centro(Box(*cast("Any", registros[0]["box"]))))
        for indice, registros in zip(ball.frames, ball.records, strict=True)
        if registros and "box" in registros[0]
    ]
    return min(candidatos, key=lambda par: par[0])[1] if candidatos else None


def _quadro_mais_perto(
    jogadores: Mapping[int, tuple[Mapping[str, object], ...]], contato: int
) -> int | None:
    com_gente = [indice for indice, caixas in jogadores.items() if caixas]
    return min(com_gente, key=lambda i: abs(i - contato)) if com_gente else None


def _recortes(
    batch: FrameBatch,
    jogadores: Mapping[int, tuple[Mapping[str, object], ...]],
    trajetorias: Mapping[int, Mapping[int, int]],
    atirador: int,
) -> tuple[_Recorte, ...]:
    recortes: list[_Recorte] = []
    for posicao, indice in enumerate(batch.indices):
        caixa = _caixa_do_track(jogadores.get(indice, ()), trajetorias.get(indice, {}), atirador)
        if caixa is None:
            continue
        pedaco = _cortar(batch, posicao, caixa)
        if pedaco is not None:
            recortes.append(_Recorte(indice, pedaco[0], pedaco[1]))
    return tuple(recortes)


def _caixa_do_track(
    caixas: Sequence[Mapping[str, object]], ids: Mapping[int, int], atirador: int
) -> Box | None:
    for posicao, caixa in enumerate(caixas):
        if ids.get(posicao) == atirador and "box" in caixa:
            return Box(*cast("Any", caixa["box"]))
    return None


def _cortar(batch: FrameBatch, posicao: int, caixa: Box) -> tuple[tuple[int, int], Any] | None:
    """O recorte sai do tensor, e a caixa vem em pixel do quadro original: ela precisa da ida."""
    fonte = batch.source_size
    bruta = (caixa.x1, caixa.y1, caixa.x2, caixa.y2)
    x1, y1, x2, y2 = bruta if fonte is None else to_tensor_box(bruta, batch.preprocess, fonte)
    margem_x, margem_y = (x2 - x1) * MARGEM, (y2 - y1) * MARGEM
    altura, largura = batch.tensor.shape[1], batch.tensor.shape[2]
    esquerda = max(0, int(x1 - margem_x))
    topo = max(0, int(y1 - margem_y))
    direita = min(largura, int(x2 + margem_x))
    baixo = min(altura, int(y2 + margem_y))
    if direita - esquerda < 2 or baixo - topo < 2:
        return None
    pedaco = np.ascontiguousarray(batch.tensor[posicao, topo:baixo, esquerda:direita])
    return (esquerda, topo), pedaco


def _no_quadro_original(pontos: Keypoints, recorte: _Recorte, batch: FrameBatch) -> Keypoints:
    """Recorte para tensor e uma soma, e tensor para quadro original e a volta do enquadramento."""
    esquerda, topo = recorte.origem
    fonte = batch.source_size
    saida: list[tuple[str, float, float, float]] = []
    for nome, x, y, confianca in pontos:
        px, py = x + esquerda, y + topo
        if fonte is None:
            saida.append((nome, px, py, confianca))
            continue
        x1, y1, _, _ = to_source_box((px, py, px, py), batch.preprocess, fonte)
        saida.append((nome, x1, y1, confianca))
    return tuple(saida)


def _peso_declarado(artifact: str | None) -> str | None:
    caminho = entrypoint_of(artifact)
    return None if caminho is None else str(caminho)


def _por_quadro(saida: ModuleOutput | None) -> dict[int, tuple[Mapping[str, object], ...]]:
    if saida is None:
        return {}
    return dict(zip(saida.frames, saida.records, strict=True))


def _track_ids(saida: ModuleOutput | None) -> dict[int, dict[int, int]]:
    """`detection_index` para `track_id`, que e a mesma chave que o `assemble` usa."""
    if saida is None:
        return {}
    return {
        indice: {
            int(cast("Any", r["detection_index"])): int(cast("Any", r["track_id"]))
            for r in registros
            if "detection_index" in r and "track_id" in r
        }
        for indice, registros in zip(saida.frames, saida.records, strict=True)
    }


def _centro(caixa: Box) -> tuple[float, float]:
    return ((caixa.x1 + caixa.x2) / 2.0, (caixa.y1 + caixa.y2) / 2.0)


def _distancia(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def keypoints_de(linha: Sequence[Sequence[float]]) -> Keypoints:
    """Nomeia a matriz 17x3 que o estimador COCO devolve, na ordem que ele promete."""
    return tuple(
        (nome, float(ponto[0]), float(ponto[1]), float(ponto[2]) if len(ponto) > 2 else 1.0)
        for nome, ponto in zip(COCO_KEYPOINTS, linha, strict=False)
    )
