"""A especie `players`: onde cada jogador esta em cada quadro da janela.

Ela e a base de tres outras. `tracking` liga as caixas dela ao longo do tempo, `pose` pendura
keypoint nelas, e `identity` poe numero de camisa. Sem ela o `assemble` monta quadro sem jogador
nenhum, e toda feature que fala de espaco, marcacao ou corpo sai `unavailable`.

O mapa de classe e parametro e nao constante porque os dois pesos que o projeto tem em vista
enumeram classes diferentes. O YOLO pre-treinado no COCO tem `person` na classe 0 e nao distingue
goleiro de arbitro; o peso de futebol em uso, descrito no manifesto da branch
`feat/deteccao-jogadores`, tem quatro classes e separa `goalkeeper`, `player` e `referee`.
Trocar de um para o outro e trocar o manifesto e o mapa, sem tocar em codigo.

`role` sai preenchido so quando o peso sabe distinguir, e nunca chutado: o contrato declara o
campo como opcional exatamente para isso, e um arbitro rotulado como jogador contaminaria toda
contagem de gente em volta do finalizador.

Nao rastreia, nao identifica e nao decide quem chutou. Uma caixa por pessoa, por quadro.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

from contracts import Box
from edge.artifacts import entrypoint_of
from edge.frames.preprocess import to_source_box
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
from edge.modules.device import preferred_device
from edge.modules.inference import shared
from edge.modules.precision import PADRAO, precisao_valida

PREPROCESS = PreprocessSpec(width=1280, height=736, color="bgr", letterbox=True, normalize="none")

COCO_PERSON: Mapping[int, str | None] = {0: None}
"""COCO so tem `person`, entao a funcao fica em aberto em vez de virar `player` por otimismo."""

FUTEBOL: Mapping[int, str | None] = {1: "goalkeeper", 2: "player", 3: "referee"}
"""As classes do peso de futebol em uso. A 0 e a bola, e nao entra na contagem de gente.

Ela fica de fora porque a metrica de qualidade da bola mede a distancia ate a caixa de gente mais
proxima, e contar a propria bola como gente daria distancia zero em todo quadro."""

MIN_CONFIDENCE = 0.25
DUPLICATE_IOU = 0.7
DUPLICATE_SMALLER_OVERLAP = 0.55


class PlayersConfigError(ModuleError): ...


@dataclass(frozen=True)
class PlayersConfig:
    imgsz: int = 1280
    conf: float = MIN_CONFIDENCE
    device: str = ""
    quantize: str = PADRAO
    """Precisao da inferencia, declarada em `[modules.players]` e medida antes de ser oferecida.

    Nesta Jetson, a 960: 45,2 ms em fp32 e 28,6 ms com `quantize='fp16'`. O padrao fica em fp32
    porque a divergencia de acuracia de meia precisao ainda nao foi medida contra gabarito.
    """

    classes: Mapping[int, str | None] = field(default_factory=lambda: COCO_PERSON)


class PlayerBackend(Protocol):
    def detect(self, frame: Any) -> tuple[tuple[Box, float, int], ...]: ...


class YoloPlayers:
    spec = ModuleSpec(
        kind="players",
        name="yolo-players",
        version="0.1.0",
        frames="window",
        preprocess=PREPROCESS,
    )

    def __init__(
        self, backend: PlayerBackend | None = None, *, config: PlayersConfig | None = None
    ) -> None:
        self._backend = backend
        self._config = config or PlayersConfig()
        self._peso = ""
        self._pronto = False

    def fingerprint(self) -> str | None:
        """O ajuste em vigor, para o cache de percepcao nao servir a percepcao da outra precisao.

        `run_module` le isto por `getattr` e sem o metodo devolve `None`, entao a chave de disco
        ficava identica em fp32 e em fp16: rodar o clipe no padrao, declarar `quantize = "fp16"` e
        rodar de novo devolvia a percepcao guardada e o modelo nem era chamado. A alavanca ficaria
        muda exatamente no uso que ela existe para servir, que e comparar as duas precisoes.

        No padrao devolve `None`, e nao `quantize=fp32`, para nao mudar de endereco quem nao pediu
        nada: com texto fixo, so acrescentar este metodo jogaria fora todo clipe de `players` ja
        guardado nesta maquina e cobraria a recontagem em GPU sem nenhum ganho de correcao.
        """
        return None if self._config.quantize == PADRAO else f"quantize={self._config.quantize}"

    def configure(self, options: Mapping[str, object]) -> None:
        """A especie ajusta a precisao da inferencia, e so ela por enquanto.

        `imgsz`, `conf` e `device` existem na `PlayersConfig` e continuam fora do vocabulario do
        TOML de proposito: nenhum deles foi calibrado por medicao aqui, e abrir o campo agora
        conviveria com o valor do manifesto sem dizer qual dos dois ganha. Nome desconhecido
        levanta em vez de ser ignorado pelo motivo de sempre: quem escreve `quantise` leria o
        arquivo, veria a execucao terminar sem reclamar e concluiria que ligou meia precisao.
        """
        desconhecidas = sorted(set(options) - {"quantize"})
        if desconhecidas:
            raise PlayersConfigError(
                f"{', '.join(desconhecidas)}: a especie players ajusta quantize"
            )
        if "quantize" in options:
            self._config = replace(
                self._config,
                quantize=precisao_valida(
                    "modules.players.quantize", options["quantize"], PlayersConfigError
                ),
            )

    def load(self) -> None:
        if self._backend is None:
            peso = entrypoint_of(self.spec.artifact)
            if peso is not None:
                self._peso = str(peso)
                from edge.modules.players.yolo import UltralyticsPlayers

                self._backend = UltralyticsPlayers.from_weights(
                    str(peso),
                    imgsz=self._config.imgsz,
                    conf=self._config.conf,
                    device=self._config.device or preferred_device(),
                    quantize=self._config.quantize,
                )
                self._config = replace(
                    self._config, classes=_classes_do_peso(self._backend, self._config)
                )
        self._pronto = True

    def close(self) -> None:
        self._backend = None
        self._peso = ""
        self._pronto = False

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        if not batch.indices:
            return ModuleOutput(kind=self.spec.kind, frames=(), records=())
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=tuple(
                self._quadro(batch, posicao, indice) for posicao, indice in enumerate(batch.indices)
            ),
        )

    def _quadro(
        self, batch: FrameBatch, posicao: int, indice: int
    ) -> tuple[Mapping[str, object], ...]:
        """Ordem estavel por confianca, porque `detection_index` e a chave que `tracking` usa.

        Sem ordem declarada, duas execucoes sobre a mesma entrada poderiam numerar as deteccoes de
        formas diferentes, o `track_id` cairia noutro jogador e a suite de conformidade reprovaria
        no teste de determinismo. A confianca desempata pela caixa para o caso de empate exato.
        """
        if self._backend is None:
            return ()
        backend = self._backend
        quadro = batch.tensor[posicao]
        brutas = (
            shared().get(
                weights=self._peso,
                preprocess=batch.preprocess,
                frame_index=indice,
                compute=lambda: backend.detect(quadro),
            )
            if self._peso
            else backend.detect(quadro)
        )
        fonte = batch.source_size
        registros: list[Mapping[str, object]] = []
        for caixa, confianca, classe in brutas:
            if classe not in self._config.classes or confianca < self._config.conf:
                continue
            no_quadro = (
                caixa
                if fonte is None
                else Box(*to_source_box(_as_tuple(caixa), batch.preprocess, fonte))
            )
            registro: dict[str, object] = {
                "box": _as_tuple(no_quadro),
                "confidence": float(confianca),
            }
            papel = self._config.classes[classe]
            if papel is not None:
                registro["role"] = papel
            registros.append(registro)
        registros.sort(key=_por_confianca)
        return _deduplicar(registros)


def _por_confianca(registro: Mapping[str, object]) -> tuple[float, float, float]:
    caixa = cast("tuple[float, float, float, float]", registro["box"])
    return (-float(cast("Any", registro["confidence"])), caixa[0], caixa[1])


def _deduplicar(
    registros: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Remove duas classes para a mesma pessoa sem juntar jogadores aglomerados.

    O peso pode devolver caixas quase iguais como `player` e `referee`. A maior confianca vence.
    Para duas caixas `player`, que podem ser pessoas diferentes numa disputa, so a sobreposicao
    forte permite remover uma delas.
    """
    mantidos: list[Mapping[str, object]] = []
    for registro in registros:
        caixa = Box(*cast("Any", registro["box"]))
        papel = registro.get("role")
        duplicado = False
        for outro in mantidos:
            outra_caixa = Box(*cast("Any", outro["box"]))
            outro_papel = outro.get("role")
            if _iou(caixa, outra_caixa) >= DUPLICATE_IOU or (
                (papel != outro_papel or papel != "player")
                and _sobreposicao_da_menor(caixa, outra_caixa) >= DUPLICATE_SMALLER_OVERLAP
            ):
                duplicado = True
                break
        if not duplicado:
            mantidos.append(registro)
    return tuple(mantidos)


def _intersecao(a: Box, b: Box) -> float:
    largura = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    altura = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return largura * altura


def _area(caixa: Box) -> float:
    return max(0.0, caixa.x2 - caixa.x1) * max(0.0, caixa.y2 - caixa.y1)


def _iou(a: Box, b: Box) -> float:
    intersecao = _intersecao(a, b)
    uniao = _area(a) + _area(b) - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


def _sobreposicao_da_menor(a: Box, b: Box) -> float:
    menor = min(_area(a), _area(b))
    return _intersecao(a, b) / menor if menor > 0 else 0.0


def _as_tuple(box: Box) -> tuple[float, float, float, float]:
    return (box.x1, box.y1, box.x2, box.y2)


PAPEIS = {"goalkeeper": "goalkeeper", "player": "player", "referee": "referee", "person": None}


def _classes_do_peso(backend: PlayerBackend, config: PlayersConfig) -> Mapping[int, str | None]:
    """Le do peso quais classes sao gente, em vez de exigir que a configuracao saiba.

    O COCO chama tudo de `person` e nao distingue funcao; o peso de futebol separa goleiro,
    jogador e arbitro, e tem a bola na classe 0, que nao pode entrar na contagem de gente.
    """
    nomes = getattr(getattr(backend, "_model", None), "names", None)
    if not isinstance(nomes, dict):
        return config.classes
    achadas = {
        int(i): PAPEIS[str(n).strip().lower()]
        for i, n in cast("dict[int, str]", nomes).items()
        if str(n).strip().lower() in PAPEIS
    }
    return achadas or config.classes
