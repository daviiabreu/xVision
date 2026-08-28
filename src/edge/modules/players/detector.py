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

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

from contracts import Box
from edge.artifacts import entrypoint_of
from edge.frames.preprocess import to_source_box
from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleNotLoaded,
    ModuleOutput,
    ModuleSpec,
    PreprocessSpec,
    check_preprocess,
)
from edge.modules.device import preferred_device
from edge.modules.inference import shared

PREPROCESS = PreprocessSpec(width=1280, height=736, color="bgr", letterbox=True, normalize="none")

COCO_PERSON: Mapping[int, str | None] = {0: None}
"""COCO so tem `person`, entao a funcao fica em aberto em vez de virar `player` por otimismo."""

FUTEBOL: Mapping[int, str | None] = {1: "goalkeeper", 2: "player", 3: "referee"}
"""As classes do peso de futebol em uso. A 0 e a bola, e nao entra na contagem de gente.

Ela fica de fora porque a metrica de qualidade da bola mede a distancia ate a caixa de gente mais
proxima, e contar a propria bola como gente daria distancia zero em todo quadro."""

MIN_CONFIDENCE = 0.25


@dataclass(frozen=True)
class PlayersConfig:
    imgsz: int = 1280
    conf: float = MIN_CONFIDENCE
    device: str = ""
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
        return tuple(registros)


def _por_confianca(registro: Mapping[str, object]) -> tuple[float, float, float]:
    caixa = cast("tuple[float, float, float, float]", registro["box"])
    return (-float(cast("Any", registro["confidence"])), caixa[0], caixa[1])


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
