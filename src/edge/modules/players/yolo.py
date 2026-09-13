"""O adaptador do Ultralytics para o backend de deteccao de jogador.

Arquivo proprio e import tardio pelo mesmo motivo das outras duas especies: `ultralytics` puxa
torch, e a suite de conformidade instancia o modulo sem peso nenhum.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from contracts import Box
from edge.modules.precision import PADRAO, ajuste_do_predict


class UltralyticsPlayers:
    """Devolve caixa, confianca e identificador de classe, sem interpretar a classe.

    Quem sabe o que a classe significa e a `PlayersConfig`, porque isso depende do peso: o COCO
    chama 0 de `person` e o peso de futebol chama 2 de `player`.
    """

    def __init__(
        self, model: Any, *, imgsz: int, conf: float, device: str, quantize: str = PADRAO
    ) -> None:
        self._model = model
        self._imgsz = imgsz
        self._conf = conf
        self._device = device
        self._quantize = quantize

    @classmethod
    def from_weights(
        cls, weights: str, *, imgsz: int, conf: float, device: str, quantize: str = PADRAO
    ) -> UltralyticsPlayers:
        yolo = cast("Any", import_module("ultralytics")).YOLO
        return cls(yolo(weights), imgsz=imgsz, conf=conf, device=device, quantize=quantize)

    def detect(self, frame: Any) -> tuple[tuple[Box, float, int], ...]:
        """A precisao entra por `quantize`, e no padrao nao entra nada.

        Medido nesta Jetson a 960: 45,2 ms em fp32, 28,6 ms com `quantize='fp16'` e 62,4 ms com o
        `half=True` que o ultralytics 8.4 depreciou.
        """
        saidas = self._model.predict(
            source=frame,
            imgsz=self._imgsz,
            conf=self._conf,
            device=self._device,
            verbose=False,
            **ajuste_do_predict(self._quantize),
        )
        deteccoes: list[tuple[Box, float, int]] = []
        for saida in saidas:
            caixas = getattr(saida, "boxes", None)
            if caixas is None:
                continue
            linhas = zip(
                caixas.xyxy.tolist(), caixas.conf.tolist(), caixas.cls.tolist(), strict=False
            )
            for xyxy, confianca, classe in linhas:
                deteccoes.append((Box(*(float(v) for v in xyxy)), float(confianca), int(classe)))
        return tuple(deteccoes)
