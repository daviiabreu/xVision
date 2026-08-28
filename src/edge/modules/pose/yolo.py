"""O adaptador do estimador de pose do Ultralytics, sobre recortes e em lote.

Fica em arquivo proprio e com import tardio pelo mesmo motivo das outras especies: `ultralytics`
puxa torch, e a suite de conformidade instancia o modulo sem peso nenhum.

Um recorte tem uma pessoa so por construcao, entao a saida e o corpo de maior confianca de cada
um, e `None` quando o estimador nao achou ninguem ali. Devolver o primeiro da lista em vez do de
maior confianca deixaria a escolha na mao da ordem interna do modelo.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, cast

from edge.modules.pose.estimator import Keypoints, keypoints_de

IMGSZ = 256
"""O recorte de um jogador de transmissao tem cerca de 40 por 130 px, e 256 amplia sem inventar."""

CONF = 0.25


class YoloPoseEstimator:
    def __init__(self, model: Any, *, imgsz: int = IMGSZ, conf: float = CONF) -> None:
        self._model = model
        self._imgsz = imgsz
        self._conf = conf

    @classmethod
    def from_weights(cls, weights: str) -> YoloPoseEstimator:
        yolo = cast("Any", import_module("ultralytics")).YOLO
        return cls(yolo(weights))

    def estimate_many(self, crops: Sequence[Any]) -> tuple[Keypoints | None, ...]:
        if not crops:
            return ()
        from edge.modules.device import preferred_device

        saidas = self._model.predict(
            source=list(crops),
            imgsz=self._imgsz,
            conf=self._conf,
            device=preferred_device(),
            verbose=False,
        )
        return tuple(_melhor(saida) for saida in saidas)


def _melhor(saida: Any) -> Keypoints | None:
    poses = getattr(saida, "keypoints", None)
    caixas = getattr(saida, "boxes", None)
    if poses is None or caixas is None or not len(poses.data):
        return None
    confiancas = caixas.conf.tolist()
    melhor = max(range(len(confiancas)), key=lambda i: confiancas[i]) if confiancas else 0
    return keypoints_de(poses.data[melhor].tolist())
