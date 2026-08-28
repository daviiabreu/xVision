"""Detector da bola baseado em YOLO (Ultralytics).

O detector fica atras da interface `BallDetector` para o rastreador nao depender do YOLO: assim o
YOLOv8n (baseline de controle da arquitetura) e o YOLO26n (candidato) sao comparaveis sem reescrever
o rastreamento, e os testes rodam com um detector de mentira. A dependencia pesada (ultralytics e
torch) e opcional, no extra `detector`, e so e importada quando o detector YOLO e realmente
construido.

O detector devolve **todos** os candidatos a bola do quadro, ja filtrados por forma e tamanho, e
deixa a escolha final para o rastreador, que decide por continuidade de movimento. Isso e o que tira
a marca do penalti do caminho: uma caixa fora da faixa de tamanho ou de aspecto de bola nem chega ao
rastreador, e a que sobra so vence se estiver na trajetoria.

Fine tuning: o YOLO pre-treinado no COCO tem a classe `sports ball` (id 32) e serve de ponto de
partida, mas em video de transmissao a bola e pequena, rapida e distante, entao o recall perto do
contato tende a exigir ajuste fino em dados de futebol. `imgsz` alto e o ajuste fino sao as maiores
alavancas; a decisao de treinar sai da medicao de recall, nao antes dela.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from contracts.perception import BallDetection, Box

FrameArray = Any
"""Quadro decodificado (array HxWxC). Tipado como Any para nao exigir numpy no import do modulo."""


@dataclass(frozen=True)
class DetectorConfig:
    """Parametros do detector, incluindo a faixa de forma e tamanho que separa bola de artefato.

    `ball_class_id` muda quando o modelo e ajustado com classes proprias. Os limites de forma sao
    fracao da altura do quadro (toleram zoom) e razao de aspecto (bola e quase quadrada); sao alvos
    preliminares, a calibrar nos clipes reais.
    """

    weights: str
    conf: float = 0.25
    imgsz: int = 1280
    ball_class_id: int = 32
    device: str = "cpu"
    half: bool = False
    min_side_frac: float = 0.004
    max_side_frac: float = 0.06
    min_aspect: float = 0.5
    max_aspect: float = 2.0


def is_ball_shaped(box: Box, frame_height: float, config: DetectorConfig) -> bool:
    """Uma caixa parece bola quando o lado cabe na faixa de tamanho e o aspecto e quase quadrado.

    O tamanho e medido como fracao da altura do quadro para nao depender do zoom. Nao separa a marca
    do penalti vazia de uma bola parada sobre ela (ambas pequenas e redondas): isso e trabalho da
    continuidade de movimento no rastreador e, no limite, do ajuste fino.
    """
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    if width <= 0 or height <= 0 or frame_height <= 0:
        return False
    side_frac = max(width, height) / frame_height
    if not config.min_side_frac <= side_frac <= config.max_side_frac:
        return False
    aspect = width / height
    return config.min_aspect <= aspect <= config.max_aspect


class BallDetector(Protocol):
    """Recebe um quadro e devolve os candidatos a bola daquele quadro, ja filtrados por forma."""

    def detect_candidates(self, frame: FrameArray) -> tuple[BallDetection, ...]: ...


class YoloBallDetector:
    """Adaptador do Ultralytics YOLO para a interface `BallDetector`.

    Carrega os pesos uma unica vez (processo residente, como pede a pesquisa de execucao) e, a cada
    quadro, devolve os candidatos a bola que passam no filtro de forma e tamanho. Nao calcula
    `pitch_xy`: isso depende da homografia, de outra frente; aqui a deteccao sai em pixels.
    """

    def __init__(self, model: Any, config: DetectorConfig) -> None:
        self._model = model
        self._config = config

    @classmethod
    def from_config(cls, config: DetectorConfig) -> YoloBallDetector:
        from ultralytics import (
            YOLO,  # pyright: ignore[reportMissingImports, reportPrivateImportUsage]
        )

        return cls(YOLO(config.weights), config)

    def detect_candidates(self, frame: FrameArray) -> tuple[BallDetection, ...]:
        config = self._config
        results = self._model(
            frame,
            imgsz=config.imgsz,
            conf=config.conf,
            device=config.device,
            half=config.half,
            verbose=False,
        )
        candidates: list[BallDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            frame_height = float(getattr(result, "orig_shape", (0, 0))[0])
            rows = zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=False)
            for xyxy, raw_conf, raw_cls in rows:
                if int(raw_cls) != config.ball_class_id:
                    continue
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                box = Box(x1, y1, x2, y2)
                if not is_ball_shaped(box, frame_height, config):
                    continue
                candidates.append(BallDetection(box=box, confidence=float(raw_conf)))
        return tuple(candidates)

    def detect(self, frame: FrameArray) -> BallDetection | None:
        """A bola de maior confianca do quadro, ignorando a continuidade de movimento."""
        candidates = self.detect_candidates(frame)
        return max(candidates, key=lambda d: d.confidence, default=None)
