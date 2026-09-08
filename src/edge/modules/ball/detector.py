"""Detector da bola baseado em YOLO (Ultralytics).

O detector fica atras da interface `BallDetector` para o rastreador nao depender do YOLO: assim o
YOLOv8n (baseline de controle da arquitetura) e o YOLO26n (candidato) sao comparaveis sem reescrever
o rastreamento, e os testes rodam com um detector de mentira. A dependencia pesada (ultralytics e
torch) e opcional, no extra `ball`, e so e importada quando o detector YOLO e realmente
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import cv2
import numpy as np

from contracts import BallDetection, Box
from edge.artifacts import sha256_of
from edge.modules.base import ModuleError
from edge.modules.inference import as_deteccoes
from edge.runtime import (
    Array,
    InferenceBackend,
    PyTorchBackend,
    RuntimeArtifact,
    RuntimeInfo,
    RuntimeModel,
    TensorSpec,
)

FrameArray = Any
"""Quadro decodificado (array HxWxC). Tipado como Any para nao exigir numpy no import do modulo."""


class AjusteDesconhecido(ModuleError): ...


class BallModelIncompatible(ModuleError): ...


@dataclass(frozen=True)
class DetectorTelemetry:
    tiles_executed: int = 0
    raw_candidates: int = 0
    candidates_after_nms: int = 0
    inference_ms: float = 0.0
    nms_ms: float = 0.0


@dataclass(frozen=True)
class DetectorConfig:
    """Parametros do detector, incluindo a faixa de forma e tamanho que separa bola de artefato.

    `ball_class_id` muda quando o modelo e ajustado com classes proprias. Os limites de forma sao
    fracao da altura do quadro (toleram zoom) e razao de aspecto (bola e quase quadrada).

    Os padroes aqui sao os de antes da calibracao, e continuam sendo o que vale para quem constroi
    o detector sem dizer nada. Quem mede troca-os pela entrada `[modules.ball]` do TOML, via
    `com_ajustes`: cravar o numero medido no padrao apagaria a diferenca entre o valor herdado e o
    valor escolhido, e a proxima pessoa nao saberia qual dos dois esta olhando.
    """

    weights: str
    conf: float = 0.25
    """Confianca minima que o YOLO aplica antes de devolver a caixa, e o corte mais caro do arquivo.

    Medido nos 38 quadros de contato anotados: a confianca mediana da bola CORRETA e 0,154, com p25
    em 0,096 e minimo em 0,033. Com 0,25 o proprio detector descarta o candidato certo em 21 a 23
    desses 38 quadros, antes de qualquer filtro nosso. `config/pipeline.models.toml` desce para
    0,10; 0,05 foi medido e recusado, porque leva a contagem de candidatos por quadro de 1 para 3,5
    sem mover `body_part`.
    """

    imgsz: int = 1280
    ball_class_id: int = 32
    device: str = "cpu"
    half: bool = False
    min_side_frac: float = 0.004
    max_side_frac: float = 0.06
    min_aspect: float = 0.5
    max_aspect: float = 2.0
    ladrilhos: int = 1
    sobreposicao_px: int = 100
    nms_iou: float = 0.1
    """Aspecto maximo aceito, ajustavel pelo TOML e medido em 2,0 mesmo assim.

    A suspeita era que o borrao de movimento alongasse a bola nos quadros do chute e 2,0 a
    descartasse: no clipe 4996030 o mesmo objeto aparece com aspecto 2,06 / 1,99 / 1,75 / 2,09 /
    1,82 / 2,02 / 1,95 em sete quadros consecutivos.

    A medicao nao sustenta subir o limite. Nos 24 quadros de contato da dobra de ajuste, a conf
    0,10 e com o limite desligado, 56 candidatos passam no filtro de tamanho e 24 deles caem a 20
    px ou menos de uma caixa de jogador, que e o criterio de `mlops.ball_quality` para ser a bola
    do lance. O aspecto desses 24 vai de 0,67 a 1,62, e nenhum passa de 2,0. Subir para 2,2 admite
    exatamente dois candidatos a mais, ambos do 4996030 e ambos a mais de 20 px de todo jogador, e
    de 2,2 ate 99 nao entra nem sai nada.

    Na metrica de ponta a ponta os quatro limites empatam: 9/24 = 0,375 a 2,0, 2,2, 2,5 e 3,0. O
    aspecto nao e o que prende esta especie, e mexer nele com 24 clipes seria decorar o conjunto de
    ajuste. Fica em 2,0, agora como valor medido e nao como alvo preliminar.

    `ladrilhos`: quantos pedacos por eixo. Um desliga; dois divide o quadro em quatro e roda
    cada um. A bola tem cerca de 13 px de lado no quadro de 1920 por 1080, e a passada cheia a 1280
    a reduz para 8,7 px, uma celula do nivel P3 do YOLO, que tem passo 8. Rodando cada pedaco de
    1060 por 640 a 1600, a escala sobe e a bola volta a ter cerca de 20 px. Medido na dobra de
    ajuste, no quadro do contato, escolhendo o candidato de maior confianca: passada cheia a 1280
    da 12/24, cheia a 1920 da 15/24, ladrilho 2x2 a 1280 da 18/24 e ladrilho 2x2 a 1600 da 19/24.
    Custa quatro inferencias por quadro em vez de uma, e nao e invencao nossa: e a receita que a
    Roboflow publica para bola de futebol, e o SAHI reporta de 5 a 7 pontos de AP so com
    inferencia fatiada.

    `sobreposicao_px`: sem ela a bola na juncao sai partida em duas meias caixas e o filtro de forma
    descarta as duas por aspecto. Cem pixels e a receita publicada e cobre com folga uma bola de 13
    px mais o erro de caixa.

    `nms_iou`: duas deteccoes da mesma bola em pedacos vizinhos se sobrepoem muito, entao o limiar
    e baixo. Bolas diferentes no mesmo quadro nunca se tocam, entao nao ha o risco usual de um NMS
    agressivo fundir dois objetos legitimos.
    """

    def com_ajustes(self, options: Mapping[str, object]) -> DetectorConfig:
        """Aplica o que a entrada `[modules.ball]` do TOML declarou, e recusa o que ela errou.

        Chave desconhecida levanta em vez de ser ignorada porque o modo de falha calado e caro
        aqui: `conf` escrito `confidence` deixaria o detector no padrao 0,25 e a execucao inteira
        reportaria o numero de antes como se fosse o da calibracao nova.
        """
        ajustaveis = {c.name for c in fields(self)} - {"weights", "ball_class_id", "device", "half"}
        desconhecidas = sorted(set(options) - ajustaveis)
        if desconhecidas:
            raise AjusteDesconhecido(
                f"{', '.join(desconhecidas)}: o detector da bola ajusta "
                f"{', '.join(sorted(ajustaveis))}"
            )
        valores: dict[str, object] = {}
        for nome, valor in options.items():
            numero = _numero(nome, valor)
            inteiros = {"imgsz", "ladrilhos", "sobreposicao_px"}
            valores[nome] = int(numero) if nome in inteiros else float(numero)
        return replace(self, **valores)


def _numero(nome: str, valor: object) -> float:
    """`conf = "0.10"` no TOML vira texto, e `float("0.10") > 0` so falharia dentro do YOLO."""
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        raise AjusteDesconhecido(f"modules.ball.{nome} = {valor!r}, e o campo espera numero")
    if valor <= 0:
        raise AjusteDesconhecido(
            f"modules.ball.{nome} = {valor!r}, e o campo espera numero acima de zero"
        )
    return float(valor)


def backend_specs(config: DetectorConfig) -> tuple[TensorSpec, TensorSpec]:
    """Bindings fixos para o quadro ou ladrilho declarado na configuracao."""
    shapes = {
        _letterbox_shape(height, width, config.imgsz)
        for width, height in _tile_sizes(1920, 1088, config.ladrilhos, config.sobreposicao_px)
    }
    if len(shapes) != 1:
        raise BallModelIncompatible("ladrilhos produzem shapes diferentes para o runtime fixo")
    height, width = shapes.pop()
    anchors = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
    return (
        TensorSpec("images", (1, 3, height, width), "float32", "NCHW"),
        TensorSpec("predictions", (1, 5, anchors), "float32", "flat"),
    )


def _tile_sizes(
    width: int, height: int, per_axis: int, overlap: int
) -> tuple[tuple[int, int], ...]:
    if per_axis <= 1:
        return ((width, height),)
    step_x, step_y = width // per_axis, height // per_axis
    return tuple(
        (
            min(width, (i + 1) * step_x + overlap) - max(0, i * step_x - overlap),
            min(height, (j + 1) * step_y + overlap) - max(0, j * step_y - overlap),
        )
        for i in range(per_axis)
        for j in range(per_axis)
    )


def _letterbox_shape(height: int, width: int, imgsz: int) -> tuple[int, int]:
    if imgsz <= 0 or imgsz % 32:
        raise BallModelIncompatible("ball.imgsz precisa ser multiplo positivo de 32")
    gain = min(imgsz / height, imgsz / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    pad_width = (imgsz - resized_width) % 32
    pad_height = (imgsz - resized_height) % 32
    return (resized_height + pad_height, resized_width + pad_width)


def pedacos(frame: FrameArray, por_eixo: int, sobreposicao: int) -> list[tuple[Any, float, float]]:
    """Divide o quadro em pedacos com sobreposicao e diz onde cada um comeca no quadro inteiro."""
    altura, largura = int(frame.shape[0]), int(frame.shape[1])
    passo_x, passo_y = largura // por_eixo, altura // por_eixo
    saida: list[tuple[Any, float, float]] = []
    for i in range(por_eixo):
        for j in range(por_eixo):
            x1 = max(0, i * passo_x - sobreposicao)
            y1 = max(0, j * passo_y - sobreposicao)
            x2 = min(largura, (i + 1) * passo_x + sobreposicao)
            y2 = min(altura, (j + 1) * passo_y + sobreposicao)
            saida.append((frame[y1:y2, x1:x2], float(x1), float(y1)))
    return saida


def sem_repetidas(
    cruas: Sequence[tuple[Box, float, int]], limiar: float
) -> tuple[tuple[Box, float, int], ...]:
    """Supressao por sobreposicao, para a bola na juncao nao sair duas vezes."""
    ordenadas = sorted(cruas, key=lambda c: -c[1])
    mantidas: list[tuple[Box, float, int]] = []
    for candidata in ordenadas:
        if all(_sobreposicao(candidata[0], guardada[0]) < limiar for guardada in mantidas):
            mantidas.append(candidata)
    return tuple(mantidas)


def _sobreposicao(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersecao = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    uniao = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


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
    """Recebe um quadro e devolve os candidatos a bola daquele quadro, ja filtrados por forma.

    `detect_raw` e `candidates_from` existem separados de `detect_candidates` para o cache
    compartilhado poder guardar a passada crua: a mesma inferencia serve `ball` e `players` quando
    as duas usam o mesmo peso, e cada especie filtra depois o que e dela.
    """

    def detect_raw(self, frame: FrameArray) -> tuple[tuple[Box, float, int], ...]: ...

    def candidates_from(
        self, cruas: Sequence[tuple[Box, float, int]], frame_height: float
    ) -> tuple[BallDetection, ...]: ...

    def detect_candidates(self, frame: FrameArray) -> tuple[BallDetection, ...]: ...


class YoloBallDetector:
    """Adaptador do Ultralytics YOLO para a interface `BallDetector`.

    Carrega os pesos uma unica vez (processo residente, como pede a pesquisa de execucao) e, a cada
    quadro, devolve os candidatos a bola que passam no filtro de forma e tamanho. Nao calcula
    `pitch_xy`: isso depende da homografia, de outra frente; aqui a deteccao sai em pixels.
    """

    def __init__(
        self,
        model: Any | None,
        config: DetectorConfig,
        *,
        backend: InferenceBackend | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._backend = backend
        self._telemetry = DetectorTelemetry()

    @property
    def telemetry(self) -> DetectorTelemetry:
        return self._telemetry

    @property
    def runtime_info(self) -> RuntimeInfo | None:
        return self._backend.info if self._backend is not None else None

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    @classmethod
    def from_config(cls, config: DetectorConfig) -> YoloBallDetector:
        """O `ultralytics` e o extra opcional `ball`, resolvido tarde e por nome.

        Import tardio porque a suite de conformidade instancia o modulo sem peso nenhum, e um
        import no topo do arquivo obrigaria torch para rodar um teste que nao usa torch.
        `import_module` em vez de `import ultralytics` porque o pyright roda em modo estrito e
        trata todo simbolo de import nao resolvido como desconhecido, o que exigiria supressao
        linha a linha em cada uso.
        """
        path = Path(config.weights)
        input_spec, output_spec = backend_specs(config)
        backend = PyTorchBackend(
            provider="ultralytics",
            device=config.device,
            loader=_load_ultralytics,
            runtime_versions=(
                ("torch", version("torch")),
                ("ultralytics", version("ultralytics")),
            ),
        )
        backend.load(
            RuntimeArtifact(
                path=path,
                sha256=sha256_of(path),
                format="pt",
                parent_onnx_sha256=None,
                inputs=(input_spec,),
                outputs=(output_spec,),
            )
        )
        return cls(None, replace(config, ball_class_id=0), backend=backend)

    def detect_raw(self, frame: FrameArray) -> tuple[tuple[Box, float, int], ...]:
        """Tudo o que o peso viu, sem filtrar classe nem forma.

        E o que o cache compartilhado guarda: a mesma passada serve a `ball` e a `players` quando as
        duas usam o mesmo peso, e cada uma filtra o que e dela depois.

        Com `ladrilhos` maior que um, roda uma passada por pedaco e devolve tudo junto no espaco
        do quadro completo, com supressao por sobreposicao na juncao. O cache compartilhado deixa
        de servir nesse modo e nao perde nada: ladrilhar so faz sentido quando `ball` tem peso
        proprio, e ai a chave do cache ja e outra.
        """
        config = self._config
        self._telemetry = DetectorTelemetry()
        if config.ladrilhos <= 1:
            start = perf_counter()
            detections = self._passada(frame, 0.0, 0.0)
            elapsed = (perf_counter() - start) * 1000.0
            self._telemetry = DetectorTelemetry(
                tiles_executed=1,
                raw_candidates=len(detections),
                candidates_after_nms=len(detections),
                inference_ms=elapsed,
            )
            return detections
        achadas: list[tuple[Box, float, int]] = []
        inference_ms = 0.0
        tiles = pedacos(frame, config.ladrilhos, config.sobreposicao_px)
        for pedaco, x, y in tiles:
            start = perf_counter()
            achadas.extend(self._passada(pedaco, x, y))
            inference_ms += (perf_counter() - start) * 1000.0
        start_nms = perf_counter()
        mantidas = sem_repetidas(achadas, config.nms_iou)
        nms_ms = (perf_counter() - start_nms) * 1000.0
        self._telemetry = DetectorTelemetry(
            tiles_executed=len(tiles),
            raw_candidates=len(achadas),
            candidates_after_nms=len(mantidas),
            inference_ms=inference_ms,
            nms_ms=nms_ms,
        )
        return mantidas

    def _passada(
        self, frame: FrameArray, dx: float, dy: float
    ) -> tuple[tuple[Box, float, int], ...]:
        if self._backend is not None:
            return self._backend_pass(frame, dx, dy)
        if self._model is None:
            raise BallModelIncompatible("detector da bola sem modelo ou backend")
        config = self._config
        results = self._model(
            frame, imgsz=config.imgsz, conf=config.conf, device=config.device, verbose=False
        )
        cruas: list[tuple[Box, float, int]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for caixa, confianca, classe in as_deteccoes(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                cruas.append(
                    (
                        Box(caixa.x1 + dx, caixa.y1 + dy, caixa.x2 + dx, caixa.y2 + dy),
                        confianca,
                        classe,
                    )
                )
        return tuple(cruas)

    def _backend_pass(
        self, frame: FrameArray, dx: float, dy: float
    ) -> tuple[tuple[Box, float, int], ...]:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise BallModelIncompatible(
                f"ball recebeu tensor {image.shape} {image.dtype}, esperado HxWx3 uint8"
            )
        tensor, gain, padding = _letterbox_tensor(image, self._config.imgsz)
        backend = self._backend
        if backend is None:
            raise BallModelIncompatible("detector da bola sem backend")
        raw = backend.infer({"images": tensor})[0].value
        torch = cast("Any", import_module("torch"))
        nms = cast("Any", import_module("ultralytics.utils.nms"))
        detections = nms.non_max_suppression(
            torch.from_numpy(np.array(raw, copy=True)),
            conf_thres=self._config.conf,
            iou_thres=0.7,
            nc=1,
            max_det=300,
        )[0]
        if len(detections) == 0:
            return ()
        height, width = image.shape[:2]
        rows: list[tuple[Box, float, int]] = []
        for row in detections.detach().cpu().numpy():
            x1, y1, x2, y2 = _restore_box(row[:4], gain, padding, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append(
                (
                    Box(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                    float(row[4]),
                    int(row[5]),
                )
            )
        return tuple(rows)

    def candidates_from(
        self, cruas: Sequence[tuple[Box, float, int]], frame_height: float
    ) -> tuple[BallDetection, ...]:
        """O filtro de forma do Gabriel aplicado sobre deteccoes que ja existem."""
        return tuple(
            BallDetection(box=caixa, confidence=confianca)
            for caixa, confianca, classe in cruas
            if classe == self._config.ball_class_id
            and is_ball_shaped(caixa, frame_height, self._config)
        )

    def detect_candidates(self, frame: FrameArray) -> tuple[BallDetection, ...]:
        return self.candidates_from(self.detect_raw(frame), float(frame.shape[0]))

    def detect(self, frame: FrameArray) -> BallDetection | None:
        """A bola de maior confianca do quadro, ignorando continuidade de movimento."""
        candidates = self.detect_candidates(frame)
        return max(candidates, key=lambda d: d.confidence, default=None)


NOMES_DE_BOLA = ("ball", "sports ball", "soccer ball", "bola")


def check_ball_model(model: object) -> None:
    task = getattr(model, "task", None)
    names = getattr(model, "names", None)
    normalized = (
        tuple(str(name).strip().lower() for name in cast("dict[object, object]", names).values())
        if isinstance(names, dict)
        else ()
    )
    if task != "detect" or len(normalized) != 1 or normalized[0] not in NOMES_DE_BOLA:
        raise BallModelIncompatible(
            f"ball exige task=detect e uma classe ball, recebeu task={task!r} e names={names!r}"
        )


def _load_ultralytics(path: Path, device: str) -> RuntimeModel:
    ultralytics = cast("Any", import_module("ultralytics"))
    torch = cast("Any", import_module("torch"))
    yolo = ultralytics.YOLO(str(path))
    check_ball_model(yolo)
    model = yolo.model.to(device).eval()

    def infer(inputs: Mapping[str, Array]) -> Mapping[str, Array]:
        tensor = torch.from_numpy(inputs["images"]).to(device)
        with torch.inference_mode():
            output = model(tensor)
            prediction = cast("Any", output[0] if isinstance(output, tuple) else output)
            host = np.asarray(prediction.detach().float().cpu().numpy(), dtype=np.float32)
        return {"predictions": cast("Array", np.ascontiguousarray(host))}

    return infer


def _letterbox_tensor(
    image: np.ndarray[Any, Any], imgsz: int
) -> tuple[np.ndarray[Any, Any], float, tuple[float, float]]:
    height, width = image.shape[:2]
    target_height, target_width = _letterbox_shape(height, width, imgsz)
    gain = min(target_height / height, target_width / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_width - resized_width) / 2.0
    pad_y = (target_height - resized_height) / 2.0
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    rgb = padded[:, :, ::-1].transpose(2, 0, 1)[None]
    tensor = np.ascontiguousarray(rgb, dtype=np.float32) / np.float32(255.0)
    return tensor, gain, (pad_x, pad_y)


def _restore_box(
    raw: Sequence[float],
    gain: float,
    padding: tuple[float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    pad_x, pad_y = padding
    x1 = min(float(width), max(0.0, (float(raw[0]) - pad_x) / gain))
    y1 = min(float(height), max(0.0, (float(raw[1]) - pad_y) / gain))
    x2 = min(float(width), max(0.0, (float(raw[2]) - pad_x) / gain))
    y2 = min(float(height), max(0.0, (float(raw[3]) - pad_y) / gain))
    return (x1, y1, x2, y2)
