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

import re
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
from edge.modules.precision import PADRAO, ajuste_do_predict, precisao_valida
from edge.runtime import (
    Array,
    InferenceBackend,
    ModelLoader,
    PyTorchBackend,
    RuntimeArtifact,
    RuntimeInfo,
    RuntimeModel,
    TensorRTBackend,
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
    quantize: str = PADRAO
    """Precisao da inferencia, e o unico campo de texto que o TOML ajusta aqui.

    Ela substitui um campo `half: bool = False` que morreu neste lugar. Ele nunca chegou ao
    predict: estava na lista de exclusao de `com_ajustes` e nenhuma chamada o passava adiante.
    `half = true` no TOML ja era erro antes desta alavanca, justamente por estar excluido: caia em
    `desconhecidas` e levantava `AjusteDesconhecido`. O que ele aceitava calado era a construcao
    programatica, `DetectorConfig(half=True)`, que passava e nao ligava meia precisao nenhuma.

    O substituto nao e um `half` que agora funciona: o parametro `half` esta depreciado no
    ultralytics 8.4 e e mais lento que a precisao cheia. Medido nesta Jetson com o peso de
    jogadores a 960, `half=True` deu 62,4 ms contra 45,2 ms de fp32 e 28,6 ms de
    `quantize='fp16'`.

    O padrao continua fp32, aqui e no `predict`: com meia precisao a divergencia de acuracia ainda
    nao foi medida contra gabarito, e um clipe rodado sem declarar nada tem de sair identico ao de
    antes.
    """

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

    ladrilhos_por_lote: int = 4
    """Quantos ladrilhos entram num despacho do Ultralytics; 4 poe o 2x2 inteiro numa chamada so.

    O lote e a alavanca de tempo, porque troca quatro idas ao modelo por uma, mas multiplica por
    quatro a memoria de ativacao, e nesta Jetson os 8 GB sao divididos entre CPU e GPU. O venv de
    teste nao tem torch, entao esse consumo nao foi medido aqui: o valor fica ajustavel pelo TOML
    para quem medir na placa poder descer para 2 sem reescrever o detector. Valor acima do numero
    de ladrilhos nao muda nada, porque o lote nunca passa do que existe.
    """

    engine: str = ""
    """Caminho de um plano TensorRT. Vazio mantem o PyTorch, que continua sendo o padrao.

    Medido nesta Jetson Orin, lote 4 a 992x1600: o forward do PyTorch em fp16 custa 101,3 ms e o
    plano equivalente custa 45,1 ms, 55,5% a menos. Quase metade disso nao vem da precisao e sim
    da fusao de camadas: em fp32 o plano ja custa 89,2 ms contra 157,7 ms do PyTorch.

    Fica em caminho e nao em manifesto porque plano nao e artefato versionavel: tem cerca de
    140 MB, nao atravessa versao de TensorRT nem placa, e por isso e construido nesta maquina em
    vez de baixado. O `engine_onnx_sha256` abaixo amarra o plano ao ONNX que o gerou, que e a
    unica procedencia que sobrevive a serializacao.

    O padrao continua no PyTorch de proposito: ele e o oraculo contra o qual a divergencia do
    plano e medida, e e o que roda se o plano nao carregar.
    """

    engine_onnx_sha256: str = ""
    """sha256 do ONNX que originou o plano, exigido pelo `RuntimeArtifact` no formato `engine`.

    Sem ele nao ha como saber de qual peso o plano veio: o `.engine` e opaco e nao carrega o
    digest do `.pt`. So e lido quando `engine` esta preenchido.
    """

    letterbox: str = "cv2"
    """Onde o letterbox dos ladrilhos acontece: no `cv2` da CPU ou no device com `cuda`.

    Medido nesta Jetson Orin, quatro ladrilhos de um 2x2 a imgsz 1600: o caminho de `cv2` mais
    numpy custa 69,9 ms por quadro, e o de CUDA custa 18,8 ms mais 6,0 ms para descer os 73 MiB
    de volta para numpy, ou seja 24,8 ms. Sao 45 ms por quadro num quadro de 270 ms, com a bola
    respondendo por cerca de 70% do clipe.

    Descer para numpy no fim e de proposito: o `PyTorchBackend` recebe array, e manter esse
    contrato e o que deixa o mesmo binding servir ao TensorRT depois. Na Jetson a memoria e
    unificada, entao a descida custa 6 ms e nao a viagem de PCIe que custaria numa placa discreta.

    O padrao fica no `cv2` porque as duas interpolacoes nao dao o mesmo pixel: o desvio maximo
    medido e 0,003 na escala 0..1. E pouco, mas a bola tem cerca de 13 px e a divergencia de
    deteccao ainda nao foi medida contra gabarito.
    """

    def com_ajustes(self, options: Mapping[str, object]) -> DetectorConfig:
        """Aplica o que a entrada `[modules.ball]` do TOML declarou, e recusa o que ela errou.

        Chave desconhecida levanta em vez de ser ignorada porque o modo de falha calado e caro
        aqui: `conf` escrito `confidence` deixaria o detector no padrao 0,25 e a execucao inteira
        reportaria o numero de antes como se fosse o da calibracao nova.
        """
        ajustaveis = {c.name for c in fields(self)} - {"weights", "ball_class_id", "device"}
        desconhecidas = sorted(set(options) - ajustaveis)
        if desconhecidas:
            raise AjusteDesconhecido(
                f"{', '.join(desconhecidas)}: o detector da bola ajusta "
                f"{', '.join(sorted(ajustaveis))}"
            )
        valores: dict[str, object] = {}
        for nome, valor in options.items():
            if nome == "quantize":
                valores[nome] = precisao_valida("modules.ball.quantize", valor, AjusteDesconhecido)
                continue
            if nome == "letterbox":
                valores[nome] = _letterbox_valido(valor)
                continue
            if nome in {"engine", "engine_onnx_sha256"}:
                valores[nome] = _texto(nome, valor)
                continue
            numero = _numero(nome, valor)
            inteiros = {"imgsz", "ladrilhos", "sobreposicao_px", "ladrilhos_por_lote"}
            valores[nome] = int(numero) if nome in inteiros else float(numero)
        return replace(self, **valores)


LETTERBOX_ACEITOS = ("cv2", "cuda")


def _letterbox_valido(valor: object) -> str:
    """`gpu` em vez de `cuda` cairia no padrao e a execucao reportaria o tempo de antes."""
    if not isinstance(valor, str) or valor not in LETTERBOX_ACEITOS:
        raise AjusteDesconhecido(
            f"modules.ball.letterbox = {valor!r}, e o aceito e {' ou '.join(LETTERBOX_ACEITOS)}"
        )
    return valor


def _texto(nome: str, valor: object) -> str:
    """Caminho de plano e digest sao texto, e o ramo numerico os rejeitaria antes de qualquer uso.

    Erra calado de um jeito caro: sem isto, `engine` declarado no TOML levanta "espera numero",
    que aponta para o tipo do valor em vez do campo que falta tratar.
    """
    if not isinstance(valor, str) or not valor:
        raise AjusteDesconhecido(f"modules.ball.{nome} = {valor!r}, e o campo espera texto")
    return valor


def _numero(nome: str, valor: object) -> float:
    """`conf = "0.10"` no TOML vira texto, e `float("0.10") > 0` so falharia dentro do YOLO."""
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        raise AjusteDesconhecido(f"modules.ball.{nome} = {valor!r}, e o campo espera numero")
    if valor <= 0:
        raise AjusteDesconhecido(
            f"modules.ball.{nome} = {valor!r}, e o campo espera numero acima de zero"
        )
    return float(valor)


def lote_do_backend(config: DetectorConfig) -> int:
    """Quantos ladrilhos o backend recebe por despacho.

    O artefato de producao e `.pt`, e modelo PyTorch aceita qualquer lote: o lote 1 que estava aqui
    era restricao nossa, escrita neste arquivo, e nao do peso. Medido no forward do shape de
    producao (992x1600) nesta Jetson Orin: quatro despachos de lote 1 custam 220,6 ms contra
    158,3 ms de um despacho de lote 4, com pico de 375 MiB, que nao pesa nos 8 GB da placa.

    Lote que nao divide o numero de ladrilhos e recusado aqui em vez de no meio do quadro: o
    binding do backend e de shape fixo, entao um resto produziria um segundo despacho de outro
    tamanho, e a recusa viria depois de metade do trabalho feito.
    """
    ladrilhos = max(1, config.ladrilhos) ** 2
    lote = min(ladrilhos, max(1, config.ladrilhos_por_lote))
    if ladrilhos % lote:
        raise BallModelIncompatible(
            f"ladrilhos_por_lote = {config.ladrilhos_por_lote} nao divide os {ladrilhos} "
            f"ladrilhos de um {config.ladrilhos}x{config.ladrilhos}, e o binding do backend e fixo"
        )
    return lote


def backend_specs(config: DetectorConfig) -> tuple[TensorSpec, TensorSpec]:
    """Bindings fixos para o lote de ladrilhos declarado na configuracao."""
    shapes = {
        _letterbox_shape(height, width, config.imgsz)
        for width, height in _tile_sizes(1920, 1088, config.ladrilhos, config.sobreposicao_px)
    }
    if len(shapes) != 1:
        raise BallModelIncompatible("ladrilhos produzem shapes diferentes para o runtime fixo")
    height, width = shapes.pop()
    anchors = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
    lote = lote_do_backend(config)
    return (
        TensorSpec("images", (lote, 3, height, width), "float32", "NCHW"),
        TensorSpec("predictions", (lote, 5, anchors), "float32", "flat"),
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
        input_spec, output_spec = backend_specs(config)
        backend: InferenceBackend
        if config.engine:
            backend, artifact = _engine_backend(config, input_spec, output_spec)
        else:
            backend, artifact = _pytorch_backend(config, input_spec, output_spec)
        backend.load(artifact)
        return cls(None, replace(config, ball_class_id=0), backend=backend)

    def detect_raw(self, frame: FrameArray) -> tuple[tuple[Box, float, int], ...]:
        """Tudo o que o peso viu, sem filtrar classe nem forma.

        E o que o cache compartilhado guarda: a mesma passada serve a `ball` e a `players` quando as
        duas usam o mesmo peso, e cada uma filtra o que e dela depois.

        Com `ladrilhos` maior que um, os pedacos vao ao modelo em lote e voltam juntos no espaco
        do quadro completo, com supressao por sobreposicao na juncao. O cache compartilhado deixa
        de servir nesse modo e nao perde nada: ladrilhar so faz sentido quando `ball` tem peso
        proprio, e ai a chave do cache ja e outra.

        `inference_ms` mede o lote inteiro, e nao a soma de quatro medicoes: com um despacho so a
        soma nao existe mais, e o que interessa ao orcamento por quadro e o tempo de parede da
        inferencia toda.
        """
        config = self._config
        self._telemetry = DetectorTelemetry()
        if config.ladrilhos <= 1:
            start = perf_counter()
            detections = self._passada_em_lote(((frame, 0.0, 0.0),))
            elapsed = (perf_counter() - start) * 1000.0
            self._telemetry = DetectorTelemetry(
                tiles_executed=1,
                raw_candidates=len(detections),
                candidates_after_nms=len(detections),
                inference_ms=elapsed,
            )
            return detections
        achadas: list[tuple[Box, float, int]] = []
        tiles = pedacos(frame, config.ladrilhos, config.sobreposicao_px)
        por_lote = max(1, config.ladrilhos_por_lote)
        start = perf_counter()
        for inicio in range(0, len(tiles), por_lote):
            achadas.extend(self._passada_em_lote(tiles[inicio : inicio + por_lote]))
        inference_ms = (perf_counter() - start) * 1000.0
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

    def _passada_em_lote(
        self, ladrilhos: Sequence[tuple[FrameArray, float, float]]
    ) -> tuple[tuple[Box, float, int], ...]:
        """Um despacho para varios ladrilhos, ja devolvendo as caixas no espaco do quadro inteiro.

        O Ultralytics aceita a lista de imagens de uma vez e devolve um result por imagem na mesma
        ordem da entrada, entao o deslocamento de cada ladrilho casa por indice. Medido nesta
        Jetson, um ladrilho a imgsz 1600 custa 70,4 ms, ou 282 ms por quadro nos quatro despachos
        sequenciais que isto substitui. Um ladrilho so continua indo cru, fora de lista, para nao
        mudar nada quando `ladrilhos` esta desligado.

        O caminho do backend tambem bateleia, e o `TensorSpec` declara o lote em `backend_specs`.
        A versao anterior desta docstring afirmava que batelar ali exigiria exportar um artefato
        novo: isso vale para ONNX e para engine, onde o shape fica assado no arquivo, e nao para o
        `.pt` de producao, que aceita qualquer lote. O lote 1 era restricao escrita aqui.

        A ordem da saida e a ordem dos ladrilhos, e ela e contratual: `sem_repetidas` ordena por
        confianca com `sorted` estavel, entao o empate se desfaz pela ordem de insercao e vira o
        `detection_index` que o rastreamento usa como chave.
        """
        if self._backend is not None:
            return self._backend_em_lote(ladrilhos)
        if self._model is None:
            raise BallModelIncompatible("detector da bola sem modelo ou backend")
        config = self._config
        imagens = [pedaco for pedaco, _, _ in ladrilhos]
        results = self._model(
            imagens[0] if len(imagens) == 1 else imagens,
            imgsz=config.imgsz,
            conf=config.conf,
            device=config.device,
            verbose=False,
            **ajuste_do_predict(config.quantize),
        )
        cruas: list[tuple[Box, float, int]] = []
        for result, (_, dx, dy) in zip(results, ladrilhos, strict=True):
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

    def _backend_em_lote(
        self, ladrilhos: Sequence[tuple[FrameArray, float, float]]
    ) -> tuple[tuple[Box, float, int], ...]:
        """Empilha os ladrilhos num tensor e faz um despacho, restaurando cada caixa no seu pedaco.

        Cada ladrilho tem letterbox proprio, com ganho e enchimento proprios, entao a geometria e
        guardada por indice e usada na volta: misturar as geometrias poria a caixa de um pedaco no
        espaco de outro, e o resultado continuaria sendo quatro floats plausiveis.

        O `non_max_suppression` do Ultralytics devolve uma lista com uma entrada por item do lote,
        na ordem da entrada. O `strict=True` no zip existe porque essa correspondencia e a unica
        coisa que liga deteccao a ladrilho: se ela mudar, o deslocamento errado seria somado a
        caixa sem erro nenhum aparecer.
        """
        backend = self._backend
        if backend is None:
            raise BallModelIncompatible("detector da bola sem backend")
        tensores: list[Any] = []
        geometrias: list[tuple[float, tuple[float, float], int, int, float, float]] = []
        for pedaco, dx, dy in ladrilhos:
            image = np.asarray(pedaco)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise BallModelIncompatible(
                    f"ball recebeu tensor {image.shape} {image.dtype}, esperado HxWx3 uint8"
                )
            tensor, gain, padding = _letterbox_tensor(
                image, self._config.imgsz, self._config.letterbox
            )
            tensores.append(tensor)
            altura, largura = image.shape[:2]
            geometrias.append((gain, padding, int(largura), int(altura), dx, dy))
        empilhado = np.ascontiguousarray(np.concatenate(tensores, axis=0))
        raw = backend.infer({"images": empilhado})[0].value
        torch = cast("Any", import_module("torch"))
        nms = cast("Any", import_module("ultralytics.utils.nms"))
        por_ladrilho = nms.non_max_suppression(
            torch.from_numpy(np.array(raw, copy=True)),
            conf_thres=self._config.conf,
            iou_thres=0.7,
            nc=1,
            max_det=300,
        )
        achadas: list[tuple[Box, float, int]] = []
        for deteccoes, geometria in zip(por_ladrilho, geometrias, strict=True):
            gain, padding, largura, altura, dx, dy = geometria
            if len(deteccoes) == 0:
                continue
            for row in deteccoes.detach().cpu().numpy():
                x1, y1, x2, y2 = _restore_box(row[:4], gain, padding, largura, altura)
                if x2 <= x1 or y2 <= y1:
                    continue
                achadas.append(
                    (
                        Box(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                        float(row[4]),
                        int(row[5]),
                    )
                )
        return tuple(achadas)

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


def carregador_ultralytics(quantize: str = PADRAO) -> ModelLoader:
    """O loader do backend ja fechado sobre a precisao declarada.

    Existe porque `ModelLoader` e `(caminho, dispositivo)` e a precisao nao cabe nessa assinatura:
    ela e do artefato e nao da chamada, e o `PyTorchBackend` mora em `edge.runtime`, que nao sabe
    de especie nenhuma.
    """

    def carregar(path: Path, device: str) -> RuntimeModel:
        return _load_ultralytics(path, device, quantize)

    return carregar


def _load_ultralytics(path: Path, device: str, quantize: str = PADRAO) -> RuntimeModel:
    """A meia precisao mora aqui, e nao no `predict`, porque producao nao passa pelo `predict`.

    `from_config` sempre monta o `PyTorchBackend`, entao o caminho do `predict` neste arquivo so
    roda com modelo injetado. O binding do artefato continua float32 dos dois lados: `TensorSpec`
    declara float32 na entrada e o `PyTorchBackend` cobra isso, e a saida volta por `.float()`
    antes do numpy. O que muda em fp16 sao o peso e o tensor, dentro desta funcao.
    """
    ultralytics = cast("Any", import_module("ultralytics"))
    torch = cast("Any", import_module("torch"))
    yolo = ultralytics.YOLO(str(path))
    check_ball_model(yolo)
    model = yolo.model.to(device).eval()
    meia = quantize != PADRAO
    if meia:
        model = model.half()

    def infer(inputs: Mapping[str, Array]) -> Mapping[str, Array]:
        tensor = torch.from_numpy(inputs["images"]).to(device)
        if meia:
            tensor = tensor.half()
        with torch.inference_mode():
            output = model(tensor)
            prediction = cast("Any", output[0] if isinstance(output, tuple) else output)
            host = np.asarray(prediction.detach().float().cpu().numpy(), dtype=np.float32)
        return {"predictions": cast("Array", np.ascontiguousarray(host))}

    return infer


def _letterbox_geometria(
    height: int, width: int, imgsz: int
) -> tuple[float, tuple[float, float], tuple[int, int]]:
    """O ganho, o enchimento e o tamanho redimensionado, num lugar so.

    Os dois letterbox precisam da MESMA geometria, porque `_restore_box` desfaz o ganho e o
    enchimento na volta. Se cada caminho calculasse por conta, uma divergencia de arredondamento
    poria a caixa no espaco errado, e o resultado continuaria sendo quatro floats plausiveis.
    """
    target_height, target_width = _letterbox_shape(height, width, imgsz)
    gain = min(target_height / height, target_width / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    pad_x = (target_width - resized_width) / 2.0
    pad_y = (target_height - resized_height) / 2.0
    return gain, (pad_x, pad_y), (resized_height, resized_width)


def _enchimento(pad: float) -> tuple[int, int]:
    """Os dois lados do enchimento, com o desempate de 0,1 que o Ultralytics usa."""
    return round(pad - 0.1), round(pad + 0.1)


def _letterbox_tensor(
    image: np.ndarray[Any, Any], imgsz: int, modo: str = "cv2"
) -> tuple[np.ndarray[Any, Any], float, tuple[float, float]]:
    """Despacha entre o letterbox da CPU e o do device, com a mesma saida em numpy."""
    if modo == "cuda":
        return _letterbox_em_cuda(image, imgsz)
    return _letterbox_em_cv2(image, imgsz)


def _letterbox_em_cv2(
    image: np.ndarray[Any, Any], imgsz: int
) -> tuple[np.ndarray[Any, Any], float, tuple[float, float]]:
    height, width = image.shape[:2]
    gain, (pad_x, pad_y), (resized_height, resized_width) = _letterbox_geometria(
        height, width, imgsz
    )
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left, right = _enchimento(pad_x)
    top, bottom = _enchimento(pad_y)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    rgb = padded[:, :, ::-1].transpose(2, 0, 1)[None]
    tensor = np.ascontiguousarray(rgb, dtype=np.float32) / np.float32(255.0)
    return tensor, gain, (pad_x, pad_y)


def _letterbox_em_cuda(
    image: np.ndarray[Any, Any], imgsz: int
) -> tuple[np.ndarray[Any, Any], float, tuple[float, float]]:
    """O mesmo letterbox no device, subindo uint8 e descendo float32.

    A ordem importa para a banda de memoria: sobe o quadro em uint8, que sao 2 MiB por ladrilho, e
    a conversao para float32 acontece no device. Subir ja convertido custaria 19 MiB por ladrilho,
    dez vezes mais trafego, e o ganho todo estaria nisso.

    A interpolacao nao e a do `cv2`: o desvio maximo medido por pixel e 0,003 na escala 0..1. Por
    isso este caminho e declarado no TOML e nao herdado, e a divergencia de deteccao precisa ser
    medida antes de virar padrao.
    """
    torch = cast("Any", import_module("torch"))
    funcional = cast("Any", import_module("torch.nn.functional"))
    height, width = image.shape[:2]
    gain, (pad_x, pad_y), (resized_height, resized_width) = _letterbox_geometria(
        height, width, imgsz
    )
    device = torch.from_numpy(np.ascontiguousarray(image)).cuda()
    rgb = device[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float()
    escalado = funcional.interpolate(
        rgb, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    left, right = _enchimento(pad_x)
    top, bottom = _enchimento(pad_y)
    enchido = funcional.pad(escalado, (left, right, top, bottom), value=114.0)
    tensor = (enchido / 255.0).cpu().numpy()
    return np.ascontiguousarray(tensor, dtype=np.float32), gain, (pad_x, pad_y)


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


def _pytorch_backend(
    config: DetectorConfig, input_spec: TensorSpec, output_spec: TensorSpec
) -> tuple[InferenceBackend, RuntimeArtifact]:
    path = Path(config.weights)
    backend = PyTorchBackend(
        provider="ultralytics",
        device=config.device,
        loader=carregador_ultralytics(config.quantize),
        runtime_versions=(
            ("torch", version("torch")),
            ("ultralytics", version("ultralytics")),
        ),
    )
    artifact = RuntimeArtifact(
        path=path,
        sha256=sha256_of(path),
        format="pt",
        parent_onnx_sha256=None,
        inputs=(input_spec,),
        outputs=(output_spec,),
    )
    return backend, artifact


def _engine_backend(
    config: DetectorConfig, input_spec: TensorSpec, output_spec: TensorSpec
) -> tuple[InferenceBackend, RuntimeArtifact]:
    """Monta o caminho TensorRT, e recusa cedo o que so falharia na primeira inferencia.

    O formato de 64 hex do `engine_onnx_sha256` e checado aqui e nao no `RuntimeArtifact` porque
    la o campo so precisa existir; o formato e o que separa "declarei a procedencia" de "colei
    qualquer coisa". O `quantize` levanta em vez de ser ignorado: a precisao de um plano foi
    fixada na construcao, e aceitar o ajuste calado faria o TOML descrever algo que nao acontece.
    """
    from edge.trt import tensorrt_loader

    plano = Path(config.engine)
    if not plano.is_file():
        raise AjusteDesconhecido(f"engine: {plano} nao existe")
    if re.fullmatch(r"[0-9a-f]{64}", config.engine_onnx_sha256) is None:
        raise AjusteDesconhecido(
            "engine_onnx_sha256: um plano exige o sha256 do ONNX que o gerou, com 64 hex"
        )
    if config.quantize != PADRAO:
        raise AjusteDesconhecido(
            f"quantize={config.quantize!r} nao se aplica a um plano: a precisao foi fixada "
            "quando ele foi construido, e declara-la aqui descreveria algo que nao acontece"
        )
    backend = TensorRTBackend(
        provider="tensorrt",
        device=config.device,
        loader=tensorrt_loader(),
        runtime_versions=(("tensorrt", version("tensorrt")),),
    )
    artifact = RuntimeArtifact(
        path=plano,
        sha256=sha256_of(plano),
        format="engine",
        parent_onnx_sha256=config.engine_onnx_sha256,
        inputs=(input_spec,),
        outputs=(output_spec,),
    )
    return backend, artifact
