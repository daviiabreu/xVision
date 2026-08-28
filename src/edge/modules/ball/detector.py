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
from typing import Any, Protocol, cast

from contracts import BallDetection, Box
from edge.modules.base import ModuleError
from edge.modules.inference import as_deteccoes

FrameArray = Any
"""Quadro decodificado (array HxWxC). Tipado como Any para nao exigir numpy no import do modulo."""


class AjusteDesconhecido(ModuleError): ...


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

    def __init__(self, model: Any, config: DetectorConfig) -> None:
        self._model = model
        self._config = config

    @classmethod
    def from_config(cls, config: DetectorConfig) -> YoloBallDetector:
        """O `ultralytics` e o extra opcional `ball`, resolvido tarde e por nome.

        Import tardio porque a suite de conformidade instancia o modulo sem peso nenhum, e um
        import no topo do arquivo obrigaria torch para rodar um teste que nao usa torch.
        `import_module` em vez de `import ultralytics` porque o pyright roda em modo estrito e
        trata todo simbolo de import nao resolvido como desconhecido, o que exigiria supressao
        linha a linha em cada uso.
        """
        yolo = cast("Any", import_module("ultralytics")).YOLO
        modelo = yolo(config.weights)
        return cls(modelo, replace(config, ball_class_id=_classe_da_bola(modelo, config)))

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
        if config.ladrilhos <= 1:
            return self._passada(frame, 0.0, 0.0)
        achadas: list[tuple[Box, float, int]] = []
        for pedaco, x, y in pedacos(frame, config.ladrilhos, config.sobreposicao_px):
            achadas.extend(self._passada(pedaco, x, y))
        return sem_repetidas(achadas, config.nms_iou)

    def _passada(
        self, frame: FrameArray, dx: float, dy: float
    ) -> tuple[tuple[Box, float, int], ...]:
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
        config = self._config
        results = self._model(
            frame,
            imgsz=config.imgsz,
            conf=config.conf,
            device=config.device,
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
        """A bola de maior confianca do quadro, ignorando continuidade de movimento."""
        candidates = self.detect_candidates(frame)
        return max(candidates, key=lambda d: d.confidence, default=None)


NOMES_DE_BOLA = ("ball", "sports ball", "soccer ball", "bola")


def _classe_da_bola(modelo: Any, config: DetectorConfig) -> int:
    """O peso declara os nomes das proprias classes, e a configuracao nao precisa saber deles.

    O `yolov8n` do COCO chama a bola de `sports ball` na classe 32; o peso de futebol do
    roboflow/sports chama de `ball` na classe 0. Cravar o numero na configuracao faria a troca de
    peso exigir uma segunda edicao, e esquecer dela nao levanta erro nenhum: o detector passa a
    procurar outra coisa e devolve zero bola, ou pior, devolve arbitro.
    """
    nomes = getattr(modelo, "names", None)
    if not isinstance(nomes, dict):
        return config.ball_class_id
    for indice, nome in cast("dict[int, str]", nomes).items():
        if str(nome).strip().lower() in NOMES_DE_BOLA:
            return int(indice)
    return config.ball_class_id
