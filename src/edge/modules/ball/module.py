"""A especie `ball` como modulo de percepcao, em cima do detector e do rastreador do Gabriel.

O que este arquivo acrescenta ao trabalho da branch `feat/ball-tracking` e so a casca: la o
detector e o rastreador eram dirigidos por um script que abria video por conta propria, e a suite
de conformidade proibe exatamente isso. Aqui os quadros chegam prontos pelo `FrameBatch` e a saida
sai no formato que `edge.modules.records` cobra.

Quatro pontos que a passagem para a fundacao obrigou a resolver.

O primeiro e o espaco da caixa. O detector ve o tensor ja redimensionado, e `records.py` cobra
pixel do quadro original. A volta usa `edge.frames.preprocess.to_source_box` com o `source_size`
que o `FrameProvider` agora carrega no lote, e ela acontece **antes** do filtro de forma, para que
os limites de fracao de altura que o Gabriel calibrou continuem significando o que ele escreveu.

O segundo e o ponto interpolado. O `TrackPoint` interpolado sai com `box=None` e `confidence=None`,
e o `assemble` faz `Box(*registro["box"])` fora do `try` da etapa: um `None` ali derruba a execucao
inteira do clipe. O modulo sintetiza um quadrado de `LADO_PADRAO_PX` em volta do centro
interpolado, e declara `source` no registro. Posicao interpolada e informacao util para a
trajetoria que o modulo `shot` consome, e escondê-la custaria mais do que declara-la.

O terceiro e a unidade de tempo. O gate de salto do rastreador e em pixel por segundo, e alimenta-lo
com indice de quadro no lugar de segundo o afrouxa pelo fator do fps: medido nos clipes do parceiro,
com o gate assim desligado o rastreador aceitava salto de 125 px por quadro, contra os cerca de 14
que a fisica permite para bola a 30 m/s num quadro de 1920 px. O `source_fps` do lote resolve, e
sem ele o modulo cai para unidade de quadro em vez de mentir.

O quarto e a injecao do detector. O YOLO entra por parametro, com a construcao adiada, porque a
suite de conformidade instancia o modulo sem peso nenhum e porque o nucleo do rastreador ja era
exercitavel sem GPU na branch de origem. Quem roda em producao passa o detector de verdade.

O quinto veio depois, da primeira calibracao: `conf`, `imgsz` e os limites de forma vinham cravados
no `DetectorConfig` e nao tinham como ser medidos sem editar `src/`. Agora chegam por `configure`,
com o vocabulario e a validacao no proprio `DetectorConfig`, e o limite efetivo e o mesmo nas duas
passadas do filtro de forma, a do tensor e a do quadro original.

O sexto e a escolha entre os candidatos, que ate a emenda 15 era do rastreador e so olhava
proximidade da posicao prevista. Agora `edge.modules.ball.ranking` pontua cada candidato e o
rastreador recebe os melhores, na ordem do ranking.

O padrao do codigo e desligado, e quem liga e `config/pipeline.models.toml`, pelo mesmo motivo de
`conf` e `imgsz`: o valor escolhido por medicao precisa aparecer no arquivo que descreve a execucao,
onde a proxima pessoa le, e nao num argumento nomeado que ninguem abre. A escolha esta sob revisao,
porque o ranking dobra o acerto de posicao e custa cerca de 2,5 vezes o tempo do modulo.

Medido fora da dobra nas seis partidas de desenvolvimento, ele levou o acerto de 93 para 186 em 317
posicoes visiveis, com dois quadros sem previsao contra um da linha de base. O ranking precisa do
pixel do quadro original, e nao do tensor: e por isso que
`to_source_image` existe, e e por isso que sem `source_size` o modulo nao pontua, em vez de pontuar
no espaco errado.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from time import perf_counter
from typing import Any, cast

import numpy as np

from contracts import BallDetection, Box
from edge.artifacts import entrypoint_of
from edge.frames.preprocess import to_source_box, to_source_image
from edge.modules.ball.detector import (
    BallDetector,
    DetectorConfig,
    DetectorTelemetry,
    is_ball_shaped,
)
from edge.modules.ball.ranking import CandidateRanker, RankerArtifact, load_ranker
from edge.modules.ball.tracking import BallObservation, BallTracker, TrackerConfig, TrackPoint
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

PREPROCESS = PreprocessSpec(width=1920, height=1088, color="bgr", letterbox=True, normalize="none")
"""Resolucao nativa, porque reduzir antes de detectar joga fora um terco do diametro da bola.

A bola tem cerca de 13 px de lado no quadro de 1920 por 1080. O valor anterior era 1280 por 736,
uma escala de 0,667, que a levava para 8,7 px: uma celula do nivel P3 do YOLO, que tem passo 8.
Medido no quadro do contato dos 24 clipes da dobra de ajuste, escolhendo o candidato de maior
confianca, a passada cheia a 1280 acerta 12 e a 1920 acerta 15.

Nao e a resolucao de inferencia, e sim o que chega ao modulo. `imgsz` continua sendo do detector, e
com ladrilho ele e aplicado por pedaco. Sem o quadro inteiro em resolucao nativa nao ha o que
ladrilhar: o pedaco sairia de um tensor onde a informacao ja se perdeu.

1088 e o multiplo de 32 acima de 1080, e o letterbox acrescenta quatro linhas em cima e quatro
embaixo, que `to_source_box` desconta.

Isto deixa `ball` e `players` com pre-processamentos diferentes, entao a passada nao e mais
compartilhada. O compartilhamento ja tinha acabado quando `ball` ganhou peso proprio: a chave de
`SharedInference` e (peso, impressao do pre-processamento, quadro), e o peso ja era outro.
"""

INTERPOLADO = "interpolated"
DETECTADO = "detected"
CONFIANCA_INTERPOLADA = 0.0
LADO_PADRAO_PX = 10.0
"""Lado do quadrado sintetizado em volta de um centro interpolado, fixo e nao herdado.

Fixo porque o ponto interpolado nao carrega tamanho: `TrackPoint` interpolado tem `box=None`, e
guardar a ultima caixa vista faria a caixa sintetica mentir com mais aparencia de medida do que
mente hoje. Dez pixels e a ordem de grandeza certa e nada mais: a bola mede 13,5 a 14,1 px de lado
no quadro de 1920x1080 dos clipes do parceiro. Quem consome distingue os dois casos pelo campo
`source`, e nao pelo tamanho da caixa.
"""


class YoloBall:
    spec = ModuleSpec(
        kind="ball",
        name="yolo-ball",
        version="0.1.0",
        frames="coarse",
        preprocess=PREPROCESS,
    )

    def __init__(
        self,
        detector: BallDetector | None = None,
        *,
        detector_config: DetectorConfig | None = None,
        tracker_config: TrackerConfig | None = None,
        ranking: bool = False,
        ranking_candidates: int = 1,
    ) -> None:
        self._detector = detector
        self._detector_config = detector_config
        self._tracker_config = tracker_config or TrackerConfig()
        self._ranking = ranking
        self._ranking_candidates = ranking_candidates
        self._ranker: RankerArtifact | None = None
        self._ajustes: Mapping[str, object] = {}
        self._efetiva: DetectorConfig | None = detector_config
        self._peso = ""
        self._chave = ""
        self._pronto = False
        self._telemetry: dict[str, int | float] = {}
        self._ranking_stats: dict[str, int | float] = {}
        self._candidate_stats: dict[str, int | float] = {}

    @property
    def telemetry(self) -> Mapping[str, int | float]:
        return dict(self._telemetry)

    def fingerprint(self) -> str:
        """O ajuste em vigor, para o cache de percepcao nao servir medida de outro limiar.

        Sem isto, baixar `conf` de 0,25 para 0,10 no TOML deixava a chave do cache identica, porque
        ela cobre o arquivo do modulo e o digest do peso e nenhum dos dois muda. A execucao seguinte
        devolvia a percepcao antiga e o relatorio publicava o numero de antes como se fosse o novo.
        """
        ordenados = sorted((str(k), repr(v)) for k, v in self._ajustes.items())
        return (
            ";".join(f"{k}={v}" for k, v in ordenados)
            + f";tracker={asdict(self._tracker_config)!r}"
            + f";ranking={self._ranking!r},{self._ranking_candidates:d}"
        )

    def configure(self, options: Mapping[str, object]) -> None:
        """Guarda o ajuste em vez de aplicar, e separa o que e do detector do que e daqui.

        Guardar porque `configure` roda no registro, antes de `load`, e o peso so aparece la. Quem
        valida o vocabulario do detector e o proprio `DetectorConfig`, dono dele: o modulo nao tem
        como saber que `max_aspect` existe e `aspecto_maximo` nao.

        Os ajustes temporais e de ranking saem daqui antes, porque o `DetectorConfig` recusaria
        nome que nao e dele e a recusa apareceria como erro de digitacao do usuario.
        """
        detector_options = dict(options)
        config = self._tracker_config
        if "ranking" in detector_options:
            ranking = detector_options.pop("ranking")
            if not isinstance(ranking, bool):
                raise ValueError("ball.ranking espera booleano")
            self._ranking = ranking
        if "ranking_candidates" in detector_options:
            quantos = detector_options.pop("ranking_candidates")
            if isinstance(quantos, bool) or not isinstance(quantos, int) or quantos < 1:
                raise ValueError("ball.ranking_candidates espera inteiro maior ou igual a um")
            self._ranking_candidates = quantos
        if "max_ball_speed_px_per_s" in detector_options:
            speed = detector_options.pop("max_ball_speed_px_per_s")
            if (
                isinstance(speed, bool)
                or not isinstance(speed, int | float)
                or not math.isfinite(speed)
                or speed <= 0
            ):
                raise ValueError("ball.max_ball_speed_px_per_s espera numero finito positivo")
            config = replace(config, max_ball_speed_px_per_s=float(speed))
        self._tracker_config = config
        self._ajustes = detector_options

    def load(self) -> None:
        config = self._config_com_peso()
        if self._detector is None and config is not None:
            self._peso = config.weights
            self._chave = chave_do_cache(config)
            from edge.modules.ball.detector import YoloBallDetector

            self._detector = YoloBallDetector.from_config(config)
        sem_peso = DetectorConfig(weights="").com_ajustes(self._ajustes)
        self._efetiva = config if config is not None else sem_peso
        self._ranker = load_ranker() if self._ranking else None
        self._pronto = True

    def _tracker_efetivo(self, ordenado: bool) -> TrackerConfig:
        """Com o ranking ligado, a ordem dos candidatos e do ranking e o rastreador nao reescolhe.

        Reescolher por proximidade da posicao prevista desfaz a decisao: medido nas 328 posicoes
        anotadas, entregar os dois melhores e desempatar por proximidade cai para 161 acertos contra
        177 respeitando a ordem. Com um candidato so as duas regras coincidem, entao ligar isto
        junto com o ranking nao muda o caminho ja medido.

        O parametro e o que o `_observar` acabou de fazer, e nao o que o TOML pediu: sem
        `source_size` a caixa fica no espaco do tensor, o modulo nao pontua, e ai a ordem da lista e
        do detector. Respeitar uma ordem que ninguem estabeleceu seria pior do que a proximidade.
        """
        if not ordenado:
            return self._tracker_config
        return replace(self._tracker_config, candidate_order=True)

    def _config_com_peso(self) -> DetectorConfig | None:
        """O peso vem do manifesto declarado, e o ajuste vem da entrada do TOML sobre o padrao.

        O padrao do `DetectorConfig` e o de antes de qualquer calibracao, `conf = 0.25`, que e
        acima da confianca mediana da bola correta nos quadros de contato (0,154 medido em 38
        quadros anotados). Ele fica onde esta e quem mede o desce em
        `config/pipeline.models.toml`, porque o valor calibrado precisa aparecer no arquivo que
        descreve a execucao e nao no codigo do detector.
        """
        if self._detector_config is not None:
            return self._detector_config
        peso = entrypoint_of(self.spec.artifact)
        if peso is None:
            return None
        base = DetectorConfig(weights=str(peso), device=preferred_device())
        return base.com_ajustes(self._ajustes)

    def close(self) -> None:
        if self._detector is not None:
            close = getattr(self._detector, "close", None)
            if callable(close):
                close()
        self._detector = None
        self._ranker = None
        self._efetiva = self._detector_config
        self._peso = ""
        self._chave = ""
        self._pronto = False
        self._telemetry = {}
        self._candidate_stats = {}
        self._ranking_stats = {}

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        if not self._pronto:
            raise ModuleNotLoaded(self.spec.name)
        check_preprocess(batch, self.spec.preprocess)
        self._ranking_stats = {
            "ranked_frames": 0,
            "ranking_motion_frames": 0,
            "ranking_score_sum": 0.0,
        }
        self._candidate_stats = {
            "tiles_executed": 0,
            "raw_candidates": 0,
            "candidates_after_nms": 0,
            "candidates_after_filter": 0,
            "inference_ms": 0.0,
            "nms_ms": 0.0,
        }
        if not batch.indices:
            self._telemetry = self._report(0, 0.0, 0.0, BallTracker().telemetry)
            return ModuleOutput(kind=self.spec.kind, frames=(), records=())

        detector_start = perf_counter()
        observacoes = self._observar(batch)
        detector_ms = (perf_counter() - detector_start) * 1000.0
        tracker = BallTracker(self._tracker_efetivo(bool(self._ranking_stats["ranked_frames"])))
        resolver_start = perf_counter()
        pontos = {p.frame_index: p for p in tracker.track(observacoes)}
        resolver_ms = (perf_counter() - resolver_start) * 1000.0
        self._telemetry = self._report(
            len(batch.indices), detector_ms, resolver_ms, tracker.telemetry
        )
        return ModuleOutput(
            kind=self.spec.kind,
            frames=batch.indices,
            records=tuple(registro_de(pontos.get(indice)) for indice in batch.indices),
        )

    def _observar(self, batch: FrameBatch) -> list[BallObservation]:
        """Um `BallObservation` por quadro, com os candidatos ja no espaco do quadro original.

        Sem detector o modulo nao inventa nada: todo quadro vira observacao sem candidato, o
        rastreador registra lacuna e a especie sai sem posicao. E o que a suite de conformidade
        exercita, e e tambem o que acontece quando o peso nao esta em disco.
        """
        fonte = batch.source_size
        taxa = batch.source_fps
        ranker = (
            None
            if self._ranker is None or fonte is None
            else CandidateRanker(self._ranker, keep=self._ranking_candidates)
        )
        observacoes: list[BallObservation] = []
        for posicao, indice in enumerate(batch.indices):
            candidatos = self._candidatos(batch.tensor[posicao], batch, fonte, indice)
            if ranker is not None and fonte is not None:
                original = to_source_image(batch.tensor[posicao], batch.preprocess, fonte)
                candidatos = ranker.rank(original, indice, candidatos)
                self._ranking_stats["ranked_frames"] += 1
            observacoes.append(
                BallObservation(
                    frame_index=indice,
                    timestamp_s=indice / taxa if taxa else float(indice),
                    candidates=candidatos,
                )
            )
        if ranker is not None:
            self._ranking_stats["ranking_motion_frames"] = ranker.motion_frames
            self._ranking_stats["ranking_score_sum"] = sum(ranker.scores)
        return observacoes

    def _candidatos(
        self,
        quadro: np.ndarray[Any, Any],
        batch: FrameBatch,
        fonte: tuple[int, int] | None,
        indice: int,
    ) -> tuple[BallDetection, ...]:
        if self._detector is None:
            return ()
        config = self._efetiva or DetectorConfig(weights="")
        detector = self._detector
        cruas = (
            shared().get(
                weights=self._chave,
                preprocess=batch.preprocess,
                frame_index=indice,
                compute=lambda: detector.detect_raw(quadro),
            )
            if self._peso
            else detector.detect_raw(quadro)
        )
        detector_telemetry = getattr(detector, "telemetry", None)
        if isinstance(detector_telemetry, DetectorTelemetry):
            self._accumulate_detector(detector_telemetry)
        else:
            config_tiles = max(1, config.ladrilhos) ** 2
            self._candidate_stats["tiles_executed"] += config_tiles
            self._candidate_stats["raw_candidates"] += len(cruas)
            self._candidate_stats["candidates_after_nms"] += len(cruas)
        altura_do_tensor = float(batch.tensor.shape[1])
        brutos = detector.candidates_from(cruas, altura_do_tensor)
        if fonte is None:
            self._candidate_stats["candidates_after_filter"] += len(brutos)
            return tuple(brutos)
        altura = float(fonte[1])
        no_espaco_do_quadro = (
            BallDetection(
                box=Box(*to_source_box(_as_tuple(d.box), batch.preprocess, fonte)),
                confidence=d.confidence,
                pitch_xy=d.pitch_xy,
            )
            for d in brutos
        )
        filtrados = tuple(
            d
            for d in no_espaco_do_quadro
            if is_ball_shaped(d.box, altura, config) and _center_inside(d.box, fonte)
        )
        self._candidate_stats["candidates_after_filter"] += len(filtrados)
        return filtrados

    def _accumulate_detector(self, telemetry: DetectorTelemetry) -> None:
        for name in ("tiles_executed", "raw_candidates", "candidates_after_nms"):
            self._candidate_stats[name] += int(getattr(telemetry, name))
        for name in ("inference_ms", "nms_ms"):
            self._candidate_stats[name] += float(getattr(telemetry, name))

    def _report(
        self,
        frames_requested: int,
        detector_ms: float,
        resolver_ms: float,
        tracker: object,
    ) -> dict[str, int | float]:
        report = dict(self._candidate_stats)
        report.update(self._ranking_stats)
        report.update(
            {
                "frames_requested": frames_requested,
                "detector_ms": detector_ms,
                "resolver_ms": resolver_ms,
            }
        )
        for name in (
            "detected_points",
            "interpolated_points",
            "missing_points",
            "segments",
            "resets",
            "rejected_jumps",
            "camera_cuts",
            "max_gap_frames",
        ):
            report[name] = int(getattr(tracker, name))
        return report


def chave_do_cache(config: DetectorConfig) -> str:
    """A chave do cache compartilhado precisa do `conf` e do `imgsz`, e nao so do caminho do peso.

    O `SharedInference` guarda por (peso, impressao do pre-processamento, quadro), e o
    `PreprocessSpec` nao carrega nem `conf` nem `imgsz`. Com o mesmo `yolov8n.pt` e o mesmo
    1280x736 em `ball` e em `players`, as duas especies casam na chave: `players` roda primeiro na
    ordem do TOML, grava a passada dele a `conf = 0.25` e `ball` consome aquilo em silencio. Sem
    esta chave, descer `conf` para 0,10 na configuracao da bola nao muda um unico candidato, e a
    medicao registra a melhora que nao aconteceu.

    O preco esta medido e e real: as duas especies deixam de compartilhar a passada, o peso roda
    duas vezes por quadro e a dobra de ajuste sobe de 68,3 s para 76,7 s de CPU nos mesmos 24
    clipes, cerca de 12%. Medido com `conf` igual nos dois lados, para isolar o custo da passada a
    mais do efeito do limiar novo. Passada repetida custa 12%; passada errada custa o numero.

    A configuracao de producao deixou de estar nesse caso: `ball` tem peso proprio, dedicado a bola,
    entao as chaves ja nao colidem por outro motivo e este preco nao esta sendo pago. A guarda fica
    porque voltar as duas especies ao mesmo peso e uma linha de configuracao, e a colisao voltaria
    calada.
    """
    return f"{config.weights}#conf={config.conf:g}#imgsz={config.imgsz:d}"


def _as_tuple(box: Box) -> tuple[float, float, float, float]:
    return (box.x1, box.y1, box.x2, box.y2)


def _center_inside(box: Box, size: tuple[int, int]) -> bool:
    center_x = (box.x1 + box.x2) / 2.0
    center_y = (box.y1 + box.y2) / 2.0
    return 0.0 <= center_x < size[0] and 0.0 <= center_y < size[1]


def registro_de(ponto: TrackPoint | None) -> tuple[dict[str, object], ...]:
    """Quadro sem posicao sai sem registro, e nunca com caixa de zeros.

    Caixa de zeros passaria em `check_records`, viraria `BallDetection` no `assemble` e chegaria a
    feature como bola no canto superior esquerdo do campo. Ausencia declarada e a unica saida que o
    consumidor consegue distinguir de uma medida.
    """
    if ponto is None or ponto.center is None:
        return ()
    registro: dict[str, object] = {
        "box": _as_tuple(ponto.box)
        if ponto.box is not None
        else _quadrado(ponto.center, LADO_PADRAO_PX),
        "confidence": (
            float(cast("Any", ponto.confidence)) if ponto.box is not None else CONFIANCA_INTERPOLADA
        ),
        "source": DETECTADO if ponto.box is not None else INTERPOLADO,
    }
    if ponto.trajectory_segment is not None:
        registro["trajectory_segment"] = ponto.trajectory_segment
    if ponto.gap_frames is not None:
        registro["gap_frames"] = ponto.gap_frames
    if ponto.velocity_xy_px_s is not None:
        registro["velocity_xy_px_s"] = ponto.velocity_xy_px_s
    if ponto.velocity_source is not None:
        registro["velocity_source"] = ponto.velocity_source
    return (registro,)


def _quadrado(centro: Sequence[float], lado: float) -> tuple[float, float, float, float]:
    x, y = centro[0], centro[1]
    metade = lado / 2.0
    return (x - metade, y - metade, x + metade, y + metade)
