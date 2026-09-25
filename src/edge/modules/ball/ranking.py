"""Escolhe qual candidato do detector e a bola, antes de o rastreador olhar o tempo.

O detector devolve varios candidatos por quadro e o rastreador so sabe desempatar por proximidade
da posicao prevista. Medido nas 328 posicoes anotadas das dobras 0 a 2: em 154 dos 224 erros da
linha de base o candidato certo estava na lista e o rastreador escolheu outro. O gargalo era a
escolha dentro do quadro, nao a continuidade entre quadros, e nenhum ajuste de limiar de
velocidade resolveu isso (o melhor chegou a 117 acertos contra 93).

Este modulo pontua cada candidato com uma regressao logistica sobre catorze medidas de aparencia
e, opcionalmente, cinco do residuo de movimento da camera. O rastreador continua dono do tempo:
ele recebe um unico candidato por quadro e segue decidindo lacuna, interpolacao e salto.

As catorze de aparencia saem do proprio quadro: logito da confianca, largura e altura em fracao da
altura do quadro, razao de aspecto, centro normalizado e seus quadrados, media e desvio de H, S e V
num recorte de raio igual ao maior lado da caixa. Elas separam bola de linha do campo, de mancha do
gramado e de parte clara do uniforme, que e o que o detector confunde.

As cinco de movimento comparam o quadro com o anterior alinhado por homografia: um objeto que se
move destoa do fundo estatico depois que o movimento da camera e removido. Elas custam a estimativa
ORB/RANSAC por quadro e por isso sao opcionais; `estimate_motion` recusa sozinha quando nao consegue
alinhar, e nesse caso as cinco entram zeradas, que e como o modelo foi ajustado.

Os coeficientes vivem em `candidate-ranker-v1.json`, ao lado deste arquivo. Sao 19 pares de media e
escala mais 20 coeficientes ajustados nas seis partidas de desenvolvimento; nao ha quadro, recorte
nem identificador do parceiro dentro deles. A estimativa de generalizacao que sustenta o padrao e
`leave-one-match-out` sobre essas mesmas seis partidas, e esta na pagina do modelo, nao aqui.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from contracts import BallDetection

ARTIFACT = Path(__file__).with_name("candidate-ranker-v1.json")
ARTIFACTS = {
    "aparencia-e-camera": ARTIFACT,
    "aparencia": Path(__file__).with_name("candidate-ranker-aparencia-v1.json"),
}
"""Os artefatos que o TOML escolhe por nome em `ball.ranking_artifact`.

O so de aparencia existe porque a estimativa de movimento da camera custa cerca de 15 s por clipe
no pipeline, e sem ela o ranking fica perto do com camera no piloto: fora da partida, 175 acertos
com dois candidatos contra 186 com um, e 93 na linha de base, em 317 visiveis.
"""

APPEARANCE_FEATURES = (
    "confidence_logit",
    "width_over_frame_height",
    "height_over_frame_height",
    "aspect_ratio",
    "center_x",
    "center_y",
    "center_x_squared",
    "center_y_squared",
    "hue_mean",
    "saturation_mean",
    "value_mean",
    "hue_std",
    "saturation_std",
    "value_std",
)

MOTION_FEATURES = (
    "motion_available",
    "residual_mean",
    "residual_p90",
    "residual_contrast",
    "residual_active_fraction",
)

MOTION_ZEROS = (0.0,) * len(MOTION_FEATURES)

MAX_MOTION_GAP_FRAMES = 12
"""Distancia maxima entre dois quadros para o residuo de movimento valer alguma coisa.

O ajuste usou pares de tres quadros a 59,94 fps, e a execucao real usa o conjunto `dense`, a 15 fps,
que da quatro. Doze e o triplo disso: acima dele a homografia entre os dois quadros ainda pode ser
estimavel, mas o residuo deixa de separar objeto movel de fundo porque o jogo inteiro andou.
"""

Image = np.ndarray[Any, Any]


class RankerError(ValueError): ...


@dataclass(frozen=True)
class RankerArtifact:
    """Media, escala e coeficientes de uma logistica, com os nomes das medidas na ordem do ajuste.

    Guardar os nomes junto e o que impede o erro silencioso: trocar a ordem das medidas no extrator
    sem trocar o arquivo daria numero plausivel e errado, e nenhum teste de forma pegaria.
    """

    version: str
    features: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    coefficient: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.features)
        if size == 0:
            raise RankerError("artefato sem nenhuma medida declarada")
        if len(self.center) != size or len(self.scale) != size:
            raise RankerError(f"center e scale precisam ter {size} valores")
        if len(self.coefficient) != size + 1:
            raise RankerError(
                f"coefficient precisa ter {size + 1} valores, o primeiro e o termo constante"
            )
        if any(value <= 0 or not math.isfinite(value) for value in self.scale):
            raise RankerError("scale tem valor nao positivo ou nao finito")
        if not all(math.isfinite(value) for value in (*self.center, *self.coefficient)):
            raise RankerError("artefato tem coeficiente nao finito")

    @property
    def uses_camera_motion(self) -> bool:
        return MOTION_FEATURES[0] in self.features

    def probability(self, values: Sequence[float]) -> float:
        if len(values) != len(self.features):
            raise RankerError(f"esperava {len(self.features)} medidas e recebeu {len(values)}")
        logit = self.coefficient[0]
        for value, mean, scale, weight in zip(
            tuple(values), self.center, self.scale, self.coefficient[1:], strict=True
        ):
            logit += weight * (value - mean) / scale
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


def load_ranker(path: Path = ARTIFACT) -> RankerArtifact:
    if not path.is_file():
        raise RankerError(f"{path} nao existe")
    document = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    missing = {"version", "features", "center", "scale", "coefficient"} - set(document)
    if missing:
        raise RankerError(f"{path.name} sem os campos {sorted(missing)}")
    return RankerArtifact(
        version=str(document["version"]),
        features=tuple(str(name) for name in document["features"]),
        center=tuple(float(value) for value in document["center"]),
        scale=tuple(float(value) for value in document["scale"]),
        coefficient=tuple(float(value) for value in document["coefficient"]),
    )


def appearance_features(
    image: Image, box: tuple[float, float, float, float], confidence: float
) -> list[float]:
    """As catorze medidas de aparencia, no espaco de pixel do quadro original."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise RankerError(f"caixa degenerada: {box!r}")
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    radius = max(box_width, box_height)
    patch = image[
        _clamp(center_y - radius, height) : _clamp(center_y + radius, height),
        _clamp(center_x - radius, width) : _clamp(center_x + radius, width),
    ]
    color = _patch_color(patch)
    normalized_x, normalized_y = center_x / width, center_y / height
    probability = min(max(confidence, 1e-6), 1 - 1e-6)
    return [
        math.log(probability / (1 - probability)),
        box_width / height,
        box_height / height,
        box_width / box_height,
        normalized_x,
        normalized_y,
        normalized_x * normalized_x,
        normalized_y * normalized_y,
        *color,
    ]


def _clamp(value: float, limit: int) -> int:
    """Sem o piso em zero um limite negativo viraria indice a partir do fim, e uma caixa inteira
    fora do quadro produziria recorte grande em vez de recorte vazio."""
    return max(0, min(limit, round(value)))


def _patch_color(patch: Image) -> list[float]:
    """Media e desvio de H, S e V. Recorte vazio nunca acontece com candidato dentro do quadro,
    mas cair em `nan` aqui contaminaria a pontuacao inteira sem erro visivel."""
    if patch.size == 0:
        return [0.0] * 6
    hsv = cast("Image", cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)).astype(float)
    return [*hsv.mean(axis=(0, 1)).tolist(), *hsv.std(axis=(0, 1)).tolist()]


def estimate_motion(source: Image | None, target: Image | None) -> Image | None:
    """Homografia que leva `source` em `target`, ou `None` quando a estimativa nao se sustenta.

    Cada recusa e um caso real medido no piloto: pouca textura, poucos pares, poucos inliers,
    correspondencias concentradas num canto, erro de reprojecao alto e quadrilatero degenerado.
    Recusar e o comportamento certo porque o consumidor sabe lidar com residuo ausente, e uma
    homografia ruim viraria residuo ruim sem nenhum aviso.
    """
    if source is None or target is None or source.shape != target.shape:
        return None
    height, width = source.shape[:2]
    scale = min(1.0, 1280 / width)
    size = (round(width * scale), round(height * scale))
    before, left_descriptors = _orb(cv2.resize(source, size))
    after, right_descriptors = _orb(cv2.resize(target, size))
    if left_descriptors is None or right_descriptors is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_descriptors, right_descriptors, k=2)
    matches = [
        pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(matches) < 24:
        return None
    left = np.asarray([before[match.queryIdx].pt for match in matches], np.float32)
    right = np.asarray([after[match.trainIdx].pt for match in matches], np.float32)
    matrix, mask = cast(
        "tuple[Image | None, Image | None]", cv2.findHomography(left, right, cv2.RANSAC, 3.0)
    )
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 16 or inliers.mean() < 0.5:
        return None
    if cv2.contourArea(cv2.convexHull(left[inliers])) / (size[0] * size[1]) < 0.03:
        return None
    projected = cast(
        "Image", cv2.perspectiveTransform(left[inliers].reshape(-1, 1, 2), matrix)
    ).reshape(-1, 2)
    if _median_norm(projected - right[inliers]) / scale > 3:
        return None
    corners: Image = np.array(
        [[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]], np.float32
    )
    moved = cast("Image", cv2.perspectiveTransform(corners, matrix))[0]
    area = cv2.contourArea(moved, oriented=True) / (size[0] * size[1])
    if not 0.3 < area < 3 or not cv2.isContourConvex(moved):
        return None
    scaling = np.diag([scale, scale, 1.0])
    return cast("Image", np.linalg.inv(scaling) @ matrix @ scaling)


def _median_norm(values: Image) -> float:
    norms = cast("Image", np.linalg.norm(values, axis=1))
    return float(np.median(norms))


def _orb(image: Image) -> tuple[Any, Any]:
    """ORB restrito ao gramado: fora dele a plateia e a publicidade dominam os pontos e a
    homografia passa a descrever a arquibancada, que nao se move junto com o campo."""
    hsv = cast("Image", cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    mask = cv2.inRange(hsv, np.array([20, 25, 20]), np.array([105, 255, 255]))
    mask = cv2.dilate(mask, np.ones((15, 15), dtype=np.uint8))
    return cv2.ORB.create(nfeatures=5000, fastThreshold=8).detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask
    )


def motion_residual(
    source: Image, target: Image, matrix: Image | None
) -> tuple[Image, Image] | None:
    """Diferenca absoluta entre o quadro anterior alinhado e o atual, com a mascara do que e valido.

    A mascara existe porque a borda do quadro alinhado nao tem origem: contar aquilo como diferenca
    marcaria como movel qualquer candidato encostado na lateral.
    """
    if matrix is None:
        return None
    size = (target.shape[1], target.shape[0])
    left = cast("Image", cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
    right = cast("Image", cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
    warped = cv2.warpPerspective(left, matrix, size)
    ones: Image = np.ones((int(left.shape[0]), int(left.shape[1])), dtype=np.uint8)
    valid = cv2.warpPerspective(ones, matrix, size, flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8)).astype(bool)
    delta = (
        cv2.absdiff(cv2.GaussianBlur(warped, (3, 3), 0), cv2.GaussianBlur(right, (3, 3), 0)).astype(
            float
        )
        / 255
    )
    return delta, valid


def motion_features(
    box: tuple[float, float, float, float], difference: tuple[Image, Image] | None
) -> list[float]:
    """As cinco medidas de movimento, zeradas quando nao ha residuo confiavel para esta caixa."""
    if difference is None:
        return list(MOTION_ZEROS)
    delta, valid = difference
    height, width = delta.shape
    x1, y1 = int(math.floor(box[0])), int(math.floor(box[1]))
    x2, y2 = int(math.ceil(box[2])), int(math.ceil(box[3]))
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        return list(MOTION_ZEROS)
    if not valid[y1:y2, x1:x2].all():
        return list(MOTION_ZEROS)
    radius = max(x2 - x1, y2 - y1)
    inner = delta[y1:y2, x1:x2]
    top, bottom = max(0, y1 - radius), min(height, y2 + radius)
    left, right = max(0, x1 - radius), min(width, x2 + radius)
    around = delta[top:bottom, left:right][valid[top:bottom, left:right]]
    return [
        1.0,
        float(inner.mean()),
        float(np.percentile(inner, 90)),
        float(inner.mean() - around.mean()),
        float((inner > 20 / 255).mean()),
    ]


class CandidateRanker:
    """Estado de um clipe: guarda o quadro anterior e devolve os melhores candidatos, em ordem.

    Quantos e decidido por `keep`, e a ordem importa: quem consome tem de respeitar a ordem em vez
    de reescolher. Medido nas 328 posicoes anotadas, entregar os dois melhores e deixar o rastreador
    desempatar por proximidade da posicao prevista cai para 161 acertos, contra 177 respeitando a
    ordem e 186 entregando so o primeiro. A ordem e a decisao; ela nao pode ser metade aqui e metade
    no consumidor.

    O empate de probabilidade e resolvido pela posicao original na lista do detector, para a saida
    nao depender da ordem de iteracao de um dicionario nem do `sort` da plataforma.
    """

    def __init__(self, artifact: RankerArtifact, *, keep: int = 1) -> None:
        if keep < 1:
            raise RankerError(f"keep={keep}, e o ranking entrega pelo menos um candidato")
        self._artifact = artifact
        self._keep = keep
        self._camera_motion = artifact.uses_camera_motion
        self._previous: Image | None = None
        self._previous_index: int | None = None
        self._scores: list[float] = []
        self._motion_frames = 0

    def reset(self) -> None:
        """Esquece o quadro anterior, porque nao ha o que alinhar atraves de um corte de camera.

        Nao ha chamador em producao hoje, e isso e uma lacuna declarada e nao um esquecimento: o
        pipeline nao tem sinal de corte de camera, `BallObservation.camera_cut` chega sempre falso,
        e calibrar um detector de corte esta na lista de pendencias da pagina. Comeco de
        clipe nao precisa disto, porque `_observar` constroi um ranker novo por lote.

        Quem usa e a auditoria, que tem os cortes marcados na trajetoria congelada e precisa deles
        para reproduzir os 186 acertos medidos.
        """
        self._previous = None
        self._previous_index = None

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(self._scores)

    @property
    def motion_frames(self) -> int:
        return self._motion_frames

    def rank(
        self, image: Image, frame_index: int, candidates: Sequence[BallDetection]
    ) -> tuple[BallDetection, ...]:
        if not candidates:
            self._guarda(image, frame_index)
            return ()
        difference = self._difference(image, frame_index)
        self._guarda(image, frame_index)
        scored: list[tuple[float, int, BallDetection]] = []
        for position, candidate in enumerate(candidates):
            box = (candidate.box.x1, candidate.box.y1, candidate.box.x2, candidate.box.y2)
            values = appearance_features(image, box, candidate.confidence)
            if self._camera_motion:
                values.extend(motion_features(box, difference))
            scored.append((self._artifact.probability(values), position, candidate))
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        self._scores.append(scored[0][0])
        return tuple(entry[2] for entry in scored[: self._keep])

    def _guarda(self, image: Image, frame_index: int) -> None:
        """Guarda o quadro mesmo sem candidato, para o proximo nao alinhar contra um quadro velho.

        Sair antes disso economizaria a estimativa e mudaria o significado do resultado: o passo
        entre quadros deixaria de ser o passo da amostragem.
        """
        if self._camera_motion:
            self._previous, self._previous_index = image, frame_index

    def _difference(self, image: Image, frame_index: int) -> tuple[Image, Image] | None:
        if not self._camera_motion or self._previous is None or self._previous_index is None:
            return None
        step = frame_index - self._previous_index
        if step <= 0 or step > MAX_MOTION_GAP_FRAMES:
            return None
        difference = motion_residual(self._previous, image, estimate_motion(self._previous, image))
        if difference is not None:
            self._motion_frames += 1
        return difference
