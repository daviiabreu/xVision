"""Segunda passada cega da anotacao da bola, e a concordancia dela com a primeira.

A primeira passada do piloto foi assistida: o detector propunha a posicao e a pessoa confirmava ou
corrigia. Isso mede a bola, mas nao mede quanto a proposta puxou a marcacao para perto dela. A spec
da bola pede reanotar, sem ver nada disso, todos os quadros de contato e 25% dos demais, e reportar
a discordancia do centro em diametros aparentes e a concordancia de visibilidade.

A ferramenta serve os quadros por HTTP local, para funcionar sem backend de janela: o wheel
headless do lock nao abre `cv2.imshow`. Escuta so em 127.0.0.1 e grava em `var/`, porque cada
quadro e material restrito da Footstats.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from edge.frames.provider import MissingFrame, VideoSource
from mlops.ball_trajectory import SCHEMA, BallTrajectoryLabel, Visibility

FRACAO = 0.25
SEMENTE = 20260922
LIMIAR_CENTRO = 0.5
"""Discordancia mediana do centro, em diametros aparentes, acima da qual o protocolo e revisto."""
LIMIAR_VISIBILIDADE = 0.80
BANDEIRAS = ("human-annotated", "blind-second-pass")
CONTEXTO = 30
"""Quantos quadros nativos para cada lado a pessoa pode ver, para achar a bola pelo movimento.

A marca so e aceita no quadro alvo. O contexto existe porque a primeira passada teve o clipe
inteiro, e negar movimento na segunda mediria a falta dele, nao a concordancia.
"""
VISIBILIDADES: tuple[Visibility, ...] = ("visible", "occluded", "not-visible", "ambiguous")
PAGINA = Path(__file__).with_name("bola_cega.html")

Chave = tuple[str, int]


class BolaCegaError(ValueError): ...


@dataclass(frozen=True)
class Alvo:
    clip_id: str
    frame_index: int
    contato: bool


@dataclass(frozen=True)
class Marca:
    visibility: Visibility
    center_xy: tuple[float, float] | None
    apparent_diameter_px: float | None
    motion_blur: bool
    camera_cut: bool


@dataclass(frozen=True)
class Concordancia:
    n_pares: int
    n_contatos: int
    visibilidade: float
    kappa: float | None
    n_centros: int
    centro_mediana: float | None
    centro_p95: float | None
    centro_max: float | None
    contato_mediana: float | None
    diametro_mediana: float | None
    confusao: Mapping[str, int]

    @property
    def aprovada(self) -> bool | None:
        if self.centro_mediana is None:
            return None
        return self.centro_mediana <= LIMIAR_CENTRO and self.visibilidade >= LIMIAR_VISIBILIDADE

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "n_pares": self.n_pares,
            "n_contatos": self.n_contatos,
            "visibilidade": self.visibilidade,
            "kappa": self.kappa,
            "n_centros": self.n_centros,
            "centro_mediana_diametros": self.centro_mediana,
            "centro_p95_diametros": self.centro_p95,
            "centro_max_diametros": self.centro_max,
            "contato_mediana_diametros": self.contato_mediana,
            "diametro_mediana_relativa": self.diametro_mediana,
            "confusao": dict(self.confusao),
            "limiares": {"centro": LIMIAR_CENTRO, "visibilidade": LIMIAR_VISIBILIDADE},
            "aprovada": self.aprovada,
        }


def contatos_da_fila(fila: Path) -> dict[str, int]:
    """O quadro de contato de cada clipe do piloto, como a fila da primeira passada registrou."""
    documento = cast("dict[str, Any]", json.loads(fila.read_text(encoding="utf-8")))
    contatos: dict[str, int] = {}
    for entrada in cast("list[dict[str, Any]]", documento.get("ball", [])):
        clip_id, contato = str(entrada["clip_id"]), int(entrada["contact_frame"])
        if contatos.setdefault(clip_id, contato) != contato:
            raise BolaCegaError(f"a fila da dois contatos para o clipe {clip_id}")
    if not contatos:
        raise BolaCegaError(f"{fila} nao tem entrada de bola")
    return contatos


def amostra(
    primeira: Sequence[BallTrajectoryLabel],
    contatos: Mapping[str, int],
    *,
    fracao: float = FRACAO,
    semente: int = SEMENTE,
) -> tuple[Alvo, ...]:
    """Todo contato e `fracao` dos demais quadros de cada clipe, embaralhados entre clipes.

    O sorteio e por clipe para nenhum clipe ficar sem amostra. A ordem final mistura os clipes para
    o quadro anterior nao servir de ancora para o seguinte.
    """
    sorteio = random.Random(semente)
    por_clipe: dict[str, list[int]] = {}
    for label in primeira:
        por_clipe.setdefault(label.clip_id, []).append(label.frame_index)
    alvos: list[Alvo] = []
    for clip_id in sorted(por_clipe):
        quadros = sorted(por_clipe[clip_id])
        contato = contatos.get(clip_id)
        if contato is None or contato not in quadros:
            raise BolaCegaError(f"o contato do clipe {clip_id} nao esta na primeira passada")
        demais = [quadro for quadro in quadros if quadro != contato]
        alvos.append(Alvo(clip_id, contato, True))
        escolhidos = sorteio.sample(demais, round(len(demais) * fracao))
        alvos.extend(Alvo(clip_id, quadro, False) for quadro in sorted(escolhidos))
    sorteio.shuffle(alvos)
    return tuple(alvos)


def registro(
    label: BallTrajectoryLabel, marca: Marca, annotator: str, agora: datetime
) -> dict[str, object]:
    """O documento `ball-trajectory-v1` da segunda passada, com a identidade da primeira.

    Da primeira sai so o que identifica o quadro. Posicao, diametro e visibilidade vem da marca.
    """
    visivel = marca.visibility == "visible"
    if visivel and (marca.center_xy is None or not marca.apparent_diameter_px):
        raise BolaCegaError("bola visivel precisa de centro e diametro")
    if not annotator:
        raise BolaCegaError("a segunda passada precisa dizer quem anotou")
    centro = marca.center_xy if visivel else None
    return {
        "schema_version": "1.0",
        "clip_id": label.clip_id,
        "match_id": label.match_id,
        "video_sha256": label.video_sha256,
        "frame_index": label.frame_index,
        "timestamp_s": label.timestamp_s,
        "center_x": None if centro is None else float(centro[0]),
        "center_y": None if centro is None else float(centro[1]),
        "apparent_diameter_px": float(marca.apparent_diameter_px or 0) if visivel else None,
        "visibility": marca.visibility,
        "trajectory_segment": 0 if visivel else None,
        "is_keyframe": True,
        "motion_blur": marca.motion_blur,
        "camera_cut": marca.camera_cut,
        "annotator": annotator,
        "reviewed_by": None,
        "created_at": agora.isoformat(),
        "quality_flags": list(BANDEIRAS),
    }


def grava(destino: Path, documento: Mapping[str, object]) -> None:
    """Valida contra o schema e reescreve o arquivo ordenado, trocando a marca do mesmo quadro."""
    validador = Draft202012Validator(
        json.loads(SCHEMA.read_text(encoding="utf-8")), format_checker=FormatChecker()
    )
    erros = list(cast("Any", validador).iter_errors(dict(documento)))
    if erros:
        raise BolaCegaError(f"marca fora do schema: {erros[0].message}")
    documentos = _documentos(destino)
    documentos[(str(documento["clip_id"]), cast("int", documento["frame_index"]))] = dict(documento)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".tmp")
    linhas = (json.dumps(documentos[chave], ensure_ascii=False) for chave in sorted(documentos))
    temporario.write_text("".join(f"{linha}\n" for linha in linhas), encoding="utf-8")
    temporario.replace(destino)


def marcas_gravadas(destino: Path) -> dict[Chave, dict[str, Any]]:
    return _documentos(destino)


def medir(
    primeira: Sequence[BallTrajectoryLabel],
    segunda: Sequence[BallTrajectoryLabel],
    contatos: Mapping[str, int],
) -> Concordancia:
    """Compara as duas passadas nos quadros que ambas marcaram.

    A distancia do centro e dividida pelo diametro da primeira passada, que e a mesma regua do PCK
    da avaliacao: discordar meio diametro e discordar menos do que o avaliador tolera.
    """
    base = {(label.clip_id, label.frame_index): label for label in primeira}
    pares = [
        (base[(label.clip_id, label.frame_index)], label)
        for label in segunda
        if (label.clip_id, label.frame_index) in base
    ]
    if not pares:
        raise BolaCegaError("nenhum quadro da segunda passada existe na primeira")
    iguais = sum(a.visibility == b.visibility for a, b in pares)
    confusao = Counter(f"{a.visibility}->{b.visibility}" for a, b in pares)
    centros: list[float] = []
    contato: list[float] = []
    diametros: list[float] = []
    for a, b in pares:
        if a.center_xy is None or b.center_xy is None or not a.apparent_diameter_px:
            continue
        distancia = float(
            np.hypot(a.center_xy[0] - b.center_xy[0], a.center_xy[1] - b.center_xy[1])
        )
        centros.append(distancia / a.apparent_diameter_px)
        if contatos.get(a.clip_id) == a.frame_index:
            contato.append(distancia / a.apparent_diameter_px)
        if b.apparent_diameter_px:
            diametros.append(
                abs(b.apparent_diameter_px - a.apparent_diameter_px) / a.apparent_diameter_px
            )
    return Concordancia(
        n_pares=len(pares),
        n_contatos=sum(contatos.get(a.clip_id) == a.frame_index for a, _ in pares),
        visibilidade=iguais / len(pares),
        kappa=_kappa([(a.visibility, b.visibility) for a, b in pares]),
        n_centros=len(centros),
        centro_mediana=_percentil(centros, 50),
        centro_p95=_percentil(centros, 95),
        centro_max=max(centros) if centros else None,
        contato_mediana=_percentil(contato, 50),
        diametro_mediana=_percentil(diametros, 50),
        confusao=dict(sorted(confusao.items())),
    )


class Sessao:
    """O estado da anotacao: a fila sorteada, os videos e o arquivo de destino."""

    def __init__(
        self,
        alvos: Sequence[Alvo],
        primeira: Mapping[Chave, BallTrajectoryLabel],
        videos: Mapping[str, Path],
        destino: Path,
        annotator: str,
        agora: Callable[[], datetime],
    ) -> None:
        faltando = sorted({alvo.clip_id for alvo in alvos} - set(videos))
        if faltando:
            raise BolaCegaError(f"sem video para {', '.join(faltando)}")
        self.alvos = tuple(alvos)
        self._primeira = primeira
        self._videos = videos
        self._destino = destino
        self._annotator = annotator
        self._agora = agora
        self._fonte: tuple[str, VideoSource] | None = None

    def estado(self) -> dict[str, object]:
        """A fila e as marcas desta passada. Nada da primeira passada sai daqui."""
        gravadas = marcas_gravadas(self._destino)
        return {
            "annotator": self._annotator,
            "contexto": CONTEXTO,
            "alvos": [
                {
                    "clip_id": alvo.clip_id,
                    "frame_index": alvo.frame_index,
                    "marca": _marca_publica(gravadas.get((alvo.clip_id, alvo.frame_index))),
                }
                for alvo in self.alvos
            ],
        }

    def quadro(self, posicao: int, deslocamento: int) -> bytes:
        alvo = self._alvo(posicao)
        if abs(deslocamento) > CONTEXTO:
            raise BolaCegaError(f"o contexto vai ate {CONTEXTO} quadros")
        indice = alvo.frame_index + deslocamento
        imagem = self._fonte_de(alvo.clip_id).read([indice])[indice]
        ok, codificado = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise BolaCegaError(f"nao consegui codificar o quadro {indice}")
        return codificado.tobytes()

    def marcar(self, posicao: int, corpo: Mapping[str, Any]) -> None:
        alvo = self._alvo(posicao)
        visibilidade = corpo.get("visibility")
        if visibilidade not in VISIBILIDADES:
            raise BolaCegaError(f"visibilidade desconhecida: {visibilidade}")
        centro = corpo.get("center")
        marca = Marca(
            visibility=visibilidade,
            center_xy=None if centro is None else (float(centro[0]), float(centro[1])),
            apparent_diameter_px=(
                None if corpo.get("diameter") is None else float(corpo["diameter"])
            ),
            motion_blur=bool(corpo.get("motion_blur", False)),
            camera_cut=bool(corpo.get("camera_cut", False)),
        )
        label = self._primeira[(alvo.clip_id, alvo.frame_index)]
        grava(self._destino, registro(label, marca, self._annotator, self._agora()))

    def close(self) -> None:
        if self._fonte is not None:
            self._fonte[1].close()
            self._fonte = None

    def _alvo(self, posicao: int) -> Alvo:
        if not 0 <= posicao < len(self.alvos):
            raise BolaCegaError(f"a fila tem {len(self.alvos)} quadros")
        return self.alvos[posicao]

    def _fonte_de(self, clip_id: str) -> VideoSource:
        if self._fonte is None or self._fonte[0] != clip_id:
            self.close()
            self._fonte = (clip_id, VideoSource(self._videos[clip_id]))
        return self._fonte[1]


def servidor(sessao: Sessao, porta: int) -> HTTPServer:
    """Um servidor de uma thread so, porque o decode de video nao e seguro entre threads."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            consulta = parse_qs(url.query)
            try:
                if url.path == "/":
                    self._responde(PAGINA.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/estado":
                    self._json(sessao.estado())
                elif url.path == "/quadro":
                    posicao = int(consulta["i"][0])
                    deslocamento = int(consulta.get("d", ["0"])[0])
                    self._responde(sessao.quadro(posicao, deslocamento), "image/jpeg")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (BolaCegaError, MissingFrame, KeyError, ValueError) as erro:
                self._json({"erro": str(erro)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            if url.path != "/marca":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                tamanho = int(self.headers.get("Content-Length", "0"))
                corpo = cast("dict[str, Any]", json.loads(self.rfile.read(tamanho) or b"{}"))
                sessao.marcar(int(parse_qs(url.query)["i"][0]), corpo)
                self._json({"ok": True})
            except (BolaCegaError, KeyError, ValueError, TypeError) as erro:
                self._json({"erro": str(erro)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None: ...

        def _json(self, corpo: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._responde(json.dumps(corpo).encode(), "application/json", status)

        def _responde(self, corpo: bytes, tipo: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

    return HTTPServer(("127.0.0.1", porta), Handler)


def _documentos(destino: Path) -> dict[Chave, dict[str, Any]]:
    if not destino.is_file():
        return {}
    documentos: dict[Chave, dict[str, Any]] = {}
    for linha in destino.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            documento = cast("dict[str, Any]", json.loads(linha))
            documentos[(str(documento["clip_id"]), int(documento["frame_index"]))] = documento
    return documentos


def _marca_publica(documento: Mapping[str, Any] | None) -> dict[str, object] | None:
    if documento is None:
        return None
    centro = (
        None if documento["center_x"] is None else [documento["center_x"], documento["center_y"]]
    )
    return {
        "visibility": documento["visibility"],
        "center": centro,
        "diameter": documento["apparent_diameter_px"],
        "motion_blur": documento["motion_blur"],
        "camera_cut": documento["camera_cut"],
    }


def _kappa(pares: Sequence[tuple[str, str]]) -> float | None:
    """Kappa de Cohen: a concordancia descontada da que duas passadas teriam por acaso."""
    total = len(pares)
    observada = sum(a == b for a, b in pares) / total
    primeira = Counter(a for a, _ in pares)
    segunda = Counter(b for _, b in pares)
    esperada = sum(primeira[c] * segunda[c] for c in VISIBILIDADES) / (total * total)
    return None if esperada >= 1 else (observada - esperada) / (1 - esperada)


def _percentil(valores: Sequence[float], percentil: int) -> float | None:
    return None if not valores else float(np.percentile(valores, percentil))
