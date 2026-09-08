"""Render compartilhado: preserva a visao superior existente e acrescenta a bola."""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cv2

from contracts import Perception
from edge.artifacts import ArtifactError, read_manifest, resolve

if sys.version_info >= (3, 11):  # pragma: no cover - o piso e 3.10, e e nele que a suite roda
    import tomllib
else:
    import tomli as tomllib

MODELOS = Path("models")

SECAO = "render.top_view"
SUFIXO = "_tracking_visao_superior.mp4"
MODULO = "top_view.create_top_view"
MODULO_HOMOGRAFIA = "top_view.homography_view"

logger = logging.getLogger("xvision.api")

_desenho = threading.Lock()


class RenderUnavailable(RuntimeError): ...


@dataclass(frozen=True)
class RenderConfig:
    players: str = ""
    field: str = ""


class _TopView(Protocol):
    IMGSZ: int
    CONFIDENCE: float
    PLAYER_DEVICE: str
    OUTPUT_DIR: Path

    def process_video(
        self, tracker: Any, video_path: Path, input_dir: Path, homography_model: Any
    ) -> None: ...

    def PlayerTracker(  # noqa: N802 - o nome e o da classe importada no script
        self,
        weights: Path,
        *,
        image_size: int,
        confidence_threshold: float,
        device: str,
    ) -> Any: ...


class _Homography(Protocol):
    def load_model(self, path: Path) -> Any: ...


def read_config(config: Path) -> RenderConfig | None:
    documento = tomllib.loads(config.read_text(encoding="utf-8"))
    secao = cast("dict[str, Any] | None", documento.get("render", {}).get("top_view"))
    if secao is None:
        return None
    source = secao.get("source", "legacy")
    if source != "legacy":
        raise RenderUnavailable(f"[{SECAO}].source espera legacy; configure players e field")
    faltando = {"players", "field"} - set(secao)
    if faltando:
        raise RenderUnavailable(f"[{SECAO}] nao declara {', '.join(sorted(faltando))}")
    for campo in ("players", "field"):
        if not isinstance(secao[campo], str):
            raise RenderUnavailable(f"[{SECAO}].{campo} = {secao[campo]!r}, e o campo espera texto")
    return RenderConfig(players=str(secao["players"]), field=str(secao["field"]))


def render_video(
    clipe: Path, destino: Path, config: RenderConfig, *, perception: Perception | None = None
) -> Path:
    """A oficina fica em `var/`, e o arquivo final aparece pronto ou nao aparece.

    O script escreve o mp4 cru e so o apaga depois de o ffmpeg terminar, entao rodar sobre o
    `OUTPUT_DIR` que ele declara deixava temporario de execucao interrompida dentro de
    `src/models`, que e area de codigo. Apontar `OUTPUT_DIR` para uma pasta descartavel resolve
    isso e paga um segundo preco: origem e destino passam a ficar no mesmo sistema de arquivos,
    onde `os.replace` e atomico. `shutil.move` entre dispositivos copia, e a copia interrompida
    deixava um mp4 truncado com o nome final, que `_endereco_do_render` anunciava e a rota servia
    com 200.

    O desenho acontece sob `_desenho` porque as duas coisas que ele mexe sao globais: `OUTPUT_DIR`
    e atributo do modulo do script, e `redirect_stdout`, que cala a barra de progresso que o script
    imprime com `\r`, troca o `sys.stdout` do processo. Com mais de um worker, dois desenhos
    simultaneos trocariam a pasta um do outro e devolveriam o stdout fora de ordem, deixando o log
    de acesso do uvicorn escrevendo num buffer que ninguem le. Esperar na fila custa menos.
    """
    jogadores = _weights(config.players, "players")
    campo = _weights(config.field, "field")
    script, homografia = _script()
    destino.parent.mkdir(parents=True, exist_ok=True)
    oficina = destino.parent / f"oficina-{destino.stem}"
    inicio = time.monotonic()
    with _desenho:
        declarado = script.OUTPUT_DIR
        try:
            shutil.rmtree(oficina, ignore_errors=True)
            oficina.mkdir(parents=True)
            script.OUTPUT_DIR = oficina
            with contextlib.redirect_stdout(io.StringIO()):
                script.process_video(
                    script.PlayerTracker(
                        jogadores,
                        image_size=script.IMGSZ,
                        confidence_threshold=script.CONFIDENCE,
                        device=script.PLAYER_DEVICE,
                    ),
                    clipe,
                    clipe.parent,
                    homografia.load_model(campo),
                )
            produzido = oficina / f"{clipe.stem}{SUFIXO}"
            if not produzido.is_file():
                raise RenderUnavailable(
                    f"o script terminou e nao deixou {produzido.name} em OUTPUT_DIR"
                )
            if perception is None:
                os.replace(produzido, destino)
            else:
                from edge.ball_video import write_video

                width = _source_width(clipe, produzido)
                write_video(produzido, destino, perception, image_width=width)
        finally:
            script.OUTPUT_DIR = declarado
            shutil.rmtree(oficina, ignore_errors=True)
    logger.info(
        "video=%s visao superior em %.0f ms, %.1f MB",
        destino.stem,
        (time.monotonic() - inicio) * 1000,
        destino.stat().st_size / 1e6,
    )
    return destino


def _source_width(source: Path, rendered: Path) -> int:
    original, canvas = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(rendered))
    try:
        if not original.isOpened() or not canvas.isOpened():
            raise RenderUnavailable("video indisponivel para integrar bola e top view")
        for prop in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_HEIGHT):
            if abs(original.get(prop) - canvas.get(prop)) > 0.001:
                raise RenderUnavailable("top view alterou FPS, duracao ou altura do video")
        width = int(original.get(cv2.CAP_PROP_FRAME_WIDTH))
        if width <= 0 or canvas.get(cv2.CAP_PROP_FRAME_WIDTH) <= width:
            raise RenderUnavailable("top view nao produziu o painel ao lado do video")
        return width
    finally:
        original.release()
        canvas.release()


def _weights(manifesto: str, especie: str) -> Path:
    caminho = Path(manifesto)
    if not caminho.is_file():
        raise RenderUnavailable(f"[{SECAO}].{especie} aponta para {caminho}, que nao existe")
    try:
        return resolve(read_manifest(caminho))
    except (ArtifactError, FileNotFoundError) as erro:
        raise RenderUnavailable(f"o peso de {especie} nao esta pronto: {erro}") from erro


def _script() -> tuple[_TopView, _Homography]:
    pasta = MODELOS.resolve()
    if not (pasta / "top_view" / "create_top_view.py").is_file():
        raise RenderUnavailable(
            f"{pasta} nao tem o script da visao superior. Ele mora em src/models e o servico o "
            "procura relativo a raiz Python: rode a partir de src/"
        )
    if str(pasta) not in sys.path:
        sys.path.insert(0, str(pasta))
    try:
        script = importlib.import_module(MODULO)
        homografia = importlib.import_module(MODULO_HOMOGRAFIA)
    except ImportError as erro:
        raise RenderUnavailable(
            f"o script da visao superior nao importa: {erro}. Ele pede torch, ultralytics e "
            "imageio-ffmpeg, que entram com o extra `topview`"
        ) from erro
    return cast("_TopView", script), cast("_Homography", homografia)
