"""Os comandos de jogadores, que antes eram um executavel proprio.

Eles vinham de `xvision-player-detection`, um segundo ponto de entrada em `argparse` que morava
dentro de `edge/`. Nenhum outro subpacote de `edge/` tem CLI, e o `brain/code-guidelines.md` diz
que terminal e trabalho de `cli/`: dois executaveis significavam dois lugares para descobrir um
comando e duas formas de escrever opcao.

Moram num modulo proprio, e nao em `cli/__init__.py`, porque aquele arquivo ja passa do teto de
800 linhas que o mesmo guia declara.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from edge.cache.perception import PerceptionCache
from edge.frames.provider import FrameProvider, VideoSource, probe
from edge.graph.framesets import read_framesets
from edge.graph.runner import run_pipeline
from edge.modules.registry import load_modules, read_config
from edge.participants.batch import process_batch
from edge.participants.identification import process_identification
from edge.participants.tracking import PlayerTracker
from edge.participants.types import frame_to_dict
from edge.participants.video import process_video, render_perception
from mlops.tracking_diagnostic import diagnosticar, ler_rotulos, ler_run

players_app = typer.Typer(help="Deteccao, tracking e identificacao de jogadores num clipe.")

PESO = typer.Option(..., help="Peso do detector de jogadores.")
DEVICE = typer.Option("auto", help="auto, cpu, mps ou cuda.")
CONFIANCA = typer.Option(0.25, help="Limiar de confianca do detector.")
IMAGEM = typer.Option(1280, help="Lado maior da imagem na inferencia.")
RASTROS = typer.Option(False, "--rastros", help="Desenha a trajetoria de cada jogador.")
SOBRESCREVER = typer.Option(False, "--sobrescrever", help="Refaz clipe que ja tem saida.")
AMOSTRAS = typer.Option(5, help="Quadros amostrados por trajetoria para decidir a equipe.")
AMOSTRAS_MINIMAS = typer.Option(2, help="Abaixo disso a trajetoria fica sem equipe.")
CONFIGURACAO_CENTRAL = typer.Option(
    Path("config/pipeline.players-debug.toml"),
    "--config",
    help="Pipeline central reduzido a players e tracking.",
)
ROTULOS = typer.Option(
    Path("../var/annotations/player-pilot-v1/agent-labels.json"),
    help="Rotulos do agente; sem o arquivo, o diagnostico sai sem trocas e costura.",
)
REFERENCIA = typer.Option(
    Path("../var/runs/player-pilot-v1/refined-bytetrack"),
    help="Run sobre o qual os rotulos foram feitos.",
)
COLUNAS = ("ids", "pico", "curtas", "retomadas", "arbitros", "goleiros", "sem_id")
COLUNAS_ROTULADAS = ("trocas", "costuras_S", "costuras_D", "costuras_sem")


def _tracker(peso: Path, device: str, confianca: float, imagem: int) -> PlayerTracker:
    return PlayerTracker(peso, image_size=imagem, confidence_threshold=confianca, device=device)


@players_app.command("run")
def players_run(
    entrada: Path,
    saida: Path,
    peso: Path = PESO,
    device: str = DEVICE,
    confianca: float = CONFIANCA,
    imagem: int = IMAGEM,
    rastros: bool = RASTROS,
) -> None:
    """Processa um clipe e grava as deteccoes e o video anotado."""
    resultado = process_video(
        entrada, saida, _tracker(peso, device, confianca, imagem), show_trails=rastros
    )
    typer.echo(
        f"{resultado['clip']['frame_count']} quadros a "
        f"{resultado['metrics']['processing_fps']:.2f} FPS"
    )


@players_app.command("batch")
def players_batch(
    entrada: Path,
    saida: Path,
    peso: Path = PESO,
    device: str = DEVICE,
    confianca: float = CONFIANCA,
    imagem: int = IMAGEM,
    rastros: bool = RASTROS,
    sobrescrever: bool = SOBRESCREVER,
) -> None:
    """Processa todos os MP4 de um diretorio, retomando de onde parou."""
    resumo = process_batch(
        entrada,
        saida,
        _tracker(peso, device, confianca, imagem),
        overwrite=sobrescrever,
        show_trails=rastros,
    )
    typer.echo(
        f"{resumo['processed']} processados, {resumo['skipped']} ignorados, "
        f"{resumo['failed']} com erro"
    )
    if resumo["failed"]:
        raise typer.Exit(code=1)


@players_app.command("debug")
def players_debug(
    entrada: Path,
    saida: Path,
    config: Path = CONFIGURACAO_CENTRAL,
    rastros: bool = RASTROS,
) -> None:
    """Roda players e tracking centrais e desenha exatamente a percepcao produzida."""
    if not config.is_file():
        typer.echo(f"{config} nao existe", err=True)
        raise typer.Exit(code=1)
    clip_id = entrada.stem.rsplit("_", 1)[-1]
    clip = probe(entrada, clip_id=clip_id)
    provider = FrameProvider(VideoSource(entrada))
    try:
        resultado = run_pipeline(
            clip=clip,
            modules=load_modules(read_config(config)),
            framesets=read_framesets(config),
            provider=provider,
            cache=PerceptionCache(),
        )
    finally:
        provider.close()

    falhas = [etapa for etapa in resultado.stages if etapa.status != "ok"]
    if falhas:
        for etapa in falhas:
            typer.echo(f"{etapa.kind}: {etapa.reason}", err=True)
        raise typer.Exit(code=1)

    video = render_perception(entrada, saida, resultado.perception, show_trails=rastros)
    documento = {
        "schema_version": "central-player-tracking.v1",
        "clip": {
            "id": clip.clip_id,
            "fps": clip.fps,
            "width": clip.width,
            "height": clip.height,
            "frame_count": clip.frame_count,
        },
        "pipeline": resultado.pipeline.to_dict(),
        "stages": [
            {
                "kind": etapa.kind,
                "name": etapa.name,
                "version": etapa.version,
                "frames": etapa.frames,
                "status": etapa.status,
                "reason": etapa.reason,
                "duration_ms": etapa.duration_ms,
            }
            for etapa in resultado.stages
        ],
        "frames": [frame_to_dict(frame) for frame in resultado.perception.frames],
    }
    destino = saida / "detections.json"
    temporario = saida / "detections.json.tmp"
    temporario.write_text(json.dumps(documento, ensure_ascii=False, indent=2), encoding="utf-8")
    temporario.replace(destino)
    typer.echo(str(video))
    typer.echo(str(destino))


@players_app.command("identify")
def players_identify(
    entrada: Path,
    tracking: Path,
    saida: Path,
    amostras: int = AMOSTRAS,
    amostras_minimas: int = AMOSTRAS_MINIMAS,
) -> None:
    """Classifica as trajetorias por equipe, sem precisar do peso do detector."""
    resultado = process_identification(
        entrada,
        tracking,
        saida,
        max_samples=amostras,
        minimum_samples=amostras_minimas,
    )
    classificadas = sum(t["status"] == "available" for t in resultado["tracks"])
    typer.echo(f"{classificadas} trajetorias classificadas")


@players_app.command("diagnose")
def players_diagnose(run: Path, rotulos: Path = ROTULOS, referencia: Path = REFERENCIA) -> None:
    """Imprime a fragmentacao do tracking por clipe a partir dos detections.json de um run."""
    clipes = ler_run(run)
    if not clipes:
        typer.echo(f"{run} nao tem */detections.json", err=True)
        raise typer.Exit(code=1)
    rotulado = ler_rotulos(rotulos, referencia) if rotulos.is_file() else None
    colunas = COLUNAS + (COLUNAS_ROTULADAS if rotulado else ())
    linhas = [diagnosticar(clipe, rotulado) for clipe in clipes]
    typer.echo(" ".join(["clipe".ljust(8), *(c.rjust(12) for c in colunas)]))
    for linha in linhas:
        typer.echo(
            " ".join([str(linha["clipe"]).ljust(8), *(str(linha[c]).rjust(12) for c in colunas)])
        )
    total = {c: sum(int(linha[c]) for linha in linhas) for c in (*colunas, "deteccoes")}
    typer.echo(" ".join(["total".ljust(8), *(str(total[c]).rjust(12) for c in colunas)]))
    typer.echo(f"IDs por pessoa visivel: {total['ids'] / max(total['pico'], 1):.2f}")
    typer.echo(f"deteccoes sem ID: {total['sem_id'] / max(total['deteccoes'], 1):.1%}")
    if rotulado is not None:
        julgadas = total["costuras_S"] + total["costuras_D"]
        precisao = f"{total['costuras_S'] / julgadas:.2f}" if julgadas else "sem costura julgada"
        typer.echo(f"precisao da costura, rotulo de agente: {precisao}")
