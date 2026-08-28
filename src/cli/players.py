"""Os comandos de jogadores, que antes eram um executavel proprio.

Eles vinham de `xvision-player-detection`, um segundo ponto de entrada em `argparse` que morava
dentro de `edge/`. Nenhum outro subpacote de `edge/` tem CLI, e o `brain/code-guidelines.md` diz
que terminal e trabalho de `cli/`: dois executaveis significavam dois lugares para descobrir um
comando e duas formas de escrever opcao.

Moram num modulo proprio, e nao em `cli/__init__.py`, porque aquele arquivo ja passa do teto de
800 linhas que o mesmo guia declara.
"""

from __future__ import annotations

from pathlib import Path

import typer

from edge.participants.batch import process_batch
from edge.participants.identification import process_identification
from edge.participants.tracking import PlayerTracker
from edge.participants.video import process_video

players_app = typer.Typer(help="Deteccao, tracking e identificacao de jogadores num clipe.")

PESO = typer.Option(..., help="Peso do detector de jogadores.")
DEVICE = typer.Option("auto", help="auto, cpu, mps ou cuda.")
CONFIANCA = typer.Option(0.25, help="Limiar de confianca do detector.")
IMAGEM = typer.Option(1280, help="Lado maior da imagem na inferencia.")
RASTROS = typer.Option(False, "--rastros", help="Desenha a trajetoria de cada jogador.")
SOBRESCREVER = typer.Option(False, "--sobrescrever", help="Refaz clipe que ja tem saida.")
AMOSTRAS = typer.Option(5, help="Quadros amostrados por trajetoria para decidir a equipe.")
AMOSTRAS_MINIMAS = typer.Option(2, help="Abaixo disso a trajetoria fica sem equipe.")


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
