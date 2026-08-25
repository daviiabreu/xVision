"""Prova ao vivo do runner: monta o grafo da configuracao, roda e imprime o que aconteceu.

`--cycle` forca um ciclo no grafo para mostrar a mensagem que nomeia os dois culpados.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from contracts import Clip
from edge.frames.provider import FrameProvider, VideoSource
from edge.graph.framesets import read_framesets
from edge.graph.runner import RunResult, run_pipeline
from edge.modules.registry import load_modules, read_config
from edge.paths import out_dir
from edge.testing.pitch import clip_truth


def run(config: Path, clip_path: Path, cycle: bool) -> RunResult:
    gabarito = clip_truth()
    clip = Clip(
        clip_id=clip_path.stem,
        path=str(clip_path),
        fps=float(gabarito.fps),
        frame_count=gabarito.frame_count,
        width=gabarito.width,
        height=gabarito.height,
    )
    if not clip_path.is_file():
        raise SystemExit(
            f"{clip_path} nao existe. Gere a fixture canonica antes: uv run xvision clip"
        )
    modules = load_modules(read_config(config))
    if cycle:
        modules["players"].spec = replace(modules["players"].spec, needs=("field",))
        modules["field"].spec = replace(modules["field"].spec, needs=("players",))

    provider = FrameProvider(VideoSource(clip_path))
    try:
        return run_pipeline(
            clip=clip,
            modules=modules,
            framesets=read_framesets(config),
            provider=provider,
        )
    finally:
        provider.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="edge.graph.demo", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.toml"))
    parser.add_argument("--clip", type=Path, default=out_dir() / "synthetic.mp4")
    parser.add_argument("--cycle", action="store_true")
    arguments = parser.parse_args(argv)

    resultado = run(arguments.config, arguments.clip, arguments.cycle)
    print(f"{'etapa':10} {'conjunto':9} {'estado':8} {'ms':>7}  nome@versao")
    for etapa in resultado.stages:
        print(
            f"{etapa.kind:10} {etapa.frames:9} {etapa.status:8} {etapa.duration_ms:7.1f}  "
            f"{etapa.name}@{etapa.version}" + (f"  <- {etapa.reason}" if etapa.reason else "")
        )
    quadros = resultado.perception.frames
    if not quadros:
        print("\nnenhum quadro na percepcao: toda etapa degradou")
        return
    primeiro, ultimo = quadros[0].frame_index, quadros[-1].frame_index
    print(f"\nquadros na percepcao: {len(quadros)}, de {primeiro} a {ultimo}")
    contato = next((q for q in quadros if q.field is not None and q.ball is not None), None)
    if contato is None:
        print("nenhum quadro calibrado: sem homografia nao existe metro")
        return
    print(f"primeiro quadro calibrado: {contato.frame_index}, {len(contato.players)} jogadores")
    if contato.ball is not None:
        print(f"bola em metros: {contato.ball.pitch_xy}")


if __name__ == "__main__":
    main()
