"""Interface de linha de comando da baseline de detecção."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from xvision_players_yolo.batch import process_batch
from xvision_players_yolo.identification import process_identification
from xvision_players_yolo.tracking import PlayerTracker
from xvision_players_yolo.video import process_video


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=1280)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xvision-player-detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="processa um único vídeo")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--show-trails", action="store_true")
    add_model_arguments(run)

    batch = subparsers.add_parser("batch", help="processa todos os MP4 de um diretório")
    batch.add_argument("--input-dir", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--overwrite", action="store_true")
    batch.add_argument("--show-trails", action="store_true")
    add_model_arguments(batch)

    identify = subparsers.add_parser(
        "identify",
        help="classifica jogadores por equipe a partir do tracking",
    )
    identify.add_argument("--input", type=Path, required=True)
    identify.add_argument("--tracking", type=Path, required=True)
    identify.add_argument("--output-dir", type=Path, required=True)
    identify.add_argument("--max-samples", type=int, default=5)
    identify.add_argument("--minimum-samples", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "identify":
            result = process_identification(
                args.input,
                args.tracking,
                args.output_dir,
                max_samples=args.max_samples,
                minimum_samples=args.minimum_samples,
            )
            available = sum(track["status"] == "available" for track in result["tracks"])
            print(f"identificação concluída: {available} trajetórias classificadas")
            return 0

        tracker = PlayerTracker(
            args.weights,
            image_size=args.image_size,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        if args.command == "run":
            result = process_video(
                args.input,
                args.output_dir,
                tracker,
                show_trails=args.show_trails,
            )
            print(
                "processados "
                f"{result['clip']['frame_count']} frames a "
                f"{result['metrics']['processing_fps']:.2f} FPS"
            )
            return 0

        summary = process_batch(
            args.input_dir,
            args.output_dir,
            tracker,
            overwrite=args.overwrite,
            show_trails=args.show_trails,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1

    print(
        f"lote concluído: {summary['processed']} processados, "
        f"{summary['skipped']} ignorados, {summary['failed']} com erro"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
