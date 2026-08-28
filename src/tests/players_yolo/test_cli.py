from pathlib import Path

from xvision_players_yolo.cli import build_parser


def test_identify_command_does_not_require_model_weights() -> None:
    args = build_parser().parse_args(
        [
            "identify",
            "--input",
            "clip.mp4",
            "--tracking",
            "detections.json",
            "--output-dir",
            "output",
        ]
    )

    assert args.command == "identify"
    assert args.input == Path("clip.mp4")
    assert args.tracking == Path("detections.json")
    assert args.max_samples == 5
    assert args.minimum_samples == 2
