"""Varredura de limiar de confianca do detector da bola, com UMA passada de inferencia.

Roda a inferencia uma vez (conf baixo), guarda as caixas com suas confiancas e depois varre varios
limiares em memoria, medindo cobertura e falsos positivos por limiar. Assim nao reprocessa o clipe a
cada conf (rapido) e roda com um comando curto — util em terminal de celular.

Descobre sozinho o `best.pt` mais recente em `runs/` e o clipe em `src/edge/ball/`,
entao normalmente basta:

    uv run python scripts/sweep_conf.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO  # pyright: ignore[reportMissingImports]  # extra opcional detector

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from contracts.perception import BallDetection, Box  # noqa: E402
from edge.ball.tracking import (  # noqa: E402
    BallObservation,
    BallTracker,
    DetectorConfig,
    TrackerConfig,
    is_ball_shaped,
)


def _find_best() -> str | None:
    runs = Path("runs")
    candidates = sorted(
        runs.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return str(candidates[0]) if candidates else None


def _find_clip() -> str | None:
    folder = _HERE.parents[1] / "src" / "edge" / "ball"
    preferred = folder / "clipe.mp4"
    if preferred.exists():
        return str(preferred)
    mp4s = sorted(folder.glob("*.mp4"))
    return str(mp4s[0]) if mp4s else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep de conf do detector da bola.")
    parser.add_argument("clip", nargs="?", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--ball-class", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--end", type=float, default=15.0, help="segundos; 0 = clipe inteiro")
    parser.add_argument("--confs", default="0.05,0.10,0.15,0.25,0.35")
    args = parser.parse_args()

    weights = args.weights or _find_best()
    clip = args.clip or _find_clip()
    if not weights or not clip:
        raise SystemExit(f"faltou weights ({weights}) ou clip ({clip})")
    print(f"weights = {weights}")
    print(f"clip    = {clip}")

    model = YOLO(weights)
    shape = DetectorConfig(weights=weights)
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = total if not args.end else min(total, int(args.end * fps))

    # Uma passada de inferencia com conf baixo: guarda todas as caixas com suas confiancas.
    per_frame: list[tuple[int, float, list[tuple[Box, float]]]] = []
    for index in range(last):
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(
            frame, imgsz=args.imgsz, conf=0.01, classes=[args.ball_class], verbose=False
        )[0]
        boxes: list[tuple[Box, float]] = []
        rows = zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=False)
        for xyxy, raw_conf in rows:
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            boxes.append((Box(x1, y1, x2, y2), float(raw_conf)))
        per_frame.append((index, index / fps, boxes))
    cap.release()

    print(f"quadros processados: {len(per_frame)}\n")
    for threshold in (float(c) for c in args.confs.split(",")):
        observations: list[BallObservation] = []
        rejected = 0
        for index, timestamp, boxes in per_frame:
            kept: list[BallDetection] = []
            for box, confidence in boxes:
                if confidence < threshold:
                    continue
                if is_ball_shaped(box, float(height), shape):
                    kept.append(BallDetection(box=box, confidence=confidence))
                else:
                    rejected += 1
            observations.append(BallObservation(index, timestamp, tuple(kept)))

        counts = {"detected": 0, "interpolated": 0, "missing": 0}
        for point in BallTracker(TrackerConfig()).track(observations):
            counts[point.source] += 1
        processed = sum(counts.values())
        det = counts["detected"]
        cov = 100 * (det + counts["interpolated"]) / max(processed, 1)
        pct = 100 * det / max(processed, 1)
        print(
            f"conf={threshold:.2f}  detected={det} ({pct:.0f}%)  "
            f"interp={counts['interpolated']}  missing={counts['missing']}  "
            f"cobertura={cov:.0f}%  rej_forma={rejected}"
        )


if __name__ == "__main__":
    main()
