"""Roda o tracking da bola num clipe e grava um video anotado, para avaliacao.

Ferramenta de avaliacao, nao faz parte do pipeline da borda. Foco exclusivo na bola: o YOLO detecta
apenas a classe `sports ball` (id 32 no COCO), o filtro de forma e tamanho descarta candidatos que
nao parecem bola, e a `BallTracker` do projeto escolhe entre os que sobram por continuidade de
movimento, rejeita saltos e interpola lacunas curtas. Serve para medir o baseline sem fine tuning e
comparar com versoes futuras.

Na saida anotada: a bola escolhida aparece em vermelho, a posicao interpolada em circulo, e os
candidatos reprovados pelo filtro em cinza fino (para ver a marca do penalti sendo descartada).

Requer o extra opcional `detector` (ultralytics) e opencv. Exemplo:

    python src/models/ball/track_ball_clip.py entrada.mkv saida.mp4 --start 25 --end 50 --imgsz 1280
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO  # pyright: ignore[reportMissingImports]  # extra opcional detector

# A ferramenta roda fora de um ambiente instalado; ela mesma poe o src no sys.path.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from contracts.perception import BallDetection, Box  # noqa: E402
from edge.ball.tracking import (  # noqa: E402
    BallObservation,
    BallTracker,
    DetectorConfig,
    TrackerConfig,
    is_ball_shaped,
)


def run(
    clip: str,
    out: str,
    weights: str,
    imgsz: int,
    conf: float,
    start_s: float,
    end_s: float | None,
    ball_class: int,
) -> None:
    model = YOLO(weights)
    shape = DetectorConfig(weights=weights)
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start_s * fps)
    last = total if end_s is None else min(total, int(end_s * fps))

    # Passo 1: detecta e separa candidatos aprovados dos reprovados pelo filtro de forma.
    observations: list[BallObservation] = []
    rejected_by_frame: dict[int, list[Box]] = {}
    rejected_total = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for index in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=imgsz, conf=conf, classes=[ball_class], verbose=False)[
            0
        ]
        kept: list[BallDetection] = []
        rejected: list[Box] = []
        rows = zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=False)
        for xyxy, raw_conf in rows:
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            box = Box(x1, y1, x2, y2)
            if is_ball_shaped(box, float(height), shape):
                kept.append(BallDetection(box=box, confidence=float(raw_conf)))
            else:
                rejected.append(box)
        observations.append(BallObservation(index, index / fps, tuple(kept)))
        rejected_by_frame[index] = rejected
        rejected_total += len(rejected)

    # Passo 2: fecha a trajetoria com a BallTracker do projeto.
    points = {p.frame_index: p for p in BallTracker(TrackerConfig()).track(observations)}

    # Passo 3: reabre o clipe e desenha cada quadro.
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    counts = {"detected": 0, "interpolated": 0, "missing": 0}
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for index in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        for box in rejected_by_frame.get(index, []):
            cv2.rectangle(
                frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (150, 150, 150), 1
            )
        point = points.get(index)
        source = point.source if point is not None else "missing"
        counts[source] = counts.get(source, 0) + 1
        if point is not None and point.source == "detected" and point.box is not None:
            b = point.box
            cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 0, 255), 3)
            label = f"BOLA {point.confidence:.2f}" if point.confidence is not None else "BOLA"
            cv2.putText(
                frame,
                label,
                (int(b.x1), max(0, int(b.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        elif point is not None and point.source == "interpolated" and point.center is not None:
            cx, cy = int(point.center[0]), int(point.center[1])
            cv2.circle(frame, (cx, cy), 9, (0, 200, 255), 2)
            cv2.putText(
                frame,
                "interp",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
            )
        cv2.putText(
            frame,
            f"t={index / fps:5.1f}s  bola:{source}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    cap.release()
    writer.release()
    processed = sum(counts.values())
    det = counts["detected"]
    interp = counts["interpolated"]
    cov = 100 * (det + interp) / max(processed, 1)
    print(
        f"frames={processed} detected={det} ({100 * det / max(processed, 1):.0f}%) "
        f"interpolated={interp} missing={counts['missing']} cobertura={cov:.0f}% "
        f"rejeitados_forma={rejected_total} -> {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tracking da bola num clipe, com saida anotada.")
    parser.add_argument("clip")
    parser.add_argument("out")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument(
        "--ball-class",
        type=int,
        default=32,
        help="id da classe bola: 32 (COCO) ou 0 (modelo fine-tunado de 1 classe)",
    )
    args = parser.parse_args()
    run(
        args.clip,
        args.out,
        args.weights,
        args.imgsz,
        args.conf,
        args.start,
        args.end,
        args.ball_class,
    )


if __name__ == "__main__":
    main()
