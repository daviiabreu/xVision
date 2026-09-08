"""Exportacao auditavel em ambiente fixo, sem data volatil no ONNX entregue."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import shutil
import tempfile
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from edge.artifacts import read_manifest, resolve, sha256_of


def normalize_date(model: Any) -> None:
    """Altera somente a data de exportacao, sem tocar grafo ou outras propriedades."""
    for entry in model.metadata_props:
        if entry.key == "date":
            entry.value = "1970-01-01T00:00:00"


def export_repeated(manifest: Path, output: Path, shape: tuple[int, int]) -> dict[str, object]:
    if any(value <= 0 or value % 32 for value in shape):
        raise ValueError("shape exige altura e largura positivas e multiplas de 32")
    weight = resolve(read_manifest(manifest))
    onnx = cast("Any", import_module("onnx"))
    yolo = cast("Any", import_module("ultralytics")).YOLO
    output.mkdir(parents=True, exist_ok=False)
    files: list[Path] = []
    raw_digests: list[str] = []
    for repeat in range(2):
        with tempfile.TemporaryDirectory(prefix="export-", dir=output) as temporary:
            source = Path(temporary) / "model.pt"
            shutil.copy2(weight, source)
            with contextlib.redirect_stdout(io.StringIO()):
                exported = Path(
                    yolo(str(source)).export(
                        format="onnx",
                        imgsz=shape,
                        opset=17,
                        batch=1,
                        dynamic=False,
                        simplify=False,
                        nms=False,
                        device="cpu",
                    )
                )
            raw_digests.append(sha256_of(exported))
            model = onnx.load(str(exported))
            onnx.checker.check_model(model)
            inputs = model.graph.input
            if len(inputs) != 1:
                raise ValueError("export exige exatamente uma entrada")
            dimensions = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
            if dimensions != [1, 3, *shape]:
                raise ValueError("export produziu shape diferente do declarado")
            normalize_date(model)
            onnx.checker.check_model(model)
            target = output / f"model-{repeat + 1}.onnx"
            onnx.save(model, str(target))
            files.append(target)
    digests = [sha256_of(path) for path in files]
    if files[0].read_bytes() != files[1].read_bytes():
        raise ValueError("reexportacao nao deterministica; artefatos retidos para diagnostico")
    report: dict[str, object] = {
        "manifest_sha256": sha256_of(manifest),
        "weight_sha256": sha256_of(weight),
        "shape": [1, 3, *shape],
        "opset": 17,
        "raw_sha256": raw_digests,
        "normalized_sha256": digests,
        "bytes": files[0].stat().st_size,
        "byte_identical": True,
        "normalized_metadata": ["date"],
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "ultralytics", "onnx")},
        "exporter_sha256": sha256_of(Path(__file__)),
    }
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(export_repeated(args.manifest, args.output, (args.height, args.width)), indent=2)
    )


if __name__ == "__main__":
    main()
