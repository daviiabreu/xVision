import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlops.export_onnx import export_repeated, normalize_date


@pytest.fixture
def exporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from mlops import export_onnx

    manifest, weight = tmp_path / "manifest.toml", tmp_path / "model.pt"
    manifest.write_text("manifest")
    weight.write_bytes(b"weight")
    state = SimpleNamespace(repeats=0, change_graph=False, input_count=1, shape=[1, 3, 544, 960])

    class Yolo:
        def __init__(self, source: str) -> None:
            self.source = Path(source)
            assert self.source.name == "model.pt"
            assert self.source.read_bytes() == weight.read_bytes()

        def export(self, **options: object) -> str:
            assert options["dynamic"] is options["nms"] is options["simplify"] is False
            assert options["opset"] == 17
            assert options["imgsz"] == (544, 960)
            assert options["batch"] == state.shape[0]
            state.repeats += 1
            target = self.source.with_suffix(".onnx")
            target.write_text(f"raw-date-{state.repeats}")
            return str(target)

    def load(path: str) -> SimpleNamespace:
        assert Path(path).is_file()
        tensor = SimpleNamespace(
            shape=SimpleNamespace(dim=[SimpleNamespace(dim_value=value) for value in state.shape])
        )
        return SimpleNamespace(
            graph=SimpleNamespace(
                input=[SimpleNamespace(type=SimpleNamespace(tensor_type=tensor))]
                * state.input_count,
                value=state.repeats if state.change_graph else 0,
            ),
            metadata_props=[SimpleNamespace(key="date", value=str(state.repeats))],
        )

    def save(model: Any, path: str) -> None:
        Path(path).write_text(json.dumps([model.graph.value, model.metadata_props[0].value]))

    def check_model(model: object) -> None:
        assert model is not None

    def read_manifest(path: Path) -> Path:
        return path

    def resolve(entry: object) -> Path:
        return weight

    def version(name: str) -> str:
        return "test-version"

    modules = {
        "onnx": SimpleNamespace(
            load=load, save=save, checker=SimpleNamespace(check_model=check_model)
        ),
        "ultralytics": SimpleNamespace(YOLO=Yolo),
    }
    monkeypatch.setattr(export_onnx, "import_module", modules.__getitem__)
    monkeypatch.setattr(export_onnx, "read_manifest", read_manifest)
    monkeypatch.setattr(export_onnx, "resolve", resolve)
    monkeypatch.setattr(export_onnx, "version", version)
    state.manifest = manifest
    state.output = tmp_path / "export"
    return state


def test_only_volatile_date_changes() -> None:
    graph = object()
    model = SimpleNamespace(
        graph=graph,
        metadata_props=[
            SimpleNamespace(key="date", value="2026-09-04T12:13:14"),
            SimpleNamespace(key="names", value="{0: 'ball'}"),
            SimpleNamespace(key="license", value="AGPL-3.0"),
        ],
    )

    normalize_date(model)
    normalize_date(model)

    assert model.graph is graph
    assert [(entry.key, entry.value) for entry in model.metadata_props] == [
        ("date", "1970-01-01T00:00:00"),
        ("names", "{0: 'ball'}"),
        ("license", "AGPL-3.0"),
    ]


@pytest.mark.parametrize("shape", [(0, 960), (544, -1), (545, 960)])
def test_invalid_shape_fails_before_loading_weights(tmp_path: Path, shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="shape"):
        export_repeated(tmp_path / "missing.toml", tmp_path / "out", shape)


def test_repeated_export_keeps_graph_and_writes_auditable_report(exporter: SimpleNamespace) -> None:
    report = export_repeated(exporter.manifest, exporter.output, (544, 960))

    assert exporter.repeats == 2
    assert report["byte_identical"] is True
    assert len(set(report["raw_sha256"])) == 2  # type: ignore[arg-type]
    assert len(set(report["normalized_sha256"])) == 1  # type: ignore[arg-type]
    assert json.loads((exporter.output / "run.json").read_text()) == report
    assert sorted(path.name for path in exporter.output.iterdir()) == [
        "model-1.onnx",
        "model-2.onnx",
        "run.json",
    ]
    with pytest.raises(FileExistsError):
        export_repeated(exporter.manifest, exporter.output, (544, 960))


def test_graph_difference_is_not_hidden_by_metadata_normalization(
    exporter: SimpleNamespace,
) -> None:
    exporter.change_graph = True
    with pytest.raises(ValueError, match="nao deterministica"):
        export_repeated(exporter.manifest, exporter.output, (544, 960))
    assert not (exporter.output / "run.json").exists()
    assert sorted(path.name for path in exporter.output.iterdir()) == [
        "model-1.onnx",
        "model-2.onnx",
    ]


@pytest.mark.parametrize("input_count", [0, 2])
def test_invalid_input_count_is_rejected(exporter: SimpleNamespace, input_count: int) -> None:
    exporter.input_count = input_count
    with pytest.raises(ValueError, match="uma entrada"):
        export_repeated(exporter.manifest, exporter.output, (544, 960))
    assert list(exporter.output.iterdir()) == []


def test_wrong_export_shape_is_rejected(exporter: SimpleNamespace) -> None:
    exporter.shape = [1, 3, 960, 544]
    with pytest.raises(ValueError, match="shape"):
        export_repeated(exporter.manifest, exporter.output, (544, 960))


def test_lote_declarado_chega_ao_export_e_ao_relatorio(exporter: SimpleNamespace) -> None:
    exporter.shape = [4, 3, 544, 960]

    report = export_repeated(exporter.manifest, exporter.output, (544, 960), batch=4)

    assert report["shape"] == [4, 3, 544, 960]


def test_lote_invalido_falha_antes_de_carregar_peso(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch"):
        export_repeated(tmp_path / "missing.toml", tmp_path / "out", (544, 960), batch=0)
