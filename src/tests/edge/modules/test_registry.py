"""Testes da selecao de implementacao por configuracao."""

from __future__ import annotations

import pytest

from edge.modules import registry
from edge.modules.base import (
    ModuleError,
    ModuleSpec,
    PreprocessSpec,
    UnknownModuleKind,
)
from edge.modules.registry import (
    UnknownDependency,
    UnknownImplementation,
    load_modules,
    read_config,
)

PREPROCESS = PreprocessSpec(width=64, height=64, color="rgb", letterbox=True, normalize="unit")


class _Players:
    spec = ModuleSpec(
        kind="players", name="a", version="0.1.0", frames="coarse", preprocess=PREPROCESS
    )


class _OtherPlayers:
    spec = ModuleSpec(
        kind="players", name="b", version="0.2.0", frames="coarse", preprocess=PREPROCESS
    )


class _FieldNeedingNonsense:
    spec = ModuleSpec(
        kind="field",
        name="c",
        version="0.1.0",
        frames="coarse",
        preprocess=PREPROCESS,
        needs=("offside",),
    )


HERE = f"{__name__}"


def test_config_selects_module_by_kind() -> None:
    first = load_modules(read_config(f'[modules.players]\nimpl = "{HERE}:_Players"\n'))
    assert isinstance(first["players"], _Players)

    second = load_modules(read_config(f'[modules.players]\nimpl = "{HERE}:_OtherPlayers"\n'))
    assert isinstance(second["players"], _OtherPlayers)
    assert second["players"].spec.name == "b"


def test_unknown_impl_raises_with_the_config_key_in_the_message() -> None:
    config = read_config('[modules.players]\nimpl = "xvision_nao_existe:Module"\n')
    with pytest.raises(UnknownImplementation, match=r"modules\.players"):
        load_modules(config)


def test_module_declaring_needs_of_unknown_kind_fails_at_load() -> None:
    config = read_config(f'[modules.field]\nimpl = "{HERE}:{_FieldNeedingNonsense.__name__}"\n')
    with pytest.raises(UnknownDependency, match="offside"):
        load_modules(config)


def test_config_overrides_frames_and_artifact_without_touching_code() -> None:
    config = read_config(
        f'[modules.players]\nimpl = "{HERE}:_Players"\n'
        'frames = "window"\nartifact = "models/players/yolo11n-0.3.1.onnx"\n'
    )
    spec = load_modules(config)["players"].spec
    assert spec.frames == "window"
    assert spec.artifact == "models/players/yolo11n-0.3.1.onnx"


class _Ball:
    spec = ModuleSpec(
        kind="ball", name="d", version="0.1.0", frames="window", preprocess=PREPROCESS
    )


def test_impl_without_colon_is_resolved_as_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Entry:
        name = "stub-players"

        @staticmethod
        def load() -> type[_Players]:
            return _Players

    def _entry_points(group: str) -> list[_Entry]:
        assert group == registry.ENTRY_POINT_GROUP
        return [_Entry()]

    monkeypatch.setattr(registry, "entry_points", _entry_points)
    modules = load_modules(read_config('[modules.players]\nimpl = "stub-players"\n'))
    assert isinstance(modules["players"], _Players)

    with pytest.raises(UnknownImplementation, match=r"xvision\.modules"):
        load_modules(read_config('[modules.players]\nimpl = "nao-registrado"\n'))


def test_config_key_must_match_the_kind_the_module_declares() -> None:
    config = read_config(f'[modules.players]\nimpl = "{HERE}:{_Ball.__name__}"\n')
    with pytest.raises(UnknownModuleKind, match="ball"):
        load_modules(config)


def test_config_entry_without_impl_is_rejected() -> None:
    with pytest.raises(ModuleError, match=r"modules\.players precisa declarar impl"):
        read_config('[modules.players]\nartifact = "models/players/x.onnx"\n')


def test_a_needs_entry_that_is_not_text_is_refused_instead_of_stringified() -> None:
    """`needs = [42, true]` virava `('42', 'True')`, e a especie 42 nunca existiria."""
    with pytest.raises(ModuleError, match="needs"):
        read_config('[modules.players]\nimpl = "x:Y"\nneeds = [42, true]\n')


def test_an_artifact_that_is_not_text_is_refused() -> None:
    with pytest.raises(ModuleError, match="artifact"):
        read_config('[modules.players]\nimpl = "x:Y"\nartifact = 42\n')
