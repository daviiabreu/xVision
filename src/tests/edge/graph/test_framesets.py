"""Conjuntos de quadros nomeados em configuracao, e nao em codigo."""

from __future__ import annotations

from pathlib import Path

import pytest

from contracts import Clip
from edge.graph.framesets import (
    AroundFrames,
    FrameSetError,
    UniformFrames,
    around_indices,
    read_framesets,
    uniform_indices,
)

CLIP = Clip(clip_id="clipe-1", path="x.mp4", fps=30.0, frame_count=300, width=320, height=240)

CONFIG = """
[frames.coarse]
strategy = "uniform"
fps = 5

[frames.window]
strategy  = "around"
anchor    = "shot"
before_s  = 1.5
after_s   = 0.5
fps       = "native"
"""


def test_the_two_strategies_of_the_spec_are_read_from_the_file() -> None:
    conjuntos = read_framesets(CONFIG)

    assert conjuntos["coarse"] == UniformFrames(fps=5.0)
    assert conjuntos["window"] == AroundFrames(anchor="shot", before_s=1.5, after_s=0.5, fps=None)


def test_native_rate_is_the_absence_of_a_rate_and_not_a_number() -> None:
    assert read_framesets(CONFIG)["window"].fps is None


def test_an_unknown_strategy_is_refused_naming_the_set() -> None:
    with pytest.raises(FrameSetError, match="rapido"):
        read_framesets('[frames.rapido]\nstrategy = "aleatorio"\nfps = 5\n')


def test_an_around_set_without_anchor_is_refused() -> None:
    with pytest.raises(FrameSetError, match="anchor"):
        read_framesets('[frames.w]\nstrategy = "around"\nbefore_s = 1.0\nafter_s = 0.5\n')


def test_a_full_set_is_declared_without_touching_code() -> None:
    conjuntos = read_framesets('[frames.full]\nstrategy = "uniform"\nfps = "native"\n')

    assert uniform_indices(CLIP, conjuntos["full"].fps) == tuple(range(CLIP.frame_count))


def test_uniform_samples_the_whole_clip_at_the_declared_rate() -> None:
    indices = uniform_indices(CLIP, 5.0)

    assert indices[:3] == (0, 6, 12)
    assert len(indices) == 50
    assert max(indices) < CLIP.frame_count


def test_a_rate_above_the_clip_gives_every_frame_once() -> None:
    assert uniform_indices(CLIP, 120.0) == tuple(range(CLIP.frame_count))


def test_around_is_asymmetric_because_context_comes_before_the_shot() -> None:
    indices = around_indices(CLIP, anchor_frame=150, before_s=1.5, after_s=0.5, fps=None)

    assert indices[0] == 150 - 45
    assert indices[-1] == 150 + 15
    assert indices == tuple(range(105, 166))


def test_around_clamps_to_the_clip_instead_of_asking_for_what_does_not_exist() -> None:
    inicio = around_indices(CLIP, anchor_frame=2, before_s=1.5, after_s=0.5, fps=None)
    fim = around_indices(CLIP, anchor_frame=299, before_s=1.5, after_s=0.5, fps=None)

    assert inicio[0] == 0
    assert fim[-1] == CLIP.frame_count - 1


def test_around_at_a_declared_rate_samples_the_window() -> None:
    indices = around_indices(CLIP, anchor_frame=150, before_s=1.0, after_s=1.0, fps=10.0)

    assert indices[0] == 120
    assert all(b - a == 3 for a, b in zip(indices, indices[1:], strict=False))


def test_a_misspelled_key_is_refused_instead_of_doubling_the_work() -> None:
    """`fpss` em vez de `fps` cai em taxa nativa e a etapa decodifica o clipe inteiro.

    Sai `ok`, o evento valida, e o custo por clipe dobra sem uma linha de aviso, num projeto cujo
    numero de manchete e milissegundos por clipe. Uma letra a mais.
    """
    with pytest.raises(FrameSetError, match="fpss"):
        read_framesets('[frames.coarse]\nstrategy = "uniform"\nfpss = 5\n')


def test_a_key_from_the_other_strategy_is_refused() -> None:
    """`anchor` num conjunto uniforme nao faz nada, e quem escreveu acha que faz."""
    with pytest.raises(FrameSetError, match="anchor"):
        read_framesets('[frames.c]\nstrategy = "uniform"\nfps = 5\nanchor = "shot"\n')


def test_the_keys_the_project_really_uses_keep_working() -> None:
    """A guarda contra a guarda: recusar chave demais quebraria a configuracao do projeto."""
    conjuntos = read_framesets(Path("config/pipeline.toml"))

    assert set(conjuntos) >= {"coarse", "window"}
