"""Editar o codigo de um modulo tem de invalidar o cache dele.

A chave levava o caminho da classe, o `nome@versao`, a spec e o digest do peso, e nao levava o
codigo. Medido: editar o corpo de um stub, sem mudar nome, versao nem caminho, e rodar de novo
servia o resultado anterior. A docstring do modulo promete que a chave e o conteudo do que foi
computado e nao o nome de nada, e para modulo escrito em Python o conteudo e o codigo.

Isso importa agora porque e exatamente o fluxo desta sprint: sete pessoas vao trocar stub por
implementacao de verdade, no mesmo arquivo, e cada uma teria de saber de cor que precisa limpar
o cache entre uma tentativa e outra.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from edge.cache.perception import PerceptionCache, implementation_id, run_module
from edge.modules.base import FrameBatch, PreprocessSpec

MODULO = """
from __future__ import annotations

from edge.modules.base import (
    FrameBatch,
    ModuleInputs,
    ModuleOutput,
    ModuleSpec,
    PerFrame,
    PreprocessSpec,
)


class Detector:
    spec = ModuleSpec(
        name="detector-de-teste",
        version="0.1.0",
        kind="players",
        frames="coarse",
        needs=(),
        batching=PerFrame(),
        preprocess=PreprocessSpec(
            width=64, height=64, color="rgb", letterbox=False, normalize="unit"
        ),
        artifact=None,
    )

    def load(self) -> None: ...

    def close(self) -> None: ...

    def check_preprocess(self, batch: FrameBatch) -> None: ...

    def infer(self, batch: FrameBatch, ctx: ModuleInputs) -> ModuleOutput:
        return ModuleOutput(
            kind="players",
            frames=batch.indices,
            records=tuple(
                ({"box": (0.0, 0.0, 1.0, 1.0), "confidence": CONFIANCA},) for _ in batch.indices
            ),
        )
"""


@pytest.fixture
def pasta(tmp_path: Path) -> Iterator[Path]:
    """Modulo escrito em disco, importavel, e reimportavel de verdade.

    `dont_write_bytecode` nao e detalhe: as duas escritas do teste caem no mesmo segundo e o
    cabecalho do `.pyc` guarda mtime com resolucao de segundo, entao o Python reusaria o bytecode
    da primeira versao. Sem isto o teste falha por motivo que nao tem nada a ver com o cache de
    percepcao, que foi o que aconteceu na primeira versao dele.
    """
    raiz = tmp_path / "modulos"
    raiz.mkdir()
    sys.path.insert(0, str(raiz))
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield raiz
    finally:
        sys.dont_write_bytecode = bytecode
        sys.path.remove(str(raiz))
        sys.modules.pop("meu_modulo", None)


def _escrever(pasta: Path, confianca: str) -> object:
    """Reescreve o modulo e forca o Python a enxergar o arquivo novo.

    Sem `invalidate_caches`, o localizador reusa a listagem do diretorio e importa o arquivo
    antigo: o teste falharia por motivo que nao tem nada a ver com o cache de percepcao.
    """
    (pasta / "meu_modulo.py").write_text(MODULO.replace("CONFIANCA", confianca), encoding="utf-8")
    sys.modules.pop("meu_modulo", None)
    importlib.invalidate_caches()
    return importlib.import_module("meu_modulo").Detector()


SPEC = PreprocessSpec(width=64, height=64, color="rgb", letterbox=False, normalize="unit")


def _lote() -> FrameBatch:
    return FrameBatch(
        indices=(0,), tensor=np.zeros((1, 64, 64, 3), dtype=np.float32), preprocess=SPEC
    )


def test_editing_the_module_body_invalidates_its_cache(pasta: Path, tmp_path: Path) -> None:
    cache = PerceptionCache(tmp_path / "cache")

    antes = _escrever(pasta, "0.11")
    primeira = run_module(antes, _lote(), {}, cache=cache, clip_id="c")  # pyright: ignore[reportArgumentType]

    depois = _escrever(pasta, "0.99")
    segunda = run_module(depois, _lote(), {}, cache=cache, clip_id="c")  # pyright: ignore[reportArgumentType]

    assert primeira.records[0][0]["confidence"] == 0.11
    assert segunda.records[0][0]["confidence"] == 0.99, (
        "o cache serviu o resultado do codigo anterior depois da edicao"
    )


def test_the_identity_of_an_unchanged_module_is_stable(pasta: Path) -> None:
    """Se a identidade mudasse a cada chamada, o cache nunca acertaria e viraria so escrita."""
    modulo = _escrever(pasta, "0.11")

    assert implementation_id(modulo) == implementation_id(modulo)  # pyright: ignore[reportArgumentType]


def test_a_class_without_a_readable_file_falls_back_instead_of_exploding() -> None:
    """Classe definida em memoria ou vinda de extensao compilada nao pode derrubar a execucao."""
    definida_aqui = type("EmMemoria", (), {})

    identidade = implementation_id(definida_aqui())  # pyright: ignore[reportArgumentType]

    assert "EmMemoria" in identidade
