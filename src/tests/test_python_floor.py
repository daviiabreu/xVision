"""O piso de Python declarado precisa ser o piso que o lock e o pyright usam.

Lock resolvido acima do piso declarado escolhe dependencia sem roda para a versao mais baixa,
e a falta so aparece na Jetson. Foi o caso do `rpds-py`: a versao que ele escolhe para 3.11 ou
acima nao publica roda cp310 nenhuma, entao a borda nao instalaria em JetPack 6.
"""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

VERSION_FLOOR = re.compile(r'^requires-python\s*=\s*">=\s*(\d+)\.(\d+)', re.MULTILINE)
PYRIGHT_VERSION = re.compile(r'^pythonVersion\s*=\s*"(\d+)\.(\d+)"', re.MULTILINE)


def _floor(path: Path, pattern: re.Pattern[str]) -> tuple[int, int]:
    assert (REPOSITORY / ".gitlab-ci.yml").is_file(), f"{REPOSITORY} nao e a raiz do repositorio"

    match = pattern.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"{path} nao declara a versao no formato esperado"

    return int(match.group(1)), int(match.group(2))


def test_o_lock_resolve_no_piso_declarado() -> None:
    declarado = _floor(REPOSITORY / "pyproject.toml", VERSION_FLOOR)
    lock = _floor(REPOSITORY / "uv.lock", VERSION_FLOOR)

    assert lock == declarado, (
        f"o lock exige Python {lock[0]}.{lock[1]} e o projeto declara {declarado[0]}.{declarado[1]}"
    )


def test_o_pyright_verifica_no_piso_declarado() -> None:
    declarado = _floor(REPOSITORY / "pyproject.toml", VERSION_FLOOR)
    pyright = _floor(REPOSITORY / "pyproject.toml", PYRIGHT_VERSION)

    assert pyright == declarado, (
        f"pyright verifica em {pyright[0]}.{pyright[1]} e o projeto declara "
        f"{declarado[0]}.{declarado[1]}: o piso mais baixo fica sem verificacao de tipo"
    )
