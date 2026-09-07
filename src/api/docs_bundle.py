"""Empacota a colecao Bruno versionada para download direto pela interface."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

COLECAO = Path(__file__).resolve().parent / "docs" / "xvision"


def bruno_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as arquivo:
        for caminho in sorted(COLECAO.rglob("*")):
            if caminho.is_file():
                arquivo.write(caminho, Path("xvision") / caminho.relative_to(COLECAO))
    return buffer.getvalue()
