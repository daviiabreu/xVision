"""Escreve o OpenAPI que o codigo gera, para a revisao ver o contrato no diff da MR."""

from __future__ import annotations

import json
from pathlib import Path

from api import create_app

DESTINO = Path(__file__).resolve().parent / "docs" / "openapi.json"


def main() -> None:
    DESTINO.write_text(
        json.dumps(create_app().openapi(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(DESTINO)


if __name__ == "__main__":  # pragma: no cover - so o `python -m` chega aqui
    main()
