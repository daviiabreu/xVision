"""Localizacao e leitura do JSON Schema, sem dependencia de terceiro.

O arquivo mora **dentro** do pacote e e lido por `importlib.resources`, nao por caminho
relativo ao repositorio. Fora do pacote ele nao entra no wheel, e a leitura quebra em
qualquer instalacao que nao seja editavel, que e exatamente o caso das imagens de conteiner.

E a fonte unica do contrato: borda e nuvem validam contra este mesmo arquivo, nunca contra
uma copia.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
SCHEMA_FILENAME = "shot_event.v1.json"
SCHEMA_PATH = Path(str(files("contracts") / SCHEMA_FILENAME))


def load_schema() -> dict[str, Any]:
    content = (files("contracts") / SCHEMA_FILENAME).read_text(encoding="utf-8")
    return json.loads(content)
