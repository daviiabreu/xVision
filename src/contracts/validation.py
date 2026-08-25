"""Validacao contra o JSON Schema, usada na fronteira dos servicos e nos testes.

O `jsonschema` e importado **dentro** da funcao, nao no topo do modulo. Como o `__init__` do
pacote importa este modulo, importar no topo faria todo `import contracts` pagar o custo de
carregar o validador, inclusive em quem so monta e serializa o evento. Um teste em subprocesso
guarda essa propriedade.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from contracts.events import ContractError
from contracts.schema import load_schema


class SchemaValidationError(ContractError): ...


@lru_cache(maxsize=1)
def _validator() -> Any:
    from jsonschema import Draft202012Validator

    return Draft202012Validator(load_schema())


def validate_shot_event(payload: dict[str, Any]) -> None:
    from jsonschema.exceptions import ValidationError

    try:
        _validator().validate(payload)
    except ValidationError as error:
        raise SchemaValidationError(str(error)) from error
