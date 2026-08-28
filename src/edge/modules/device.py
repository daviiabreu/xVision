"""Onde a inferencia roda, escolhido na chamada e nunca cravado no codigo do modulo.

Os sete donos desenvolvem em tres sistemas diferentes e o alvo e uma Jetson, entao cravar `cuda`
quebra o Mac, cravar `mps` quebra o Linux e cravar `cpu` desperdicia a placa. A variavel de
ambiente ganha de tudo porque quem opera precisa poder forcar sem editar arquivo, e e o que
permite a suite rodar identica no CI, que nao tem placa nenhuma.

Nao importa torch no topo: quem instala so a borda sem extra de modelo nao carrega torch, e um
import aqui obrigaria todo mundo a ter a pilha inteira para ler um caminho de arquivo.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, cast

DEVICE_ENV = "XVISION_DEVICE"


def preferred_device() -> str:
    """`cpu` e o piso, e ele nunca falha. Acelerador so entra quando o torch instalado confirma."""
    declarado = os.environ.get(DEVICE_ENV)
    if declarado:
        return declarado
    try:
        torch = cast("Any", import_module("torch"))
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
