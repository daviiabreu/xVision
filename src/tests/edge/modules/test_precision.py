"""O vocabulario de precisao que as quatro especies com peso do Ultralytics compartilham.

Uma tabela so porque o erro de digitacao tem de doer igual em `ball`, `players`, `field` e `pose`:
com cada especie inventando o seu, `quantize = "FP16"` passaria numa e levantaria noutra, e a
divergencia de acuracia entre elas viraria efeito de configuracao em vez de efeito de precisao.
"""

from __future__ import annotations

import pytest

from edge.modules.base import ModuleError
from edge.modules.precision import ACEITAS, PADRAO, ajuste_do_predict, precisao_valida


def test_o_padrao_e_precisao_cheia() -> None:
    """Meia precisao ainda nao foi medida contra gabarito, entao ninguem a ganha sem pedir."""
    assert PADRAO == "fp32"
    assert ACEITAS == ("fp32", "fp16")


def test_a_precisao_declarada_volta_como_texto() -> None:
    assert precisao_valida("modules.ball.quantize", "fp16", ModuleError) == "fp16"


@pytest.mark.parametrize("valor", ["fp8", "FP16", True, 16, None, "half"])
def test_valor_fora_do_vocabulario_levanta_dizendo_o_que_e_aceito(valor: object) -> None:
    """Aviso calado aqui e meia precisao silenciosa, e ninguem descobre pela saida do modelo."""
    with pytest.raises(ModuleError, match="fp32.*fp16") as erro:
        precisao_valida("modules.ball.quantize", valor, ModuleError)

    assert "modules.ball.quantize" in str(erro.value)


def test_o_padrao_nao_manda_quantize_ao_predict() -> None:
    """Nao mandar e a prova de que o caminho de hoje continua identico, byte a byte de kwargs."""
    assert ajuste_do_predict(PADRAO) == {}


def test_meia_precisao_vira_o_parametro_que_o_ultralytics_entende() -> None:
    """`half=True` esta depreciado no 8.4 e mediu 62,4 ms contra 28,6 ms de `quantize='fp16'`."""
    assert ajuste_do_predict("fp16") == {"quantize": "fp16"}
