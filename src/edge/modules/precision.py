"""A precisao de inferencia que cada especie declara no TOML, num vocabulario so.

Quatro especies carregam peso do Ultralytics (`ball`, `players`, `field` e `pose`) e as quatro
pagam o mesmo custo de tempo pela precisao cheia. Medido nesta Jetson Orin, com ultralytics 8.4:
`players` a 960 custa 45,2 ms em fp32 e 28,6 ms com `quantize='fp16'`, e um ladrilho da bola a
1600 cai de 66,8 ms para 53,3 ms. O detector da bola sozinho e cerca de 70% do tempo de um clipe.

O parametro e `quantize` e nao `half`. O `half` esta depreciado no 8.4 e, alem de depreciado, e
mais lento: os mesmos 960 de `players` medem 62,4 ms com `half=True`, contra os 45,2 ms de fp32.
Ele nao e um sinonimo velho do caminho novo, e sim outro caminho.

O padrao e fp32 e nao muda: a divergencia de acuracia de meia precisao ainda nao foi medida contra
gabarito, e ninguem pode herdar meia precisao sem ter pedido. Por isso `ajuste_do_predict` devolve
dicionario vazio no padrao, em vez de mandar `quantize='fp32'`: assim a chamada de quem nao
declarou nada continua sendo, argumento por argumento, a mesma de antes desta alavanca existir.

O vocabulario para aqui, em duas palavras. O Ultralytics tambem aceita 8, 'int8' e os esquemas
`w8a8` e afins, mas para um peso `.pt` em `predict` so 16 muda alguma coisa (o predictor le
`quantize == 16`): declarar `int8` seria aceito e nao faria nada, que e o modo de falha calado que
este projeto recusa em configuracao.
"""

from __future__ import annotations

PADRAO = "fp32"
ACEITAS = (PADRAO, "fp16")


def precisao_valida(chave: str, valor: object, erro: type[Exception]) -> str:
    """Recusa o que nao esta no vocabulario, com o tipo de erro que a especie ja usa.

    O tipo vem de fora porque cada especie tem o seu (`AjusteDesconhecido` na bola,
    `FieldConfigError` no campo) e quem le a mensagem esta lendo a secao dela no TOML. A mensagem
    diz o que e aceito porque o erro tipico e vizinho do certo: `fp8`, `16`, `True`, `half`.
    """
    if not isinstance(valor, str) or valor not in ACEITAS:
        raise erro(f"{chave} = {valor!r}, e a precisao aceita e {' ou '.join(ACEITAS)}")
    return valor


def ajuste_do_predict(precisao: str) -> dict[str, str]:
    """O que entra nos kwargs do `predict`, que no padrao e nada."""
    return {} if precisao == PADRAO else {"quantize": precisao}
