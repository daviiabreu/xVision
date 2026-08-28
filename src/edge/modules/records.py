"""O minimo que cada especie promete devolver, conferido por etapa.

O runner le chaves dos registros para montar a percepcao, e ate aqui lia sem conferir: um modulo
que devolvesse `box` sem `confidence` passava nos seis testes de conformidade, reportava `ok`, e
matava a execucao inteira com `KeyError` em `assemble`, que roda fora do try. Erro que passa como
acerto e depois some do relatorio e o pior modo de falha que este runner tem.

A tabela diz o minimo, nunca o maximo. `players` que devolva `velocity` continua valido, porque
chave extra e informacao a mais e nao contrato quebrado. Faixa de valor tambem nao e cobrada:
`confidence` calibrada e exigencia que nem todo detector cumpre, e para `field`, que sai de ajuste
de homografia com erro de reprojecao, confianca nao significa nada.

Nao e JSON Schema de proposito. Registro nunca cruza fronteira de processo, e validar 633 clipes
com `jsonschema` custa tempo na Jetson sem comprar nada: aqui e diferenca de conjuntos.

`box` e `x1, y1, x2, y2` em pixel do quadro original, e nao do tensor pre-processado.
"""

from __future__ import annotations

from collections.abc import Mapping

from edge.modules.base import ModuleError, ModuleOutput

REQUIRED: Mapping[str, frozenset[str]] = {
    "players": frozenset({"box", "confidence"}),
    "ball": frozenset({"box", "confidence"}),
    "field": frozenset(),
    "tracking": frozenset({"track_id"}),
    "identity": frozenset({"track_id"}),
    "pose": frozenset({"track_id", "keypoints"}),
    "shot": frozenset({"frame_index"}),
}


class RecordError(ModuleError): ...


def check_records(output: ModuleOutput) -> None:
    exigidas = REQUIRED[output.kind]
    if not exigidas:
        return
    for quadro, registros in zip(output.frames, output.records, strict=True):
        for posicao, registro in enumerate(registros):
            faltando = exigidas - set(registro)
            if faltando:
                raise RecordError(
                    f"{output.kind} devolveu no quadro {quadro}, registro {posicao}, sem "
                    f"{', '.join(sorted(faltando))}. A especie promete "
                    f"{', '.join(sorted(exigidas))}"
                )
