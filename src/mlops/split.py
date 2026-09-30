"""Particao por partida, por hash salgado, para oito pessoas usarem a mesma sem coordenar nada.

Agrupa por partida e nao por clipe porque os 604 clipes uteis moram em 28 partidas, e dentro de
uma partida se repetem gramado, angulo de camera, iluminacao e os mesmos 22 jogadores.
Particionar por clipe poe vizinhos quase identicos dos dois lados e infla toda metrica.

Usa sal e nao lista porque a lista das 28 partidas e dado do parceiro. O sal resolve
reprodutibilidade e sigilo de uma vez, e o arquivo versionado fica com quatro linhas sem um dado
da Footstats dentro.

Sao cinco folds e nao 70/15/15 porque a validacao de uma particao 70/15/15 fica com 91 clipes e,
dentro dela, oito positivos de `looked_before`. F1 sobre oito positivos muda de 0,0 para 0,5 com
um acerto, entao o numero que diria que o time melhorou seria ruido.

Nao guarda identificador do parceiro, nao escolhe o sal em tempo de execucao e nao obriga ninguem
a usar fold: quem avalia regra sobre tudo ignora o campo.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from edge.paths import config_file

if sys.version_info >= (3, 11):  # pragma: no cover - o piso e 3.10, e e nele que a suite roda
    import tomllib
else:
    import tomli as tomllib

AGRUPAMENTO = "match"
SPLIT_NOME = "split.toml"


def split_config() -> Path:
    """Funcao, e nao constante: congelar no import faz a variavel de ambiente nao ter efeito,
    e derivar de `__file__` quebra quando o pacote e instalado como wheel.
    """
    return config_file(SPLIT_NOME)


DIGITOS = 16


class SplitError(ValueError): ...


@dataclass(frozen=True)
class SplitConfig:
    salt: str
    folds: int
    grouped_by: str = AGRUPAMENTO

    def __post_init__(self) -> None:
        if not self.salt:
            raise SplitError("a particao precisa de salt, que e o que a torna reproduzivel")
        if self.folds < 2:
            raise SplitError(f"folds={self.folds}, e particao com menos de dois nao particiona")
        if self.grouped_by != AGRUPAMENTO:
            raise SplitError(
                f"grouped_by={self.grouped_by!r}: a particao agrupa por {AGRUPAMENTO!r}, porque "
                "clipes da mesma partida dividem gramado, camera e jogadores, e separa-los infla "
                "toda metrica"
            )


def read_split(source: str | Path) -> SplitConfig:
    texto = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    documento = cast("dict[str, object]", tomllib.loads(texto))
    return SplitConfig(
        salt=cast("str", documento.get("salt", "")),
        folds=int(cast("int", documento.get("folds", 0))),
        grouped_by=cast("str", documento.get("grouped_by", AGRUPAMENTO)),
    )


def fold_of(match_id: str, cfg: SplitConfig) -> int:
    bruto = hashlib.sha256(f"{cfg.salt}:{match_id}".encode()).hexdigest()[:DIGITOS]
    return int(bruto, 16) % cfg.folds


DOBRAS: Mapping[str, frozenset[int]] = {
    "ajuste": frozenset({0, 1, 2}),
    "reservado": frozenset({3, 4}),
    "tudo": frozenset({0, 1, 2, 3, 4}),
}
"""Quem calibra limiar ve `ajuste`; o numero que vale sai de `reservado`, que ninguem otimizou.

A dobra e por partida e nao por clipe, entao um clipe de ajuste e um reservado nunca vem do mesmo
gramado, da mesma camera nem da mesma iluminacao. Sem isso, um limiar de cor de gramado ajustado
num jogo apareceria como generalizacao quando so decorou o estadio.

Reservar duas das cinco dobras deixa 24 clipes para ajustar e 14 para medir. Catorze e pouco e o
intervalo de Wilson sobre eles e largo; o alternativa era nao reservar nada, e ai o numero final
nao significaria nada.

Mora aqui, e nao no `cli`, porque `mlops` tambem precisa dela e nao pode importar o `cli`: a
calibracao do INT8 so pode ver `ajuste`, e uma segunda copia do conjunto seria o lugar por onde o
pacote reservado entraria calado.
"""
