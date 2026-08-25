"""As duas raizes do sistema de arquivos, com significado unico cada uma.

`data/` e o que o parceiro mandou e nada nosso escreve la. `var/` e tudo que o sistema produz:
cache de percepcao, pesos baixados, execucoes e saida. A separacao existe porque o dump da
Footstats e material restrito de 3,5 GB, e misturar o que escrevemos com ele faz o contexto de um
`docker build` passar a incluir video de partida.

As duas sao relativas ao diretorio de trabalho, e nao ao arquivo. Derivar de `__file__` parece
mais robusto e nao e: instalado como wheel, a raiz cairia dentro de `site-packages` e o cache
gravaria na camada da imagem, ou estouraria em filesystem somente leitura. Relativo ao diretorio
de trabalho vale igual instalado ou nao, e quem precisa de outro lugar tem as duas variaveis.

So existe aqui o que tem chamador: cache de percepcao, saida, execucoes e pesos de modelo. A
API usa `var_dir()` direto para o que e dela, o banco de videos enviados e os arquivos em
transito, porque a borda nao pode conhecer conceito da API.

`var/` nao guarda fonte de verdade. Apagar a pasta inteira deixa o sistema funcionando, so mais
lento, e isso e verificado por teste. Ela tambem nao tem teto nem expiracao: cresce, e quem limpa
e a pessoa.

Nada aqui e resolvido no import. Congelar o valor faria `monkeypatch.setenv` num teste nao ter
efeito sobre modulo ja importado, e o teste apagaria o cache real da maquina achando que estava
isolado.
"""

from __future__ import annotations

import os
from pathlib import Path

VAR_ENV = "XVISION_VAR_DIR"
DATA_ENV = "XVISION_DATA_DIR"
CONFIG_ENV = "XVISION_CONFIG_DIR"

DEFAULT_VAR = Path("var")
DEFAULT_DATA = Path("data") / "footstats"
DEFAULT_CONFIG = Path("config")


class DataUnavailable(FileNotFoundError): ...


def var_dir() -> Path:
    declarado = os.environ.get(VAR_ENV)
    return Path(declarado) if declarado else DEFAULT_VAR


def data_root(*, required: bool = True) -> Path:
    declarado = os.environ.get(DATA_ENV)
    raiz = Path(declarado) if declarado else DEFAULT_DATA
    if required and not raiz.is_dir():
        raise DataUnavailable(
            f"o conjunto do parceiro nao esta em {raiz.resolve()}. Ele e restrito e nao vem no "
            f"repositorio: ponha o dump ali, ou aponte {DATA_ENV} para onde ele esta"
        )
    return raiz


class ConfigUnavailable(FileNotFoundError): ...


def config_dir() -> Path:
    """A terceira raiz: o que a pessoa edita, e nem `data/` nem `var/`.

    Ela e relativa ao diretorio de trabalho pelo mesmo motivo das outras duas, e o motivo tem
    caso medido: `mlops` derivava o caminho de `__file__` com `parents[2]`, e instalado como
    wheel isso resolvia para dentro de `site-packages`, onde `config/` nao existe. `xvision run
    --all` morria com `FileNotFoundError` cru apontando para o venv.

    O `compose.yaml` monta `./config:/app/config:ro`, entao esta e a raiz que quem opera
    substitui sem tocar em codigo.
    """
    declarado = os.environ.get(CONFIG_ENV)
    return Path(declarado) if declarado else DEFAULT_CONFIG


def config_file(nome: str) -> Path:
    caminho = config_dir() / nome
    if not caminho.is_file():
        raise ConfigUnavailable(
            f"{caminho.resolve()} nao existe. Rode a partir da raiz do repositorio, ou aponte "
            f"{CONFIG_ENV} para a pasta onde a configuracao esta"
        )
    return caminho


PIPELINE_NOME = "pipeline.toml"


def pipeline_config() -> Path:
    """O caminho do `pipeline.toml`, resolvido na chamada e nao no import.

    Constante de modulo congelava o valor antes de qualquer variavel de ambiente ser lida, entao
    `XVISION_CONFIG_DIR` era aceita e ignorada: medido, apontando a variavel para uma copia com
    outro `impl`, as sete etapas saiam `ok` e o evento trazia o modulo antigo. O operador trocava
    o modelo, o relatorio dizia que rodou, e rodou o de antes.
    """
    return config_dir() / PIPELINE_NOME


def cache_dir() -> Path:
    return var_dir() / "cache" / "perception"


def out_dir() -> Path:
    return var_dir() / "out"


def runs_dir() -> Path:
    return var_dir() / "runs"


def models_dir() -> Path:
    return var_dir() / "models"
