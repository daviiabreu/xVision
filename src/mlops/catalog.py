"""Um registro por finalizacao, juntando o clipe, os rotulos do parceiro e a particao.

Le os CSVs com o modulo `csv` da biblioteca padrao, de proposito: pandas acrescentaria dependencia
que precisa de roda ARM64 para uma juncao de duas tabelas.

Duas chaves parecidas, e uma delas e armadilha silenciosa. `match_shots_xg.id` casa 0 de 2557 com
`match_shots.id`, e um merge por ela nao levanta erro nenhum: devolve tabela vazia, e quem
escreveu procura o defeito no lugar errado. A chave certa e `match_shots_xg.id_match_shot`.

`clip_id` e o `match_shots.id`, que e o quinto campo do nome do arquivo, e nunca o nome do
arquivo. O quarto campo e o desfecho da finalizacao, `off`, `on`, `block`, `gol` ou `wood`, e em
xG o desfecho e o alvo: com o nome como identificador, o rotulo viaja dentro do `clip_id`, vira
prefixo do `shot_id` de todo evento e chega ate a chave do cache de percepcao.

`labels` guarda o texto cru. As colunas booleanas do parceiro sao o texto `true` e `false`, com
vazio para ausente, entao `bool(valor)` devolve `True` para `'false'`. A conversao mora em
`labels`, declarada, e nao escondida aqui no leitor.

`big_enough` compara tamanho e nao abre nada, e o nome diz isso: os 24 arquivos truncados do
conjunto tem 261 bytes e o menor abrivel tem 2 MB, entao o limiar acerta hoje por estar num vao
de quatro ordens de grandeza. Download interrompido de 5 MB passaria, e quem quer certeza chama
`edge.frames.provider.probe`.

Nao abre video, nao decodifica quadro e nao importa cv2. Nao normaliza vocabulario de rotulo. Nao
escreve nada em disco.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from edge.paths import data_root
from mlops.split import SplitConfig, fold_of, read_split, split_config

SHOTS = "0005_match_shots_1390.csv"
XG = "0006_match_shots_xg_1390.csv"
CLIPS = "cortes"
SHOT_ID = "id"
XG_SHOT_ID = "id_match_shot"
MATCH_ID = "id_match"
CAMPOS_NO_NOME = 5
PLAYABLE_MIN_BYTES = 4096
COLUNAS_SHOTS = frozenset({SHOT_ID, MATCH_ID})
COLUNAS_XG = frozenset({XG_SHOT_ID})


class CatalogError(Exception): ...


@dataclass(frozen=True)
class ShotRecord:
    shot_id: str
    match_id: str
    clip_path: Path
    big_enough: bool
    fold: int
    labels: Mapping[str, str]


@dataclass(frozen=True)
class CatalogCounts:
    rows: int
    with_clip: int
    without_clip: int
    big_enough: int
    matches: int
    ignored_files: tuple[str, ...] = ()


def load_catalog(
    root: Path | None = None, split: SplitConfig | None = None
) -> tuple[ShotRecord, ...]:
    raiz = data_root() if root is None else root
    particao = read_split(split_config()) if split is None else split
    tiros = _read(raiz / "db" / SHOTS, COLUNAS_SHOTS)
    xg = {linha[XG_SHOT_ID]: linha for linha in _read(raiz / "db" / XG, COLUNAS_XG)}
    clipes, _ = _clips(raiz / CLIPS)
    registros: list[ShotRecord] = []
    for tiro in tiros:
        shot_id = tiro[SHOT_ID]
        caminho = clipes.get(shot_id)
        if caminho is None:
            continue
        rotulos = {k: v for k, v in tiro.items() if k != SHOT_ID}
        extra = xg.get(shot_id)
        if extra is not None:
            rotulos.update({k: v for k, v in extra.items() if k not in (SHOT_ID, XG_SHOT_ID)})
        registros.append(
            ShotRecord(
                shot_id=shot_id,
                match_id=tiro[MATCH_ID],
                clip_path=caminho,
                big_enough=caminho.stat().st_size >= PLAYABLE_MIN_BYTES,
                fold=fold_of(tiro[MATCH_ID], particao),
                labels=rotulos,
            )
        )
    return tuple(registros)


def catalog_counts(root: Path | None = None) -> CatalogCounts:
    raiz = data_root() if root is None else root
    tiros = _read(raiz / "db" / SHOTS, COLUNAS_SHOTS)
    clipes, ignorados = _clips(raiz / CLIPS)
    com_clipe = [t for t in tiros if t[SHOT_ID] in clipes]
    return CatalogCounts(
        rows=len(tiros),
        with_clip=len(com_clipe),
        without_clip=len(tiros) - len(com_clipe),
        big_enough=sum(
            1 for t in com_clipe if clipes[t[SHOT_ID]].stat().st_size >= PLAYABLE_MIN_BYTES
        ),
        matches=len({t[MATCH_ID] for t in com_clipe}),
        ignored_files=ignorados,
    )


def _read(caminho: Path, exigidas: frozenset[str]) -> list[dict[str, str]]:
    if not caminho.is_file():
        raise CatalogError(f"{caminho} nao existe, e o catalogo precisa dele")
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        faltando = exigidas - set(leitor.fieldnames or ())
        if faltando:
            raise CatalogError(
                f"{caminho.name} nao tem {', '.join(sorted(faltando))}. O dump do parceiro mudou "
                "de esquema, ou e outro arquivo"
            )
        return list(leitor)


def _clips(raiz: Path) -> tuple[dict[str, Path], tuple[str, ...]]:
    encontrados: dict[str, Path] = {}
    ignorados: list[str] = []
    for caminho in sorted(raiz.rglob("*.mp4")):
        campos = caminho.stem.split("_")
        if len(campos) != CAMPOS_NO_NOME or not campos[-1].isdigit():
            ignorados.append(caminho.name)
            continue
        encontrados[campos[-1]] = caminho
    return encontrados, tuple(ignorados)
