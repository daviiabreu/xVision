"""O catalogo do parceiro: um registro por finalizacao, com clipe, rotulo e particao.

A amostra destes testes e sintetica e tem as mesmas colunas, as mesmas chaves e o mesmo formato
de nome do conjunto real. O dado real e restrito, nunca entra no repositorio nem no CI, entao o
que a suite cobra e o formato, e nao o conteudo.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mlops.catalog import CatalogError, load_catalog

TIROS = [
    {
        "id": "10",
        "id_match": "900",
        "body_part": "PERNA_DIREITA",
        "meters": "10",
        "is_goal": "true",
    },
    {"id": "11", "id_match": "900", "body_part": "CABECA", "meters": "25", "is_goal": "false"},
    {"id": "12", "id_match": "901", "body_part": "", "meters": "", "is_goal": "false"},
]
XG = [
    {
        "id": "555",
        "id_match_shot": "10",
        "expected_goals": "0.42",
        "marked": "true",
        "looked_before": "false",
        "occupied_area": "true",
        "goalkeaper_position": "false",
        "assist": "true",
        "type_of_play": "open play",
    },
    {
        "id": "556",
        "id_match_shot": "11",
        "expected_goals": "0.05",
        "marked": "",
        "looked_before": "",
        "occupied_area": "",
        "goalkeaper_position": "",
        "assist": "",
        "type_of_play": "set piece",
    },
]


def _csv(caminho: Path, linhas: list[dict[str, str]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


@pytest.fixture
def conjunto(tmp_path: Path) -> Path:
    _csv(tmp_path / "db" / "0005_match_shots_1390.csv", TIROS)
    _csv(tmp_path / "db" / "0006_match_shots_xg_1390.csv", XG)
    for pasta, nome, tamanho in (
        ("900", "900_004_Mexico_gol_10.mp4", 5000),
        ("900", "900_017_Mexico_off_11.mp4", 5000),
        ("901", "901_022_Suica_block_12.mp4", 200),
    ):
        destino = tmp_path / "cortes" / pasta / nome
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b"\0" * tamanho)
    return tmp_path


def test_the_clip_id_is_the_shot_id_and_never_the_file_name(conjunto: Path) -> None:
    """O quarto campo do nome e o desfecho, e em xG o desfecho e o alvo.

    Medido no conjunto real: 78 dos 633 arquivos tem `gol` no nome. Com `clip_id = path.stem`, o
    rotulo viaja dentro do identificador, vira prefixo do `shot_id` de todo evento e chega ate a
    chave do cache de percepcao. Qualquer coisa que aprenda sobre o identificador ganha o alvo.
    """
    catalogo = load_catalog(conjunto)

    identificadores = {r.shot_id for r in catalogo}
    assert identificadores == {"10", "11", "12"}
    for registro in catalogo:
        assert "gol" not in registro.shot_id
        assert registro.clip_path.stem not in registro.shot_id


def test_the_join_uses_id_match_shot_and_not_id(conjunto: Path) -> None:
    """`xg.id` casa 0 de 2557 com `match_shots.id` e nao levanta erro, so devolve tabela vazia."""
    por_id = {r.shot_id: r for r in load_catalog(conjunto)}

    assert por_id["10"].labels["expected_goals"] == "0.42"
    assert por_id["11"].labels["type_of_play"] == "set piece"


def test_a_shot_without_an_xg_row_keeps_its_own_columns(conjunto: Path) -> None:
    doze = {r.shot_id: r for r in load_catalog(conjunto)}["12"]

    assert doze.labels["is_goal"] == "false"
    assert "expected_goals" not in doze.labels


def test_labels_keep_the_raw_text_because_false_is_not_falsy(conjunto: Path) -> None:
    """As colunas booleanas do parceiro sao o texto 'true' e 'false', e `bool('false')` e True."""
    onze = {r.shot_id: r for r in load_catalog(conjunto)}["11"]

    assert onze.labels["looked_before"] == ""
    assert isinstance(onze.labels["is_goal"], str)


def test_a_truncated_clip_is_listed_and_marked_unplayable(conjunto: Path) -> None:
    """Os 24 truncados do conjunto real precisam aparecer na contagem, e nao sumir da lista."""
    por_id = {r.shot_id: r for r in load_catalog(conjunto)}

    assert por_id["12"].big_enough is False
    assert por_id["10"].big_enough is True


def test_every_record_carries_the_match_it_came_from(conjunto: Path) -> None:
    por_id = {r.shot_id: r for r in load_catalog(conjunto)}

    assert (por_id["10"].match_id, por_id["12"].match_id) == ("900", "901")


def test_a_dataset_without_the_csv_says_which_file_is_missing(tmp_path: Path) -> None:
    (tmp_path / "cortes").mkdir()

    with pytest.raises(CatalogError, match="match_shots"):
        load_catalog(tmp_path)


def test_every_record_carries_the_fold_so_nobody_has_to_compute_it(conjunto: Path) -> None:
    """Sem o campo, todo consumidor teria que lembrar de chamar `fold_of` com o config certo,
    que e exatamente a coordenacao que o sal existe para eliminar."""
    folds = {r.fold for r in load_catalog(conjunto)}

    assert folds <= set(range(5))


def test_clips_in_the_same_match_share_a_fold(conjunto: Path) -> None:
    por_id = {r.shot_id: r for r in load_catalog(conjunto)}

    assert por_id["10"].fold == por_id["11"].fold


def test_the_count_says_how_many_rows_had_no_clip(conjunto: Path) -> None:
    """2047 das 2680 linhas somem por nao terem clipe, e some-las caladas faz o dono procurar
    o defeito no glob."""
    from mlops.catalog import catalog_counts

    contagem = catalog_counts(conjunto)

    assert contagem.rows == 3
    assert contagem.with_clip == 3
    assert contagem.without_clip == 0


def test_a_file_outside_the_naming_pattern_is_listed_and_does_not_kill_the_catalog(
    conjunto: Path,
) -> None:
    """Um `.DS_Store` renomeado ou um download pela metade nao pode impedir os 633 de carregar."""
    from mlops.catalog import catalog_counts

    (conjunto / "cortes" / "900" / "avulso.mp4").write_bytes(b"\0" * 5000)

    assert len(load_catalog(conjunto)) == 3
    assert catalog_counts(conjunto).ignored_files == ("avulso.mp4",)


def test_a_shot_without_a_clip_is_counted_and_not_only_dropped(conjunto: Path) -> None:
    """Somem 2047 das 2680 linhas por nao ter clipe, e some-las caladas manda procurar
    o defeito no lugar errado."""
    import csv as csvmod

    from mlops.catalog import catalog_counts

    caminho = conjunto / "db" / "0005_match_shots_1390.csv"
    with caminho.open("a", newline="", encoding="utf-8") as arquivo:
        csvmod.DictWriter(arquivo, fieldnames=list(TIROS[0])).writerow(
            {
                "id": "99",
                "id_match": "902",
                "body_part": "CABECA",
                "meters": "5",
                "is_goal": "false",
            }
        )

    contagem = catalog_counts(conjunto)

    assert (contagem.rows, contagem.with_clip, contagem.without_clip) == (4, 3, 1)
    assert len(load_catalog(conjunto)) == 3


def test_a_renamed_partner_column_says_so_instead_of_raising_key_error(conjunto: Path) -> None:
    """A falha mais provavel na vida real e o proximo dump mudar de esquema."""
    caminho = conjunto / "db" / "0006_match_shots_xg_1390.csv"
    caminho.write_text(
        caminho.read_text(encoding="utf-8").replace("id_match_shot", "shot"), encoding="utf-8"
    )

    with pytest.raises(CatalogError, match="id_match_shot"):
        load_catalog(conjunto)


def test_two_different_matches_do_not_share_a_fold_by_accident(conjunto: Path) -> None:
    """Sem isto, `fold=0` cravado passava verde e toda avaliacao fora de fold viraria dentro."""
    from mlops.split import fold_of, read_split, split_config

    particao = read_split(split_config())
    por_id = {r.shot_id: r for r in load_catalog(conjunto)}

    assert por_id["10"].fold == fold_of("900", particao)
    assert por_id["12"].fold == fold_of("901", particao)
