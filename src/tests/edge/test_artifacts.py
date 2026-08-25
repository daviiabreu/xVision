"""O peso ganha endereco pelo conteudo, e nao pelo nome do arquivo que o descreve."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from edge.artifacts import (
    ArtifactError,
    ArtifactManifest,
    ArtifactMissing,
    DigestMismatch,
    read_manifest,
    resolve,
)

MANIFESTO = """
schema_version = "1.0"
kind = "players"
name = "demo"
version = "0.0.1"
entrypoint = "model.onnx"

[[files]]
path = "model.onnx"
sha256 = "{a}"
bytes = {na}

[[files]]
path = "model.onnx.data"
sha256 = "{b}"
bytes = {nb}
"""


def _sha(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


@pytest.fixture
def manifesto(tmp_path: Path) -> Path:
    a, b = b"peso", b"dados externos"
    caminho = tmp_path / "demo-0.0.1.toml"
    caminho.write_text(
        MANIFESTO.format(a=_sha(a), na=len(a), b=_sha(b), nb=len(b)), encoding="utf-8"
    )
    return caminho


def _guardado(raiz: Path, manifesto: ArtifactManifest) -> Path:
    assert manifesto.digest is not None
    return raiz / manifesto.digest


def _grava(destino: Path, a: bytes = b"peso", b: bytes = b"dados externos") -> None:
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "model.onnx").write_bytes(a)
    (destino / "model.onnx.data").write_bytes(b)


def test_a_manifest_describes_more_than_one_file_because_two_stacks_need_it(
    manifesto: Path,
) -> None:
    """SavedModel e diretorio, OpenVINO e `.xml` mais `.bin`, ONNX externo tem `.onnx.data`.

    Manifesto de arquivo unico excluiria duas pilhas por acidente, e o pedido e generico.
    """
    lido = read_manifest(manifesto)

    assert [f.path for f in lido.files] == ["model.onnx", "model.onnx.data"]
    assert lido.entrypoint == "model.onnx"


def test_two_different_weights_produce_different_digests(manifesto: Path, tmp_path: Path) -> None:
    """Era o defeito: trocar o peso sem trocar o nome do manifesto servia saida velha do cache."""
    outro = tmp_path / "outro.toml"
    outro.write_text(
        MANIFESTO.format(a=_sha(b"peso novo"), na=9, b=_sha(b"dados externos"), nb=14),
        encoding="utf-8",
    )

    assert read_manifest(manifesto).digest != read_manifest(outro).digest


def test_the_digest_does_not_change_when_the_manifest_file_is_renamed(
    manifesto: Path, tmp_path: Path
) -> None:
    copia = tmp_path / "com-outro-nome.toml"
    copia.write_text(manifesto.read_text(encoding="utf-8"), encoding="utf-8")

    assert read_manifest(manifesto).digest == read_manifest(copia).digest


def test_resolving_checks_every_file_against_its_declared_hash(
    manifesto: Path, tmp_path: Path
) -> None:
    lido = read_manifest(manifesto)
    raiz = tmp_path / "var-models"
    _grava(_guardado(raiz, lido), a=b"peso adulterado")

    with pytest.raises(DigestMismatch, match="model.onnx"):
        resolve(lido, root=raiz)


def test_resolving_returns_the_entrypoint_when_every_hash_matches(
    manifesto: Path, tmp_path: Path
) -> None:
    lido = read_manifest(manifesto)
    raiz = tmp_path / "var-models"
    _grava(_guardado(raiz, lido))

    assert resolve(lido, root=raiz) == _guardado(raiz, lido) / "model.onnx"


def test_a_missing_weight_says_exactly_where_to_put_it(manifesto: Path, tmp_path: Path) -> None:
    """Peso derivado de dado restrito nao se baixa sozinho: a saida e a pessoa por a mao."""
    lido = read_manifest(manifesto)
    raiz = tmp_path / "var-models"

    with pytest.raises(ArtifactMissing) as erro:
        resolve(lido, root=raiz)

    assert str(_guardado(raiz, lido)) in str(erro.value)


def test_a_manifest_from_a_hub_may_omit_the_digest(tmp_path: Path) -> None:
    """Metade dos donos vai obter o primeiro modelo de hub, e proibir empurraria a mentir."""
    caminho = tmp_path / "hub.toml"
    caminho.write_text(
        'schema_version = "1.0"\nkind = "players"\nname = "yolo"\nversion = "11n"\n'
        'entrypoint = "yolo11n.pt"\n\n[source]\ntype = "hub"\nref = "ultralytics/yolo11n"\n',
        encoding="utf-8",
    )

    lido = read_manifest(caminho)

    assert lido.files == ()
    assert lido.digest is None
    assert lido.source["type"] == "hub"


def test_a_manifest_without_the_fields_that_identify_it_is_refused(tmp_path: Path) -> None:
    caminho = tmp_path / "torto.toml"
    caminho.write_text('schema_version = "1.0"\nkind = "players"\n', encoding="utf-8")

    with pytest.raises(ArtifactError, match="name"):
        read_manifest(caminho)


def test_a_manifest_for_an_unknown_species_is_refused(tmp_path: Path) -> None:
    caminho = tmp_path / "especie.toml"
    caminho.write_text(
        'schema_version = "1.0"\nkind = "goleiro"\nname = "x"\nversion = "1"\n'
        'entrypoint = "m.onnx"\n',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="goleiro"):
        read_manifest(caminho)


def test_the_digest_is_unambiguous_even_when_a_path_contains_the_separators() -> None:
    """`"|".join(f"{p}:{s}")` colidia: um arquivo com `:` e `|` no nome imitava dois arquivos."""
    from edge.artifacts import ArtifactFile

    def com(files: tuple[ArtifactFile, ...]) -> str | None:
        return ArtifactManifest(
            kind="players", name="n", version="1", entrypoint="e", files=files
        ).digest

    x, y = "a" * 64, "b" * 64
    dois = com((ArtifactFile("a", x, 6), ArtifactFile("b", y, 7)))
    um = com((ArtifactFile(f"a:{x}|b", y, 7),))

    assert dois != um


def test_the_hash_reads_past_the_first_block(tmp_path: Path) -> None:
    """Toda fixture do repositorio cabe num bloco, entao o laco nunca rodava duas vezes.

    Dois pesos de 2 MB que so diferem depois do primeiro MiB dariam o mesmo hash sem isto, e essa
    e exatamente a faixa de tamanho que este modulo existe para identificar.
    """
    from edge.artifacts import CHUNK, sha256_of

    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"\0" * CHUNK + b"final a")
    b.write_bytes(b"\0" * CHUNK + b"final b")

    assert sha256_of(a) != sha256_of(b)


def test_an_entrypoint_that_is_not_among_the_files_is_refused(tmp_path: Path) -> None:
    """Um comando cuja funcao e conferir e devolver o entrypoint nao pode sair 0 apontando vazio."""
    caminho = tmp_path / "torto.toml"
    caminho.write_text(
        'schema_version = "1.0"\nkind = "players"\nname = "n"\nversion = "1"\n'
        'entrypoint = "nao-existe.onnx"\n\n[[files]]\npath = "m.onnx"\n'
        f'sha256 = "{"a" * 64}"\nbytes = 4\n',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="nao-existe.onnx"):
        read_manifest(caminho)


def test_a_path_that_escapes_the_digest_directory_is_refused(tmp_path: Path) -> None:
    """`Path(base) / "/etc/passwd"` e `/etc/passwd`, e `../` sai do diretorio do digest."""
    for fuga in ("/etc/passwd", "../fora.bin"):
        caminho = tmp_path / "fuga.toml"
        caminho.write_text(
            'schema_version = "1.0"\nkind = "players"\nname = "n"\nversion = "1"\n'
            f'entrypoint = "{fuga}"\n\n[[files]]\npath = "{fuga}"\n'
            f'sha256 = "{"a" * 64}"\nbytes = 4\n',
            encoding="utf-8",
        )

        with pytest.raises(ArtifactError, match="fora do diretorio"):
            read_manifest(caminho)


def test_a_sha256_that_is_not_a_sha256_is_refused(tmp_path: Path) -> None:
    caminho = tmp_path / "hash.toml"
    caminho.write_text(
        'schema_version = "1.0"\nkind = "players"\nname = "n"\nversion = "1"\n'
        'entrypoint = "m.onnx"\n\n[[files]]\npath = "m.onnx"\nsha256 = "curto"\nbytes = 4\n',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="sha256"):
        read_manifest(caminho)


def test_a_hub_manifest_resolves_to_the_entrypoint_instead_of_blowing_up(tmp_path: Path) -> None:
    """O plano trata hub como caso legitimo: a identidade sai ausente rio abaixo, nao aborta."""
    caminho = tmp_path / "hub.toml"
    caminho.write_text(
        'schema_version = "1.0"\nkind = "players"\nname = "yolo"\nversion = "11n"\n'
        'entrypoint = "yolo11n.pt"\n\n[source]\ntype = "hub"\nref = "ultralytics/yolo11n"\n',
        encoding="utf-8",
    )
    raiz = tmp_path / "var-models"
    (raiz / "hub" / "yolo-11n").mkdir(parents=True)
    (raiz / "hub" / "yolo-11n" / "yolo11n.pt").write_bytes(b"baixado pelo framework")

    assert resolve(read_manifest(caminho), root=raiz).name == "yolo11n.pt"


def test_a_manifest_field_of_the_wrong_type_says_so_instead_of_raising_key_error(
    tmp_path: Path,
) -> None:
    caminho = tmp_path / "tipo.toml"
    caminho.write_text(
        'schema_version = "1.0"\nkind = "players"\nname = 42\nversion = "1"\n'
        'entrypoint = "m.onnx"\n',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="name"):
        read_manifest(caminho)


def test_a_hub_manifest_whose_weight_was_not_downloaded_says_so(tmp_path: Path) -> None:
    """Manifesto de hub nao lista arquivo, entao so esta checagem impede devolver caminho vazio."""
    caminho = tmp_path / "hub.toml"
    caminho.write_text(
        'schema_version = "1.0"\nkind = "players"\nname = "yolo"\nversion = "11n"\n'
        'entrypoint = "yolo11n.pt"\n\n[source]\ntype = "hub"\nref = "u/yolo11n"\n',
        encoding="utf-8",
    )
    raiz = tmp_path / "var-models"
    (raiz / "hub" / "yolo-11n").mkdir(parents=True)

    with pytest.raises(ArtifactMissing, match="yolo11n.pt"):
        resolve(read_manifest(caminho), root=raiz)
