"""O peso enderecado pelo conteudo, e nao pelo nome do arquivo que o descreve.

Ate aqui a chave do cache de percepcao usava `ModuleSpec.artifact`, que e o caminho do manifesto.
Trocar o peso dentro dele sem trocar o nome mantinha a chave e servia saida velha, calada. O
digest e sobre os pares de caminho e sha256 dos arquivos, ordenados, entao ele muda quando o peso
muda e nao muda quando o manifesto e renomeado.

Sao varios arquivos de proposito. TensorFlow SavedModel e diretorio, OpenVINO IR e `.xml` mais
`.bin`, e ONNX com dado externo tem `.onnx.data` ao lado. Manifesto de arquivo unico excluiria
duas pilhas por acidente, e a fundacao serve seis pessoas com pilhas diferentes.

Peso de hub pode omitir o digest. Pelo menos metade dos donos vai obter o primeiro modelo assim,
com `ultralytics` baixando sozinho ou `from_pretrained`, e proibir empurraria a pessoa a mentir no
manifesto. A garantia some, mas some visivel: o `run.json` mostra execucao sem identidade de peso
em vez de fingir que tem uma.

Nao carrega o modelo, que e o `load()` do dono com o framework dele. Nao converte formato, nao
escolhe device e nao treina. Nao baixa de URL publica, e quando o arquivo nao esta la levanta
`ArtifactMissing` com o caminho exato onde por a mao, que e a saida obrigatoria para peso derivado
de dado restrito.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from edge.modules.base import MODULE_KINDS
from edge.paths import models_dir

if sys.version_info >= (3, 11):  # pragma: no cover - o piso e 3.10, e e nele que a suite roda
    import tomllib
else:
    import tomli as tomllib

CHUNK = 1 << 20


class ArtifactError(ValueError): ...


class ArtifactMissing(FileNotFoundError): ...


class DigestMismatch(ArtifactError): ...


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ArtifactManifest:
    kind: str
    name: str
    version: str
    entrypoint: str
    files: tuple[ArtifactFile, ...] = ()
    source: Mapping[str, str] = field(default_factory=lambda: cast("Mapping[str, str]", {}))

    @property
    def digest(self) -> str | None:
        if not self.files:
            return None
        canonico = json.dumps(
            [[f.path, f.sha256] for f in sorted(self.files, key=_por_caminho)],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonico.encode()).hexdigest()


def _por_caminho(arquivo: ArtifactFile) -> str:
    return arquivo.path


def read_manifest(path: Path) -> ArtifactManifest:
    if not path.is_file():
        raise ArtifactMissing(f"{path} nao existe")
    documento = cast("dict[str, object]", tomllib.loads(path.read_text(encoding="utf-8")))
    faltando = {"kind", "name", "version", "entrypoint"} - set(documento)
    if faltando:
        raise ArtifactError(f"{path.name} nao declara {', '.join(sorted(faltando))}")
    for campo in ("kind", "name", "version", "entrypoint"):
        if not isinstance(documento[campo], str):
            raise ArtifactError(
                f"{path.name}.{campo} = {documento[campo]!r}, e o campo espera texto"
            )
    kind = cast("str", documento["kind"])
    if kind not in MODULE_KINDS:
        raise ArtifactError(
            f"{path.name} declara kind {kind!r}, e as especies conhecidas sao "
            f"{', '.join(sorted(MODULE_KINDS))}"
        )
    declarados = documento.get("files", [])
    if not isinstance(declarados, list):
        raise ArtifactError(f"{path.name}.files precisa ser lista de tabelas")
    arquivos = tuple(
        _file(path.name, cast("dict[str, object]", entrada))
        for entrada in cast("list[object]", declarados)
    )
    entrypoint = cast("str", documento["entrypoint"])
    _dentro(path.name, entrypoint)
    if arquivos and entrypoint not in {f.path for f in arquivos}:
        raise ArtifactError(
            f"{path.name} declara entrypoint {entrypoint!r}, que nao esta entre os arquivos: "
            f"{', '.join(f.path for f in arquivos)}"
        )
    return ArtifactManifest(
        kind=kind,
        name=cast("str", documento["name"]),
        version=cast("str", documento["version"]),
        entrypoint=entrypoint,
        files=arquivos,
        source=cast("dict[str, str]", documento.get("source", {})),
    )


SHA256 = re.compile(r"[0-9a-f]{64}")


def _file(nome: str, entrada: dict[str, object]) -> ArtifactFile:
    faltando = {"path", "sha256"} - set(entrada)
    if faltando:
        raise ArtifactError(f"{nome} tem um [[files]] sem {', '.join(sorted(faltando))}")
    caminho, sha = entrada["path"], entrada["sha256"]
    if not isinstance(caminho, str) or not isinstance(sha, str):
        raise ArtifactError(f"{nome} tem um [[files]] com path ou sha256 que nao e texto")
    if not SHA256.fullmatch(sha):
        raise ArtifactError(f"{nome}: {sha!r} nao e um sha256 de 64 digitos hexadecimais")
    _dentro(nome, caminho)
    return ArtifactFile(path=caminho, sha256=sha, bytes=int(cast("int", entrada.get("bytes", 0))))


def _dentro(nome: str, caminho: str) -> None:
    """`Path(base) / "/etc/passwd"` e `/etc/passwd`, e `../` sai do diretorio do digest."""
    relativo = Path(caminho)
    if relativo.is_absolute() or ".." in relativo.parts:
        raise ArtifactError(
            f"{nome} declara {caminho!r}, que aponta para fora do diretorio do peso"
        )


def resolve(manifest: ArtifactManifest, *, root: Path | None = None) -> Path:
    base = _base(manifest, root)
    if not base.is_dir():
        raise ArtifactMissing(
            f"o peso de {manifest.name}-{manifest.version} nao esta em {base}. Ele nao se baixa "
            "sozinho porque e material derivado de dado restrito: ponha os arquivos ali"
        )
    for arquivo in manifest.files:
        destino = base / arquivo.path
        if not destino.is_file():
            raise ArtifactMissing(f"{destino} nao existe, e o manifesto o declara")
        lido = sha256_of(destino)
        if lido != arquivo.sha256:
            raise DigestMismatch(
                f"{arquivo.path} tem sha256 {lido[:12]} e o manifesto declara {arquivo.sha256[:12]}"
            )
    entrypoint = base / manifest.entrypoint
    if not entrypoint.exists():
        raise ArtifactMissing(f"{entrypoint} nao existe, e e o entrypoint que o manifesto declara")
    return entrypoint


def _base(manifest: ArtifactManifest, root: Path | None) -> Path:
    """Peso de hub nao tem digest, entao ele e enderecado por nome e versao.

    Abortar seria pior: o plano trata hub como caso legitimo de metade do time, e a garantia some
    de forma visivel rio abaixo, com `artifact_digest` nulo no registro da execucao.
    """
    raiz = models_dir() if root is None else root
    digest = manifest.digest
    if digest is None:
        return raiz / "hub" / f"{manifest.name}-{manifest.version}"
    return raiz / digest


def sha256_of(path: Path) -> str:
    digestor = hashlib.sha256()
    with path.open("rb") as arquivo:
        while bloco := arquivo.read(CHUNK):
            digestor.update(bloco)
    return digestor.hexdigest()
