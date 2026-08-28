"""Saida de modulo gravada em disco, endereçada pela identidade do que foi computado.

Quem esta escrevendo uma feature itera em segundos em vez de esperar a inferencia inteira a cada
tentativa. A chave e o conteudo do que foi computado, e nao o nome de nada: clipe, spec de
pre-processamento, indices de quadro, saida dos modulos de que ele depende e o digest do peso.
Assim uma entrada so e reaproveitada quando descreve exatamente a mesma computacao.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from edge.artifacts import ArtifactError, read_manifest
from edge.modules.base import (
    FrameBatch,
    ModuleError,
    ModuleInputs,
    ModuleOutput,
    ModuleSpec,
    PerceptionModule,
)
from edge.paths import cache_dir


def implementation_id(module: PerceptionModule) -> str:
    """Quem implementa, e em que estado o codigo dela estava.

    So o caminho da classe nao basta, e o buraco aparece exatamente no fluxo desta sprint. Medido:
    editar o corpo de um stub, sem mudar nome nem versao nem caminho, e rodar de novo servia o
    resultado anterior. A docstring do modulo promete que a chave e o conteudo do que foi
    computado, e para modulo escrito em Python o conteudo e o codigo.

    O digest do peso cobre modelo com arquivo de pesos, e nao cobre a fase de trocar stub por
    implementacao, que e onde o time esta.

    O arquivo inteiro entra, e nao so a classe: um modulo que importe funcao auxiliar do mesmo
    arquivo muda de comportamento sem que a classe mude uma linha. Trocar dependencia de outro
    arquivo continua fora do alcance, e isso esta declarado: quem mexer em varios arquivos limpa
    o cache, ou espera resultado de antes.

    Sem arquivo legivel, o que acontece com classe definida em memoria ou vinda de extensao
    compilada, a identidade cai de volta no caminho da classe, que e o comportamento de antes.
    """
    classe = type(module)
    caminho = f"{classe.__module__}:{classe.__qualname__}"
    return f"{caminho}@{_fonte_de(classe)}"


def _fonte_de(classe: type) -> str:
    try:
        bruto = Path(inspect.getfile(classe)).read_bytes()
    except (OSError, TypeError, ValueError):
        return "sem-fonte"
    return hashlib.sha256(bruto).hexdigest()[:16]


def perception_key(
    clip_id: str,
    spec: ModuleSpec,
    indices: Sequence[int],
    ctx: ModuleInputs,
    impl: str,
    *,
    artifact_digest: str | None = None,
    tuning: str | None = None,
) -> str:
    """A chave do cache. `tuning` e o ajuste em vigor do modulo, e sem ele a chave mente.

    `implementation_id` hasheia o arquivo do modulo e o digest cobre o peso, mas o ajuste que vem do
    TOML nao aparece em nenhum dos dois: com `conf` na configuracao, baixar o limiar de 0,25 para
    0,10 deixava a chave identica e a execucao seguinte servia a percepcao antiga, medindo o valor
    antigo como se fosse o novo. O mesmo vale para `imgsz` e para o numero de ladrilhos."""
    canonico = json.dumps(
        [
            clip_id,
            impl,
            f"{spec.name}@{spec.version}",
            spec.kind,
            spec.frames,
            list(spec.needs),
            repr(spec.batching),
            artifact_digest or spec.artifact,
            spec.preprocess.fingerprint,
            tuning or "",
            list(indices),
            {kind: _to_json(ctx[kind]) for kind in sorted(spec.needs) if kind in ctx},
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonico.encode()).hexdigest()


class PerceptionCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = cache_dir() if root is None else root

    def get(self, key: str) -> ModuleOutput | None:
        try:
            guardado = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            return _from_json(guardado)
        except (KeyError, TypeError, AttributeError, ModuleError):
            return None

    def put(self, key: str, output: ModuleOutput) -> None:
        destino = self._path(key)
        destino.parent.mkdir(parents=True, exist_ok=True)
        parcial = destino.with_suffix(f".{os.getpid()}.part")
        parcial.write_text(json.dumps(_to_json(output)), encoding="utf-8")
        parcial.replace(destino)

    def clear(self) -> int:
        arquivos = sorted(self.root.glob("*.json")) + sorted(self.root.glob("*.part"))
        for arquivo in arquivos:
            arquivo.unlink()
        return len(arquivos)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"


def run_module(
    module: PerceptionModule,
    batch: FrameBatch,
    ctx: ModuleInputs,
    *,
    cache: PerceptionCache,
    clip_id: str,
) -> ModuleOutput:
    key = perception_key(
        clip_id,
        module.spec,
        batch.indices,
        ctx,
        implementation_id(module),
        artifact_digest=artifact_digest_of(module.spec.artifact),
        tuning=tuning_of(module),
    )
    guardado = cache.get(key)
    if guardado is not None and _responde_ao_pedido(guardado, module.spec.kind, batch.indices):
        return guardado
    saida = _como_o_cache_devolve(module.infer(batch, ctx))
    cache.put(key, saida)
    return saida


def tuning_of(module: PerceptionModule) -> str | None:
    """O ajuste em vigor do modulo, lido por `getattr` e nao por metodo do protocolo.

    Alargar `PerceptionModule` obrigaria os sete donos e todo duble de teste a crescer um metodo que
    a maioria nao usa: `field` e `identity` sao stubs sem nada para ajustar. Modulo que nao declara
    ajuste devolve `None` e a chave fica como era, entao nada existente muda de endereco.
    """
    impressao = getattr(module, "fingerprint", None)
    if not callable(impressao):
        return None
    valor: object = impressao()
    return None if valor is None else str(valor)


def artifact_digest_of(artifact: str | None) -> str | None:
    """O digest do peso, quando o manifesto existe e o descreve.

    Sem isto o parametro fica solto: a chave continua usando o caminho do manifesto, trocar o peso
    dentro dele nao muda nada, e o cache serve saida velha depois de retreinar. Manifesto ausente
    ou de hub devolve None e a chave cai de volta no caminho, que e o comportamento de antes e o
    unico possivel sem identidade de peso.
    """
    if not artifact:
        return None
    caminho = Path(artifact)
    if not caminho.is_file():
        return None
    try:
        return read_manifest(caminho).digest
    except ArtifactError:
        return None


def _como_o_cache_devolve(output: ModuleOutput) -> ModuleOutput:
    return _from_json(_to_json(output))


def _responde_ao_pedido(saida: ModuleOutput, kind: str, indices: tuple[int, ...]) -> bool:
    return saida.kind == kind and saida.frames == indices


def _to_json(output: ModuleOutput) -> dict[str, Any]:
    return {
        "kind": output.kind,
        "frames": list(output.frames),
        "records": [[dict(registro) for registro in quadro] for quadro in output.records],
    }


def _from_json(guardado: object) -> ModuleOutput:
    dados = cast("dict[str, Any]", guardado)
    return ModuleOutput(
        kind=dados["kind"],
        frames=tuple(dados["frames"]),
        records=tuple(tuple(quadro) for quadro in dados["records"]),
    )
