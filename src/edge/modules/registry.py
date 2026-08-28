"""Selecao da implementacao de cada especie a partir de `config/pipeline.toml`.

Trocar qual modelo roda em `players` e trocar uma linha de configuracao. Nenhum
arquivo de codigo muda, e nenhum dono de outra especie fica sabendo.

O que a entrada declara alem de `impl`, `artifact`, `frames` e `needs` vai para `options`, e o
registro nao interpreta nada disso: cada especie tem vocabulario proprio, e uma lista fechada de
campos aqui obrigaria este arquivo a mudar toda vez que um dono acrescentasse um limiar ao seu
modelo. O caso que abriu o campo: `conf` e os limites de forma do detector da bola estavam cravados
em `src/`, entao calibrar o limiar contra a dobra de ajuste exigia editar codigo, e o valor
calibrado nao aparecia em lugar nenhum da configuracao que descreve a execucao.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from edge.modules.base import (
    MODULE_KINDS,
    ModuleError,
    PerceptionModule,
    UnknownModuleKind,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - a borda suporta 3.10, onde tomllib ainda nao existe
    import tomli as tomllib

ENTRY_POINT_GROUP = "xvision.modules"


class UnknownImplementation(ModuleError): ...


class UnknownDependency(ModuleError): ...


class ModuleNotConfigurable(ModuleError): ...


CAMPOS_DA_ENTRADA = frozenset({"impl", "artifact", "frames", "needs"})
"""O que o registro entende. Todo o resto e ajuste da especie, e passa inteiro para ela."""

SEM_AJUSTE: Mapping[str, object] = MappingProxyType({})
"""Padrao imutavel: `dict` como `default_factory` deixaria o tipo parcialmente desconhecido."""


@runtime_checkable
class ConfigurableModule(Protocol):
    """Modulo que aceita ajuste vindo do TOML, alem de peso e conjunto de quadros.

    Opcional de proposito: `field` e `identity` sao stubs sem numero para ajustar, e obrigar todos
    a implementar um metodo vazio trocaria um erro claro na configuracao por seis linhas mortas.
    """

    def configure(self, options: Mapping[str, object]) -> None: ...


@dataclass(frozen=True)
class ModuleConfig:
    kind: str
    impl: str
    artifact: str | None = None
    frames: str | None = None
    needs: tuple[str, ...] | None = None
    options: Mapping[str, object] = SEM_AJUSTE


def read_config(source: str | Path) -> tuple[ModuleConfig, ...]:
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    documento = cast("dict[str, object]", tomllib.loads(text))
    modules = cast("dict[str, dict[str, object]]", documento.get("modules", {}))
    entries: list[ModuleConfig] = []
    for kind, entry in modules.items():
        key = f"modules.{kind}"
        impl = entry.get("impl")
        if not isinstance(impl, str):
            raise ModuleError(f"{key} precisa declarar impl")
        needs = cast("list[object] | None", entry.get("needs"))
        entries.append(
            ModuleConfig(
                kind=kind,
                impl=impl,
                artifact=_optional_str(key, "artifact", entry.get("artifact")),
                frames=_optional_str(key, "frames", entry.get("frames")),
                needs=_needs(key, needs),
                options={k: v for k, v in entry.items() if k not in CAMPOS_DA_ENTRADA},
            )
        )
    return tuple(entries)


def load_modules(configs: tuple[ModuleConfig, ...]) -> dict[str, PerceptionModule]:
    registry: dict[str, PerceptionModule] = {}
    for config in configs:
        key = f"modules.{config.kind}"
        module = _resolve(key, config.impl)()
        spec = module.spec
        if spec.kind != config.kind:
            raise UnknownModuleKind(f"{key} recebeu um modulo de kind {spec.kind!r}")
        module.spec = replace(
            spec,
            artifact=config.artifact if config.artifact is not None else spec.artifact,
            frames=config.frames if config.frames is not None else spec.frames,
            needs=config.needs if config.needs is not None else spec.needs,
        )
        unknown = tuple(n for n in module.spec.needs if n not in MODULE_KINDS)
        if unknown:
            raise UnknownDependency(f"{key} depende de especie desconhecida: {', '.join(unknown)}")
        if config.options:
            _configurar(key, module, config.options)
        registry[config.kind] = module
    return registry


def _configurar(key: str, module: PerceptionModule, options: Mapping[str, object]) -> None:
    """Ajuste declarado para modulo que nao aceita ajuste e erro, e nao um campo a ignorar.

    Ignorar seria o mesmo modo de falha que motivou o campo: quem escreveu `conf = 0.10` na secao
    errada leria o numero no arquivo, veria a execucao terminar sem reclamar e concluiria que o
    limiar mudou.
    """
    if not isinstance(module, ConfigurableModule):
        raise ModuleNotConfigurable(
            f"{key} declara {', '.join(sorted(options))}, e {type(module).__name__} nao aceita "
            "ajuste por configuracao"
        )
    module.configure(options)


def _needs(key: str, needs: list[object] | None) -> tuple[str, ...] | None:
    """`str(42)` produzia a especie `'42'`, que nunca existiria e so falharia mais tarde."""
    if needs is None:
        return None
    for item in needs:
        if not isinstance(item, str):
            raise ModuleError(f"{key}.needs tem {item!r}, e cada dependencia e o nome da especie")
    return tuple(cast("list[str]", needs))


def _optional_str(key: str, campo: str, value: object) -> str | None:
    """Devolver None calado para valor que nao e texto faz `artifact = 42` sumir do manifesto."""
    if value is None or isinstance(value, str):
        return value
    raise ModuleError(f"{key}.{campo} = {value!r}, e o campo espera texto")


def _resolve(key: str, impl: str) -> Callable[[], PerceptionModule]:
    if ":" in impl:
        path, _, attribute = impl.partition(":")
        try:
            return getattr(import_module(path), attribute)
        except (ImportError, AttributeError) as exc:
            raise UnknownImplementation(f"{key} aponta para {impl!r}: {exc}") from exc
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        if entry.name == impl:
            return entry.load()
    raise UnknownImplementation(
        f"{key} aponta para o entry point {impl!r}, ausente do grupo {ENTRY_POINT_GROUP!r}"
    )
