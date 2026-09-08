"""Checkpoint de uma execucao longa do catalogo, para ela poder ser retomada.

Sem isto, uma execucao do catalogo inteiro nao termina. Medido no proprio projeto: a reproducao da
linha de base de `type_of_play` sobre os 604 clipes completos foi interrompida com 46 eventos
escritos apos 2.168 s, e a projecao era de cerca de 8,6 h. O arquivo parcial sobrevive, mas o
avaliador exige `events.jsonl` e nao le `events.jsonl.parcial`, entao o trabalho daquelas horas
nao vale nada e a proxima tentativa comeca do primeiro clipe.

O checkpoint e a fonte da verdade, e nao o arquivo parcial. A ordem de escrita e: evento no parcial,
`flush`, e de tantos em tantos clipes o checkpoint. Se a queda acontece entre os dois, o parcial tem
mais linhas do que o checkpoint conhece, e a retomada corta o excesso e reprocessa aqueles clipes.
O contrario, checkpoint na frente do parcial, produziria `run.json` com tempo de etapa de um clipe
cujo evento nao esta no arquivo, e ai o relatorio somaria o que ninguem consegue conferir.

A retomada recusa configuracao diferente. Continuar uma execucao com outro `config_sha256` produz um
`events.jsonl` unico com metade dos eventos de cada configuracao, e `run.json` declarando so uma
delas: o numero sai plausivel e descreve uma execucao que nunca existiu.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

CHECKPOINT_FILE = "progress.json"
SCHEMA = "1.0"


class CheckpointError(ValueError): ...


@dataclass
class Checkpoint:
    """O que a retomada precisa saber, e nao so quantos clipes faltam.

    Os agregados entram aqui porque eles vivem em memoria durante a execucao: sem persistir tempo
    por etapa, identidade de modulo, falha por etapa e contagem de procedencia, o `run.json` de uma
    execucao retomada descreveria apenas o ultimo trecho e passaria por completo.
    """

    config_sha256: str
    population: int
    started_at: str
    selection_sha256: str = ""
    """Digest dos identificadores selecionados, que a contagem sozinha nao distingue.

    `population` e `len(escolhidos)`, calculado depois de `--dobra`, `--gabarito` e `--limit`, e
    `config_sha256` cobre o TOML e nao a linha de comando. Sem este campo, `--all --limit 24` e
    `--all --gabarito --dobra ajuste` sao intercambiaveis para a guarda quando caem na mesma
    contagem, e `--limit` da ao operador controle exato sobre isso.
    """
    features_sha256: str = ""
    """Digest de nome e versao das features; o TOML nao declara as versoes."""
    done: list[str] = field(default_factory=list[str])
    ok: int = 0
    failed: int = 0
    resumed: int = 0
    origens: dict[str, int] = field(default_factory=dict[str, int])
    fontes: dict[str, str] = field(default_factory=dict[str, str])
    """Procedencia do instante por clipe, que os agregados de `origens` nao conseguem devolver.

    Sem ela, `eval-instant` chamava `measure` sem `fontes` e as 38 medidas saiam como
    `desconhecida`, enquanto o `run.json` da mesma execucao sabia que 37 vinham de `ball-break` e
    uma de `midpoint`. O avaliador media junto o contato detectado na trajetoria e o ponto medio
    preenchido, que sao coisas diferentes.
    """
    tempos: dict[str, list[float]] = field(default_factory=dict[str, list[float]])
    identidade: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    falhas: dict[str, list[Any]] = field(default_factory=dict[str, list[Any]])

    @property
    def events_written(self) -> int:
        """Quantas linhas do parcial este checkpoint responde por.

        E `ok`, e nao `len(done)`: um clipe que falhou entra em `done`, para nao ser reprocessado,
        mas nao escreveu evento nenhum.
        """
        return self.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "config_sha256": self.config_sha256,
            "population": self.population,
            "started_at": self.started_at,
            "selection_sha256": self.selection_sha256,
            "features_sha256": self.features_sha256,
            "done": self.done,
            "ok": self.ok,
            "failed": self.failed,
            "resumed": self.resumed,
            "origens": self.origens,
            "fontes": self.fontes,
            "tempos": self.tempos,
            "identidade": self.identidade,
            "falhas": self.falhas,
        }


def write_checkpoint(destino: Path, checkpoint: Checkpoint) -> Path:
    """Escreve por arquivo temporario e `replace`, que e atomico no mesmo sistema de arquivos.

    Gravar em cima do checkpoint bom e ser interrompido no meio deixaria a execucao sem nenhum
    checkpoint valido, que e pior do que nao ter checkpoint: a pessoa acredita que tem retomada.
    """
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / CHECKPOINT_FILE
    parcial = caminho.with_suffix(".json.parcial")
    parcial.write_text(json.dumps(checkpoint.to_dict(), indent=2) + "\n", encoding="utf-8")
    parcial.replace(caminho)
    return caminho


def read_checkpoint(destino: Path) -> Checkpoint:
    caminho = destino / CHECKPOINT_FILE
    if not caminho.is_file():
        raise CheckpointError(
            f"{caminho} nao existe: esta execucao nao tem checkpoint para retomar"
        )
    try:
        decoded: object = json.loads(caminho.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as erro:
        raise CheckpointError(f"{caminho.name}: JSON invalido: {erro}") from erro
    if not isinstance(decoded, dict):
        raise CheckpointError(f"{caminho.name}: exige um objeto JSON")
    documento = cast("dict[str, Any]", decoded)
    if documento.get("schema_version") != SCHEMA:
        raise CheckpointError(
            f"{caminho.name}: schema {documento.get('schema_version')!r}, esperado {SCHEMA!r}"
        )
    faltando = {"config_sha256", "population", "started_at", "done", "ok"} - set(documento)
    if faltando:
        raise CheckpointError(f"{caminho.name}: sem os campos {sorted(faltando)}")
    _validate_progress(documento)
    _validate_aggregates(documento)
    return Checkpoint(
        config_sha256=documento["config_sha256"],
        population=documento["population"],
        started_at=documento["started_at"],
        selection_sha256=documento.get("selection_sha256", ""),
        features_sha256=documento.get("features_sha256", ""),
        done=documento["done"],
        ok=documento["ok"],
        failed=documento.get("failed", 0),
        resumed=documento.get("resumed", 0),
        origens=documento.get("origens", {}),
        fontes=documento.get("fontes", {}),
        tempos=documento.get("tempos", {}),
        identidade=documento.get("identidade", {}),
        falhas=documento.get("falhas", {}),
    )


def _invalid(name: str) -> CheckpointError:
    return CheckpointError(f"{CHECKPOINT_FILE}: {name} invalido ou incoerente")


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _invalid(name)
    return value


def _mapping(document: dict[str, Any], name: str) -> dict[str, Any]:
    value: object = document.get(name, {})
    if not isinstance(value, dict):
        raise _invalid(name)
    result = cast("dict[str, Any]", value)
    if any(not key for key in result):
        raise _invalid(name)
    return result


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise _invalid(name)
    return cast("list[Any]", value)


def _validate_progress(document: dict[str, Any]) -> None:
    for name in ("config_sha256", "started_at", "selection_sha256", "features_sha256"):
        if not isinstance(document.get(name, ""), str):
            raise _invalid(name)
    for name in ("population", "ok", "failed", "resumed"):
        _count(document.get(name, 0), name)
    done = _list(document["done"], "done")
    if any(not isinstance(item, str) or not item for item in done):
        raise _invalid("done")
    if len(set(done)) != len(done) or len(done) != document["ok"] + document.get("failed", 0):
        raise _invalid("done, ok e failed")
    if len(done) > document["population"]:
        raise _invalid("done e population")
    sources = _mapping(document, "fontes")
    if any(not isinstance(value, str) or not value for value in sources.values()):
        raise _invalid("fontes")
    if len(sources) != document["ok"] or not set(sources).issubset(done):
        raise _invalid("fontes, done e ok")
    origins = _mapping(document, "origens")
    for value in origins.values():
        _count(value, "origens")
    if dict(Counter(sources.values())) != origins:
        raise _invalid("origens e fontes")


def _validate_aggregates(document: dict[str, Any]) -> None:
    timings = _mapping(document, "tempos")
    identities = _mapping(document, "identidade")
    failures = _mapping(document, "falhas")
    if set(timings) != set(identities) or not set(failures).issubset(timings):
        raise _invalid("tempos, identidade e falhas")
    for raw in timings.values():
        values = _list(raw, "tempos")
        if not values or len(values) > document["ok"]:
            raise _invalid("tempos")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise _invalid("tempos")
    for raw in identities.values():
        values = _list(raw, "identidade")
        if len(values) != 2 or any(not isinstance(value, str) or not value for value in values):
            raise _invalid("identidade")
    for kind, raw in failures.items():
        values = _list(raw, "falhas")
        if len(values) != 2:
            raise _invalid("falhas")
        if _count(values[0], "falhas") > len(timings[kind]):
            raise _invalid("falhas e ok")
        if values[1] is not None and not isinstance(values[1], str):
            raise _invalid("falhas")


def truncate_to(parcial: Path, linhas: int, *, expected_clips: set[str] | None = None) -> int:
    """Corta o parcial no numero de eventos que o checkpoint responde por, e devolve o que sobrou.

    A escrita e por arquivo temporario e `replace`, pelo mesmo motivo de `write_checkpoint`: ser
    interrompido gravando em cima do parcial deixaria menos linhas completas do que o checkpoint
    declara, e ai a retomada seguinte para na propria guarda desta funcao, sem caminho de volta.
    Este arquivo vale mais do que o checkpoint, porque ele e o unico lugar onde os eventos estao.

    Duas coisas sao descartadas aqui. A primeira e o evento escrito depois do ultimo checkpoint: ele
    e valido, mas o checkpoint nao tem o tempo de etapa dele, entao reprocessar custa um clipe e
    manter custaria a integridade do relatorio. A segunda e uma ultima linha truncada, de uma queda
    no meio da escrita: anexar depois dela produziria um JSON invalido no meio do arquivo, e o
    avaliador so descobriria isso horas depois.
    """
    _count(linhas, "quantidade de eventos")
    if not parcial.is_file():
        if linhas:
            raise CheckpointError(
                f"{parcial.name} nao existe, mas o checkpoint declara {linhas} eventos"
            )
        return 0
    completas: list[str] = []
    for linha in parcial.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        try:
            json.loads(linha)
        except json.JSONDecodeError:
            break
        completas.append(linha)
    if len(completas) < linhas:
        raise CheckpointError(
            f"{parcial.name} tem {len(completas)} eventos completos e o checkpoint declara "
            f"{linhas}: o parcial foi truncado ou substituido"
        )
    mantidas = completas[:linhas]
    if expected_clips is not None:
        _validate_prefix(mantidas, expected_clips)
    temporario = parcial.with_suffix(".parcial.novo")
    temporario.write_text("".join(linha + "\n" for linha in mantidas), encoding="utf-8")
    temporario.replace(parcial)
    return len(mantidas)


def _validate_prefix(lines: list[str], expected_clips: set[str]) -> None:
    clips: set[str] = set()
    for line in lines:
        event: object = json.loads(line)
        if not isinstance(event, dict):
            raise CheckpointError("prefixo do parcial contem evento invalido")
        clip = cast("dict[str, object]", event).get("clip_id")
        if not isinstance(clip, str) or not clip or clip in clips:
            raise CheckpointError("prefixo do parcial contem clip_id invalido ou duplicado")
        clips.add(clip)
    if clips != expected_clips:
        raise CheckpointError("prefixo do parcial e checkpoint descrevem outros clipes")
