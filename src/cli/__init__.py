"""O ponto de entrada de linha de comando, que junta a borda e o mlops.

Cada comando novo entra aqui em vez de virar um executavel proprio, senao oito pessoas publicam
oito pontos de entrada que nao compartilham nem ajuda nem convencao de opcao.

E a razao de `cli` existir separado de `edge` e de `mlops`: ele importa as duas e nenhuma delas
importa ele. A borda nunca le CSV, entao ela nao importa `mlops`; o caminho contrario e
permitido e usado, porque `mlops` precisa saber onde o conjunto do parceiro mora.

A `api` e a outra area de composicao, e existe separada porque serve HTTP: quem instala a borda
para a Jetson nao carrega servidor web junto.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import typer

from contracts import ShotEvent
from contracts.events import ContractError
from edge.artifacts import ArtifactError, ArtifactMissing, read_manifest, resolve, sha256_of
from edge.cache.perception import PerceptionCache
from edge.features.base import FeatureError
from edge.features.registry import load_extractors, read_feature_config
from edge.frames.provider import ClipUnavailable, FrameProvider, VideoSource, probe
from edge.graph.framesets import FrameSetError, read_framesets
from edge.graph.runner import StageRun
from edge.modules.base import MODULE_KINDS, ModuleError
from edge.modules.registry import load_modules, read_config
from edge.paths import (
    ConfigUnavailable,
    DataUnavailable,
    cache_dir,
    out_dir,
    pipeline_config,
    runs_dir,
)
from edge.pipeline import PipelineError, process_clip
from edge.testing.make_clip import make_synthetic_clip
from mlops.catalog import CatalogError, ShotRecord, catalog_counts, load_catalog
from mlops.evaluate import EvaluationError, evaluate
from mlops.labels import LabelError, Labels, labels_config, read_labels
from mlops.run import (
    EVENTS_FILE,
    RunRecord,
    environment,
    git_state,
    new_run_id,
    write_run,
)
from mlops.split import SplitError, read_split, split_config

CLIP_OUT = typer.Option(None, help="Onde escrever o clipe.", show_default="var/out/synthetic.mp4")
CLIP_SEED = typer.Option(0, help="Muda o tom do gramado, nao a geometria.")
CACHE_ROOT_OPTION = typer.Option(
    None, help="Raiz do cache de percepcao.", show_default="var/cache/perception"
)

app = typer.Typer(help="Extrai features de finalizacao a partir de video, na borda.")
cache_app = typer.Typer(help="Cache de percepcao em disco.")
app.add_typer(cache_app, name="cache")
data_app = typer.Typer(help="O conjunto do parceiro.")
app.add_typer(data_app, name="data")
artifact_app = typer.Typer(help="Pesos de modelo, enderecados pelo conteudo.")
app.add_typer(artifact_app, name="artifact")


@app.command()
def clip(out: Path | None = CLIP_OUT, seed: int = CLIP_SEED) -> None:
    """Escreve a fixture de video sintetica e canonica."""
    destino = out_dir() / "synthetic.mp4" if out is None else out
    typer.echo(str(make_synthetic_clip(destino, seed=seed)))


RUN_CLIP = typer.Option(None, help="Clipe a processar.")
RUN_ALL = typer.Option(False, "--all", help="Processa o catalogo do parceiro.")
RUN_LIMIT = typer.Option(0, help="Para depois de N clipes. Zero e sem limite.")
RUN_OUT = typer.Option(None, help="Onde gravar a execucao.", show_default="var/runs")


# Toda excecao que significa "a entrada ou a configuracao esta errada", e nao "o programa tem um
# defeito". Elas viram mensagem de uma linha; o resto continua subindo como traceback, porque
# traceback de defeito nosso e informacao e traceback de erro de digitacao e ruido.
#
# A lista curta de antes deixava escapar uma duzia de erros de configuracao com mensagem
# excelente, entregues como 60 linhas de Rich: implementacao inexistente, especie desconhecida,
# conjunto de quadros nao declarado, feature quebrada e a configuracao ausente.
def _config(escolhido: Path | None) -> Path:
    """A opcao explicita ganha; o padrao sai de `edge.paths`, que le a variavel de ambiente."""
    return escolhido if escolhido is not None else pipeline_config()


def _labels(escolhido: Path | None) -> Path:
    return escolhido if escolhido is not None else labels_config()


def _split(escolhido: Path | None) -> Path:
    return escolhido if escolhido is not None else split_config()


ERROS_DE_USO = (
    ArtifactError,
    ArtifactMissing,
    CatalogError,
    ClipUnavailable,
    ConfigUnavailable,
    ContractError,
    DataUnavailable,
    FeatureError,
    FrameSetError,
    LabelError,
    ModuleError,
    PipelineError,
    SplitError,
)

# `None` e nao o caminho: o padrao do Typer e avaliado no import, e congelar ali faria
# `XVISION_CONFIG_DIR` ser aceita e ignorada. Quem passa a opcao explicita continua mandando.
RUN_CONFIG = typer.Option(None, help="Modulos, quadros e features. Padrao: pipeline.toml.")
LABELS_OPTION = typer.Option(None, help="Rotulos do parceiro. Padrao: labels.toml.")
SPLIT_OPTION = typer.Option(None, help="Particao por partida. Padrao: split.toml.")


@app.command()
def run(
    clip: Path | None = RUN_CLIP,
    out: Path | None = RUN_OUT,
    config: Path | None = RUN_CONFIG,
    todos: bool = RUN_ALL,
    limit: int = RUN_LIMIT,
) -> None:
    """Processa um clipe, ou o catalogo inteiro com --all."""
    config = _config(config)
    if todos:
        _run_catalog(out, config, limit)
        return
    if clip is None:
        typer.echo("passe --clip com um clipe, ou --all para o catalogo", err=True)
        raise typer.Exit(code=1)
    if not config.is_file():
        typer.echo(f"{config} nao existe: passe --config com o caminho do arquivo", err=True)
        raise typer.Exit(code=1)
    destino = (runs_dir() if out is None else out) / nome_da_execucao(clip)
    try:
        evento, etapas = _processa(clip, config)
    except ERROS_DE_USO as erro:
        typer.echo(f"{clip}: {erro}", err=True)
        raise typer.Exit(code=1) from erro
    for etapa in etapas:
        motivo = "" if etapa.reason is None else f"  {etapa.reason}"
        typer.echo(f"  {etapa.kind:<10} {etapa.status:<8} {etapa.duration_ms:>8.1f} ms{motivo}")
    destino.mkdir(parents=True, exist_ok=True)
    escrito = destino / "events.jsonl"
    parcial = escrito.with_suffix(".jsonl.parcial")
    parcial.write_text(json.dumps(evento.to_dict()) + "\n", encoding="utf-8")
    parcial.replace(escrito)
    typer.echo(str(escrito))


def _run_catalog(out: Path | None, config: Path, limit: int) -> None:
    if not config.is_file():
        typer.echo(f"{config} nao existe: passe --config com o caminho do arquivo", err=True)
        raise typer.Exit(code=1)
    try:
        tabela = read_labels(labels_config())
        catalogo = _populacao(load_catalog(), tabela)
    except ERROS_DE_USO as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    escolhidos = catalogo[:limit] if limit else catalogo
    destino = (runs_dir() if out is None else out) / new_run_id()
    destino.mkdir(parents=True, exist_ok=True)
    eventos = destino / EVENTS_FILE
    parcial = eventos.with_suffix(".jsonl.parcial")
    tempos: dict[str, list[float]] = {}
    identidade: dict[str, tuple[str, str]] = {}
    falhas: dict[str, tuple[int, str | None]] = {}
    ok = falhou = 0
    with parcial.open("w", encoding="utf-8") as arquivo:
        for registro in escolhidos:
            try:
                evento, etapas = _processa_registro(registro, config)
            except ERROS_DE_USO as erro:
                falhou += 1
                typer.echo(f"  {registro.shot_id}: {erro}", err=True)
                continue
            arquivo.write(json.dumps(evento.to_dict()) + "\n")
            ok += 1
            for etapa in etapas:
                tempos.setdefault(etapa.kind, []).append(etapa.duration_ms)
                identidade[etapa.kind] = (etapa.name, etapa.version)
                if etapa.status != "ok":
                    quantas, motivo = falhas.get(etapa.kind, (0, None))
                    falhas[etapa.kind] = (quantas + 1, motivo or etapa.reason)
    parcial.replace(eventos)
    modulos = resumo_dos_modulos(tempos, identidade, falhas)
    sha, sujo = git_state()
    host, python, plataforma = environment()
    write_run(
        destino,
        RunRecord(
            run_id=destino.name,
            started_at=datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            git_sha=sha,
            git_dirty=sujo,
            config_sha256=sha256_of(config),
            dataset={"clips": ok, "failed": falhou, "population": f"n={len(escolhidos)}"},
            modules=modulos,
            features=tuple(
                (e.spec.name, e.spec.version) for e in load_extractors(read_feature_config(config))
            ),
            host=host,
            python=python,
            platform=plataforma,
        ),
    )
    typer.echo(f"  {ok} eventos, {falhou} falhas")
    typer.echo(f"  {destino}")


def resumo_dos_modulos(
    tempos: dict[str, list[float]],
    identidade: dict[str, tuple[str, str]],
    falhas: dict[str, tuple[int, str | None]],
) -> dict[str, dict[str, object]]:
    """Uma amostra de um clipe nao serve num projeto cujo numero de manchete e ms por clipe.

    E guardar so o ultimo faz uma execucao em que a etapa falhou em metade dos clipes aparecer
    como `ok`, porque o ultimo passou.
    """
    resumo: dict[str, dict[str, object]] = {}
    for especie, amostras in sorted(tempos.items()):
        nome, versao = identidade[especie]
        quantas, motivo = falhas.get(especie, (0, None))
        resumo[especie] = {
            "name": nome,
            "version": versao,
            "status": "ok" if quantas == 0 else "failed",
            "failed_clips": quantas,
            "reason": motivo,
            "duration_ms": {
                "n": len(amostras),
                "min": min(amostras),
                "max": max(amostras),
                "mean": sum(amostras) / len(amostras),
            },
        }
    return resumo


def _populacao(catalogo: Sequence[ShotRecord], tabela: Labels) -> list[ShotRecord]:
    obrigatorios = tabela.required()
    return [
        r
        for r in catalogo
        if r.big_enough and all(r.labels.get(o.column, "") not in o.missing for o in obrigatorios)
    ]


def _processa_registro(
    registro: ShotRecord, config: Path
) -> tuple[ShotEvent, tuple[StageRun, ...]]:
    clipe = probe(registro.clip_path, clip_id=registro.shot_id)
    provider = FrameProvider(VideoSource(registro.clip_path))
    try:
        return process_clip(
            clip=replace(clipe, match_id=registro.match_id),
            modules=load_modules(read_config(config)),
            framesets=read_framesets(config),
            provider=provider,
            extractors=load_extractors(read_feature_config(config)),
            cache=PerceptionCache(cache_dir()),
        )
    finally:
        provider.close()


def nome_da_execucao(clip: Path) -> str:
    return clip.stem


def _processa(clip: Path, config: Path) -> tuple[ShotEvent, tuple[StageRun, ...]]:
    provider = FrameProvider(VideoSource(clip))
    try:
        return process_clip(
            clip=probe(clip),
            modules=load_modules(read_config(config)),
            framesets=read_framesets(config),
            provider=provider,
            extractors=load_extractors(read_feature_config(config)),
            cache=PerceptionCache(cache_dir()),
        )
    finally:
        provider.close()


@data_app.command("check")
def data_check(labels: Path | None = LABELS_OPTION, split: Path | None = SPLIT_OPTION) -> None:
    """Conta o que o conjunto do parceiro tem, e por fold."""
    labels, split = _labels(labels), _split(split)
    for arquivo, opcao in ((labels, "--labels"), (split, "--split")):
        if not arquivo.is_file():
            typer.echo(f"{arquivo} nao existe: passe {opcao} com o caminho do arquivo", err=True)
            raise typer.Exit(code=1)
    try:
        tabela = read_labels(labels)
        particao = read_split(split)
        contagem = catalog_counts()
        catalogo = load_catalog(split=particao)
    except ERROS_DE_USO as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro

    obrigatorios = tabela.required()
    completos = [
        r
        for r in catalogo
        if r.big_enough and all(r.labels.get(o.column, "") not in o.missing for o in obrigatorios)
    ]
    typer.echo(f"  {contagem.rows} finalizacoes na tabela do parceiro")
    typer.echo(f"  {contagem.with_clip} com clipe, {contagem.without_clip} sem")
    typer.echo(
        f"  {contagem.big_enough} abriveis, {contagem.with_clip - contagem.big_enough} truncados"
    )
    typer.echo(f"  {len(completos)} completos, com os {len(obrigatorios)} rotulos obrigatorios")
    typer.echo(f"  {contagem.matches} partidas")
    if contagem.ignored_files:
        typer.echo(f"  {len(contagem.ignored_files)} arquivos fora do padrao do nome, ignorados")
    por_fold: dict[int, int] = {}
    for registro in completos:
        por_fold[registro.fold] = por_fold.get(registro.fold, 0) + 1
    for fold in sorted(por_fold):
        typer.echo(f"  fold {fold}: {por_fold[fold]} clipes")


EVAL_TRUTH = typer.Option(None, help="Gabarito ja extraido, em JSON. Sem isto, le o catalogo.")


@app.command("eval")
def evaluate_run(
    run_dir: Path,
    labels: Path | None = LABELS_OPTION,
    truth: Path | None = EVAL_TRUTH,
) -> None:
    """Casa os eventos de uma execucao com o gabarito e conta."""
    labels = _labels(labels)
    eventos = run_dir / EVENTS_FILE
    if not eventos.is_file():
        typer.echo(f"{eventos} nao existe: passe o diretorio de uma execucao", err=True)
        raise typer.Exit(code=1)
    for arquivo, opcao in ((labels, "--labels"), (truth, "--truth")):
        if arquivo is not None and not arquivo.is_file():
            typer.echo(f"{arquivo} nao existe: passe {opcao} com o caminho do arquivo", err=True)
            raise typer.Exit(code=1)
    try:
        tabela = read_labels(labels)
        gabarito = _gabarito(truth, tabela)
    except (*ERROS_DE_USO, json.JSONDecodeError) as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro

    predicoes: dict[str, dict[str, tuple[object, str | None]]] = {}
    try:
        linhas = eventos.read_text(encoding="utf-8").splitlines()
        carregadas = [json.loads(linha) for linha in linhas]
    except (OSError, json.JSONDecodeError) as erro:
        typer.echo(f"{eventos}: {erro}", err=True)
        raise typer.Exit(code=1) from erro
    for evento in carregadas:
        predicoes[evento["clip_id"]] = {
            f["name"]: (f["value"] if f["status"] != "unavailable" else None, f["reason"])
            for f in evento["features"]
        }
    try:
        relatorio = evaluate(predicoes, gabarito, tabela)
    except EvaluationError as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro

    typer.echo(f"  {'feature':<24}{'n':>6}{'cobertura':>11}{'acuracia':>10}{'maioria':>10}")
    for metrica in relatorio.metrics:
        medida = metrica.scores.get("accuracy", metrica.scores.get("mae"))
        maioria = (metrica.baseline_majority or {}).get("accuracy")
        typer.echo(
            f"  {metrica.feature:<24}{metrica.n_labeled:>6}{metrica.coverage:>10.1%}"
            f"{'-' if medida is None else f'{medida:>10.3f}':>10}"
            f"{'-' if maioria is None else f'{maioria:>10.3f}':>10}"
        )
    destino = run_dir / "report.json"
    destino.write_text(json.dumps(relatorio.to_dict(), indent=1) + "\n", encoding="utf-8")
    typer.echo(f"  {destino}")


def _gabarito(truth: Path | None, tabela: Labels) -> dict[str, dict[str, str]]:
    """Sem `--truth`, a populacao e a mesma que o `data check` chama de completos.

    Misturar clipe truncado ou sem rotulo faz cada metrica ter um `n` diferente, e comparar duas
    execucoes de populacao diferente e o jeito mais facil de publicar um delta sem sentido.
    """
    if truth is not None:
        return cast("dict[str, dict[str, str]]", json.loads(truth.read_text(encoding="utf-8")))
    obrigatorios = tabela.required()
    return {
        r.shot_id: dict(r.labels)
        for r in load_catalog()
        if r.big_enough and all(r.labels.get(o.column, "") not in o.missing for o in obrigatorios)
    }


IGNORADOS = frozenset({".DS_Store", "Thumbs.db", "__pycache__", ".ipynb_checkpoints"})


def _lixo(caminho: Path, raiz: Path) -> bool:
    """Metadado de gerenciador de arquivos mudava o digest do mesmo peso, entre duas maquinas."""
    return any(parte in IGNORADOS for parte in caminho.relative_to(raiz).parts)


ART_KIND = typer.Option(..., help="Especie do modulo que usa o peso.")
ART_NAME = typer.Option(..., help="Nome do modelo.")
ART_VERSION = typer.Option(..., help="Versao do modelo.")
ART_ENTRY = typer.Option(..., help="Arquivo que o `load` do dono abre.")
ART_OUT = typer.Option(Path("models"), help="Onde escrever o manifesto.")
ART_ROOT = typer.Option(None, help="Onde os pesos ficam.", show_default="var/models")


@artifact_app.command("add")
def artifact_add(
    diretorio: Path,
    kind: str = ART_KIND,
    name: str = ART_NAME,
    version: str = ART_VERSION,
    entrypoint: str = ART_ENTRY,
    out: Path = ART_OUT,
) -> None:
    """Descreve um peso em disco num manifesto versionavel."""
    if not diretorio.is_dir():
        typer.echo(f"{diretorio} nao e um diretorio", err=True)
        raise typer.Exit(code=1)
    if kind not in MODULE_KINDS:
        typer.echo(f"{kind!r} nao e especie conhecida: {', '.join(sorted(MODULE_KINDS))}", err=True)
        raise typer.Exit(code=1)
    arquivos = sorted(p for p in diretorio.rglob("*") if p.is_file() and not _lixo(p, diretorio))
    if not arquivos:
        typer.echo(f"{diretorio} esta vazio", err=True)
        raise typer.Exit(code=1)
    relativos = [p.relative_to(diretorio).as_posix() for p in arquivos]
    if entrypoint not in relativos:
        typer.echo(
            f"{entrypoint!r} nao esta em {diretorio}: os arquivos sao {', '.join(relativos)}",
            err=True,
        )
        raise typer.Exit(code=1)
    linhas = [
        'schema_version = "1.0"',
        f'kind = "{kind}"',
        f"name = {json.dumps(name)}",
        f"version = {json.dumps(version)}",
        f"entrypoint = {json.dumps(entrypoint)}",
    ]
    for arquivo, relativo in zip(arquivos, relativos, strict=True):
        linhas += [
            "",
            "[[files]]",
            f"path = {json.dumps(relativo)}",
            f'sha256 = "{sha256_of(arquivo)}"',
            f"bytes = {arquivo.stat().st_size}",
        ]
    destino = out / kind / f"{name}-{version}.toml"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    typer.echo(f"  {len(arquivos)} arquivos descritos")
    typer.echo(str(destino))


@artifact_app.command("get")
def artifact_get(manifesto: Path, root: Path | None = ART_ROOT) -> None:
    """Confere o peso contra o manifesto e imprime o caminho do entrypoint."""
    try:
        lido = read_manifest(manifesto)
        caminho = resolve(lido, root=root)
    except (ArtifactError, ArtifactMissing) as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    typer.echo(str(caminho))


@cache_app.command("clear")
def cache_clear(root: Path | None = CACHE_ROOT_OPTION) -> None:
    """Apaga as entradas do cache de percepcao."""
    cache = PerceptionCache(root)
    typer.echo(f"{cache.clear()} entradas removidas de {cache.root.resolve()}")
