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

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import IO, Any, NamedTuple, cast

import typer

from cli.players import players_app
from contracts import Perception
from contracts.events import ContractError
from edge.artifacts import ArtifactError, ArtifactMissing, read_manifest, resolve, sha256_of
from edge.cache.perception import PerceptionCache
from edge.features.base import FeatureError
from edge.features.registry import load_extractors, read_feature_config
from edge.frames.provider import ClipUnavailable, FrameProvider, VideoSource, probe
from edge.graph.framesets import FrameSetError, read_framesets
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
from edge.pipeline import ClipResult, PipelineError, process_clip
from edge.testing.make_clip import make_synthetic_clip
from edge.video import RenderUnavailable
from mlops.catalog import CatalogError, ShotRecord, catalog_counts, load_catalog
from mlops.checkpoint import (
    CHECKPOINT_FILE,
    Checkpoint,
    CheckpointError,
    read_checkpoint,
    truncate_to,
    write_checkpoint,
)
from mlops.evaluate import EvaluationError, evaluate
from mlops.instant import (
    OBSERVATIONS_FILE,
    TOLERANCIA_ALVO,
    Anotacao,
    BaselineReport,
    InstantError,
    Resolucao,
    agreement,
    baselines,
    consensus,
    diagnose,
    labels_dir,
    measure,
    read,
    read_spots,
    resolution,
    sample,
    taxonomy,
    wilson,
    write,
)
from mlops.labels import LabelError, Labels, labels_config, read_labels
from mlops.run import (
    EVENTS_FILE,
    RunRecord,
    environment,
    git_state,
    hardware,
    new_run_id,
    runtimes,
    write_run,
)
from mlops.run_lock import exclusive_run
from mlops.sheets import coarse_indices, contact_sheet, fine_indices
from mlops.split import SplitError, read_split, split_config

CLIP_OUT = typer.Option(
    None, help="Onde escrever o clipe.", show_default="../var/out/synthetic.mp4"
)
CLIP_SEED = typer.Option(0, help="Muda o tom do gramado, nao a geometria.")
CACHE_ROOT_OPTION = typer.Option(
    None, help="Raiz do cache de percepcao.", show_default="../var/cache/perception"
)

app = typer.Typer(help="Extrai features de finalizacao a partir de video, na borda.")
cache_app = typer.Typer(help="Cache de percepcao em disco.")
app.add_typer(cache_app, name="cache")
data_app = typer.Typer(help="O conjunto do parceiro.")
app.add_typer(data_app, name="data")
artifact_app = typer.Typer(help="Pesos de modelo, enderecados pelo conteudo.")
app.add_typer(artifact_app, name="artifact")
label_app = typer.Typer(help="Gabarito manual do instante do contato.")
app.add_typer(label_app, name="label")
app.add_typer(players_app, name="players")


@app.command()
def clip(out: Path | None = CLIP_OUT, seed: int = CLIP_SEED) -> None:
    """Escreve a fixture de video sintetica e canonica."""
    destino = out_dir() / "synthetic.mp4" if out is None else out
    typer.echo(str(make_synthetic_clip(destino, seed=seed)))


RUN_CLIP = typer.Option(None, help="Clipe a processar.")
RUN_ALL = typer.Option(False, "--all", help="Processa o catalogo do parceiro.")
RUN_GABARITO = typer.Option(
    False, "--gabarito", help="So os clipes com instante anotado a mao, para o experimento."
)
RUN_LIMIT = typer.Option(0, help="Para depois de N clipes. Zero e sem limite.")
RUN_DOBRA = typer.Option(
    None,
    help="ajuste (dobras 0 a 2), reservado (3 e 4) ou tudo. Obrigatorio junto com --gabarito.",
    show_default="tudo",
)
RUN_RESUME = typer.Option(
    None, help="Retoma uma execucao interrompida, no diretorio dela. Exige a mesma configuracao."
)
PROGRESSO_A_CADA = 10
"""De quantos em quantos clipes o checkpoint e gravado e o progresso e impresso.

Dez porque o custo de gravar e de milissegundos e o custo de perder e um clipe de dezenas de
segundos: no ritmo medido, dez clipes sao cerca de nove minutos de trabalho no pior caso, e esse e
o teto do que uma queda joga fora.
"""
RUN_OUT = typer.Option(None, help="Onde gravar a execucao.", show_default="../var/runs")
RUN_RENDER = typer.Option(
    False, "--render", help="Gera annotated.mp4 com top view e bola desta execucao."
)


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
    RenderUnavailable,
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
    gabarito: bool = RUN_GABARITO,
    dobra: str | None = RUN_DOBRA,
    resume: Path | None = RUN_RESUME,
    render: bool = RUN_RENDER,
) -> None:
    """Processa um clipe, ou o catalogo inteiro com --all.

    `--gabarito` restringe aos clipes com instante anotado a mao. Serve ao experimento que troca o
    detector de instante pelo oraculo (`config/pipeline.oracle.toml`): rodar o catalogo inteiro com
    ele gastaria dezesseis vezes mais tempo para produzir evento sem instante em todo clipe que
    ninguem anotou.

    `--dobra` restringe por dobra do split. Ele existe porque `--gabarito` sozinho abre os 38
    clipes anotados, e 14 deles sao o pacote reservado: sem o filtro, medir uma melhoria no instante
    durante o desenvolvimento consumiria o holdout sem ninguem decidir isso.
    """
    config = _config(config)
    if render and todos:
        typer.echo("--render exige um unico --clip; nao pode ser usado com --all", err=True)
        raise typer.Exit(code=1)
    if todos:
        _run_catalog(out, config, limit, gabarito, dobra, resume)
        return
    if resume is not None:
        typer.echo("--resume vale para --all: um clipe so nao tem o que retomar", err=True)
        raise typer.Exit(code=1)
    if clip is None:
        typer.echo("passe --clip com um clipe, ou --all para o catalogo", err=True)
        raise typer.Exit(code=1)
    if not config.is_file():
        typer.echo(f"{config} nao existe: passe --config com o caminho do arquivo", err=True)
        raise typer.Exit(code=1)
    destino = (runs_dir() if out is None else out) / nome_da_execucao(clip)

    def write_render(perception: Perception) -> None:
        from edge.video import read_config, render_video

        target = destino / "annotated.mp4"
        render_config = read_config(config)
        if render_config is None:
            raise RenderUnavailable("--render exige [render.top_view] com players e field")
        render_video(clip, target, render_config, perception=perception)
        typer.echo(str(target))

    try:
        resultado = (
            _processa(clip, config, on_perception=write_render)
            if render
            else _processa(clip, config)
        )
    except ERROS_DE_USO as erro:
        typer.echo(f"{clip}: {erro}", err=True)
        raise typer.Exit(code=1) from erro
    for etapa in resultado.stages:
        motivo = "" if etapa.reason is None else f"  {etapa.reason}"
        typer.echo(f"  {etapa.kind:<10} {etapa.status:<8} {etapa.duration_ms:>8.1f} ms{motivo}")
    destino.mkdir(parents=True, exist_ok=True)
    escrito = destino / "events.jsonl"
    parcial = escrito.with_suffix(".jsonl.parcial")
    parcial.write_text(json.dumps(resultado.event.to_dict()) + "\n", encoding="utf-8")
    parcial.replace(escrito)
    typer.echo(str(escrito))


def _run_catalog(
    out: Path | None,
    config: Path,
    limit: int,
    gabarito: bool = False,
    dobra: str | None = None,
    resume: Path | None = None,
) -> None:
    if not config.is_file():
        typer.echo(f"{config} nao existe: passe --config com o caminho do arquivo", err=True)
        raise typer.Exit(code=1)
    if gabarito and dobra is None:
        typer.echo(
            "--gabarito abre os 38 clipes anotados, e 14 deles sao o pacote reservado: passe "
            f"--dobra explicitamente, com {', '.join(DOBRAS_DA_BOLA)}",
            err=True,
        )
        raise typer.Exit(code=1)
    dobras = dobras_de(dobra or "tudo")
    try:
        tabela = read_labels(labels_config())
        catalogo = _populacao(load_catalog(), tabela)
    except ERROS_DE_USO as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    if dobras != DOBRAS_DA_BOLA["tudo"]:
        catalogo = [r for r in catalogo if r.fold in dobras]
        if not catalogo:
            typer.echo(f"nenhum clipe na dobra {dobra}", err=True)
            raise typer.Exit(code=1)
    if gabarito:
        anotados = set(consensus(read()))
        catalogo = [r for r in catalogo if r.shot_id in anotados]
        if not catalogo:
            typer.echo("nenhum clipe tem instante anotado; rode `xvision label scrub`", err=True)
            raise typer.Exit(code=1)
    escolhidos = catalogo[:limit] if limit else catalogo
    destino = resume if resume is not None else (runs_dir() if out is None else out) / new_run_id()
    try:
        with exclusive_run(destino):
            _run_selected_catalog(escolhidos, config, destino, resume=resume is not None)
    except CheckpointError as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro


def _run_selected_catalog(
    escolhidos: Sequence[ShotRecord], config: Path, destino: Path, *, resume: bool
) -> None:
    digest = sha256_of(config)
    selecao = hashlib.sha256("\0".join(sorted(r.shot_id for r in escolhidos)).encode()).hexdigest()
    feature_digest = _feature_digest(config)
    agora = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    if not resume:
        estado = Checkpoint(
            config_sha256=digest,
            population=len(escolhidos),
            started_at=agora,
            selection_sha256=selecao,
            features_sha256=feature_digest,
        )
        modo = "w"
    else:
        estado, modo = _retomada(
            destino,
            digest,
            len(escolhidos),
            selecao,
            feature_digest,
            selected_clips={record.shot_id for record in escolhidos},
        )
        estado.resumed += 1
        write_checkpoint(destino, estado)
    destino.mkdir(parents=True, exist_ok=True)
    eventos = destino / EVENTS_FILE
    parcial = eventos.with_suffix(".jsonl.parcial")

    tempos: dict[str, list[float]] = {k: list(v) for k, v in estado.tempos.items()}
    identidade: dict[str, tuple[str, str]] = {
        k: (v[0], v[1]) for k, v in estado.identidade.items() if len(v) == 2
    }
    falhas: dict[str, tuple[int, str | None]] = {
        k: (int(v[0]), None if v[1] is None else str(v[1]))
        for k, v in estado.falhas.items()
        if len(v) == 2
    }
    ajustes: dict[str, str] = {}
    origens: dict[str, int] = dict(estado.origens)
    fontes: dict[str, str] = dict(estado.fontes)
    apoios: dict[str, str] = dict(estado.apoios)
    candidatos: dict[str, list[dict[str, Any]]] = {k: list(v) for k, v in estado.candidatos.items()}
    pools: dict[str, list[int]] = {k: list(v) for k, v in estado.pools.items()}
    feitos = set(estado.done)
    ok, falhou = estado.ok, estado.failed
    pendentes = [r for r in escolhidos if r.shot_id not in feitos]
    if resume:
        typer.echo(f"  retomando em {len(feitos)}/{len(escolhidos)}: {len(pendentes)} pendentes")
    comeco = perf_counter()

    def checkpoint() -> None:
        estado.done = sorted(feitos)
        estado.ok, estado.failed = ok, falhou
        estado.origens = dict(origens)
        estado.fontes = dict(fontes)
        estado.apoios = dict(apoios)
        estado.candidatos = {k: list(v) for k, v in candidatos.items()}
        estado.pools = {k: list(v) for k, v in pools.items()}
        estado.tempos = {k: list(v) for k, v in tempos.items()}
        estado.identidade = {k: [v[0], v[1]] for k, v in identidade.items()}
        estado.falhas = {k: [v[0], v[1]] for k, v in falhas.items()}
        write_checkpoint(destino, estado)

    with parcial.open(modo, encoding="utf-8") as arquivo:
        for posicao, registro in enumerate(pendentes, start=1):
            try:
                resultado = _processa_registro(registro, config)
            except ERROS_DE_USO as erro:
                falhou += 1
                feitos.add(registro.shot_id)
                typer.echo(f"  {registro.shot_id}: {erro}", err=True)
                _marco(arquivo, checkpoint, posicao, len(feitos), len(escolhidos), comeco)
                continue
            arquivo.write(json.dumps(resultado.event.to_dict()) + "\n")
            ok += 1
            feitos.add(registro.shot_id)
            fonte = resultado.shot_source or "sem-origem"
            origens[fonte] = origens.get(fonte, 0) + 1
            fontes[registro.shot_id] = fonte
            apoios[registro.shot_id] = resultado.ball_support or "sem-apoio"
            candidatos[registro.shot_id] = [dict(c) for c in resultado.shot_candidates]
            pools[registro.shot_id] = list(resultado.shot_candidate_pool)
            ajustes.update(resultado.tunings)
            for etapa in resultado.stages:
                tempos.setdefault(etapa.kind, []).append(etapa.duration_ms)
                identidade[etapa.kind] = (etapa.name, etapa.version)
                if etapa.status != "ok":
                    quantas, motivo = falhas.get(etapa.kind, (0, None))
                    falhas[etapa.kind] = (quantas + 1, motivo or etapa.reason)
            _marco(arquivo, checkpoint, posicao, len(feitos), len(escolhidos), comeco)
        arquivo.flush()
        checkpoint()
    _escreve_procedencia(destino, fontes, apoios, candidatos, pools)
    modulos = resumo_dos_modulos(tempos, identidade, falhas, ajustes)
    sha, sujo = git_state()
    host, python, plataforma = environment()
    write_run(
        destino,
        RunRecord(
            run_id=destino.name,
            started_at=estado.started_at,
            git_sha=sha,
            git_dirty=sujo,
            config_sha256=digest,
            dataset={
                "clips": ok,
                "failed": falhou,
                "population": f"n={len(escolhidos)}",
                "shot_source": dict(sorted(origens.items())),
                "resumed": estado.resumed,
            },
            modules=modulos,
            features=tuple(
                (e.spec.name, e.spec.version) for e in load_extractors(read_feature_config(config))
            ),
            host=host,
            python=python,
            platform=plataforma,
            hardware=hardware(),
            runtime=runtimes(),
        ),
    )
    parcial.replace(eventos)
    (destino / CHECKPOINT_FILE).unlink(missing_ok=True)
    typer.echo(f"  {ok} eventos, {falhou} falhas")
    typer.echo(f"  {destino}")


def _marco(
    arquivo: IO[Any],
    checkpoint: Callable[[], None],
    posicao: int,
    feitos: int,
    total: int,
    comeco: float,
) -> None:
    """Checkpoint e progresso a cada `PROGRESSO_A_CADA` clipes, inclusive nos que falharam.

    Inclusive nos que falharam porque `continue` saltava este bloco: se as falhas caissem nos
    multiplos de dez, a granularidade da retomada degradava calada, e no pior caso uma execucao de
    horas ficava sem nenhum checkpoint no meio do laco.
    """
    if posicao % PROGRESSO_A_CADA:
        return
    arquivo.flush()
    checkpoint()
    typer.echo(_progresso(feitos, total, comeco, posicao))


def _escreve_procedencia(
    destino: Path,
    fontes: Mapping[str, str],
    apoios: Mapping[str, str],
    candidatos: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    pools: Mapping[str, Sequence[int]] | None = None,
) -> Path:
    """Procedencia do instante por clipe, que o agregado do `run.json` nao consegue devolver.

    Sem este arquivo, `eval-instant` mede junto o contato detectado na trajetoria e o ponto medio
    preenchido, e as duas coisas somem dentro de uma mediana so.

    `candidates` entra na mesma linha, e nao em arquivo proprio, porque ele responde sobre o mesmo
    clipe e na mesma execucao: dois arquivos precisariam concordar sobre quais clipes existem, e
    essa concordancia ja e conferida uma vez em `_procedencia`.
    """
    caminho = destino / OBSERVATIONS_FILE
    parcial = caminho.with_suffix(".jsonl.parcial")
    parcial.write_text(
        "".join(
            json.dumps(
                {
                    "clip_id": clip,
                    "shot_source": fonte,
                    "ball_support": apoios.get(clip, "sem-apoio"),
                    "candidates": [dict(c) for c in (candidatos or {}).get(clip, ())],
                    "candidate_pool": list((pools or {}).get(clip, ())),
                }
            )
            + "\n"
            for clip, fonte in sorted(fontes.items())
        ),
        encoding="utf-8",
    )
    parcial.replace(caminho)
    return caminho


def _feature_digest(config: Path) -> str:
    """Identifica as versoes das features que o TOML seleciona."""
    pairs = sorted(
        f"{e.spec.name}@{e.spec.version}" for e in load_extractors(read_feature_config(config))
    )
    return hashlib.sha256("\0".join(pairs).encode()).hexdigest()


def _retomada(
    destino: Path,
    digest: str,
    populacao: int,
    selecao: str,
    feature_digest: str,
    *,
    selected_clips: set[str],
) -> tuple[Checkpoint, str]:
    """Le o checkpoint, recusa configuracao ou populacao diferente, e corta o parcial no
    ponto que o checkpoint responde por."""
    if (destino / EVENTS_FILE).is_file():
        typer.echo(f"  {destino} ja esta completa: {destino / EVENTS_FILE}")
        raise typer.Exit(code=0)
    try:
        estado = read_checkpoint(destino)
    except CheckpointError as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    if estado.config_sha256 != digest:
        typer.echo(
            f"a execucao em {destino} usou config_sha256 {estado.config_sha256[:12]} e agora "
            f"chegou {digest[:12]}: retomar misturaria duas configuracoes num events.jsonl so",
            err=True,
        )
        raise typer.Exit(code=1)
    if estado.population != populacao:
        typer.echo(
            f"a execucao em {destino} tinha {estado.population} clipes e a selecao atual tem "
            f"{populacao}: retomar mediria outra populacao",
            err=True,
        )
        raise typer.Exit(code=1)
    if not estado.selection_sha256 or not estado.features_sha256:
        typer.echo(
            "checkpoint sem identidade de selecao ou features: inicie outra execucao; "
            "o parcial original foi preservado",
            err=True,
        )
        raise typer.Exit(code=1)
    if estado.selection_sha256 != selecao:
        typer.echo(
            f"a execucao em {destino} selecionou outros clipes, com a mesma contagem de "
            f"{populacao}: retomar mediria outra populacao",
            err=True,
        )
        raise typer.Exit(code=1)
    if estado.features_sha256 != feature_digest:
        typer.echo(
            f"a execucao em {destino} rodou features {estado.features_sha256[:12]} e agora chegou "
            f"{feature_digest[:12]}: retomar misturaria duas versoes de feature "
            "num events.jsonl so",
            err=True,
        )
        raise typer.Exit(code=1)
    if not set(estado.done).issubset(selected_clips):
        typer.echo("checkpoint contem clipes fora da selecao atual", err=True)
        raise typer.Exit(code=1)
    try:
        truncate_to(
            (destino / EVENTS_FILE).with_suffix(".jsonl.parcial"),
            estado.events_written,
            expected_clips=set(estado.fontes),
        )
    except CheckpointError as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    return estado, "a"


def _progresso(feitos: int, total: int, comeco: float, nesta_sessao: int) -> str:
    """Progresso com estimativa, calculada so com o que esta sessao mediu.

    Usar o tempo do trecho anterior estragaria a estimativa depois de uma retomada, porque a maquina
    e a carga podem ter mudado entre as duas.
    """
    decorrido = perf_counter() - comeco
    por_clipe = decorrido / nesta_sessao
    restam = total - feitos
    return (
        f"  {feitos}/{total} clipes, {por_clipe:.1f} s por clipe, "
        f"faltam cerca de {restam * por_clipe / 60:.0f} min"
    )


def _mediana(ordenado: Sequence[float]) -> float:
    """A mesma convencao de `mlops.instant`, que resolve n par pelo elemento de cima.

    Duas definicoes de mediana no mesmo repositorio produziriam numeros diferentes para a mesma
    amostra dependendo de qual relatorio a pessoa leu, que e a classe de defeito que este projeto
    persegue em todo lugar.
    """
    return ordenado[len(ordenado) // 2]


def _p95(ordenado: Sequence[float]) -> float:
    return ordenado[min(len(ordenado) - 1, int(0.95 * (len(ordenado) - 1)))]


def resumo_dos_modulos(
    tempos: dict[str, list[float]],
    identidade: dict[str, tuple[str, str]],
    falhas: dict[str, tuple[int, str | None]],
    ajustes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Uma amostra de um clipe nao serve num projeto cujo numero de manchete e ms por clipe.

    E guardar so o ultimo faz uma execucao em que a etapa falhou em metade dos clipes aparecer
    como `ok`, porque o ultimo passou.

    A mediana e o p95 entram ao lado da media porque a media e o estatistico errado aqui, e o
    proprio projeto ja argumenta isso no teste de meia precisao: o primeiro clipe paga carga de
    peso e compilacao de kernel, entao ele estica a media de uma amostra pequena sem dizer nada
    sobre o custo em regime.

    `tuning` e o ajuste em vigor da especie. Sem ele, dois tempos de execucoes com precisao
    diferente chegam ao relatorio com a mesma cara, e foi o que impediu de reproduzir as medicoes
    da Sprint 3 sem a apresentacao oral do lado.
    """
    resumo: dict[str, dict[str, object]] = {}
    for especie, amostras in sorted(tempos.items()):
        nome, versao = identidade[especie]
        quantas, motivo = falhas.get(especie, (0, None))
        ordenado = sorted(amostras)
        resumo[especie] = {
            "name": nome,
            "version": versao,
            "status": "ok" if quantas == 0 else "failed",
            "failed_clips": quantas,
            "reason": motivo,
            "tuning": (ajustes or {}).get(especie),
            "duration_ms": {
                "n": len(amostras),
                "min": ordenado[0],
                "max": ordenado[-1],
                "mean": sum(amostras) / len(amostras),
                "median": _mediana(ordenado),
                "p95": _p95(ordenado),
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


def _processa_registro(registro: ShotRecord, config: Path) -> ClipResult:
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


def _processa(
    clip: Path, config: Path, *, on_perception: Callable[[Perception], None] | None = None
) -> ClipResult:
    provider = FrameProvider(VideoSource(clip))
    try:
        return process_clip(
            clip=probe(clip),
            modules=load_modules(read_config(config)),
            framesets=read_framesets(config),
            provider=provider,
            extractors=load_extractors(read_feature_config(config)),
            cache=PerceptionCache(cache_dir()),
            on_perception=on_perception,
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
EVAL_DOBRA = typer.Option("tudo", help="ajuste (dobras 0 a 2), reservado (3 e 4) ou tudo.")


@app.command("eval")
def evaluate_run(
    run_dir: Path,
    labels: Path | None = LABELS_OPTION,
    truth: Path | None = EVAL_TRUTH,
    dobra: str = EVAL_DOBRA,
) -> None:
    """Casa os eventos de uma execucao com o gabarito e conta.

    `--dobra` existe porque as specs desta sprint calibram tudo nas dobras 0 a 2 e mantem 3 e 4
    fechadas, e sem o filtro a unica forma de respeitar isso era recortar um `--truth` a mao. Um
    numero medido na populacao inteira nao e o numero que o gate pede, e os dois se parecem.
    """
    labels = _labels(labels)
    dobras = dobras_de(dobra)
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
        if dobras != DOBRAS_DA_BOLA["tudo"]:
            gabarito = _pela_dobra(gabarito, dobras, dobra)
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
            f["name"]: (_valor_para_avaliacao(f, tabela), f["reason"]) for f in evento["features"]
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


def _valor_para_avaliacao(feature: Mapping[str, object], tabela: Labels) -> object:
    """Usa o escore guardado na evidencia para permitir calibrar features booleanas.

    O evento continua publicando `value` como booleano. Durante a avaliacao, quando o extractor
    tambem guardou `evidence.score`, o numero preserva a curva de limiar e produz `best_threshold`
    sem alterar a decisao publicada no video ou no JSON.
    """
    if feature.get("status") == "unavailable":
        return None
    nome = feature.get("name")
    rotulo = tabela.label_for(str(nome))
    evidencia_bruta = feature.get("evidence")
    evidencia = (
        cast("Mapping[str, object]", evidencia_bruta)
        if isinstance(evidencia_bruta, Mapping)
        else None
    )
    escore = evidencia.get("score") if evidencia is not None else None
    if rotulo is not None and rotulo.type == "bool" and isinstance(escore, (float, int)):
        return float(escore)
    return feature.get("value")


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


def _pela_dobra(
    gabarito: dict[str, dict[str, str]], dobras: frozenset[int], nome: str
) -> dict[str, dict[str, str]]:
    """Restringe o gabarito por dobra, e recusa em vez de medir zero clipe."""
    folds = {r.shot_id: r.fold for r in load_catalog() if r.big_enough}
    recortado = {c: v for c, v in gabarito.items() if folds.get(c) in dobras}
    if not recortado:
        typer.echo(f"nenhum clipe do gabarito esta na dobra {nome}", err=True)
        raise typer.Exit(code=1)
    return recortado


IGNORADOS = frozenset({".DS_Store", "Thumbs.db", "__pycache__", ".ipynb_checkpoints"})


def _lixo(caminho: Path, raiz: Path) -> bool:
    """Metadado de gerenciador de arquivos mudava o digest do mesmo peso, entre duas maquinas."""
    return any(parte in IGNORADOS for parte in caminho.relative_to(raiz).parts)


ART_KIND = typer.Option(..., help="Especie do modulo que usa o peso.")
ART_NAME = typer.Option(..., help="Nome do modelo.")
ART_VERSION = typer.Option(..., help="Versao do modelo.")
ART_ENTRY = typer.Option(..., help="Arquivo que o `load` do dono abre.")
ART_OUT = typer.Option(Path("models/manifests"), help="Onde escrever o manifesto.")
ART_ROOT = typer.Option(None, help="Onde os pesos ficam.", show_default="../var/models")


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


LABEL_LIMIT = typer.Option(40, help="Quantos clipes sortear para anotar.")
LABEL_SEED = typer.Option(20260825, help="Semente do sorteio, para a amostra ser reproduzivel.")
LABEL_FPS = typer.Option(5.0, help="Taxa da folha grossa.")
LABEL_ANCORA = typer.Option(None, help="Quadro aproximado, para a folha fina em torno dele.")
LABEL_QUEM = typer.Option("", help="Quem anotou, para medir a concordancia entre duas pessoas.")
LABEL_REFAZER = typer.Option(
    False, "--refazer", help="Mostra tambem os clipes que voce ja anotou, para revisar."
)
LABEL_RUN = typer.Option(
    None, help="So para inspecionar o erro depois: anotar vendo o palpite do modelo enviesa."
)


@label_app.command("sheet")
def label_sheet(
    clip_id: str | None = typer.Argument(None, help="Um clipe so; sem ele, a amostra inteira."),
    limit: int = LABEL_LIMIT,
    seed: int = LABEL_SEED,
    fps: float = LABEL_FPS,
    around: int | None = LABEL_ANCORA,
) -> None:
    """Escreve a folha de contato dos quadros candidatos, em PNG, sem abrir janela."""
    try:
        catalogo = {r.shot_id: r for r in load_catalog() if r.big_enough}
    except (CatalogError, DataUnavailable) as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    alvos = (
        (clip_id,)
        if clip_id is not None
        else sample([(r.shot_id, r.match_id) for r in catalogo.values()], limit, seed)
    )
    destino_base = labels_dir() / "sheets"
    for alvo in alvos:
        registro = catalogo.get(alvo)
        if registro is None:
            typer.echo(f"{alvo} nao esta no catalogo", err=True)
            raise typer.Exit(code=1)
        clip = probe(registro.clip_path, clip_id=alvo)
        indices = fine_indices(clip, around) if around is not None else coarse_indices(clip, fps)
        sufixo = "fina" if around is not None else "grossa"
        caminho = contact_sheet(clip, indices, destino_base / f"{alvo}-{sufixo}.png")
        typer.echo(f"  {alvo}: {len(indices)} quadros -> {caminho}")
    typer.echo(f"{len(alvos)} folhas em {destino_base.resolve()}")


@label_app.command("scrub")
def label_scrub(
    limit: int = LABEL_LIMIT,
    seed: int = LABEL_SEED,
    annotator: str = LABEL_QUEM,
    refazer: bool = LABEL_REFAZER,
    run: Path | None = LABEL_RUN,
) -> None:
    """Navega o clipe quadro a quadro numa janela e grava o instante do contato.

    Precisa de janela, entao roda no Mac e nao no wheel de Linux, que sai com `GUI: NONE`. Quem
    esta sem janela usa `xvision label sheet`, que escreve PNG e nao depende de GUI.

    Sem `--run` a janela abre no meio do clipe e nao mostra palpite nenhum, que e como o gabarito
    tem que ser construido: quem anota vendo onde o modelo achou que era anota perto dali, e o erro
    medido contra esse gabarito sai menor do que o real. `--run` serve depois, para olhar um clipe
    especifico e entender por que o modelo errou, e nao para produzir a anotacao.

    Roda de novo e ele comeca no que falta: clipe que voce ja anotou sai da fila, entao a sessao
    seguinte cai exatamente nos que voce pulou ou nao alcancou. `--refazer` traz todos de volta.

    O quadro a marcar e o ultimo em que a bola ainda encosta na parte do corpo, antes de separar, e
    nao o primeiro depois. A razao e dimensional: `body_part` decide comparando a distancia entre a
    bola e cada articulacao, e o sinal que ele procura tem 10 a 25 px, o raio da bola mais a folga
    do keypoint. Um chute de 25 m/s move a bola 35 a 45 px por quadro na escala da transmissao,
    entao um quadro de atraso injeta erro do dobro do tamanho do sinal. Um jeito pratico de achar:
    avanca ate ver a bola claramente saindo, e volta um.

    Toque de conducao, passe que chega, defesa do goleiro e desvio em zagueiro nao sao o contato.
    Quando o clipe tem mais de um candidato, a faixa mostra o que o parceiro marcou (origem,
    distancia, area e desfecho) e e por ela que se decide qual dos dois e o evento.

    Clipe sem finalizacao visivel se pula com `s`. Chutar um quadro poe linha falsa no gabarito, e o
    gabarito e o que mede todo o resto. Medido: os dois pulados dos primeiros 40 eram `is_blocked`
    de fora da area, e chute travado na saida nao parece chute.
    """
    try:
        catalogo = {r.shot_id: r for r in load_catalog() if r.big_enough}
    except (CatalogError, DataUnavailable) as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    from mlops.scrub import SemJanela, anotar, contexto

    sugestoes = _instantes_da_execucao(run, catalogo)
    amostra = sample([(r.shot_id, r.match_id) for r in catalogo.values()], limit, seed)
    anteriores: set[str] = set()
    if not refazer:
        anteriores = {a.clip_id for a in read() if not annotator or a.annotator == annotator}
    feitos = anteriores & set(amostra)
    alvos = tuple(c for c in amostra if c not in feitos)
    if not alvos:
        typer.echo(
            f"os {len(amostra)} clipes da amostra ja foram anotados; use --refazer para rever"
        )
        return
    clipes = [
        (probe(catalogo[c].clip_path, clip_id=c), sugestoes.get(c)) for c in alvos if c in catalogo
    ]
    marcacoes = {c: contexto(catalogo[c].labels) for c in alvos if c in catalogo}
    typer.echo(
        f"  {len(clipes)} clipes a anotar, {len(feitos)} ja anotados.\n"
        "  [setas] 1  [a/d] 10  [A/D] 60  [espaco] marca  [s] pula  [v] volta  [q] sai"
    )
    try:
        marcados = anotar(clipes, annotator, marcacoes)
    except SemJanela as erro:
        typer.echo(f"{erro}", err=True)
        raise typer.Exit(code=1) from erro
    typer.echo(f"{marcados} clipes anotados em {labels_dir().resolve() / 'instants.jsonl'}")


def _instantes_da_execucao(run: Path | None, catalogo: Mapping[str, ShotRecord]) -> dict[str, int]:
    """O instante que o modelo achou, so para posicionar a janela onde ele acha que e o lance.

    Nao e sugestao de resposta: a regua desenha os dois, o do modelo em verde e o atual em laranja,
    e quem anota ve a diferenca. Comecar do meio do clipe faria a pessoa navegar mais, e comecar
    escondendo o palpite do modelo tiraria dela a informacao de que ele existe.
    """
    if run is None:
        return {}
    eventos = run / EVENTS_FILE
    if not eventos.is_file():
        return {}
    instantes: dict[str, int] = {}
    for linha in eventos.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        evento = cast("dict[str, Any]", json.loads(linha))
        clip_id = str(evento["clip_id"])
        registro = catalogo.get(clip_id)
        if registro is None:
            continue
        fps = probe(registro.clip_path, clip_id=clip_id).fps
        instantes[clip_id] = round(float(evento["shot_timestamp_s"]) * fps)
    return instantes


@label_app.command("set")
def label_set(clip_id: str, frame_index: int, annotator: str = LABEL_QUEM) -> None:
    """Registra o quadro do contato que a pessoa leu na folha."""
    if frame_index < 0:
        typer.echo("o quadro do contato e um indice nao negativo", err=True)
        raise typer.Exit(code=1)
    caminho = write(Anotacao(clip_id=clip_id, frame_index=frame_index, annotator=annotator))
    typer.echo(str(caminho))


@label_app.command("stats")
def label_stats() -> None:
    """Quantos clipes ja tem gabarito, de quantas partidas, e quanto dois anotadores discordam."""
    try:
        anotacoes = read()
    except InstantError as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    if not anotacoes:
        typer.echo("nenhuma anotacao ainda: comece por xvision label sheet")
        return
    por_clipe = consensus(anotacoes)
    typer.echo(f"  {len(por_clipe)} clipes anotados, {len(anotacoes)} anotacoes")
    concordancia = agreement(anotacoes)
    if concordancia is None:
        typer.echo("  nenhum clipe anotado duas vezes: o gabarito esta sem barra de erro propria")
        return
    typer.echo(
        f"  concordancia em {int(concordancia['n_clipes'])} clipes: "
        f"mediana {concordancia['mediana_quadros']:.0f} quadros, "
        f"maximo {concordancia['max_quadros']:.0f}"
    )


def dobras_de(dobra: str) -> frozenset[int]:
    """Traduz o nome da dobra em conjunto de indices, ou encerra com a lista do que existe.

    Um so lugar porque `eval-ball`, `eval-field`, `run` e `eval-instant` cobram o mesmo vocabulario,
    e um deles aceitando `desenvolvimento` calado mediria a populacao errada sem avisar.
    """
    dobras = DOBRAS_DA_BOLA.get(dobra)
    if dobras is None:
        typer.echo(f"dobra {dobra!r} nao existe: use {', '.join(DOBRAS_DA_BOLA)}", err=True)
        raise typer.Exit(code=1)
    return dobras


DOBRAS_DA_BOLA: Mapping[str, frozenset[int]] = {
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
"""

BALL_CONFIG = typer.Option(None, help="Modulos a medir. Padrao: pipeline.toml.")
BALL_LIMIT = typer.Option(0, help="Para depois de N clipes anotados. Zero e sem limite.")
BALL_JSON = typer.Option(None, help="Grava o resultado em JSON, para comparar duas execucoes.")
BALL_BASE = typer.Option(
    None, help="JSON de uma execucao anterior, para ver o que ganhou e perdeu."
)
BALL_DOBRA = typer.Option(
    "ajuste", help="ajuste (dobras 0 a 2, para calibrar), reservado (3 e 4) ou tudo."
)


@app.command("eval-ball")
def eval_ball(
    config: Path | None = BALL_CONFIG,
    limit: int = BALL_LIMIT,
    saida: Path | None = BALL_JSON,
    base: Path | None = BALL_BASE,
    dobra: str = BALL_DOBRA,
) -> None:
    """Mede se a bola detectada no instante do contato e a bola do lance.

    `xvision eval` mede a feature no fim da cadeia e nao diz onde ela quebrou. Este mede o elo que
    esta quebrado: numa finalizacao a bola encosta no finalizador por definicao, entao no quadro do
    contato anotado a deteccao tem que cair sobre ou colada a uma caixa de jogador.
    """
    from edge.graph.runner import run_pipeline
    from mlops.ball_quality import Contato, compara, contato_de, measure, quadro_do_contato

    caminho = _config(config)
    if not caminho.is_file():
        typer.echo(f"{caminho} nao existe: passe --config com o caminho do arquivo", err=True)
        raise typer.Exit(code=1)
    try:
        gabarito = consensus(read())
        catalogo = {r.shot_id: r for r in load_catalog() if r.big_enough}
    except ERROS_DE_USO as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    dobras = dobras_de(dobra)
    alvos = sorted(c for c in gabarito if c in catalogo and catalogo[c].fold in dobras)
    if limit:
        alvos = alvos[:limit]
    if not alvos:
        typer.echo("nenhum clipe anotado; rode `xvision label scrub`", err=True)
        raise typer.Exit(code=1)
    modulos = {
        kind: module
        for kind, module in load_modules(read_config(caminho)).items()
        if kind in {"players", "ball"}
    }
    framesets = read_framesets(caminho)
    contatos: list[Contato] = []
    carregados: list[Any] = []
    try:
        for modulo in modulos.values():
            modulo.load()
            carregados.append(modulo)
        for clip_id in alvos:
            registro = catalogo[clip_id]
            clip = probe(registro.clip_path, clip_id=clip_id)
            provider = FrameProvider(VideoSource(registro.clip_path))
            try:
                percepcao = run_pipeline(
                    clip=clip, modules=modulos, framesets=framesets, provider=provider
                ).perception
            finally:
                provider.close()
            contatos.append(
                contato_de(
                    clip_id,
                    quadro_do_contato(percepcao.frames, gabarito[clip_id]),
                    clip.height,
                )
            )
    finally:
        for modulo in reversed(carregados):
            modulo.close()
    relatorio = measure(contatos)
    typer.echo(json.dumps(relatorio.to_dict(), indent=2, ensure_ascii=False))
    typer.echo(f"  dobra {dobra}: {sorted(dobras)}")
    typer.echo(
        f"  encostada em {relatorio.encostada}/{relatorio.n} = {relatorio.taxa:.3f}   "
        f"(contando interpolacao seriam {relatorio.encostada_incluindo_inventada})"
    )
    if base is not None:
        _comparar_bola(base, contatos, relatorio.taxa, compara)
    if saida is not None:
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(
            json.dumps(
                {
                    "config": str(caminho),
                    "resumo": relatorio.to_dict(),
                    "por_clipe": {
                        c.clip_id: {
                            "tem_bola": c.tem_bola,
                            "distancia_px": c.distancia_px,
                            "y_relativo": c.y_relativo,
                            "encostada": c.encostada,
                            "detectada": c.detectada,
                        }
                        for c in contatos
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        typer.echo(f"  {saida}")


BALL_TRAJECTORY_LABELS = typer.Option(..., help="JSONL validado pelo schema ball-trajectory-v1.")
BALL_TRAJECTORY_CONFIG = typer.Option(None, help="Configuracao com o modulo ball a medir.")
BALL_TRAJECTORY_JSON = typer.Option(None, help="Grava somente o relatorio agregado em JSON.")
BALL_TRAJECTORY_DOBRA = typer.Option(
    "ajuste", help="ajuste (dobras 0 a 2), reservado (3 e 4) ou tudo."
)


@app.command("eval-ball-trajectory")
def eval_ball_trajectory(
    labels: Path = BALL_TRAJECTORY_LABELS,
    config: Path | None = BALL_TRAJECTORY_CONFIG,
    saida: Path | None = BALL_TRAJECTORY_JSON,
    dobra: str = BALL_TRAJECTORY_DOBRA,
) -> None:
    """Mede posicao e continuidade em rotulos privados, sem depender de jogadores."""
    from edge.graph.runner import assemble
    from edge.modules import inference
    from mlops.ball_trajectory import (
        BallTrajectoryLabelError,
        aggregate_telemetry,
        read_ball_trajectory_labels,
        validate_match_folds,
    )
    from mlops.ball_trajectory import measure as measure_trajectory

    path = _config(config)
    folds = dobras_de(dobra)
    try:
        annotations = read_ball_trajectory_labels(labels)
        catalog = {record.shot_id: record for record in load_catalog() if record.big_enough}
        validate_match_folds((record.match_id, record.fold) for record in catalog.values())
        modules = load_modules(read_config(path))
    except ERROS_DE_USO + (BallTrajectoryLabelError,) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    ball = modules.get("ball")
    if ball is None:
        typer.echo(f"{path} nao declara modules.ball", err=True)
        raise typer.Exit(code=1)
    selected = tuple(
        annotation
        for annotation in annotations
        if annotation.clip_id in catalog
        and catalog[annotation.clip_id].fold in folds
        and catalog[annotation.clip_id].match_id == annotation.match_id
    )
    if not selected:
        typer.echo("nenhum frame anotado pertence a dobra e ao catalogo selecionados", err=True)
        raise typer.Exit(code=1)

    grouped: dict[str, list[Any]] = {}
    for annotation in selected:
        grouped.setdefault(annotation.clip_id, []).append(annotation)
    predictions: dict[tuple[str, int], Any] = {}
    telemetry_runs: list[Mapping[str, int | float]] = []
    ball.load()
    try:
        for clip_id, group in grouped.items():
            inference.reset()
            record = catalog[clip_id]
            if sha256_of(record.clip_path) != group[0].video_sha256:
                raise BallTrajectoryLabelError(
                    "video_sha256 do clipe anotado nao corresponde ao arquivo local"
                )
            clip = replace(probe(record.clip_path, clip_id=clip_id), match_id=record.match_id)
            provider = FrameProvider(VideoSource(record.clip_path))
            try:
                indices = tuple(sorted(annotation.frame_index for annotation in group))
                batch = provider.frames(indices, ball.spec.preprocess)
                output = ball.infer(batch, {})
                perception = assemble(clip, {"ball": output})
            finally:
                provider.close()
            predictions.update(
                ((clip_id, frame.frame_index), frame.ball)
                for frame in perception.frames
                if frame.ball is not None
            )
            telemetry_runs.append(cast("Mapping[str, int | float]", getattr(ball, "telemetry", {})))
    except (BallTrajectoryLabelError,) + ERROS_DE_USO as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    finally:
        ball.close()

    report = measure_trajectory(selected, predictions)
    document = {
        "population": f"folds={sorted(folds)}",
        "telemetry": aggregate_telemetry(telemetry_runs),
        **report.to_dict(),
    }
    typer.echo(json.dumps(document, indent=2, ensure_ascii=False))
    if saida is not None:
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"  {saida}")


def _comparar_bola(
    base: Path,
    contatos: Sequence[Any],
    taxa: float,
    compara: Callable[[Mapping[str, bool], Mapping[str, bool]], Mapping[str, int]],
) -> None:
    """A media sozinha esconde troca: uma mudanca pode ganhar cinco clipes e perder cinco.

    Sem o par ganhou/perdeu, duas correcoes com a mesma media parecem equivalentes quando uma
    consertou o que a outra quebrou, e a proxima iteracao escolhe pelo numero errado.
    """
    if not base.is_file():
        typer.echo(f"{base} nao existe, sem comparacao", err=True)
        return
    documento = cast("dict[str, Any]", json.loads(base.read_text(encoding="utf-8")))
    antes = {
        k: bool(v["encostada"])
        for k, v in cast("dict[str, dict[str, Any]]", documento["por_clipe"]).items()
    }
    depois = {c.clip_id: bool(c.encostada) for c in contatos}
    delta = compara(antes, depois)
    anterior = float(documento["resumo"]["taxa_encostada"])
    typer.echo(
        f"  contra {base.name}: {anterior:.3f} -> {taxa:.3f}   "
        f"ganhou {delta['ganhou']}, perdeu {delta['perdeu']}, em {delta['n']} clipes comuns"
    )
    ganhos = sorted(c for c in antes if c in depois and depois[c] and not antes[c])
    perdas = sorted(c for c in antes if c in depois and antes[c] and not depois[c])
    if ganhos:
        typer.echo(f"    ganhou: {' '.join(ganhos)}")
    if perdas:
        typer.echo(f"    perdeu: {' '.join(perdas)}")


FIELD_LABELS = typer.Option(..., help="JSONL validado pelo schema field-calibration-v1.")
FIELD_CONFIG = typer.Option(None, help="Configuracao com o modulo field a medir.")
FIELD_JSON = typer.Option(None, help="Grava somente o relatorio agregado em JSON.")
FIELD_METHOD = typer.Option(None, help="Sobrescreve least-squares, ransac ou magsac.")
FIELD_DOBRA = typer.Option("ajuste", help="ajuste (dobras 0 a 2), reservado (3 e 4) ou tudo.")
FIELD_PROPOSALS = typer.Option(
    False, help="Inclui propostas em diagnostico exploratorio explicito."
)


@app.command("eval-field")
def eval_field(
    labels: Path = FIELD_LABELS,
    config: Path | None = FIELD_CONFIG,
    saida: Path | None = FIELD_JSON,
    method: str | None = FIELD_METHOD,
    dobra: str = FIELD_DOBRA,
    include_proposals: bool = FIELD_PROPOSALS,
) -> None:
    """Mede homografia em linhas confirmadas por humano, preservando a procedencia."""
    from contracts import FieldGeometry
    from edge.artifacts import sha256_of
    from edge.graph.runner import assemble
    from edge.modules.registry import ConfigurableModule
    from mlops.field_quality import (
        FieldLabelError,
        evaluation_labels,
        read_field_labels,
        validate_label_video,
    )
    from mlops.field_quality import measure as measure_field

    path = _config(config)
    folds = dobras_de(dobra)
    try:
        annotations = read_field_labels(labels)
        catalog = {record.shot_id: record for record in load_catalog() if record.big_enough}
        modules = load_modules(read_config(path))
    except ERROS_DE_USO + (FieldLabelError,) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    field = modules.get("field")
    if field is None:
        typer.echo(f"{path} nao declara modules.field", err=True)
        raise typer.Exit(code=1)
    if method is not None:
        if not isinstance(field, ConfigurableModule):
            typer.echo("o modulo field nao aceita trocar method", err=True)
            raise typer.Exit(code=1)
        field.configure({"method": method})
    selected = tuple(
        annotation
        for annotation in annotations
        if annotation.clip_id in catalog
        and catalog[annotation.clip_id].fold in folds
        and catalog[annotation.clip_id].match_id == annotation.match_id
    )
    if not selected:
        typer.echo("nenhum frame anotado pertence a dobra e ao catalogo selecionados", err=True)
        raise typer.Exit(code=1)
    grouped: dict[str, list[Any]] = {}
    for annotation in evaluation_labels(selected, include_proposals=include_proposals):
        grouped.setdefault(annotation.clip_id, []).append(annotation)
    geometries: dict[tuple[str, int], FieldGeometry] = {}
    field.load()
    try:
        for clip_id, group in grouped.items():
            record = catalog[clip_id]
            clip = replace(probe(record.clip_path, clip_id=clip_id), match_id=record.match_id)
            video_sha256 = sha256_of(record.clip_path)
            for annotation in group:
                validate_label_video(annotation, clip, video_sha256)
            provider = FrameProvider(VideoSource(record.clip_path))
            try:
                indices = tuple(annotation.frame_index for annotation in group)
                batch = provider.frames(indices, field.spec.preprocess)
                output = field.infer(batch, {})
                perception = assemble(clip, {"field": output})
            finally:
                provider.close()
            geometries.update(
                ((clip_id, frame.frame_index), frame.field)
                for frame in perception.frames
                if frame.field is not None
            )
    except FieldLabelError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    finally:
        field.close()
    report = measure_field(selected, geometries, include_proposals=include_proposals)
    document = {"population": f"folds={sorted(folds)}", **report.to_dict()}
    typer.echo(json.dumps(document, indent=2, ensure_ascii=False))
    if saida is not None:
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"  {saida}")


PLACAR_RUN = typer.Option(..., help="Execucao com --gabarito de onde ler as features.")
PLACAR_BOLA = typer.Option(..., help="JSON gravado por eval-ball com --saida.")


@app.command("placar")
def placar(run_dir: Path = PLACAR_RUN, bola: Path = PLACAR_BOLA) -> None:
    """Junta bola, body_part de tres classes e de duas classes na mesma tabela.

    Medir uma correcao pede tres comandos e a comparacao entre eles some no terminal. Sem juntar,
    uma mudanca que sobe a taxa da bola e derruba `body_part` passa por vitoria.
    """
    from mlops.ball_quality import ACASO
    from mlops.placar import colapsa, linha_de_bola, linha_de_feature, previsoes, tabela

    try:
        catalogo = {r.shot_id: r for r in load_catalog() if r.big_enough}
    except ERROS_DE_USO as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    if not bola.is_file():
        typer.echo(f"{bola} nao existe: rode eval-ball com --saida", err=True)
        raise typer.Exit(code=1)
    documento = cast("dict[str, Any]", json.loads(bola.read_text(encoding="utf-8")))
    linhas = [linha_de_bola(cast("dict[str, Any]", documento["resumo"]), ACASO)]
    eventos = run_dir / EVENTS_FILE
    for feature, transforma in (("body_part", str), ("body_part", colapsa), ("type_of_play", str)):
        previsto = {c: transforma(v) for c, v in previsoes(eventos, feature).items()}
        verdade = {c: transforma(catalogo[c].labels[feature]) for c in previsto if c in catalogo}
        nome = feature if transforma is str else f"{feature} (cabeca x pe)"
        achada = linha_de_feature(nome, previsto, verdade)
        if achada is not None:
            linhas.append(achada)
    typer.echo(tabela(linhas))


INSTANT_DOBRA = typer.Option("tudo", help="ajuste (dobras 0 a 2), reservado (3 e 4) ou tudo.")


@app.command("eval-instant")
def evaluate_instant(run: Path, dobra: str = INSTANT_DOBRA) -> None:
    """Mede o instante detectado contra o gabarito manual, com o ponto medio como piso.

    `--dobra` filtra na medicao, e nao na execucao: um `run` que contenha clipe reservado ainda
    pode ser lido so na parte de desenvolvimento, sem consumir o holdout na decisao.
    """
    eventos = run / "events.jsonl"
    if not eventos.is_file():
        typer.echo(f"{eventos} nao existe: passe o diretorio de uma execucao", err=True)
        raise typer.Exit(code=1)
    dobras = dobras_de(dobra)
    try:
        anotacoes = read()
    except InstantError as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    gabarito = consensus(anotacoes)
    if not gabarito:
        typer.echo("sem gabarito manual nao ha contra o que medir o instante", err=True)
        raise typer.Exit(code=1)
    if dobras != DOBRAS_DA_BOLA["tudo"]:
        try:
            folds = {r.shot_id: r.fold for r in load_catalog() if r.big_enough}
        except ERROS_DE_USO as erro:
            typer.echo(str(erro), err=True)
            raise typer.Exit(code=1) from erro
        gabarito = {c: v for c, v in gabarito.items() if folds.get(c) in dobras}
        if not gabarito:
            typer.echo(f"nenhum clipe anotado na dobra {dobra}", err=True)
            raise typer.Exit(code=1)

    previsto: dict[str, int] = {}
    clips: set[str] = set()
    for linha in eventos.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        evento = cast("dict[str, Any]", json.loads(linha))
        clip = str(evento["clip_id"])
        if clip in clips:
            typer.echo("events.jsonl: clipe duplicado impede conferir a procedencia", err=True)
            raise typer.Exit(code=1)
        clips.add(clip)
        fps = _fps_do_clipe(clip)
        if fps is None:
            continue
        previsto[clip] = round(float(evento["shot_timestamp_s"]) * fps)

    procedencia = _procedencia(run, clips)
    relatorio = measure(
        previsto,
        gabarito,
        fontes=procedencia.fontes,
        concordancia=agreement(anotacoes),
    )
    if relatorio.n == 0:
        typer.echo("nenhum clipe anotado aparece nesta execucao", err=True)
        raise typer.Exit(code=1)
    baixo, alto = wilson(round(relatorio.dentro_da_tolerancia * relatorio.n), relatorio.n)
    typer.echo(json.dumps(relatorio.to_dict(), ensure_ascii=False, indent=2))
    typer.echo(
        f"  dentro de {TOLERANCIA_ALVO} quadros: {relatorio.dentro_da_tolerancia:.3f} "
        f"(IC95 {baixo:.3f} a {alto:.3f}, n={relatorio.n})"
    )
    _mostra_nulos(
        baselines(gabarito, _comprimentos(gabarito)),
        resolution(round(relatorio.dentro_da_tolerancia * relatorio.n), relatorio.n),
    )
    causas = taxonomy(
        diagnose(
            previsto,
            gabarito,
            fontes=procedencia.fontes,
            candidatos=procedencia.candidatos,
        ),
        gabarito=gabarito,
        pools=procedencia.pools,
    )
    typer.echo(json.dumps(causas.to_dict(), ensure_ascii=False, indent=2))
    typer.echo(
        f"  causas nomeadas: {causas.cobertura:.3f} de {causas.erros} erros"
        if causas.erros
        else "  nenhum erro acima da tolerancia nesta execucao"
    )


class Procedencia(NamedTuple):
    """O que o arquivo de procedencia sabe sobre cada clipe da execucao.

    Os candidatos vem no mesmo objeto porque saem da mesma linha do mesmo arquivo: separa-los em
    dois leitores faria a segunda leitura repetir, ou pior esquecer, a conferencia de que
    procedencia e eventos falam dos mesmos clipes.
    """

    fontes: dict[str, str]
    candidatos: dict[str, list[dict[str, object]]]
    pools: dict[str, list[int]]


SPOTS_DOBRA = typer.Option("ajuste", help="ajuste (dobras 0 a 2), reservado (3 e 4) ou tudo.")


@app.command("eval-spots")
def evaluate_spots(spots: Path, dobra: str = SPOTS_DOBRA) -> None:
    """Mede um detector de instante de fora deste repositorio contra o gabarito manual.

    O arquivo tem uma linha por clipe, com `clip_id` e `frame_index` em quadro nativo. Serve para
    avaliar alternativa cujo codigo nao pode entrar em `src/`, como o T-DEED, que e GPL-3.0 contra
    o CC-BY-4.0 deste projeto. O detector roda fora, escreve o arquivo, e a comparacao com a regra
    em vigor acontece aqui, com o mesmo gabarito, a mesma tolerancia e os mesmos nulos.

    O padrao e `--dobra ajuste` e nao `tudo`, ao contrario do `eval-instant`: alternativa externa e
    exatamente o caso em que abrir o reservado por descuido custa o holdout do projeto.
    """
    try:
        previsto = read_spots(spots)
        anotacoes = read()
    except InstantError as erro:
        typer.echo(str(erro), err=True)
        raise typer.Exit(code=1) from erro
    gabarito = consensus(anotacoes)
    if not gabarito:
        typer.echo("sem gabarito manual nao ha contra o que medir o instante", err=True)
        raise typer.Exit(code=1)
    dobras = dobras_de(dobra)
    if dobras != DOBRAS_DA_BOLA["tudo"]:
        try:
            folds = {r.shot_id: r.fold for r in load_catalog() if r.big_enough}
        except ERROS_DE_USO as erro:
            typer.echo(str(erro), err=True)
            raise typer.Exit(code=1) from erro
        gabarito = {c: v for c, v in gabarito.items() if folds.get(c) in dobras}
        if not gabarito:
            typer.echo(f"nenhum clipe anotado na dobra {dobra}", err=True)
            raise typer.Exit(code=1)

    relatorio = measure(previsto, gabarito, concordancia=agreement(anotacoes))
    if relatorio.n == 0:
        typer.echo("nenhum clipe anotado aparece neste arquivo de spots", err=True)
        raise typer.Exit(code=1)
    baixo, alto = wilson(round(relatorio.dentro_da_tolerancia * relatorio.n), relatorio.n)
    typer.echo(json.dumps(relatorio.to_dict(), ensure_ascii=False, indent=2))
    typer.echo(
        f"  dentro de {TOLERANCIA_ALVO} quadros: {relatorio.dentro_da_tolerancia:.3f} "
        f"(IC95 {baixo:.3f} a {alto:.3f}, n={relatorio.n})"
    )
    _mostra_nulos(
        baselines(gabarito, _comprimentos(gabarito)),
        resolution(round(relatorio.dentro_da_tolerancia * relatorio.n), relatorio.n),
    )


def _procedencia(run: Path, clips: set[str]) -> Procedencia:
    """Le a procedencia por clipe da execucao, quando ela existe.

    Execucao antiga nao tem o arquivo, e ai o relatorio continua dizendo `desconhecida` em vez de
    inventar `ball-break`: a diferenca entre nao saber e saber que foi medido e o que este arquivo
    resolve, e ela nao pode ser preenchida por otimismo. O mesmo vale um nivel abaixo para
    `candidates`, que execucao anterior ao campo nao tem: a lista fica ausente e a taxonomia diz
    `nao-declarado`, em vez de ler ausencia como lista vazia e concluir que nao havia quebra.
    """
    caminho = run / OBSERVATIONS_FILE
    if not caminho.is_file():
        return Procedencia({}, {}, {})
    fontes: dict[str, str] = {}
    candidatos: dict[str, list[dict[str, object]]] = {}
    pools: dict[str, list[int]] = {}
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            decoded: object = json.loads(linha)
        except json.JSONDecodeError as erro:
            typer.echo(f"{caminho.name}:{numero}: JSON invalido: {erro.msg}", err=True)
            raise typer.Exit(code=1) from erro
        if not isinstance(decoded, dict):
            typer.echo(f"{caminho.name}:{numero}: registro de procedencia invalido", err=True)
            raise typer.Exit(code=1)
        documento = cast("dict[str, object]", decoded)
        clip = documento.get("clip_id")
        fonte = documento.get("shot_source")
        if not isinstance(clip, str) or not isinstance(fonte, str) or not fonte:
            typer.echo(f"{caminho.name}:{numero}: registro de procedencia invalido", err=True)
            raise typer.Exit(code=1)
        if clip in fontes:
            typer.echo(f"{caminho.name}:{numero}: procedencia duplicada", err=True)
            raise typer.Exit(code=1)
        fontes[clip] = fonte
        lista = documento.get("candidates")
        if lista is not None:
            if not isinstance(lista, list):
                typer.echo(f"{caminho.name}:{numero}: candidates precisa ser lista", err=True)
                raise typer.Exit(code=1)
            candidatos[clip] = [
                cast("dict[str, object]", item)
                for item in cast("list[object]", lista)
                if isinstance(item, dict)
            ]
        conjunto = documento.get("candidate_pool")
        if conjunto is not None:
            if not isinstance(conjunto, list):
                typer.echo(f"{caminho.name}:{numero}: candidate_pool precisa ser lista", err=True)
                raise typer.Exit(code=1)
            pools[clip] = [
                item
                for item in cast("list[object]", conjunto)
                if isinstance(item, int) and not isinstance(item, bool)
            ]
    if set(fontes) != clips:
        typer.echo("procedencia e eventos precisam conter os mesmos clipes", err=True)
        raise typer.Exit(code=1)
    return Procedencia(fontes, candidatos, pools)


def _comprimentos(clips: Iterable[str]) -> dict[str, int]:
    """Quantos quadros tem cada clipe, que e o denominador do nulo do quadro ao acaso."""
    catalogo = {r.shot_id: r for r in load_catalog() if r.big_enough}
    medidos: dict[str, int] = {}
    for clip in clips:
        registro = catalogo.get(clip)
        if registro is None:
            continue
        try:
            medidos[clip] = probe(registro.clip_path, clip_id=clip).frame_count
        except ERROS_DE_USO:
            continue
    return medidos


def _mostra_nulos(relatorio: BaselineReport, resolucao: Resolucao) -> None:
    """Imprime as linhas de base e a resolucao junto da medida, sempre.

    Separado num helper porque os dois comandos de instante precisam imprimir o mesmo, e a
    primeira versao desta sprint publicou uma taxa sem nulo e ela foi lida como o dobro do que era.
    A resolucao entrou pelo mesmo motivo um nivel acima: sem ela, mudanca de um clipe em 24 e
    apresentada como ganho.
    """
    typer.echo(
        json.dumps(
            {"linhas_de_base": relatorio.to_dict(), "resolucao": resolucao.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )
    typer.echo(
        f"  este conjunto nao distingue mudanca menor que {resolucao.clipes:.1f} clipes; "
        f"a comparacao pareada precisa de {resolucao.discordantes_para_sinal} discordantes"
    )


def _fps_do_clipe(clip_id: str) -> float | None:
    try:
        registro = next(r for r in load_catalog() if r.shot_id == clip_id and r.big_enough)
    except (StopIteration, CatalogError, DataUnavailable):
        return None
    return probe(registro.clip_path, clip_id=clip_id).fps
