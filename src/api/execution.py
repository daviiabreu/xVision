"""Roda o pipeline fora do ciclo da requisicao e escreve o resultado no estado.

O processamento nao cabe numa requisicao HTTP: um clipe leva segundos e um proxy corta a conexao
antes disso. Entao o handler responde `202` com o identificador e o trabalho segue aqui.

A concorrencia e limitada de proposito, e o padrao e um. Medido no servico sincrono anterior:
tres envios simultaneos de 300 MB levaram o processo a 2,81 GB de pico, porque o pool de threads
do Starlette aceita quarenta chamadas bloqueantes ao mesmo tempo e cada pipeline abre o seu
proprio decodificador. Uma fila de um transforma pico de memoria em espera na fila, que e a troca
certa para um dispositivo de borda, onde memoria e o recurso escasso e latencia de fila nao e.

Nenhuma excecao sobe daqui. O que sobe de um worker morre sem leitor e deixaria o video parado em
`Em processamento` para sempre, entao toda falha vira estado `Falhou` com motivo.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from api import render, store
from api.schemas import Metrica, Resultado, tipo_de
from edge.cache.perception import PerceptionCache
from edge.features.registry import load_extractors, read_feature_config
from edge.frames.provider import ClipUnavailable, FrameProvider, VideoSource, probe
from edge.graph.framesets import read_framesets
from edge.modules.registry import load_modules, read_config
from edge.paths import cache_dir
from edge.pipeline import PipelineError, process_clip

WORKERS_ENV = "XVISION_API_WORKERS"
FILA_ENV = "XVISION_API_MAX_FILA"
FILA_PADRAO = 32
SEM_CAMINHO = "<arquivo enviado>"
FALHA_INTERNA = "o video nao pode ser processado por um erro interno do servico"

logger = logging.getLogger("xvision.api")


class FilaCheia(RuntimeError): ...


# O teto so e teto se conferir e enfileirar acontecerem juntos. Medido sem o cadeado: com teto
# de 4 e 16 clientes numa barreira, os 16 foram aceitos e nenhum recusado, porque `submeter` e
# sincrono e `_pendentes` muda em mais de uma thread: quem enfileira (o handler `receber` do PUT,
# a recuperacao de orfaos) le o tamanho antes de inserir, e o `add_done_callback` remove no pool.
_porta = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_pendentes: set[Future[None]] = set()
_desligado = False


def workers() -> int:
    declarado = os.environ.get(WORKERS_ENV)
    return max(1, int(declarado)) if declarado else 1


def executor() -> ThreadPoolExecutor:
    """So cria o pool se ninguem desligou. Depois de `desligar`, submeter e erro e nao ressurreicao.

    Recriando o pool sob demanda, uma tarefa que chegasse depois do desligamento levantava uma
    thread nova que ninguem mais desliga, e o caminho de saida que este modulo existe para
    garantir deixava de valer.
    """
    global _executor
    if _desligado:
        raise FilaCheia("o servico esta desligando e nao aceita trabalho novo")
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=workers(), thread_name_prefix="xvision")
    return _executor


def teto_da_fila() -> int:
    declarado = os.environ.get(FILA_ENV)
    return max(1, int(declarado)) if declarado else FILA_PADRAO


def submeter(funcao: Callable[..., None], *argumentos: Any) -> Future[None]:
    """Enfileira, e recusa quando a fila esta cheia.

    Sem teto, medido: 200 envios foram aceitos em 0,28 s e chegaram a 186 arquivos simultaneos em
    `var/uploads`. Com o teto de 128 MB por arquivo, isso e ate 23 GB de disco em transito, e
    nada devolvia 429 nem 503. Aceitar trabalho que nao se consegue fazer e pior que recusa-lo:
    quem chamou fica esperando um resultado que nunca vem.
    """
    with _porta:
        if len(_pendentes) >= teto_da_fila():
            raise FilaCheia(f"a fila esta cheia: {len(_pendentes)} trabalhos esperando")
        tarefa = executor().submit(funcao, *argumentos)
        _pendentes.add(tarefa)
    tarefa.add_done_callback(_pendentes.discard)
    return tarefa


def aguardar(timeout: float = 60.0) -> None:
    """Esperar a fila esvaziar, que e o que o teste precisa e o desligamento tambem.

    O prazo e absoluto, e nao por rodada do laco. Com prazo por rodada, uma tarefa que nunca
    termina faz `wait` voltar por tempo esgotado e o laco recomecar para sempre: em vez de falhar,
    a suite pendura sem dizer o que esta esperando. Aconteceu.
    """
    limite = time.monotonic() + timeout
    while _pendentes:
        restante = limite - time.monotonic()
        if restante <= 0:
            raise TimeoutError(
                f"{len(_pendentes)} tarefa(s) ainda na fila depois de {timeout:.0f}s"
            )
        wait(set(_pendentes), timeout=restante)


def religar() -> None:
    """Reabre a fila. Existe para o teste, e para quem embute o servico e sobe de novo."""
    global _desligado
    _desligado = False


def desligar() -> None:
    """Espera o que ja comecou e cancela o que ainda nao.

    `shutdown(wait=True)` sem cancelar executa a fila inteira antes de sair: medido, cinco clipes
    enfileirados levaram o desligamento a 30,2 s. `docker stop` mata em 10 s e o Kubernetes em 30,
    entao o processo apanhava SIGKILL no meio e deixava videos presos, que e o problema que a
    recuperacao de queda existe para limpar. Cancelando o que nao comecou, o desligamento passa a
    durar o que durar o clipe em voo, e os cancelados voltam na proxima subida.
    """
    global _executor, _desligado
    with _porta:
        _desligado = True
        pool, _executor = _executor, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
    _pendentes.clear()


def _sem_caminho(mensagem: str, caminho: Path) -> str:
    """A mensagem da borda cita o caminho do arquivo, e ele nao pode chegar ao cliente.

    Medido no servico anterior: um envio que nao era video devolvia o caminho absoluto completo,
    expondo o diretorio do autor, o layout da implantacao e o nome do temporario.
    """
    return mensagem.replace(str(caminho), SEM_CAMINHO).replace(caminho.name, SEM_CAMINHO)


def processar_video(
    identificador: int, caminho: Path, config: Path, banco: Path | None = None
) -> None:
    inicio = time.monotonic()
    try:
        store.marcar(identificador, store.PROCESSANDO, caminho=banco)
        logger.info("video=%s iniciou", identificador)
        resultados, motivo = _analisar(identificador, caminho, config)
        _renderizar(identificador, caminho, config)
    except ClipUnavailable as erro:
        logger.warning("video=%s recusado: %s", identificador, type(erro).__name__)
        store.marcar(
            identificador,
            store.FALHOU,
            motivo=_sem_caminho(str(erro), caminho),
            caminho=banco,
        )
    except Exception:
        logger.exception("video=%s morreu por erro interno", identificador)
        store.marcar(identificador, store.FALHOU, motivo=FALHA_INTERNA, caminho=banco)
    else:
        gasto = (time.monotonic() - inicio) * 1000
        if resultados is None:
            logger.info("video=%s falhou ms=%.0f motivo=%s", identificador, gasto, motivo)
            store.marcar(identificador, store.FALHOU, motivo=motivo, caminho=banco)
        else:
            logger.info(
                "video=%s terminou etapas=%d ms=%.0f", identificador, len(resultados), gasto
            )
            store.marcar(identificador, store.FINALIZADO, resultados=resultados, caminho=banco)
    finally:
        caminho.unlink(missing_ok=True)


def _analisar(
    identificador: int, caminho: Path, config: Path
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """`ClipUnavailable` continua subindo: clipe que nao abre nao produz evento nem imagem."""
    try:
        return _rodar(identificador, caminho, config), None
    except PipelineError as erro:
        logger.warning("video=%s sem evento: %s", identificador, type(erro).__name__)
        return None, _sem_caminho(str(erro), caminho)


def _renderizar(identificador: int, caminho: Path, config: Path) -> None:
    """Falha aqui nao derruba o resultado da analise."""
    try:
        configuracao = render.read_config(config)
        if configuracao is None:
            return
        render.render(identificador, caminho, configuracao)
    except render.RenderUnavailable as erro:
        logger.warning("video=%s sem visao superior: %s", identificador, erro)
    except Exception:
        logger.exception("video=%s: a visao superior morreu por erro interno", identificador)


def _rodar(identificador: int, caminho: Path, config: Path) -> list[dict[str, Any]]:
    extractors = load_extractors(read_feature_config(config))
    clipe = probe(caminho, clip_id=f"video-{identificador}")
    provider = FrameProvider(VideoSource(caminho))
    try:
        evento, etapas = process_clip(
            clip=clipe,
            modules=load_modules(read_config(config)),
            framesets=read_framesets(config),
            provider=provider,
            extractors=extractors,
            cache=PerceptionCache(cache_dir()),
        )
    finally:
        provider.close()

    unidades = {e.spec.name: e.spec.unit for e in extractors}
    medidas = [
        Metrica(
            nome=feature.name,
            unidade_de_medida=unidades.get(feature.name, ""),
            tipo=tipo_de(feature.value),
            valor=feature.value,
            status=feature.status,
        )
        for feature in evento.features
    ]
    return [
        Resultado(
            nome_processo=etapa.kind,
            status=etapa.status == "ok",
            frame=None,
            metricas=medidas if etapa.kind == "shot" else [],
        ).model_dump()
        for etapa in etapas
    ]
