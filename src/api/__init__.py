"""O servico HTTP que serve o frontend, rodando o pipeline da borda de forma assincrona.

Na arquitetura de destino a borda processa perto de onde o video chega e um servico central
recebe o evento pronto. Este e a ponte da sprint 2, em que as duas pontas rodam na mesma maquina.

O envio e um caminho so: pedir uma URL em `POST /videos/upload-url` e mandar o binario cru nela
por `PUT`, que transmite o arquivo direto para o disco. Houve um segundo, que aceitava o video em
base64 dentro de um campo JSON, porque duas especificacoes chegaram do time do frontend e o
servico preferiu atender as duas a escolher uma e descobrir a divergencia no dia da integracao. A
interface fechou no `PUT`, o caminho base64 ficou sem nenhum cliente, e ele saiu: base64 infla
33%, o corpo inteiro precisa existir em memoria antes de virar bytes, e manter uma superficie que
ninguem chama custa teste e leitura em toda revisao.

O endereco de envio aponta para o proprio servico e sai como caminho relativo a raiz dele. Ele ja
saiu absoluto, montado com o `Host` que o cliente mandou, e isso escrevia no corpo da resposta um
dado que nao e nosso: proxy que reescreve `Host` devolvia o nome interno do servico, e com TLS
terminado acima devolvia `http://`, que o navegador recusa como conteudo misto. Quem chama ja sabe
a base, porque precisou dela para chegar aqui.

Ele e area de composicao como o `cli`: importa `edge`, e ninguem o importa fora do gerador de
OpenAPI. A documentacao dele mora ao lado, em `docs/`: `openapi.json` e a copia versionada do
contrato e ha teste que falha quando ela e o codigo discordam, e a colecao do Bruno cobre as
rotas uma a uma.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from api import auth, execution, render, seguranca, store
from api.schemas import (
    Aceito,
    Erro,
    LoginRequest,
    PrincipalResponse,
    RegistroRequest,
    ResultadoResponse,
    SessaoResponse,
    StatusResponse,
    UploadUrlRequest,
    UploadUrlResponse,
)
from edge.frames.provider import ClipUnavailable, probe
from edge.paths import pipeline_config, var_dir

FAMILIA = "video/"
TITULO = "xVision"

VERSAO_MAIOR = 1
VERSAO = f"{VERSAO_MAIOR}.0.0"
PREFIXO = f"/v{VERSAO_MAIOR}"

LOG_ENV = "XVISION_LOG_LEVEL"
MP4 = "video/mp4"

ORIGENS_ENV = "XVISION_CORS_ORIGINS"
ORIGENS_PADRAO = "http://localhost:5173,http://localhost:3000"
UPLOAD_MB_ENV = "XVISION_API_MAX_UPLOAD_MB"
UPLOAD_MB_PADRAO = 128
MB = 1024 * 1024

logger = logging.getLogger("xvision.api")

# Hash de uma senha que ninguem tem, para o login gastar o mesmo tempo quando o email nao existe.
SENHA_FALSA = auth.hash_de_senha(secrets.token_urlsafe(32))

# O ciclo de vida roda sem argumento nosso, entao a config escolhida no `create_app` fica aqui.
CONFIG_ATUAL: list[Path | None] = [None]

RECUSAS = {
    400: "O corpo da requisicao chegou vazio.",
    404: "Nao existe video com esse identificador, ou o endereco de envio e desconhecido.",
    413: "O arquivo passa do teto aceito.",
    401: "Credencial ausente, invalida, expirada ou revogada.",
    409: "Ja existe conta com esse email, ou o video ja saiu da fila.",
    415: "O tipo enviado nao e um dos que o pipeline abre.",
    429: "Chamadas demais vindas deste endereco. Retry-After diz quando voltar.",
    503: "A fila esta cheia. O cabecalho Retry-After diz quando voltar.",
}

TENTATIVAS_ENV = "XVISION_LOGIN_TENTATIVAS"
REGISTROS_ENV = "XVISION_REGISTRO_TENTATIVAS"
JANELA_ENV = "XVISION_LOGIN_JANELA_S"
TENTATIVAS_PADRAO = 20
REGISTROS_PADRAO = 100
JANELA_PADRAO = 300
CHAVES_MAXIMAS = 10_000

_erradas: dict[str, list[float]] = {}


def zerar_tentativas() -> None:
    """Solta os bloqueios de login em vigor, para quem operar nao precisar reiniciar o servico."""
    _erradas.clear()


def chaves_de_tentativa() -> tuple[str, ...]:
    """As chaves que a contagem guarda agora, para quem opera medir o que ela custa de memoria."""
    return tuple(_erradas)


def uploads_dir() -> Path:
    return var_dir() / "uploads"


def _teto(variavel: str, padrao: int) -> int:
    return _inteiro(variavel, padrao) * MB


def _inteiro(variavel: str, padrao: int) -> int:
    declarado = os.environ.get(variavel)
    return int(declarado) if declarado else padrao


def _origens() -> list[str]:
    return [item.strip() for item in os.environ.get(ORIGENS_ENV, ORIGENS_PADRAO).split(",") if item]


def health() -> dict[str, str]:
    """Diz se o servico subiu."""
    return {"status": "ok"}


def raiz() -> dict[str, Any]:
    """Lista as versoes que este servico responde.

    Existe para que uma chamada sem versao, que e o erro mais provavel de quem integra, encontre
    o caminho em vez de receber 404 seco.
    """
    return {
        "servico": TITULO,
        "versoes": [PREFIXO],
        "atual": PREFIXO,
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


def _rota_upload_url(
    pedido: UploadUrlRequest, principal: seguranca.Autenticado
) -> UploadUrlResponse:
    """Reserva um identificador e devolve o endereco para onde mandar o binario."""
    _conferir_tipo(pedido.contentType)
    video = store.criar(pedido.fileName, pedido.contentType, principal_id=principal.id)
    return UploadUrlResponse(id=video.id, uploadUrl=f"{PREFIXO}/videos/upload/{video.token}")


def _conferir_tipo(content_type: str) -> None:
    """A porta faz a triagem barata; quem decide de verdade e o decodificador.

    Lista fixa de container erra nas duas direcoes. Erra recusando: o decodificador desta imagem
    abre mov, avi, mkv e o mov em HEVC que o iPhone grava, e uma lista com `video/mp4` devolveria
    415 para todos eles. E erra aceitando, porque `contentType` e afirmacao do cliente: um `.txt`
    renomeado para `.mp4` chega aqui declarando `video/mp4` e passa.

    Entao aqui so cai o que nem vídeo diz ser, e a prova real e `_conferir_decodavel`.
    """
    if not content_type.startswith(FAMILIA):
        raise HTTPException(
            status_code=415,
            detail=f"{content_type!r} nao e video: envie algum {FAMILIA}*",
        )


def _conferir_decodavel(arquivo: Path, video_id: int) -> None:
    """Abre o arquivo antes de enfileirar, para a recusa chegar na resposta e nao meia hora depois.

    Custa milissegundos, porque le so o cabecalho. Sem isto, arquivo que nao e video sobe inteiro,
    entra na fila, e quem enviou descobre consultando o status, sem saber se a culpa foi do
    arquivo dele ou do servico.

    O registro ja existe quando isto roda, entao ele vai para `Falhou` junto: deixa-lo em
    `Pendente` faria o frontend consultar para sempre um trabalho que ninguem vai pegar.
    """
    try:
        probe(arquivo, clip_id="verificacao")
    except ClipUnavailable as erro:
        arquivo.unlink(missing_ok=True)
        motivo = "o arquivo enviado nao abriu como video"
        store.marcar(video_id, store.FALHOU, motivo=motivo)
        raise HTTPException(
            status_code=415,
            detail=(
                f"{motivo}. Verifique se ele esta completo e se o formato e um que o servico "
                f"decodifica ({_sem_caminho(str(erro), arquivo)})"
            ),
        ) from erro


def _sem_caminho(mensagem: str, caminho: Path) -> str:
    return mensagem.replace(str(caminho), "arquivo enviado").replace(caminho.name, "arquivo")


def _handler_envio(config: Path) -> Callable[[Request, str], Awaitable[Aceito]]:
    async def receber(request: Request, token: str) -> Aceito:
        """Recebe o binario do video e enfileira o processamento."""
        if store.ler_por_token(token) is None:
            raise HTTPException(status_code=404, detail="endereco de envio desconhecido")
        video = store.reivindicar(token)
        if video is None:
            raise HTTPException(
                status_code=409, detail="este endereco de envio ja foi usado: peca outro"
            )
        destino = uploads_dir()
        destino.mkdir(parents=True, exist_ok=True)
        arquivo = destino / f"{video.id}.mp4"
        try:
            await _gravar(request, arquivo)
        except BaseException as erro:
            _abandonar(video.id, arquivo, _motivo_de(erro))
            raise
        _conferir_decodavel(arquivo, video.id)
        _enfileirar(video.id, arquivo, config)
        return Aceito(status=202, id=video.id)

    return receber


def _motivo_de(erro: BaseException) -> str:
    if isinstance(erro, HTTPException):
        return str(erro.detail)
    return "o envio foi interrompido antes de terminar"


def _enfileirar(video_id: int, arquivo: Path, config: Path) -> None:
    """Aceitar trabalho que nao se consegue fazer e pior que recusa-lo.

    Sem teto, medido: 200 envios aceitos em 0,28 s e 186 arquivos simultaneos em `var/uploads`.
    Com 128 MB por arquivo isso e ate 23 GB de disco em transito, e quem chamou fica esperando um
    resultado que nunca vem. O 503 com `Retry-After` diz a verdade e diz quando voltar.
    """
    try:
        execution.submeter(execution.processar_video, video_id, arquivo, config)
    except execution.FilaCheia as erro:
        _abandonar(video_id, arquivo, "o servico estava sem capacidade e recusou o trabalho")
        raise HTTPException(
            status_code=503,
            detail=f"{erro}. Tente de novo em instantes, pedindo um endereco novo",
            headers={"Retry-After": "30"},
        ) from erro


def _abandonar(video_id: int, arquivo: Path, motivo: str) -> None:
    """Envio que nao completou deixa o registro em `Falhou`, e nao esperando para sempre.

    Cliente que fecha a aba no meio do PUT levanta `ClientDisconnect`, que nao e `HTTPException`:
    medido, o arquivo de 3 MB ficava no disco e o registro ficava em `Pendente` para sempre. Toda
    aba fechada deixava lixo permanente.

    O endereco de envio nao volta a valer. Reabri-lo traria de volta a corrida que a posse fecha,
    e pedir outro custa uma requisicao.
    """
    arquivo.unlink(missing_ok=True)
    with suppress(store.StoreError):
        store.marcar(video_id, store.FALHOU, motivo=motivo)


async def _gravar(request: Request, destino: Path) -> None:
    """Transmitir para o disco, e nao acumular em memoria, e o que segura o teto de 128 MB."""
    teto = _teto(UPLOAD_MB_ENV, UPLOAD_MB_PADRAO)
    total = 0
    with destino.open("wb") as saida:
        async for pedaco in request.stream():
            total += len(pedaco)
            if total > teto:
                raise HTTPException(
                    status_code=413, detail=f"o video passa do teto de {teto // MB} MB"
                )
            saida.write(pedaco)
    if total == 0:
        raise HTTPException(status_code=400, detail="o corpo da requisicao veio vazio")


def _rota_registro(pedido: RegistroRequest, request: Request) -> PrincipalResponse:
    """Cria a conta, e recusa email repetido sem dizer que ele existe pelo tempo de resposta.

    O teto e por origem, e nao por par (origem, email) como no login: aqui o email e novo a cada
    chamada, entao uma chave que o incluisse nunca acumularia duas tentativas. Esta e a rota nao
    autenticada mais cara do servico, porque roda `scrypt` e grava uma linha, e sem teto ela
    aceitava trabalho ate o pool de threads do Starlette encher.

    A contagem inclui o cadastro que deu certo, porque ele custa o mesmo `scrypt` do que falhou, e
    por isso o teto e o de `REGISTROS_ENV` e nao o do login. `REGISTROS_PADRAO` sai do preco medido
    do trabalho, 17 ms de CPU por derivacao nesta maquina: cem chamadas em 300 s sao 1,7 s de CPU
    por origem na janela, menos de 1% de um nucleo. O numero e alto porque atras de um NAT a origem
    e uma so para a turma inteira, e vinte cadastros em cinco minutos trancariam a demonstracao.
    """
    chave = f"registro|{_origem(request)}"
    _cobrar_tentativa(chave, REGISTROS_ENV, REGISTROS_PADRAO)
    _erradas.setdefault(chave, []).append(time.monotonic())
    try:
        criado = store.criar_principal(
            "usuario", pedido.email, senha_hash=auth.hash_de_senha(pedido.senha)
        )
    except store.NomeEmUso as erro:
        raise HTTPException(status_code=409, detail="este email ja tem conta") from erro
    return PrincipalResponse(id=criado.id, tipo="usuario", nome=criado.nome)


def _origem(request: Request) -> str:
    """`X-Forwarded-For` e escolhido por quem ataca, entao a origem e a do socket."""
    return request.client.host if request.client else "desconhecido"


def _chave(request: Request, email: str) -> str:
    """A contagem do login e por par (origem, email), para o vizinho de NAT nao trancar a conta."""
    return f"{_origem(request)}|{email.strip().lower()}"


def _varrer(agora: float, janela: int) -> None:
    """A contagem mora em memoria e a chave carrega o email, que quem chama escolhe.

    Sem teto de tamanho, uma rajada de emails distintos vindos de uma rota sem credencial deixa
    uma entrada por email, para sempre: podar os carimbos de uma chave nao devolve as outras. A
    varredura so roda quando o dicionario passa do tamanho declarado, entao o caso normal nao
    paga nada. `CHAVES_MAXIMAS` e o teto de memoria: cada entrada e uma chave de texto e uma lista
    curta de carimbos, entao dez mil delas cabem em poucos megabytes, que e o maximo que a contagem
    pode custar num dispositivo de borda.

    Quando ainda sobra gente dentro da janela, sai primeiro quem tem menos carimbos, e nao quem
    entrou antes. A ordem importa porque ela decide quem paga a conta da inundacao: despejando
    pela ordem do dicionario, que e a de primeira insercao, a chave que vinha acumulando
    tentativas ha minutos era a primeira a cair, e quem inundava com emails descartaveis desfazia
    a propria tranca de graca. Contando carimbos, cai o lixo de uma tentativa so e fica quem esta
    perto do teto. Empurrar a chave de uma vitima para fora continua sendo possivel, e passa a
    custar tantas requisicoes quanto o teto, cada uma com o `scrypt` inteiro do lado de quem
    tenta.
    """
    if len(_erradas) <= CHAVES_MAXIMAS:
        return
    for chave, carimbos in list(_erradas.items()):
        if not carimbos or agora - carimbos[-1] >= janela:
            del _erradas[chave]
    if len(_erradas) <= CHAVES_MAXIMAS:
        return
    descartaveis = sorted(_erradas, key=lambda c: (len(_erradas[c]), _erradas[c][-1]))
    for chave in descartaveis[: len(_erradas) - CHAVES_MAXIMAS]:
        del _erradas[chave]


def _cobrar_tentativa(
    chave: str, variavel: str = TENTATIVAS_ENV, padrao: int = TENTATIVAS_PADRAO
) -> None:
    """Chave sem carimbo recente sai do dicionario, em vez de ficar com uma lista vazia."""
    teto = _inteiro(variavel, padrao)
    janela = _inteiro(JANELA_ENV, JANELA_PADRAO)
    agora = time.monotonic()
    _varrer(agora, janela)
    recentes = [t for t in _erradas.get(chave, []) if agora - t < janela]
    if recentes:
        _erradas[chave] = recentes
    else:
        _erradas.pop(chave, None)
    if len(recentes) >= teto:
        raise HTTPException(
            status_code=429,
            detail="chamadas demais vindas deste endereco, espere um pouco",
            headers={"Retry-After": str(int(janela - (agora - recentes[0])) + 1)},
        )


def _rota_login(pedido: LoginRequest, request: Request) -> SessaoResponse:
    """Email desconhecido e senha errada respondem igual, e demoram igual.

    Respondendo diferente, a rota vira lista de emails cadastrados. Por isso a senha e conferida
    mesmo quando a conta nao existe: sem isso, o caminho do email desconhecido volta antes de
    rodar o `scrypt` e o tempo de resposta conta a diferenca sozinho.

    A contagem de erradas vive em memoria do processo. Com mais de uma replica cada uma conta a
    sua, e o teto efetivo multiplica; resolver isso pede estado compartilhado, que so se paga
    quando existir a segunda replica.
    """
    chave = _chave(request, pedido.email)
    _cobrar_tentativa(chave)
    linha = store.principal_por_nome(pedido.email)
    guardado = linha.senha_hash if linha is not None and linha.senha_hash else SENHA_FALSA
    confere = auth.conferir_senha(pedido.senha, guardado)
    if linha is None or linha.revogado_em is not None or not confere:
        _erradas.setdefault(chave, []).append(time.monotonic())
        raise HTTPException(
            status_code=401,
            detail="email ou senha invalidos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _erradas.pop(chave, None)
    return SessaoResponse(access_token=auth.assinar(linha.id), expira_em=auth.validade())


def _rota_eu(principal: seguranca.Autenticado) -> PrincipalResponse:
    """Diz quem o servico acha que voce e, que e o que um cliente precisa para depurar sessao."""
    return PrincipalResponse(id=principal.id, tipo=principal.tipo, nome=principal.nome)


def _rota_status(video_id: int, principal: seguranca.Autenticado) -> StatusResponse:
    """Diz em que ponto o video esta, sem devolver o resultado."""
    video = _buscar(video_id)
    seguranca.conferir_posse(video, principal)
    return StatusResponse(
        status=200,
        video_status=video.estado,  # pyright: ignore[reportArgumentType]
        motivo=video.motivo,
    )


def _rota_resultado(video_id: int, principal: seguranca.Autenticado) -> ResultadoResponse:
    """Devolve o resultado por etapa quando terminou, e o motivo quando falhou.

    Recusar `Falhou` com 409 dizendo "ainda nao terminou" era falso duas vezes: o video terminou,
    e a mensagem mandava o frontend continuar consultando um estado que nunca muda. Pior, o campo
    `motivo` so e preenchido em `Falhou`, entao ele era inalcancavel por qualquer rota e toda a
    higiene de nao vazar caminho protegia um texto que ninguem lia.
    """
    video = _buscar(video_id)
    seguranca.conferir_posse(video, principal)
    if video.estado not in store.TERMINAIS:
        raise HTTPException(
            status_code=409,
            detail=f"o video ainda nao terminou: esta em {video.estado!r}",
        )
    return ResultadoResponse(
        status=200,
        video_status=video.estado,  # pyright: ignore[reportArgumentType]
        resultados=video.resultados or [],  # pyright: ignore[reportArgumentType]
        motivo=video.motivo,
        render_url=_endereco_do_render(video_id),
    )


def _endereco_do_render(video_id: int) -> str | None:
    if not render.render_path(video_id).is_file():
        return None
    return f"{PREFIXO}/videos/{video_id}/render"


def _rota_render(video_id: int, principal: seguranca.Autenticado) -> FileResponse:
    """Devolve o video anotado da analise, e 404 quando ele nao foi produzido."""
    video = _buscar(video_id)
    seguranca.conferir_posse(video, principal)
    caminho = render.render_path(video_id)
    if not caminho.is_file():
        raise HTTPException(status_code=404, detail=f"o video {video_id} nao tem render gravado")
    return FileResponse(caminho, media_type=MP4, filename=f"render-{video_id}.mp4")


def _buscar(video_id: int) -> store.Video:
    video = store.ler(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"nao existe video com id {video_id}")
    return video


async def _recusar_corpo_grande(
    request: Request, seguir: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Recusa pelo cabecalho, antes de o corpo existir em memoria.

    Conferir o tamanho depois de o pydantic materializar a string protegia a metade menor do
    problema. Medido no servico anterior: um POST de 300 MB devolvia 413 e levava o processo de
    65 MB para 1176 MB de RSS, e quatro envios simultaneos de 150 MB chegaram a 1608 MB. Numa
    borda de 2 GB isso e OOM com dois clientes, e a resposta correta nao ajudava em nada.

    Sem `Content-Length`, o que acontece em envio fragmentado, nao ha o que conferir aqui e quem
    segura e o teto por pedaco do `_gravar`.
    """
    declarado = request.headers.get("content-length")
    if declarado is not None and declarado.isdigit():
        teto = _teto(UPLOAD_MB_ENV, UPLOAD_MB_PADRAO)
        if int(declarado) > teto:
            return JSONResponse(
                status_code=413,
                content={"detail": f"o corpo passa do teto de {teto // MB} MB"},
            )
    return await seguir(request)


async def _validacao(_: Request, erro: Exception) -> JSONResponse:
    """O pydantic devolve o corpo recebido dentro do erro, e nada garante que ele seja pequeno.

    Medido no servico anterior, quando existia a rota que aceitava o video em base64 dentro do
    JSON: um envio de 10 MB sem `contentType` voltava com 10 MB de resposta, o corpo inteiro
    ecoado no campo `input`. Aquela rota saiu, e o handler fica: ele custa nada, vale para todo
    corpo que o pydantic recusar, e a proxima rota com corpo grande nao vai lembrar de pedi-lo.
    """
    if not isinstance(erro, RequestValidationError):  # pragma: no cover - o FastAPI so
        raise erro  # pragma: no cover - registra este handler para esta excecao
    recebidos = cast("list[dict[str, Any]]", erro.errors())
    return JSONResponse(status_code=422, content={"detail": [_limpar(i) for i in recebidos]})


def _limpar(item: dict[str, Any]) -> dict[str, Any]:
    """Tira o corpo recebido, e converte o que nao vira JSON.

    Tirar so `input` nao bastava: quando a recusa vem de um validador proprio, o pydantic poe a
    excecao original em `ctx`, e objeto de excecao nao serializa. Medido: um email vazio, que o
    validador de normalizacao recusa, devolvia 500 com `Object of type ValueError is not JSON
    serializable` em vez do 422 que o cliente espera. O caminho de erro virando erro e o pior
    lugar para ter um erro.
    """
    limpo: dict[str, Any] = {}
    for chave, valor in item.items():
        if chave == "input":
            continue
        limpo[chave] = _serializavel(valor)
    return limpo


def _serializavel(valor: Any) -> Any:
    if isinstance(valor, str | int | float | bool | type(None)):
        return valor
    if isinstance(valor, dict):
        return {str(k): _serializavel(v) for k, v in cast("dict[Any, Any]", valor).items()}
    if isinstance(valor, list | tuple):
        return [_serializavel(v) for v in cast("list[Any]", valor)]
    return str(valor)


def _config_atual() -> Path:
    escolhido = CONFIG_ATUAL[0]
    return escolhido if escolhido is not None else pipeline_config()


def _configurar_log() -> None:
    """Sem isto o unico `logging` do projeto caia no logger raiz sem handler.

    O motivo de uma falha so existia como texto numa linha do SQLite, e `docker logs` mostrava
    apenas as linhas de acesso do uvicorn: quem opera nao tinha como saber se a culpa era do
    arquivo enviado ou do servico. O nivel sai de `XVISION_LOG_LEVEL`, e nada e configurado
    quando a aplicacao ja tem handler, para nao brigar com quem embute o servico.
    """
    raiz = logging.getLogger()
    if raiz.handlers:
        return
    logging.basicConfig(
        level=os.environ.get(LOG_ENV, "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _conferir_ambiente(config: Path) -> None:
    """Recusar a subir e melhor que subir verde e falhar em toda requisicao.

    Vale tambem para o segredo de assinatura: sem ele o servico nao assina sessao nenhuma, e
    subir para descobrir isso na primeira tentativa de login e descobrir tarde demais.

    Medido: com `XVISION_API_MAX_UPLOAD_MB=128MB` (o sufixo a mais) o servico subia, `/health`
    respondia ok, e o primeiro envio morria com `ValueError` virando 500. Um erro de digitacao no
    deploy passava por toda sonda de prontidao. O mesmo vale para o `config/pipeline.toml`
    ausente: a configuracao so era lida por requisicao, dentro do worker, onde a excecao virava
    "erro interno" generico. Todo video falhava e o servico se dizia saudavel.
    """
    auth.segredo()
    if not config.is_file():
        raise RuntimeError(
            f"{config.resolve()} nao existe, e o servico nao processa video sem ela. "
            "Suba a partir da raiz do repositorio, ou monte a configuracao no lugar esperado"
        )
    for variavel, padrao in ((UPLOAD_MB_ENV, UPLOAD_MB_PADRAO),):
        try:
            _teto(variavel, padrao)
        except ValueError as erro:
            declarado = os.environ.get(variavel)
            raise RuntimeError(f"{variavel}={declarado!r} nao e um numero de megabytes") from erro


def _recuperar_orfaos(config: Path) -> None:
    """Video preso em `Em processamento` por uma queda volta para a fila, ou falha com motivo.

    O arquivo enviado e a unica coisa que sobrevive a queda: com ele, o trabalho e refeito de
    verdade; sem ele, nao ha o que reprocessar e fingir `Pendente` deixaria o frontend
    consultando para sempre.

    Fila cheia durante a recuperacao para a rodada e nao derruba a subida. Sem o `try`, um numero
    de orfaos maior que o teto fazia o servico nao subir, e como os orfaos continuam la, ele nao
    subiria mais nunca: o que era recuperacao virava laco permanente.
    """
    for video in store.recuperaveis():
        arquivo = uploads_dir() / f"{video.id}.mp4"
        if arquivo.is_file():
            try:
                execution.submeter(execution.processar_video, video.id, arquivo, config)
            except execution.FilaCheia:
                logger.warning(
                    "a fila encheu durante a recuperacao: %d video(s) ficam para a proxima subida",
                    len(store.recuperaveis()),
                )
                return
            continue
        with suppress(store.StoreError):
            store.marcar(
                video.id,
                store.FALHOU,
                motivo="o servico caiu durante o processamento e o arquivo enviado nao sobrou",
            )


@asynccontextmanager
async def _ciclo_de_vida(_: FastAPI) -> AsyncGenerator[None, None]:
    """Construir a aplicacao nao pode tocar o disco, subir o servico pode.

    Tocar o banco no construtor faria qualquer teste que instancia o app criar o banco de
    verdade, e faria uma ferramenta que so le o OpenAPI escrever em `var/`.

    A fila e reaberta na subida porque o desligamento a fecha de proposito: sem isto, subir o
    servico de novo no mesmo processo, que e o que um teste faz e o que quem embute a aplicacao
    tambem faz, encontraria a fila fechada e recusaria todo trabalho.
    """
    _configurar_log()
    execution.religar()
    _conferir_ambiente(_config_atual())
    if seguranca.semear_token_de_servico():
        logger.info("token de servico semeado a partir de %s", seguranca.TOKEN_SEMENTE_ENV)
    _recuperar_orfaos(_config_atual())
    try:
        yield
    finally:
        execution.desligar()


def create_app(config: Path | None = None) -> FastAPI:
    """`None` resolve pelo `edge.paths`, que le `XVISION_CONFIG_DIR` na chamada.

    Constante de modulo congelava o caminho no import, entao a variavel era aceita e ignorada.
    """
    config = config if config is not None else pipeline_config()
    CONFIG_ATUAL[0] = config
    app = FastAPI(
        title=TITULO,
        version=VERSAO,
        summary="Extrai caracteristicas de finalizacao a partir de um clipe de video.",
        lifespan=_ciclo_de_vida,
    )
    app.middleware("http")(_recusar_corpo_grande)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origens(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Retry-After"],
    )
    app.add_exception_handler(RequestValidationError, _validacao)
    _registrar(app, config)
    return app


@dataclass(frozen=True)
class Rota:
    """Uma linha da tabela de rotas, nomeada.

    Como tupla de sete posicoes com `Any` no meio, o pyright nao conseguia casar o handler com o
    `response_model` e a tabela passava no portao de tipos sem verificar nada. Nomear os campos
    devolve a verificacao e torna impossivel trocar duas posicoes por engano.
    """

    caminho: str
    funcao: Callable[..., Any]
    metodo: str
    resumo: str
    modelo: type[BaseModel] | None
    recusas: dict[int | str, dict[str, Any]]
    sucesso: int
    resposta: type[Response] = JSONResponse
    """Sem ela o FastAPI descreve todo 200 como `application/json`."""


def _erros(*codigos: int) -> dict[int | str, dict[str, Any]]:
    """Declarar a recusa e o que impede o cliente gerado de tipar errado o caminho de erro.

    O FastAPI so documenta o que a rota declara, entao sem isto o contrato versionado descreve
    apenas o sucesso e o 422 do pydantic, e quem gera cliente pelo arquivo trata 415 como 200.
    """
    return {codigo: {"model": Erro, "description": RECUSAS[codigo]} for codigo in codigos}


def _registrar(app: FastAPI, config: Path) -> None:
    """Rota de negocio entra sob `/v1`; rota de operacao fica fora dele.

    O codigo de sucesso e declarado por rota. Sem `status_code`, o FastAPI responde 200 em tudo,
    e as rotas de envio devolviam HTTP 200 com `{"status": 202}` no corpo: o contrato versionado
    dizia 200, a documentacao dizia 202, e um cliente que trate 202 como enfileirado e 200 como
    pronto quebraria na primeira chamada.

    `/health` e `/` pertencem a implantacao e nao ao contrato: um orquestrador que consulta saude
    nao deveria precisar saber em que versao o contrato esta, e trocar para `/v2` nao pode obrigar
    a reconfigurar o healthcheck de todo mundo.

    O prefixo mora num `APIRouter`, e nao repetido em cada caminho, porque publicar a v2 passa a
    ser criar um router novo e manter o antigo respondendo, em vez de editar seis strings e
    descobrir na producao a que ficou para tras.
    """
    app.add_api_route(
        "/", raiz, methods=["GET"], summary="Lista as versoes que o servico responde."
    )
    app.add_api_route("/health", health, methods=["GET"], summary="Diz se o servico subiu.")

    v1 = APIRouter(prefix=PREFIXO, tags=[PREFIXO])
    rotas = [
        Rota(
            "/auth/register",
            _rota_registro,
            "POST",
            "Cria uma conta de usuario.",
            PrincipalResponse,
            _erros(409, 429),
            201,
        ),
        Rota(
            "/auth/login",
            _rota_login,
            "POST",
            "Troca email e senha por uma sessao.",
            SessaoResponse,
            _erros(401, 429),
            200,
        ),
        Rota(
            "/auth/me",
            _rota_eu,
            "GET",
            "Diz quem o servico acha que voce e.",
            PrincipalResponse,
            _erros(401),
            200,
        ),
        Rota(
            "/videos/upload-url",
            _rota_upload_url,
            "POST",
            "Reserva um identificador e devolve o endereco de envio.",
            UploadUrlResponse,
            _erros(401, 415),
            200,
        ),
        Rota(
            "/videos/upload/{token}",
            _handler_envio(config),
            "PUT",
            "Recebe o binario do video e enfileira o processamento.",
            Aceito,
            _erros(400, 404, 409, 413, 415, 503),
            202,
        ),
        Rota(
            "/videos/status/{video_id}",
            _rota_status,
            "GET",
            "Diz em que ponto o video esta.",
            StatusResponse,
            _erros(401, 404),
            200,
        ),
        Rota(
            "/videos/{video_id}",
            _rota_resultado,
            "GET",
            "Devolve o resultado por etapa, quando o video ja terminou.",
            ResultadoResponse,
            _erros(401, 404, 409),
            200,
        ),
        Rota(
            "/videos/{video_id}/render",
            _rota_render,
            "GET",
            "Devolve o video anotado da analise.",
            None,
            {**_erros(401, 404), 200: {"content": {MP4: {}}, "description": "O video anotado."}},
            200,
            FileResponse,
        ),
    ]
    for rota in rotas:
        v1.add_api_route(
            rota.caminho,
            rota.funcao,
            methods=[rota.metodo],
            summary=rota.resumo,
            response_model=rota.modelo,
            response_class=rota.resposta,
            responses=rota.recusas,
            status_code=rota.sucesso,
        )
    app.include_router(v1)
