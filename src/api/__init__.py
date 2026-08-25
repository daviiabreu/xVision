"""O servico HTTP que serve o frontend, rodando o pipeline da borda de forma assincrona.

Na arquitetura de destino a borda processa perto de onde o video chega e um servico central
recebe o evento pronto. Este e a ponte da sprint 2, em que as duas pontas rodam na mesma maquina.

O envio tem dois caminhos porque o time do frontend especificou dois. O recomendado e pedir uma
URL em `POST /videos/upload-url` e mandar o binario cru nela por `PUT`, que e o que o codigo em
`src/interface` ja faz e o que transmite o arquivo direto para o disco. O outro aceita o video em
base64 dentro do JSON, como o contrato escrito pede, e por isso tem teto proprio: base64 infla
33% e o corpo inteiro precisa existir em memoria antes de virar bytes.

A URL de envio aponta para o proprio servico e nao para um armazenamento de objetos. O formato e
o mesmo que um S3 assinado, entao trocar por armazenamento de verdade no dia da implantacao muda
apenas quem escreve o endereco, e nao o frontend.

Ele e area de composicao como o `cli`: importa `edge`, e ninguem o importa fora do gerador de
OpenAPI. A documentacao dele mora ao lado, em `docs/`: `openapi.json` e a copia versionada do
contrato e ha teste que falha quando ela e o codigo discordam, e a colecao do Bruno cobre as
rotas uma a uma.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import secrets
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api import auth, execution, seguranca, store
from api.schemas import (
    Aceito,
    Erro,
    LoginRequest,
    PrincipalResponse,
    ProcessamentoRequest,
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
ORIGENS_ENV = "XVISION_CORS_ORIGINS"
ORIGENS_PADRAO = "http://localhost:5173,http://localhost:3000"
UPLOAD_MB_ENV = "XVISION_API_MAX_UPLOAD_MB"
JSON_MB_ENV = "XVISION_API_MAX_JSON_MB"
UPLOAD_MB_PADRAO = 128
JSON_MB_PADRAO = 32
MB = 1024 * 1024

logger = logging.getLogger("xvision.api")

# Hash de uma senha que ninguem tem, para o login gastar o mesmo tempo quando o email nao existe.
SENHA_FALSA = auth.hash_de_senha(secrets.token_urlsafe(32))

# O ciclo de vida roda sem argumento nosso, entao a config escolhida no `create_app` fica aqui.
CONFIG_ATUAL: list[Path | None] = [None]

RECUSAS = {
    400: "O corpo chegou vazio, ou o campo file nao e base64 valido.",
    404: "Nao existe video com esse identificador, ou o endereco de envio e desconhecido.",
    413: "O arquivo passa do teto aceito.",
    401: "Credencial ausente, invalida, expirada ou revogada.",
    409: "Ja existe conta com esse email, ou o video ja saiu da fila.",
    415: "O tipo enviado nao e um dos que o pipeline abre.",
    503: "A fila esta cheia. O cabecalho Retry-After diz quando voltar.",
}


def uploads_dir() -> Path:
    return var_dir() / "uploads"


def _teto(variavel: str, padrao: int) -> int:
    declarado = os.environ.get(variavel)
    return (int(declarado) if declarado else padrao) * MB


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
    request: Request, pedido: UploadUrlRequest, principal: seguranca.Autenticado
) -> UploadUrlResponse:
    """Reserva um identificador e devolve o endereco para onde mandar o binario."""
    _conferir_tipo(pedido.contentType)
    video = store.criar(pedido.fileName, pedido.contentType, principal_id=principal.id)
    base = str(request.base_url).rstrip("/")
    return UploadUrlResponse(id=video.id, uploadUrl=f"{base}{PREFIXO}/videos/upload/{video.token}")


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


def _handler_base64(config: Path) -> Callable[..., Aceito]:
    def processar(pedido: ProcessamentoRequest, principal: seguranca.Autenticado) -> Aceito:
        """Aceita o video em base64 e enfileira o processamento."""
        _conferir_tipo(pedido.contentType)
        teto = _teto(JSON_MB_ENV, JSON_MB_PADRAO)
        if len(pedido.file) > teto:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"o campo file passa do teto de {teto // MB} MB em base64: "
                    "use POST /videos/upload-url para arquivos maiores"
                ),
            )
        try:
            bytes_do_video = base64.b64decode(pedido.file, validate=True)
        except (binascii.Error, ValueError) as erro:
            raise HTTPException(
                status_code=400, detail=f"o campo file nao e base64 valido: {erro}"
            ) from erro
        if not bytes_do_video:
            raise HTTPException(status_code=400, detail="o campo file decodificou para nada")
        video = store.criar(
            pedido.fileName,
            pedido.contentType,
            principal_id=principal.id,
            reivindicado=True,
        )
        destino = uploads_dir()
        destino.mkdir(parents=True, exist_ok=True)
        arquivo = destino / f"{video.id}.mp4"
        arquivo.write_bytes(bytes_do_video)
        _conferir_decodavel(arquivo, video.id)
        _enfileirar(video.id, arquivo, config)
        return Aceito(status=202, id=video.id)

    return processar


def _rota_registro(pedido: RegistroRequest) -> PrincipalResponse:
    """Cria a conta, e recusa email repetido sem dizer que ele existe pelo tempo de resposta."""
    try:
        criado = store.criar_principal(
            "usuario", pedido.email, senha_hash=auth.hash_de_senha(pedido.senha)
        )
    except store.NomeEmUso as erro:
        raise HTTPException(status_code=409, detail="este email ja tem conta") from erro
    return PrincipalResponse(id=criado.id, tipo="usuario", nome=criado.nome)


def _rota_login(pedido: LoginRequest) -> SessaoResponse:
    """Email desconhecido e senha errada respondem igual, e demoram igual.

    Respondendo diferente, a rota vira lista de emails cadastrados. Por isso a senha e conferida
    mesmo quando a conta nao existe: sem isso, o caminho do email desconhecido volta antes de
    rodar o `scrypt` e o tempo de resposta conta a diferenca sozinho.
    """
    linha = store.principal_por_nome(pedido.email)
    guardado = linha.senha_hash if linha is not None and linha.senha_hash else SENHA_FALSA
    confere = auth.conferir_senha(pedido.senha, guardado)
    if linha is None or linha.revogado_em is not None or not confere:
        raise HTTPException(
            status_code=401,
            detail="email ou senha invalidos",
            headers={"WWW-Authenticate": "Bearer"},
        )
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
    )


def _buscar(video_id: int) -> store.Video:
    video = store.ler(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"nao existe video com id {video_id}")
    return video


def _teto_da_rota(caminho: str) -> tuple[int, str]:
    """O teto e a orientacao de saida, que e o que evita o cliente so repetir o mesmo envio."""
    if caminho.endswith("/videos/processamento"):
        teto = _teto(JSON_MB_ENV, JSON_MB_PADRAO)
        return teto, " em base64: use POST /videos/upload-url para arquivos maiores"
    return _teto(UPLOAD_MB_ENV, UPLOAD_MB_PADRAO), ""


async def _recusar_corpo_grande(
    request: Request, seguir: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Recusa pelo cabecalho, antes de o corpo existir em memoria.

    Conferir o tamanho depois de o pydantic materializar a string protegia a metade menor do
    problema. Medido no servico anterior: um POST de 300 MB devolvia 413 e levava o processo de
    65 MB para 1176 MB de RSS, e quatro envios simultaneos de 150 MB chegaram a 1608 MB. Numa
    borda de 2 GB isso e OOM com dois clientes, e a resposta correta nao ajudava em nada.

    Sem `Content-Length`, o que acontece em envio fragmentado, nao ha o que conferir aqui e quem
    segura e o teto por pedaco do `_gravar`. O caminho base64 nao tem essa protecao, e por isso
    ele e o alternativo e nao o recomendado.
    """
    declarado = request.headers.get("content-length")
    if declarado is not None and declarado.isdigit():
        teto, saida = _teto_da_rota(request.url.path)
        if int(declarado) > teto:
            return JSONResponse(
                status_code=413,
                content={"detail": f"o corpo passa do teto de {teto // MB} MB{saida}"},
            )
    return await seguir(request)


async def _validacao(_: Request, erro: Exception) -> JSONResponse:
    """O pydantic devolve o corpo recebido dentro do erro, e o corpo aqui e um video inteiro.

    Medido no servico anterior: um envio de 10 MB sem `contentType` voltava com 10 MB de resposta,
    o base64 inteiro ecoado no campo `input`. Com 500 MB de envio seriam 500 MB de volta, para o
    cliente e para qualquer proxy que registre corpo.
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
    for variavel, padrao in ((UPLOAD_MB_ENV, UPLOAD_MB_PADRAO), (JSON_MB_ENV, JSON_MB_PADRAO)):
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origens(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )
    app.middleware("http")(_recusar_corpo_grande)
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
            _erros(409),
            201,
        ),
        Rota(
            "/auth/login",
            _rota_login,
            "POST",
            "Troca email e senha por uma sessao.",
            SessaoResponse,
            _erros(401),
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
            "/videos/processamento",
            _handler_base64(config),
            "POST",
            "Aceita o video em base64 e enfileira o processamento.",
            Aceito,
            _erros(400, 401, 413, 415, 503),
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
    ]
    for rota in rotas:
        v1.add_api_route(
            rota.caminho,
            rota.funcao,
            methods=[rota.metodo],
            summary=rota.resumo,
            response_model=rota.modelo,
            responses=rota.recusas,
            status_code=rota.sucesso,
        )
    app.include_router(v1)
