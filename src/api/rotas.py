"""Tabela versionada de rotas e recusas publicadas pelo servico."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from api.schemas import (
    Aceito,
    Erro,
    EstatisticasContaResponse,
    ListaVideosResponse,
    MensagemResponse,
    PrincipalResponse,
    ResultadoResponse,
    SessaoAtualizadaResponse,
    SessaoResponse,
    StatusResponse,
    UploadUrlResponse,
)

RECUSAS = {
    400: "O corpo da requisicao chegou vazio.",
    401: "Credencial ausente, invalida, expirada ou revogada.",
    403: "A credencial nao representa uma conta de usuario.",
    404: "Nao existe video com esse identificador, ou o endereco de envio e desconhecido.",
    409: "Ja existe conta com esse email, ou o video ja saiu da fila.",
    413: "O arquivo passa do teto aceito.",
    415: "O tipo enviado nao e um dos que o pipeline abre.",
    429: "Chamadas demais vindas deste endereco. Retry-After diz quando voltar.",
    503: "A fila esta cheia. O cabecalho Retry-After diz quando voltar.",
}


@dataclass(frozen=True)
class Handlers:
    raiz: Callable[..., Any]
    health: Callable[..., Any]
    bruno: Callable[..., Any]
    registro: Callable[..., Any]
    login: Callable[..., Any]
    esqueci_senha: Callable[..., Any]
    redefinir_senha: Callable[..., Any]
    eu: Callable[..., Any]
    atualizar_email: Callable[..., Any]
    estatisticas: Callable[..., Any]
    alterar_senha: Callable[..., Any]
    videos: Callable[..., Any]
    upload_url: Callable[..., Any]
    upload: Callable[..., Any]
    status: Callable[..., Any]
    resultado: Callable[..., Any]
    render: Callable[..., Any]


@dataclass(frozen=True)
class Rota:
    caminho: str
    funcao: Callable[..., Any]
    metodo: str
    resumo: str
    modelo: type[BaseModel] | None
    recusas: dict[int | str, dict[str, Any]]
    sucesso: int
    resposta: type[Response] = JSONResponse


def registrar(app: FastAPI, prefixo: str, handlers: Handlers) -> None:
    app.add_api_route("/", handlers.raiz, methods=["GET"], summary="Lista as versoes da API.")
    app.add_api_route("/health", handlers.health, methods=["GET"], summary="Sonda de saude.")
    app.add_api_route("/bruno.zip", handlers.bruno, methods=["GET"], summary="Colecao Bruno.")
    router = APIRouter(prefix=prefixo, tags=[prefixo])
    for rota in _definicoes(handlers):
        router.add_api_route(
            rota.caminho,
            rota.funcao,
            methods=[rota.metodo],
            summary=rota.resumo,
            response_model=rota.modelo,
            response_class=rota.resposta,
            responses=rota.recusas,
            status_code=rota.sucesso,
        )
    app.include_router(router)


def _definicoes(handlers: Handlers) -> list[Rota]:
    return [
        *_acesso(handlers),
        *_recuperacao(handlers),
        *_conta(handlers),
        *_envio(handlers),
        *_resultados(handlers),
    ]


def _acesso(h: Handlers) -> list[Rota]:
    return [
        _rota(
            "/auth/register",
            h.registro,
            "POST",
            "Cria uma conta.",
            PrincipalResponse,
            201,
            409,
            429,
        ),
        _rota("/auth/login", h.login, "POST", "Inicia uma sessao.", SessaoResponse, 200, 401, 429),
        _rota("/auth/me", h.eu, "GET", "Identifica a credencial.", PrincipalResponse, 200, 401),
    ]


def _recuperacao(h: Handlers) -> list[Rota]:
    return [
        _rota(
            "/auth/password/forgot",
            h.esqueci_senha,
            "POST",
            "Envia redefinicao.",
            MensagemResponse,
            202,
            429,
            503,
        ),
        _rota(
            "/auth/password/reset",
            h.redefinir_senha,
            "POST",
            "Redefine a senha.",
            MensagemResponse,
            200,
            400,
            429,
        ),
    ]


def _conta(h: Handlers) -> list[Rota]:
    return [
        _rota(
            "/auth/me",
            h.atualizar_email,
            "PATCH",
            "Atualiza o email.",
            PrincipalResponse,
            200,
            400,
            401,
            403,
            409,
            429,
        ),
        _rota(
            "/auth/me/statistics",
            h.estatisticas,
            "GET",
            "Agrega analises.",
            EstatisticasContaResponse,
            200,
            401,
        ),
        _rota(
            "/auth/password/change",
            h.alterar_senha,
            "POST",
            "Troca a senha.",
            SessaoAtualizadaResponse,
            200,
            400,
            401,
            403,
            429,
        ),
    ]


def _envio(h: Handlers) -> list[Rota]:
    return [
        _rota("/videos", h.videos, "GET", "Lista os videos.", ListaVideosResponse, 200, 401),
        _rota(
            "/videos/upload-url",
            h.upload_url,
            "POST",
            "Reserva um envio.",
            UploadUrlResponse,
            200,
            401,
            415,
        ),
        _rota(
            "/videos/upload/{token}",
            h.upload,
            "PUT",
            "Recebe o video.",
            Aceito,
            202,
            400,
            404,
            409,
            413,
            415,
            503,
        ),
    ]


def _resultados(h: Handlers) -> list[Rota]:
    render_responses: dict[int | str, dict[str, Any]] = {
        **_erros(401, 404),
        200: {"content": {"video/mp4": {}}, "description": "O video anotado."},
    }
    return [
        _rota(
            "/videos/status/{video_id}",
            h.status,
            "GET",
            "Consulta o andamento.",
            StatusResponse,
            200,
            401,
            404,
        ),
        _rota(
            "/videos/{video_id}",
            h.resultado,
            "GET",
            "Devolve o resultado.",
            ResultadoResponse,
            200,
            401,
            404,
            409,
        ),
        Rota(
            "/videos/{video_id}/render",
            h.render,
            "GET",
            "Devolve o render.",
            None,
            render_responses,
            200,
            FileResponse,
        ),
    ]


def _rota(
    caminho: str,
    funcao: Callable[..., Any],
    metodo: str,
    resumo: str,
    modelo: type[BaseModel],
    sucesso: int,
    *recusas: int,
) -> Rota:
    return Rota(caminho, funcao, metodo, resumo, modelo, _erros(*recusas), sucesso)


def _erros(*codigos: int) -> dict[int | str, dict[str, Any]]:
    return {codigo: {"model": Erro, "description": RECUSAS[codigo]} for codigo in codigos}
