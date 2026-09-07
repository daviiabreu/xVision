"""Limites de tentativa para as rotas abertas e operacoes sensiveis."""

from __future__ import annotations

import os
import time
from ipaddress import ip_address

from fastapi import HTTPException, Request

TENTATIVAS_ENV = "XVISION_LOGIN_TENTATIVAS"
REGISTROS_ENV = "XVISION_REGISTRO_TENTATIVAS"
REDEFINICOES_ENV = "XVISION_REDEFINICAO_TENTATIVAS"
JANELA_ENV = "XVISION_LOGIN_JANELA_S"
TRUST_PROXY_ENV = "XVISION_TRUST_PROXY"

TENTATIVAS_PADRAO = 20
REGISTROS_PADRAO = 100
REDEFINICOES_PADRAO = 20
JANELA_PADRAO = 300
CHAVES_MAXIMAS = 10_000

_tentativas: dict[str, list[float]] = {}


def origem(request: Request) -> str:
    """Aceita o IP reescrito apenas quando a implantacao declara um proxy confiavel."""
    encaminhada = request.headers.get("x-forwarded-for", "").strip()
    if os.environ.get(TRUST_PROXY_ENV) == "1" and encaminhada:
        try:
            return str(ip_address(encaminhada))
        except ValueError:
            pass
    return request.client.host if request.client else "desconhecido"


def chave_login(request: Request, email: str) -> str:
    return f"{origem(request)}|{email.strip().lower()}"


def cobrar(
    chave: str,
    variavel: str = TENTATIVAS_ENV,
    padrao: int = TENTATIVAS_PADRAO,
) -> None:
    teto = _inteiro(variavel, padrao)
    janela = _inteiro(JANELA_ENV, JANELA_PADRAO)
    agora = time.monotonic()
    _varrer(agora, janela)
    recentes = [instante for instante in _tentativas.get(chave, []) if agora - instante < janela]
    if recentes:
        _tentativas[chave] = recentes
    else:
        _tentativas.pop(chave, None)
    if len(recentes) >= teto:
        raise HTTPException(
            status_code=429,
            detail="chamadas demais vindas deste endereco, espere um pouco",
            headers={"Retry-After": str(int(janela - (agora - recentes[0])) + 1)},
        )


def registrar(chave: str) -> None:
    _tentativas.setdefault(chave, []).append(time.monotonic())


def liberar(chave: str) -> None:
    _tentativas.pop(chave, None)


def zerar() -> None:
    _tentativas.clear()


def chaves() -> tuple[str, ...]:
    return tuple(_tentativas)


def _inteiro(variavel: str, padrao: int) -> int:
    declarado = os.environ.get(variavel)
    return int(declarado) if declarado else padrao


def _varrer(agora: float, janela: int) -> None:
    if len(_tentativas) <= CHAVES_MAXIMAS:
        return
    for chave, carimbos in list(_tentativas.items()):
        if not carimbos or agora - carimbos[-1] >= janela:
            del _tentativas[chave]
    if len(_tentativas) <= CHAVES_MAXIMAS:
        return
    descartaveis = sorted(
        _tentativas,
        key=lambda chave: (len(_tentativas[chave]), _tentativas[chave][-1]),
    )
    for chave in descartaveis[: len(_tentativas) - CHAVES_MAXIMAS]:
        del _tentativas[chave]
