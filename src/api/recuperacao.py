"""Redefinicao de senha por link curto, entregue por SMTP configurado pelo operador."""

from __future__ import annotations

import os
import secrets
import smtplib
import ssl
import time
from email.message import EmailMessage
from urllib.parse import quote, urlsplit

from api import auth, store

SMTP_HOST_ENV = "XVISION_SMTP_HOST"
SMTP_PORT_ENV = "XVISION_SMTP_PORT"
SMTP_USER_ENV = "XVISION_SMTP_USER"
SMTP_PASSWORD_ENV = "XVISION_SMTP_PASSWORD"
SMTP_FROM_ENV = "XVISION_SMTP_FROM"
SMTP_STARTTLS_ENV = "XVISION_SMTP_STARTTLS"
PUBLIC_URL_ENV = "XVISION_PUBLIC_URL"
TTL_ENV = "XVISION_PASSWORD_RESET_TTL_S"

TTL_PADRAO = 30 * 60
SMTP_PORT_PADRAO = 587


def configurada() -> bool:
    """A recuperacao so e anunciada quando entrega e endereco publico foram decididos."""
    return all(
        os.environ.get(nome, "").strip() for nome in (SMTP_HOST_ENV, SMTP_FROM_ENV, PUBLIC_URL_ENV)
    )


def validar_configuracao() -> None:
    """Recusa valores invalidos na subida, antes de aceitar qualquer pedido."""
    if not configurada():
        return
    try:
        porta = int(os.environ.get(SMTP_PORT_ENV, SMTP_PORT_PADRAO))
        ttl = _ttl()
    except ValueError as erro:
        raise ValueError("porta SMTP e validade precisam ser inteiros") from erro
    if not 1 <= porta <= 65_535 or ttl < 60:
        raise ValueError("porta SMTP ou validade fora do intervalo aceito")
    publica = urlsplit(os.environ[PUBLIC_URL_ENV])
    if publica.scheme not in {"http", "https"} or not publica.netloc:
        raise ValueError("XVISION_PUBLIC_URL precisa ser uma URL HTTP completa")


def solicitar(principal: store.PrincipalRow) -> None:
    """Emite um token de uso unico, guarda apenas o resumo e envia o link."""
    token = secrets.token_urlsafe(32)
    criada = store.criar_redefinicao_senha(
        principal,
        auth.hash_de_token(token),
        int(time.time()) + _ttl(),
    )
    if criada:
        _enviar(principal.nome, _link(token))


def confirmar(token: str, senha: str) -> bool:
    """Troca a senha somente enquanto o token ainda estiver valido e sem uso."""
    return store.redefinir_senha(
        auth.hash_de_token(token),
        auth.hash_de_senha(senha),
        agora=int(time.time()),
    )


def _ttl() -> int:
    declarado = os.environ.get(TTL_ENV)
    return max(60, int(declarado)) if declarado else TTL_PADRAO


def _link(token: str) -> str:
    base = os.environ[PUBLIC_URL_ENV].rstrip("/")
    return f"{base}/reset-password?token={quote(token, safe='')}"


def _enviar(destino: str, link: str) -> None:
    mensagem = EmailMessage()
    mensagem["Subject"] = "Redefina sua senha do XVision"
    mensagem["From"] = os.environ[SMTP_FROM_ENV]
    mensagem["To"] = destino
    mensagem.set_content(
        "Recebemos um pedido para redefinir sua senha do XVision.\n\n"
        f"Abra este link, valido por {_ttl() // 60} minutos:\n{link}\n\n"
        "Se voce nao pediu a troca, ignore esta mensagem."
    )

    host = os.environ[SMTP_HOST_ENV]
    porta = int(os.environ.get(SMTP_PORT_ENV, SMTP_PORT_PADRAO))
    usuario = os.environ.get(SMTP_USER_ENV)
    senha = os.environ.get(SMTP_PASSWORD_ENV)
    with smtplib.SMTP(host, porta, timeout=10) as servidor:
        if os.environ.get(SMTP_STARTTLS_ENV, "1") != "0":
            servidor.starttls(context=ssl.create_default_context())
        if usuario and senha:
            servidor.login(usuario, senha)
        servidor.send_message(mensagem)
