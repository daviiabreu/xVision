"""Quem esta chamando, e como isso e provado. Sem HTTP aqui.

Duas credenciais entram e uma coisa sai: um `Principal`. O resto do servico nunca pergunta se
veio token de servico ou conta de usuario, so recebe quem e. Sem essa costura, cada rota teria de
lidar com dois casos e a regra de posse teria dois jeitos de estar errada.

Chave de API compartilhada nao existe de proposito. Ela e um token de servico que nao esta no
banco: nao da para revogar sem redeploy, e nao responde "quem mandou esse video". A comodidade de
por um segredo na variavel continua, e o servico semeia um token a partir dela na subida, mas ele
nasce na tabela, revogavel e atribuivel.

O token de sessao e assinado com HMAC-SHA256 e nao e um JWT. JWT carrega o algoritmo dentro do
proprio token, e a familia de erros mais conhecida da especificacao nasce dai: aceitar `none`,
confundir HS256 com RS256. Aqui os dois lados sao o mesmo processo e o algoritmo e fixo, entao
negociar algoritmo so acrescentaria a superficie sem comprar nada. A comparacao e em tempo
constante, que e o unico detalhe que realmente importa.

Senha usa `hashlib.scrypt`, da biblioteca padrao, com sal por senha. E funcao de derivacao de
chave de verdade, e nao um sha256 apressado: um vazamento do banco nao entrega as senhas.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Literal, cast

SEGREDO_ENV = "XVISION_AUTH_SECRET"
VALIDADE_ENV = "XVISION_AUTH_TTL_S"
VALIDADE_PADRAO = 8 * 3600

TIPOS = ("usuario", "servico")
TipoPrincipal = Literal["usuario", "servico"]

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SAL_BYTES = 16


class AuthError(RuntimeError): ...


class SegredoAusente(AuthError): ...


class TokenInvalido(AuthError): ...


@dataclass(frozen=True)
class Principal:
    """Quem chamou. O `id` e unico entre usuarios e servicos, porque a posse e uma so."""

    id: int
    tipo: TipoPrincipal
    nome: str


@dataclass(frozen=True)
class SessaoAssinada:
    """Identidade e versao revogavel carregadas pela credencial de sessao."""

    principal_id: int
    versao: int


def segredo() -> bytes:
    """Sem segredo o servico nao sobe, e nao inventa um padrao.

    Segredo padrao e pior que segredo ausente: ele funciona, ninguem repara, e o mesmo valor vai
    para producao com o codigo. Ausente, alguem precisa decidir uma vez.
    """
    declarado = os.environ.get(SEGREDO_ENV)
    if not declarado:
        raise SegredoAusente(
            f"{SEGREDO_ENV} nao esta definida, e o servico nao assina sessao sem ela. "
            'Gere uma com `python -c "import secrets; print(secrets.token_urlsafe(48))"`'
        )
    return declarado.encode()


def validade() -> int:
    declarado = os.environ.get(VALIDADE_ENV)
    return max(60, int(declarado)) if declarado else VALIDADE_PADRAO


def hash_de_senha(senha: str, sal: bytes | None = None) -> str:
    """`scrypt` com sal por senha, guardado junto do hash no mesmo texto."""
    usado = sal if sal is not None else secrets.token_bytes(SAL_BYTES)
    derivado = hashlib.scrypt(
        senha.encode(), salt=usado, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(usado)}${_b64(derivado)}"


def conferir_senha(senha: str, guardado: str) -> bool:
    """Comparacao em tempo constante, para o tempo de resposta nao contar quanto ela acertou."""
    try:
        marca, n, r, p, sal, esperado = guardado.split("$")
    except ValueError:
        return False
    if marca != "scrypt":
        return False
    derivado = hashlib.scrypt(
        senha.encode(),
        salt=_de_b64(sal),
        n=int(n),
        r=int(r),
        p=int(p),
        dklen=len(_de_b64(esperado)),
    )
    return hmac.compare_digest(derivado, _de_b64(esperado))


def novo_token() -> str:
    """O token de servico e sorteado e mostrado uma vez; o banco guarda so o resumo dele."""
    return secrets.token_urlsafe(32)


def hash_de_token(token: str) -> str:
    """sha256 basta aqui, e nao seria o caso para senha.

    Token e sorteado com 256 bits de entropia, entao nao ha o que adivinhar por forca bruta; senha
    e escolhida por gente, e por isso precisa de funcao lenta.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def assinar(principal_id: int, sessao_versao: int = 1, agora: int | None = None) -> str:
    momento = int(time.time()) if agora is None else agora
    corpo = {"sub": principal_id, "ver": sessao_versao, "exp": momento + validade()}
    dados = _b64(json.dumps(corpo, separators=(",", ":"), sort_keys=True).encode())
    return f"{dados}.{_b64(_assinatura(dados))}"


def verificar(token: str, agora: int | None = None) -> SessaoAssinada:
    """Devolve a sessao do token, ou recusa com `TokenInvalido`. Nunca levanta outra coisa.

    Toda falha sai como `TokenInvalido` de proposito. `binascii.Error` de base64 torto, `ValueError`
    de JSON quebrado e `KeyError` de campo ausente escapavam daqui e viravam 500 na fronteira HTTP,
    o que quebrava a promessa escrita em `seguranca._recusar`, de que ausente, malformada, expirada
    e revogada respondem igual. Medido: `Bearer abc.b` devolvia 500 e `Bearer abc.bc` devolvia 401,
    e a diferenca entre as duas conta a quem tenta que a primeira chegou mais longe.
    """
    try:
        dados, assinatura = token.split(".")
    except ValueError as erro:
        raise TokenInvalido("o token nao tem a forma esperada") from erro
    try:
        confere = hmac.compare_digest(_de_b64(assinatura), _assinatura(dados))
    except (binascii.Error, ValueError) as erro:
        raise TokenInvalido("o token nao esta em base64") from erro
    if not confere:
        raise TokenInvalido("a assinatura do token nao confere")
    try:
        corpo = cast("dict[str, Any]", json.loads(_de_b64(dados)))
        expiracao = int(corpo["exp"])
        principal = int(corpo["sub"])
        versao = int(corpo["ver"])
    except (binascii.Error, ValueError, TypeError, KeyError) as erro:
        raise TokenInvalido("o corpo do token nao tem a forma esperada") from erro
    momento = int(time.time()) if agora is None else agora
    if expiracao <= momento:
        raise TokenInvalido("a sessao expirou")
    return SessaoAssinada(principal_id=principal, versao=versao)


def _assinatura(dados: str) -> bytes:
    return hmac.new(segredo(), dados.encode(), hashlib.sha256).digest()


def _b64(bruto: bytes) -> str:
    return base64.urlsafe_b64encode(bruto).decode().rstrip("=")


def _de_b64(texto: str) -> bytes:
    return base64.urlsafe_b64decode(texto + "=" * (-len(texto) % 4))
