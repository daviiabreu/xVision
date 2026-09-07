"""A ponte entre o cabecalho da requisicao e o `Principal`, e a regra de posse.

Uma credencial entra e um `Principal` sai. Quem chama nao sabe se veio sessao de usuario ou
token de servico, e e isso que impede a regra de posse de ter dois jeitos de estar errada.

`PUT /videos/upload/{token}` fica de fora da exigencia de Bearer, e isso e decisao e nao
esquecimento. O endereco de envio ja e uma credencial: 32 bytes sorteados, valido uma vez
so, preso a um video e reivindicado por UPDATE condicional. E o mesmo desenho de uma URL
pre-assinada, que e o padrao que o codigo do frontend ja implementa. Exigir tambem o Bearer
ali quebraria `videoService.ts` sem fechar nada, porque quem tem o endereco ja provou que
passou por `upload-url`, que exige credencial.
"""

from __future__ import annotations

import os
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from api import auth, store

TOKEN_SEMENTE_ENV = "XVISION_API_TOKEN"
SEMENTE_MINIMA = 32
SERVICO_SEMENTE = "semeado-do-ambiente"
RECUSA = "credencial ausente ou invalida"


def _recusar() -> HTTPException:
    """A mesma resposta para ausente, malformada, expirada e revogada.

    Distinguir os casos conta a quem tenta o que faltou para acertar: se "token expirado" e
    diferente de "token desconhecido", ja se sabe que o token existiu. O `WWW-Authenticate` diz o
    esquema esperado, que e o que um cliente legitimo precisa.
    """
    return HTTPException(status_code=401, detail=RECUSA, headers={"WWW-Authenticate": "Bearer"})


def principal_de(credencial: str) -> store.PrincipalRow:
    """Resolve os dois tipos de credencial, e recusa os dois do mesmo jeito.

    A sessao tem forma propria (corpo e assinatura separados por ponto), entao ela e tentada
    primeiro; o que nao tem essa forma so pode ser token de servico.
    """
    linha = _por_sessao(credencial) if "." in credencial else _por_token(credencial)
    if linha is None or linha.revogado_em is not None:
        raise _recusar()
    return linha


def _por_sessao(credencial: str) -> store.PrincipalRow | None:
    try:
        sessao = auth.verificar(credencial)
    except auth.AuthError:
        return None
    linha = store.ler_principal(sessao.principal_id)
    if linha is None or linha.sessao_versao != sessao.versao:
        return None
    return linha


def _por_token(credencial: str) -> store.PrincipalRow | None:
    return store.principal_por_token(auth.hash_de_token(credencial))


def principal_atual(request: Request) -> auth.Principal:
    cabecalho = request.headers.get("authorization", "")
    esquema, _, credencial = cabecalho.partition(" ")
    if esquema.lower() != "bearer" or not credencial:
        raise _recusar()
    linha = principal_de(credencial)
    return auth.Principal(id=linha.id, tipo=cast("auth.TipoPrincipal", linha.tipo), nome=linha.nome)


Autenticado = Annotated[auth.Principal, Depends(principal_atual)]


def conferir_posse(video: store.Video, principal: auth.Principal) -> None:
    """Video de outro responde 404, e nao 403.

    Com 403, o identificador sequencial vira um oraculo: quem contar de 1 em diante descobre
    quantos videos existem e quando cada um foi enviado, mesmo sem ler nenhum. Com 404, "nao e
    seu" e "nao existe" sao indistinguiveis de fora, que e a unica coisa que quem tenta precisa
    nao saber.
    """
    if video.principal_id != principal.id:
        raise HTTPException(status_code=404, detail=f"nao existe video com id {video.id}")


def semear_token_de_servico() -> str | None:
    """A comodidade da chave de API, sem o buraco dela.

    Por o segredo numa variavel continua sendo o jeito de comecar, e o token nasce na tabela:
    revogavel sem redeploy, e atribuivel quando alguem perguntar quem mandou um video. Semear e
    idempotente, entao subir duas vezes nao cria dois.

    O valor tem piso de comprimento porque ele vira credencial exatamente como esta escrito.
    Medido antes do piso: `XVISION_API_TOKEN=x` autenticava, e `Bearer x` abria a API inteira. Um
    token sorteado tem 43 caracteres; aceitar 1 e ter senha fraca com nome de token.
    """
    declarado = (os.environ.get(TOKEN_SEMENTE_ENV) or "").strip()
    if not declarado:
        return None
    if len(declarado) < SEMENTE_MINIMA:
        raise ValueError(
            f"{TOKEN_SEMENTE_ENV} tem {len(declarado)} caracteres e o minimo e {SEMENTE_MINIMA}. "
            "Ela vira credencial verbatim, entao um valor curto e uma senha fraca com nome de "
            'token: `python -c "import secrets; print(secrets.token_urlsafe(32))"`'
        )
    if store.principal_por_token(auth.hash_de_token(declarado)) is not None:
        return None
    if store.principal_por_nome(SERVICO_SEMENTE) is not None:
        return None
    store.criar_principal("servico", SERVICO_SEMENTE, token_hash=auth.hash_de_token(declarado))
    return SERVICO_SEMENTE
