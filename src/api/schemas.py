"""O contrato HTTP que o frontend consome, e a traducao dele para os tipos da borda.

Tres especificacoes chegaram do time do frontend e nao coincidiam, entao por uma sprint o servico
atendeu as tres: dois caminhos de envio e um alias no nome do campo de tipo. A interface fechou no
`PUT` do binario cru, o caminho base64 e o alias ficaram sem cliente, e os dois sairam. O que
sobra e o que alguem chama.

Dois campos que o frontend pede e o evento nao carrega, cada um resolvido onde tem resposta.
`unidade_de_medida` vem da `FeatureSpec`, porque quem sabe a unidade e a feature, e nao a
fronteira: derivar metro de um sufixo `_m` quebraria na primeira que nao seguisse a convencao.
`tipo` vem do valor, e nao de um campo declarado na `FeatureSpec`: o valor e o que a tela desenha,
e um tipo declarado que discordasse dele nao teria como ser conferido aqui. Os tres tipos que o
evento aceita atravessam, texto inclusive.

`status` na metrica existe porque a borda promete que numero medido e numero indisponivel nunca
chegam com a mesma cara. Sem ele, uma feature de baixa confianca seria lida como medida, que e
exatamente o silencio que `edge.features.base` recusa produzir.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

TipoDeMetrica = Literal["boolean", "float", "text"]

BOOLEANO: TipoDeMetrica = "boolean"
FLUTUANTE: TipoDeMetrica = "float"
TEXTO: TipoDeMetrica = "text"

VideoStatus = Literal["Pendente", "Em processamento", "Finalizado", "Falhou"]


class UploadUrlRequest(BaseModel):
    """Os dois campos tem teto de tamanho porque os dois vao para o banco.

    Sem teto, medido: um `fileName` de 5 MB era aceito e gravado, e o banco saiu de vazio para
    5 MB com uma requisicao.
    """

    fileName: str = Field(max_length=255, description="Nome original do arquivo, so para registro.")
    contentType: str = Field(max_length=255, description="Tipo do arquivo, por exemplo video/mp4.")


class UploadUrlResponse(BaseModel):
    id: int = Field(description="Identificador do video, usado para consultar o andamento.")
    uploadUrl: str = Field(
        description="Caminho para onde mandar o binario por PUT, relativo a raiz do servico."
    )


class Aceito(BaseModel):
    """A resposta de quem so enfileirou: o trabalho comeca depois que esta resposta ja saiu."""

    status: int = Field(description="Codigo HTTP repetido no corpo, como o contrato pede.")
    id: int = Field(description="Identificador do video para consultar o andamento.")


class StatusResponse(BaseModel):
    status: int = Field(description="Codigo HTTP repetido no corpo, como o contrato pede.")
    video_status: VideoStatus = Field(description="Em que ponto o video esta.")
    motivo: str | None = Field(
        default=None, description="Por que falhou. Nulo em qualquer outro estado."
    )


class Metrica(BaseModel):
    nome: str
    unidade_de_medida: str
    tipo: TipoDeMetrica | None = Field(
        default=None,
        description="Tipo do valor medido. Nulo quando nao houve leitura, porque ele sai do valor.",
    )
    valor: bool | float | str | None = Field(
        default=None, description="Nulo quando a feature nao pode ser medida."
    )
    status: Literal["ok", "low_confidence", "unavailable"] = Field(
        default="ok", description="Numero medido e numero indisponivel nao tem a mesma cara."
    )


class Resultado(BaseModel):
    nome_processo: str
    status: bool = Field(description="Verdadeiro quando a etapa rodou sem falhar.")
    frame: str | None = Field(
        default=None,
        description="Imagem de demonstracao em base64. Nula enquanto a visualizacao nao entrar.",
    )
    metricas: list[Metrica] = []


class ResultadoResponse(BaseModel):
    status: int = Field(description="Codigo HTTP repetido no corpo, como o contrato pede.")
    video_status: VideoStatus
    resultados: list[Resultado] = Field(
        default=[], description="Vazio ate o video chegar em Finalizado."
    )
    motivo: str | None = Field(default=None, description="Preenchido apenas quando Falhou.")
    render_url: str | None = Field(
        default=None,
        description=(
            "Caminho do video anotado da analise, relativo a raiz do servico, quando ele o "
            "produziu. Nulo quando o render nao rodou, e o motivo fica no log e nao aqui: a "
            "analise nao falhou por isso."
        ),
    )


SENHA_MINIMA = 12


def _email(valor: str) -> str:
    """Normaliza antes de comparar, senao a unicidade nao e unicidade.

    Medido: `Mixed@Case.com` e `mixed@case.com` criavam duas contas, porque o `UNIQUE` do SQLite
    compara byte a byte. Duas contas para a mesma pessoa e problema de suporte e de posse: os
    videos ficam divididos entre elas e ninguem entende por que sumiram.

    Espaco em volta tambem sai: e o que o navegador manda quando alguem cola o email.
    """
    limpo = valor.strip().lower()
    if not limpo:
        raise ValueError("o email nao pode ser vazio")
    return limpo


Email = Annotated[str, AfterValidator(_email)]


class RegistroRequest(BaseModel):
    """Comprimento minimo, e nenhuma regra de simbolo obrigatorio.

    Exigir maiuscula, numero e simbolo produz `Senha@123` e nada mais: as pessoas satisfazem a
    regra pelo caminho mais curto, que e o mesmo caminho para todo mundo, e o conjunto de senhas
    provaveis encolhe. Comprimento e o unico parametro que aumenta o custo de adivinhar sem
    empurrar todo mundo para o mesmo padrao.
    """

    email: Email = Field(max_length=254, description="Serve de identificador da conta.")
    senha: str = Field(min_length=SENHA_MINIMA, max_length=256)


class LoginRequest(BaseModel):
    email: Email = Field(max_length=254)
    senha: str = Field(max_length=256)


class PrincipalResponse(BaseModel):
    id: int
    tipo: Literal["usuario", "servico"]
    nome: str


class SessaoResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expira_em: int = Field(description="Segundos ate a sessao deixar de valer.")


class Erro(BaseModel):
    """A forma de toda recusa, para quem gera cliente a partir do contrato nao errar o tipo."""

    detail: str


def tipo_de(valor: object) -> TipoDeMetrica | None:
    """Sem valor nao ha tipo. Dizer `float` para uma categoria sem leitura e inventar um numero.

    `body_part_3` e `type_of_play` sao dois dos tres rotulos `enum` de `config/labels.toml`, e
    quando eles nao medirem o antigo `else` declararia `float` para uma feature que nunca produz
    numero.
    """
    if valor is None:
        return None
    if isinstance(valor, bool):
        return BOOLEANO
    return TEXTO if isinstance(valor, str) else FLUTUANTE
