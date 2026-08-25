"""O estado dos videos enviados, que e o que torna o processamento assincrono possivel.

Quem envia o video e quem pergunta pelo resultado sao duas requisicoes HTTP diferentes, e entre
elas nao ha memoria compartilhada confiavel: o processo pode reiniciar, e o worker roda em outra
thread. O estado precisa entao morar fora do processo, e mora aqui.

E SQLite, da biblioteca padrao, e nao Postgres. A necessidade presente e uma requisicao escrever
e outra ler, num processo so, sem implantacao: SQLite entrega isso sem servico novo, sem
container novo e sem credencial. O preco esta nomeado: um arquivo nao se replica, entao no dia em
que o servico rodar em mais de uma instancia isto vira Postgres. A troca fica contida porque
nenhum chamador conhece SQL, so as sete funcoes abaixo.

O identificador e inteiro e sequencial porque o contrato do frontend pede `id: int`. Ele nao
serve de segredo: quem tem o numero le o estado. Por isso o envio nao usa o id e sim um token
opaco, que e o que autoriza gravar bytes.

Os quatro estados sao exatamente os que o frontend declara no enum, e nao ha um quinto. Video
criado e ainda sem bytes fica em `Pendente`, igual a video com bytes esperando a vez: para quem
consulta, os dois casos sao a mesma espera.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Generator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from edge.paths import var_dir

PENDENTE = "Pendente"
PROCESSANDO = "Em processamento"
FINALIZADO = "Finalizado"
FALHOU = "Falhou"

ESTADOS = (PENDENTE, PROCESSANDO, FINALIZADO, FALHOU)

TERMINAIS = frozenset({FINALIZADO, FALHOU})

PERMITIDO: Mapping[str, frozenset[str]] = {
    PENDENTE: frozenset({PROCESSANDO, FALHOU}),
    # `Em processamento` para si mesmo e permitido de proposito: e o que a recuperacao de
    # queda faz ao retomar um video que ja estava nesse estado. O que continua proibido e o
    # que causava troca de resultado, sair de um estado terminal e voltar para `Pendente`.
    PROCESSANDO: frozenset({PROCESSANDO, FINALIZADO, FALHOU}),
    FINALIZADO: frozenset(),
    FALHOU: frozenset(),
}


def jobs_db() -> Path:
    """Mora aqui, e nao em `edge.paths`, porque banco de video enviado e conceito da API.

    A borda nao pode conhecer nada da API: a fronteira entre as areas e cobrada por teste.
    """
    return var_dir() / "api" / "jobs.db"


MIGRACOES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token TEXT NOT NULL UNIQUE,
        nome_arquivo TEXT NOT NULL,
        content_type TEXT NOT NULL,
        estado TEXT NOT NULL,
        motivo TEXT,
        resultados TEXT,
        criado_em TEXT NOT NULL,
        atualizado_em TEXT NOT NULL
    );
    """,
    # A posse do endereco de envio. Fica fora do enum de propósito: o contrato do frontend tem
    # quatro estados e inventar um quinto quebraria o consumidor. Quem grava os bytes e quem
    # reivindica primeiro, e isso e decidido por um UPDATE condicional, nao por ler e depois
    # escrever.
    "ALTER TABLE videos ADD COLUMN reivindicado_em TEXT;",
    # Quem chama. Uma tabela so para os dois tipos, e nao uma por tipo, porque a posse do video
    # aponta para um espaco de identificadores unico: com duas tabelas, `videos.principal_id`
    # precisaria dizer tambem de qual delas, e toda leitura ganharia um caso a mais para errar.
    """
    CREATE TABLE IF NOT EXISTS principais (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tipo TEXT NOT NULL,
        nome TEXT NOT NULL UNIQUE,
        senha_hash TEXT,
        token_hash TEXT UNIQUE,
        criado_em TEXT NOT NULL,
        revogado_em TEXT
    );
    ALTER TABLE videos ADD COLUMN principal_id INTEGER REFERENCES principais(id);
    """,
)


def migrar(conn: sqlite3.Connection) -> int:
    """Aplica so as migracoes que faltam, e nunca a mesma duas vezes.

    `CREATE TABLE IF NOT EXISTS` sozinho cria banco novo e nao conserta banco velho: no dia em
    que uma coluna entrar, quem ja tem o arquivo continua com o esquema antigo e descobre isso em
    tempo de execucao. `PRAGMA user_version` guarda quantas migracoes ja rodaram, e o indice na
    tupla e a versao, entao acrescentar uma migracao e acrescentar uma string no fim.

    Nao usa Alembic porque Alembic traz SQLAlchemy junto, e aqui nao ha ORM nem chamador que
    escreva SQL: seriam duas dependencias para versionar uma tabela.

    Migracao existente nunca e editada, so acrescentada. Editar uma ja aplicada deixa dois bancos
    com o mesmo `user_version` e esquemas diferentes, que e pior que nao ter versao nenhuma.
    """
    atual = cast("int", conn.execute("PRAGMA user_version").fetchone()[0])
    for versao in range(atual, len(MIGRACOES)):
        conn.executescript(MIGRACOES[versao])
        conn.execute(f"PRAGMA user_version = {versao + 1}")
    return len(MIGRACOES)


class StoreError(RuntimeError): ...


class TransicaoInvalida(StoreError): ...


class NomeEmUso(StoreError): ...


@dataclass(frozen=True)
class Video:
    id: int
    token: str
    nome_arquivo: str
    content_type: str
    estado: str
    motivo: str | None
    resultados: list[dict[str, Any]] | None
    principal_id: int | None


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conexao(caminho: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Uma conexao por operacao, que e o que deixa o worker escrever de outra thread.

    Reaproveitar uma conexao entre threads exigiria `check_same_thread=False` mais um lock nosso,
    e trocaria um problema que o SQLite ja resolve por um que teriamos de manter.
    """
    destino = caminho if caminho is not None else jobs_db()
    destino.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(destino, timeout=30.0)) as conexao_aberta:
        conexao_aberta.row_factory = sqlite3.Row
        conexao_aberta.execute("PRAGMA journal_mode=WAL")
        conexao_aberta.execute("PRAGMA foreign_keys=ON")
        migrar(conexao_aberta)
        with conexao_aberta:
            yield conexao_aberta


def _para_video(linha: sqlite3.Row) -> Video:
    bruto = linha["resultados"]
    return Video(
        id=int(linha["id"]),
        token=str(linha["token"]),
        nome_arquivo=str(linha["nome_arquivo"]),
        content_type=str(linha["content_type"]),
        estado=str(linha["estado"]),
        motivo=None if linha["motivo"] is None else str(linha["motivo"]),
        resultados=None if bruto is None else json.loads(str(bruto)),
        principal_id=(None if linha["principal_id"] is None else int(linha["principal_id"])),
    )


def criar(
    nome_arquivo: str,
    content_type: str,
    caminho: Path | None = None,
    *,
    principal_id: int | None = None,
    reivindicado: bool = False,
) -> Video:
    """`reivindicado` marca o video que ja nasce com dono do trabalho, e nao so com dono da conta.

    O caminho base64 nao emite endereco de envio: os bytes chegam na mesma requisicao que cria o
    registro, entao nao ha nada a reivindicar depois. Sem marcar aqui, ele ficava de fora de
    `recuperaveis` e uma queda no meio do processamento deixava o video em `Pendente` para sempre,
    com o arquivo parado em disco. A correcao anterior cobriu so o caminho do `PUT`.
    """
    agora = _agora()
    token = secrets.token_urlsafe(32)
    with conexao(caminho) as conn:
        cursor = conn.execute(
            "INSERT INTO videos (token, nome_arquivo, content_type, estado, criado_em, "
            "atualizado_em, principal_id, reivindicado_em) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                token,
                nome_arquivo,
                content_type,
                PENDENTE,
                agora,
                agora,
                principal_id,
                agora if reivindicado else None,
            ),
        )
        identificador = cursor.lastrowid
    if identificador is None:  # pragma: no cover - o SQLite sempre devolve o rowid do INSERT
        raise StoreError("o SQLite nao devolveu identificador para o video criado")
    return Video(
        id=identificador,
        token=token,
        nome_arquivo=nome_arquivo,
        content_type=content_type,
        estado=PENDENTE,
        motivo=None,
        resultados=None,
        principal_id=principal_id,
    )


def ler(identificador: int, caminho: Path | None = None) -> Video | None:
    with conexao(caminho) as conn:
        linha = conn.execute("SELECT * FROM videos WHERE id = ?", (identificador,)).fetchone()
    return None if linha is None else _para_video(linha)


def ler_por_token(token: str, caminho: Path | None = None) -> Video | None:
    with conexao(caminho) as conn:
        linha = conn.execute("SELECT * FROM videos WHERE token = ?", (token,)).fetchone()
    return None if linha is None else _para_video(linha)


def reivindicar(token: str, caminho: Path | None = None) -> Video | None:
    """Toma a posse do endereco de envio, ou devolve nada se outro ja tomou.

    Ler o registro, conferir o estado e so entao gravar nao e atomico, e a janela nao e teorica:
    medido, dois PUT no mesmo token passavam os dois, e com a fila ocupada isso acontecia 100%
    das vezes mesmo sem simultaneidade nenhuma, porque a janela e a fila inteira. Bastava um
    cliente repetir depois de um timeout de rede. O segundo envio apagava o arquivo que o worker
    estava lendo.

    Um UPDATE condicional decide sozinho: quem viu `rowcount` igual a 1 ganhou.
    """
    agora = _agora()
    with conexao(caminho) as conn:
        cursor = conn.execute(
            "UPDATE videos SET reivindicado_em = ?, atualizado_em = ? "
            "WHERE token = ? AND reivindicado_em IS NULL AND estado = ?",
            (agora, agora, token, PENDENTE),
        )
        if cursor.rowcount != 1:
            return None
        linha = conn.execute("SELECT * FROM videos WHERE token = ?", (token,)).fetchone()
    return _para_video(linha)


def marcar(
    identificador: int,
    estado: str,
    *,
    motivo: str | None = None,
    resultados: list[dict[str, Any]] | None = None,
    caminho: Path | None = None,
) -> None:
    """Grava o estado novo, e recusa transicao que a maquina de estados nao preve.

    Validar so o valor deixava passar o que importa: sair de `Finalizado` ou de `Falhou`. Medido
    no servico anterior, dois envios no mesmo token faziam um lado marcar `Falhou` e o worker do
    outro gravar `Finalizado` por cima, e o registro terminava contando a historia errada. Estado
    terminal e terminal: quem tentar sair dele leva erro, e nao silencio.
    """
    if estado not in ESTADOS:
        raise StoreError(f"{estado!r} nao e um dos estados do contrato: {', '.join(ESTADOS)}")
    with conexao(caminho) as conn:
        linha = conn.execute("SELECT estado FROM videos WHERE id = ?", (identificador,)).fetchone()
        if linha is None:
            raise StoreError(f"nao existe video com id {identificador}")
        atual = str(linha["estado"])
        if estado not in PERMITIDO[atual]:
            saidas = ", ".join(sorted(PERMITIDO[atual])) or "lugar nenhum"
            raise TransicaoInvalida(
                f"o video {identificador} esta em {atual!r} e nao pode ir para {estado!r}. "
                f"De {atual!r} so se sai para {saidas}"
            )
        conn.execute(
            "UPDATE videos SET estado = ?, motivo = ?, resultados = ?, atualizado_em = ? "
            "WHERE id = ?",
            (
                estado,
                motivo,
                None if resultados is None else json.dumps(resultados, ensure_ascii=False),
                _agora(),
                identificador,
            ),
        )


def recuperaveis(caminho: Path | None = None) -> tuple[Video, ...]:
    """Os videos que uma queda deixou pelo meio: reivindicados e ainda sem desfecho.

    Nao basta olhar `Em processamento`. Um video que ja tinha os bytes e esperava a vez continua
    em `Pendente`, e o desligamento com prazo cancela justamente esses: eles tambem precisam
    voltar. O que separa um trabalho interrompido de um endereco de envio que o cliente ainda nao
    usou e a posse, e nao o estado.

    Antes isto se chamava `reenfileirar_orfaos` e nao reenfileirava nada: trocava a coluna para
    `Pendente` e pronto, e como ninguem pega trabalho pendente sozinho, o frontend passava a
    girar em `Pendente` em vez de girar em `Em processamento`. O arquivo enviado tambem ficava
    no disco para sempre. Trocar o nome do estado nao e recuperar o trabalho.

    Quem sabe reprocessar e a camada que tem a fila, entao aqui so se le. A decisao do que fazer
    com cada um mora em quem pode submeter.
    """
    with conexao(caminho) as conn:
        linhas = conn.execute(
            "SELECT * FROM videos WHERE reivindicado_em IS NOT NULL AND estado NOT IN (?, ?) "
            "ORDER BY id",
            (FINALIZADO, FALHOU),
        ).fetchall()
    return tuple(_para_video(linha) for linha in linhas)


@dataclass(frozen=True)
class PrincipalRow:
    id: int
    tipo: str
    nome: str
    senha_hash: str | None
    revogado_em: str | None


def _para_principal(linha: sqlite3.Row) -> PrincipalRow:
    return PrincipalRow(
        id=int(linha["id"]),
        tipo=str(linha["tipo"]),
        nome=str(linha["nome"]),
        senha_hash=None if linha["senha_hash"] is None else str(linha["senha_hash"]),
        revogado_em=None if linha["revogado_em"] is None else str(linha["revogado_em"]),
    )


def criar_principal(
    tipo: str,
    nome: str,
    *,
    senha_hash: str | None = None,
    token_hash: str | None = None,
    caminho: Path | None = None,
) -> PrincipalRow:
    """Nome repetido e recusado com erro proprio, e nao com o `IntegrityError` cru do SQLite.

    Quem chama precisa distinguir "esse email ja existe" de "o banco quebrou", e a mensagem do
    SQLite nao serve para mostrar a ninguem.
    """
    with conexao(caminho) as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO principais (tipo, nome, senha_hash, token_hash, criado_em) "
                "VALUES (?, ?, ?, ?, ?)",
                (tipo, nome, senha_hash, token_hash, _agora()),
            )
        except sqlite3.IntegrityError as erro:
            raise NomeEmUso(f"{nome!r} ja esta em uso") from erro
        identificador = cursor.lastrowid
        linha = conn.execute("SELECT * FROM principais WHERE id = ?", (identificador,)).fetchone()
    return _para_principal(linha)


def principal_por_nome(nome: str, caminho: Path | None = None) -> PrincipalRow | None:
    with conexao(caminho) as conn:
        linha = conn.execute("SELECT * FROM principais WHERE nome = ?", (nome,)).fetchone()
    return None if linha is None else _para_principal(linha)


def principal_por_token(token_hash: str, caminho: Path | None = None) -> PrincipalRow | None:
    """Procura pelo resumo, e nao pelo token: o banco nunca guardou o token em si."""
    with conexao(caminho) as conn:
        linha = conn.execute(
            "SELECT * FROM principais WHERE token_hash = ?", (token_hash,)
        ).fetchone()
    return None if linha is None else _para_principal(linha)


def ler_principal(identificador: int, caminho: Path | None = None) -> PrincipalRow | None:
    with conexao(caminho) as conn:
        linha = conn.execute("SELECT * FROM principais WHERE id = ?", (identificador,)).fetchone()
    return None if linha is None else _para_principal(linha)


def revogar_principal(identificador: int, caminho: Path | None = None) -> bool:
    """Revogar nao apaga: o video continua apontando para quem o enviou.

    Apagando a linha, a posse dos videos daquele principal viraria nula e eles ficariam sem dono,
    legiveis por ninguem e apagaveis por ninguem. A revogacao fecha a porta e preserva o registro.
    """
    with conexao(caminho) as conn:
        cursor = conn.execute(
            "UPDATE principais SET revogado_em = ? WHERE id = ? AND revogado_em IS NULL",
            (_agora(), identificador),
        )
        return cursor.rowcount == 1


def videos_de(principal_id: int, caminho: Path | None = None) -> tuple[Video, ...]:
    with conexao(caminho) as conn:
        linhas = conn.execute(
            "SELECT * FROM videos WHERE principal_id = ? ORDER BY id", (principal_id,)
        ).fetchall()
    return tuple(_para_video(linha) for linha in linhas)
