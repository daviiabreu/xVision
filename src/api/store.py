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
    """
    ALTER TABLE videos ADD COLUMN etapa_atual TEXT;
    ALTER TABLE videos ADD COLUMN etapas_concluidas INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE videos ADD COLUMN etapas_total INTEGER NOT NULL DEFAULT 0;
    """,
    """
    CREATE TABLE redefinicoes_senha (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        principal_id INTEGER NOT NULL REFERENCES principais(id),
        token_hash TEXT NOT NULL UNIQUE,
        expira_em INTEGER NOT NULL,
        criado_em TEXT NOT NULL,
        usado_em TEXT
    );
    CREATE INDEX idx_redefinicoes_senha_principal
        ON redefinicoes_senha(principal_id, usado_em);
    """,
    "ALTER TABLE principais ADD COLUMN sessao_versao INTEGER NOT NULL DEFAULT 1;",
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
    criado_em: str
    atualizado_em: str
    etapa_atual: str | None
    etapas_concluidas: int
    etapas_total: int


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
        criado_em=str(linha["criado_em"]),
        atualizado_em=str(linha["atualizado_em"]),
        etapa_atual=(None if linha["etapa_atual"] is None else str(linha["etapa_atual"])),
        etapas_concluidas=int(linha["etapas_concluidas"]),
        etapas_total=int(linha["etapas_total"]),
    )


def criar(
    nome_arquivo: str,
    content_type: str,
    caminho: Path | None = None,
    *,
    principal_id: int | None = None,
) -> Video:
    """O video nasce sem posse do trabalho: quem grava os bytes a reivindica depois, no `PUT`.

    E o que mantem fora de `recuperaveis` o endereco de envio que o cliente pediu e ainda nao
    usou, porque o que separa trabalho interrompido de espera legitima e a posse, nao o estado.
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
                None,
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
        criado_em=agora,
        atualizado_em=agora,
        etapa_atual=None,
        etapas_concluidas=0,
        etapas_total=0,
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


def expirar_reservas(
    antes_de: str,
    caminho: Path | None = None,
    *,
    motivo: str = "o endereco de envio expirou antes de receber o video",
) -> int:
    """Encerra reservas que nunca foram reivindicadas pelo cliente."""
    agora = _agora()
    with conexao(caminho) as conn:
        cursor = conn.execute(
            "UPDATE videos SET estado = ?, motivo = ?, atualizado_em = ? "
            "WHERE estado = ? AND reivindicado_em IS NULL AND criado_em <= ?",
            (FALHOU, motivo, agora, PENDENTE, antes_de),
        )
    return cursor.rowcount


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
            "UPDATE videos SET estado = ?, motivo = ?, resultados = ?, atualizado_em = ?, "
            "etapa_atual = NULL, etapas_concluidas = CASE WHEN ? = ? THEN etapas_total "
            "ELSE etapas_concluidas END "
            "WHERE id = ?",
            (
                estado,
                motivo,
                None if resultados is None else json.dumps(resultados, ensure_ascii=False),
                _agora(),
                estado,
                FINALIZADO,
                identificador,
            ),
        )


def marcar_progresso(
    identificador: int,
    etapa: str,
    concluidas: int,
    total: int,
    caminho: Path | None = None,
) -> None:
    """Atualiza o progresso somente enquanto o video esta em processamento."""
    if not etapa.strip():
        raise StoreError("a etapa atual nao pode ser vazia")
    if total <= 0 or concluidas < 0 or concluidas > total:
        raise StoreError(f"progresso invalido: {concluidas} de {total}")
    with conexao(caminho) as conn:
        cursor = conn.execute(
            "UPDATE videos SET etapa_atual = ?, etapas_concluidas = ?, etapas_total = ?, "
            "atualizado_em = ? WHERE id = ? AND estado = ?",
            (etapa, concluidas, total, _agora(), identificador, PROCESSANDO),
        )
        if cursor.rowcount != 1:
            raise StoreError(f"o video {identificador} nao esta em processamento")


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
    sessao_versao: int


@dataclass(frozen=True)
class VideoStatistics:
    total: int
    pendentes: int
    em_processamento: int
    finalizadas: int
    falhas: int


def _para_principal(linha: sqlite3.Row) -> PrincipalRow:
    return PrincipalRow(
        id=int(linha["id"]),
        tipo=str(linha["tipo"]),
        nome=str(linha["nome"]),
        senha_hash=None if linha["senha_hash"] is None else str(linha["senha_hash"]),
        revogado_em=None if linha["revogado_em"] is None else str(linha["revogado_em"]),
        sessao_versao=int(linha["sessao_versao"]),
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


def atualizar_nome_principal(
    identificador: int, nome: str, caminho: Path | None = None
) -> PrincipalRow | None:
    """Atualiza o identificador de login sem trocar a identidade que possui os videos."""
    with conexao(caminho) as conn:
        try:
            cursor = conn.execute(
                "UPDATE principais SET nome = ? "
                "WHERE id = ? AND tipo = 'usuario' AND revogado_em IS NULL",
                (nome, identificador),
            )
        except sqlite3.IntegrityError as erro:
            raise NomeEmUso(f"{nome!r} ja esta em uso") from erro
        if cursor.rowcount != 1:
            return None
        conn.execute(
            "UPDATE redefinicoes_senha SET usado_em = ? "
            "WHERE principal_id = ? AND usado_em IS NULL",
            (_agora(), identificador),
        )
        linha = conn.execute("SELECT * FROM principais WHERE id = ?", (identificador,)).fetchone()
    return _para_principal(linha)


def atualizar_senha_principal(
    identificador: int, senha_hash: str, caminho: Path | None = None
) -> PrincipalRow | None:
    agora = _agora()
    with conexao(caminho) as conn:
        cursor = conn.execute(
            "UPDATE principais SET senha_hash = ?, sessao_versao = sessao_versao + 1 "
            "WHERE id = ? AND tipo = 'usuario' AND revogado_em IS NULL",
            (senha_hash, identificador),
        )
        if cursor.rowcount != 1:
            return None
        conn.execute(
            "UPDATE redefinicoes_senha SET usado_em = ? "
            "WHERE principal_id = ? AND usado_em IS NULL",
            (agora, identificador),
        )
        linha = conn.execute("SELECT * FROM principais WHERE id = ?", (identificador,)).fetchone()
    return _para_principal(linha)


def criar_redefinicao_senha(
    principal: PrincipalRow,
    token_hash: str,
    expira_em: int,
    caminho: Path | None = None,
) -> bool:
    """Emite para a credencial solicitante ainda atual e invalida os pedidos anteriores."""
    agora = _agora()
    with conexao(caminho) as conn:
        cursor = conn.execute(
            "INSERT INTO redefinicoes_senha "
            "(principal_id, token_hash, expira_em, criado_em) "
            "SELECT id, ?, ?, ? FROM principais "
            "WHERE id = ? AND nome = ? AND sessao_versao = ? "
            "AND tipo = 'usuario' AND revogado_em IS NULL",
            (token_hash, expira_em, agora, principal.id, principal.nome, principal.sessao_versao),
        )
        if cursor.rowcount != 1:
            return False
        conn.execute(
            "UPDATE redefinicoes_senha SET usado_em = ? "
            "WHERE principal_id = ? AND token_hash != ? AND usado_em IS NULL",
            (agora, principal.id, token_hash),
        )
    return True


def redefinir_senha(
    token_hash: str,
    senha_hash: str,
    *,
    agora: int,
    caminho: Path | None = None,
) -> bool:
    """Consome o token uma vez e troca a senha na mesma transacao."""
    usado_em = _agora()
    with conexao(caminho) as conn:
        cursor = conn.execute(
            "UPDATE redefinicoes_senha SET usado_em = ? "
            "WHERE token_hash = ? AND usado_em IS NULL AND expira_em > ? "
            "AND principal_id IN ("
            "SELECT id FROM principais WHERE tipo = 'usuario' AND revogado_em IS NULL"
            ")",
            (usado_em, token_hash, agora),
        )
        if cursor.rowcount != 1:
            return False
        linha = conn.execute(
            "SELECT principal_id FROM redefinicoes_senha WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        if linha is None:  # pragma: no cover - a linha acabou de ser atualizada
            raise StoreError("a redefinicao consumida desapareceu")
        conn.execute(
            "UPDATE principais SET senha_hash = ?, sessao_versao = sessao_versao + 1 WHERE id = ?",
            (senha_hash, int(linha["principal_id"])),
        )
        conn.execute(
            "UPDATE redefinicoes_senha SET usado_em = ? "
            "WHERE principal_id = ? AND usado_em IS NULL",
            (usado_em, int(linha["principal_id"])),
        )
    return True


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


def videos_de(
    principal_id: int,
    caminho: Path | None = None,
    *,
    limite: int = 20,
    deslocamento: int = 0,
) -> tuple[Video, ...]:
    with conexao(caminho) as conn:
        linhas = conn.execute(
            "SELECT * FROM videos WHERE principal_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (principal_id, limite, deslocamento),
        ).fetchall()
    return tuple(_para_video(linha) for linha in linhas)


def contar_videos_de(principal_id: int, caminho: Path | None = None) -> int:
    with conexao(caminho) as conn:
        linha = conn.execute(
            "SELECT COUNT(*) AS total FROM videos WHERE principal_id = ?", (principal_id,)
        ).fetchone()
    return int(linha["total"])


def estatisticas_videos_de(principal_id: int, caminho: Path | None = None) -> VideoStatistics:
    """Conta todos os videos por estado em uma consulta, sem depender da pagina carregada."""
    with conexao(caminho) as conn:
        linha = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN estado = ? THEN 1 ELSE 0 END) AS pendentes, "
            "SUM(CASE WHEN estado = ? THEN 1 ELSE 0 END) AS em_processamento, "
            "SUM(CASE WHEN estado = ? THEN 1 ELSE 0 END) AS finalizadas, "
            "SUM(CASE WHEN estado = ? THEN 1 ELSE 0 END) AS falhas "
            "FROM videos WHERE principal_id = ?",
            (PENDENTE, PROCESSANDO, FINALIZADO, FALHOU, principal_id),
        ).fetchone()
    return VideoStatistics(
        total=int(linha["total"] or 0),
        pendentes=int(linha["pendentes"] or 0),
        em_processamento=int(linha["em_processamento"] or 0),
        finalizadas=int(linha["finalizadas"] or 0),
        falhas=int(linha["falhas"] or 0),
    )
