"""A fronteira: quem entra, quem nao entra, e o que cada um enxerga.

O teste mais importante deste arquivo nao e nenhum caso individual, e sim a varredura que recusa
rota de negocio publicada sem credencial. Casos individuais cobrem o que existe hoje; a varredura
cobre o que alguem acrescentar amanha sem lembrar disso.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
from api import PREFIXO, auth, create_app, store
from edge.testing.make_clip import make_synthetic_clip

from .apoio import cliente_autenticado

MP4 = "video/mp4"
SENHA = "uma-senha-longa-de-verdade"

ABERTAS = {("/", "get"), ("/health", "get")}
SEM_BEARER = {
    (PREFIXO + "/videos/upload/{token}", "put"),
    (f"{PREFIXO}/auth/register", "post"),
    (f"{PREFIXO}/auth/login", "post"),
}


@pytest.fixture
def anonimo() -> Iterator[TestClient]:
    with TestClient(create_app()) as aberto:
        yield aberto


@pytest.fixture
def cliente() -> Iterator[TestClient]:
    with cliente_autenticado(create_app()) as aberto:
        yield aberto


def _rotas_de_negocio() -> list[tuple[str, str]]:
    documento = create_app().openapi()
    return [
        (caminho, metodo)
        for caminho, metodos in documento["paths"].items()
        for metodo in metodos
        if (caminho, metodo) not in ABERTAS and (caminho, metodo) not in SEM_BEARER
    ]


def _concreto(caminho: str) -> str:
    return caminho.replace("{video_id}", "1").replace("{token}", "qualquer")


@pytest.mark.parametrize(("caminho", "metodo"), _rotas_de_negocio())
def test_every_business_route_refuses_a_request_without_a_credential(
    anonimo: TestClient, caminho: str, metodo: str
) -> None:
    """Uma rota nova sem credencial e uma porta aberta que ninguem ve no diff.

    As excecoes estao nomeadas em `ABERTAS` e `SEM_BEARER`, com o motivo escrito: saude e raiz
    pertencem a implantacao, e o `PUT` e autorizado pelo proprio endereco de envio.
    """
    resposta = anonimo.request(metodo.upper(), _concreto(caminho), json={})

    assert resposta.status_code == 401, f"{metodo.upper()} {caminho} respondeu sem credencial"
    assert resposta.headers.get("WWW-Authenticate") == "Bearer"


def test_the_open_routes_stay_open(anonimo: TestClient) -> None:
    for caminho, _ in ABERTAS:
        assert anonimo.get(caminho).status_code == 200, f"{caminho} deixou de responder"


@pytest.mark.parametrize(
    "cabecalho",
    ["", "Bearer", "Bearer ", "Basic abc", "Bearer token-que-ninguem-emitiu", "bearer "],
)
def test_a_malformed_or_unknown_credential_is_refused(anonimo: TestClient, cabecalho: str) -> None:
    resposta = anonimo.get(
        f"{PREFIXO}/auth/me", headers={"Authorization": cabecalho} if cabecalho else {}
    )

    assert resposta.status_code == 401


def test_a_valid_credential_in_the_wrong_scheme_is_refused() -> None:
    """Uma credencial boa no esquema errado precisa ser recusada pelo esquema.

    Com esquema qualquer aceito, o teste de cabecalho malformado continua passando, porque
    `Basic abc` e recusado de qualquer jeito por `abc` nao ser credencial. O que revela a falha e
    mandar uma credencial que funciona num esquema que nao e o declarado.
    """
    token = auth.novo_token()
    store.criar_principal("servico", "s", token_hash=auth.hash_de_token(token))

    with TestClient(create_app()) as cliente:
        certo = cliente.get(f"{PREFIXO}/auth/me", headers={"Authorization": f"Bearer {token}"})
        errado = cliente.get(f"{PREFIXO}/auth/me", headers={"Authorization": f"Basic {token}"})

    assert certo.status_code == 200
    assert errado.status_code == 401, "a credencial passou num esquema que o servico nao declara"


def test_rotating_the_seed_token_does_not_crash_the_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trocar `XVISION_API_TOKEN` e o caso normal de rotacionar segredo, e ele nao pode derrubar.

    O nome do principal semeado e unico, entao semear de novo com um token diferente tentaria
    criar o mesmo nome duas vezes: sem a guarda pelo nome, a subida morre com o principal ja em
    uso e o servico nao sobe mais.
    """
    monkeypatch.setenv("XVISION_API_TOKEN", "primeiro-segredo-com-tamanho-suficiente-de-token")
    with TestClient(create_app()):
        pass

    monkeypatch.setenv("XVISION_API_TOKEN", "segundo-segredo-depois-da-rotacao-com-tamanho")
    with TestClient(create_app()) as cliente:
        antigo = cliente.get(
            f"{PREFIXO}/auth/me",
            headers={"Authorization": "Bearer primeiro-segredo-com-tamanho-suficiente-de-token"},
        )

    assert antigo.status_code == 200, (
        "o token semeado antigo parou de valer sozinho, e a rotacao precisa ser explicita: "
        "emita o novo e revogue o antigo"
    )


def test_a_revoked_service_token_stops_working() -> None:
    """Revogar sem redeploy e a razao de o token estar no banco, e nao numa variavel."""
    token = auth.novo_token()
    criado = store.criar_principal("servico", "s", token_hash=auth.hash_de_token(token))
    with TestClient(create_app()) as cliente:
        cabecalho = {"Authorization": f"Bearer {token}"}
        assert cliente.get(f"{PREFIXO}/auth/me", headers=cabecalho).status_code == 200

        store.revogar_principal(criado.id)

        assert cliente.get(f"{PREFIXO}/auth/me", headers=cabecalho).status_code == 401


def test_an_account_can_be_created_and_used(anonimo: TestClient) -> None:
    criacao = anonimo.post(f"{PREFIXO}/auth/register", json={"email": "a@b.c", "senha": SENHA})
    entrada = anonimo.post(f"{PREFIXO}/auth/login", json={"email": "a@b.c", "senha": SENHA})

    assert criacao.status_code == 201, criacao.text
    assert entrada.status_code == 200, entrada.text
    sessao = entrada.json()
    assert sessao["token_type"] == "bearer"
    eu = anonimo.get(
        f"{PREFIXO}/auth/me", headers={"Authorization": f"Bearer {sessao['access_token']}"}
    )
    assert eu.json() == {"id": criacao.json()["id"], "tipo": "usuario", "nome": "a@b.c"}


def test_the_same_email_cannot_have_two_accounts(anonimo: TestClient) -> None:
    corpo = {"email": "a@b.c", "senha": SENHA}
    anonimo.post(f"{PREFIXO}/auth/register", json=corpo)

    assert anonimo.post(f"{PREFIXO}/auth/register", json=corpo).status_code == 409


def test_a_short_password_is_refused(anonimo: TestClient) -> None:
    resposta = anonimo.post(f"{PREFIXO}/auth/register", json={"email": "a@b.c", "senha": "curta"})

    assert resposta.status_code == 422


def test_an_unknown_email_and_a_wrong_password_answer_the_same(anonimo: TestClient) -> None:
    """Respondendo diferente, a rota de login vira uma lista de quem tem conta."""
    anonimo.post(f"{PREFIXO}/auth/register", json={"email": "existe@b.c", "senha": SENHA})

    desconhecido = anonimo.post(
        f"{PREFIXO}/auth/login", json={"email": "nao-existe@b.c", "senha": SENHA}
    )
    errada = anonimo.post(
        f"{PREFIXO}/auth/login", json={"email": "existe@b.c", "senha": "outra-senha-longa"}
    )

    assert desconhecido.status_code == errada.status_code == 401
    assert desconhecido.json() == errada.json()


def test_a_video_of_another_principal_answers_404_and_not_403(
    cliente: TestClient, tmp_path: Path
) -> None:
    """403 transforma o identificador sequencial num oraculo.

    Contando de 1 em diante, um 403 diz "existe e nao e seu" e um 404 diz "nao existe": com o
    primeiro da para descobrir quantos videos ha e quando cada um entrou, sem ler nenhum.
    """
    video = make_synthetic_clip(tmp_path / "s.mp4").read_bytes()
    reserva = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()
    cliente.put(reserva["uploadUrl"], content=video)

    with cliente_autenticado(create_app(), nome="outro-servico") as intruso:
        status = intruso.get(f"{PREFIXO}/videos/status/{reserva['id']}")
        resultado = intruso.get(f"{PREFIXO}/videos/{reserva['id']}")

    assert status.status_code == 404
    assert resultado.status_code == 404
    assert "nao e seu" not in status.text.lower()


def test_the_owner_still_sees_their_own_video(cliente: TestClient, tmp_path: Path) -> None:
    """A guarda da guarda: recusar demais seria tao ruim quanto recusar de menos."""
    video = make_synthetic_clip(tmp_path / "s.mp4").read_bytes()
    reserva = cliente.post(
        f"{PREFIXO}/videos/upload-url", json={"fileName": "c.mp4", "contentType": MP4}
    ).json()
    cliente.put(reserva["uploadUrl"], content=video)

    assert cliente.get(f"{PREFIXO}/videos/status/{reserva['id']}").status_code == 200


def test_the_env_seed_creates_a_revocable_token_and_not_a_shared_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A comodidade da chave de API, sem o buraco dela.

    Chave de API e um token que nao esta no banco: nao da para revogar sem redeploy, e nao
    responde "quem mandou esse video". Aqui a variavel continua sendo o jeito de comecar, e o
    token nasce na tabela.
    """
    monkeypatch.setenv("XVISION_API_TOKEN", "segredo-vindo-do-ambiente-com-tamanho-de-token")

    with TestClient(create_app()) as cliente:
        cabecalho = {"Authorization": "Bearer segredo-vindo-do-ambiente-com-tamanho-de-token"}
        assert cliente.get(f"{PREFIXO}/auth/me", headers=cabecalho).status_code == 200

        linha = store.principal_por_nome("semeado-do-ambiente")
        assert linha is not None
        store.revogar_principal(linha.id)

        assert cliente.get(f"{PREFIXO}/auth/me", headers=cabecalho).status_code == 401


def test_seeding_twice_does_not_create_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subir duas vezes e o caso normal de um reinicio, e nao pode acumular principal."""
    monkeypatch.setenv("XVISION_API_TOKEN", "segredo-vindo-do-ambiente-com-tamanho-de-token")

    for _ in range(3):
        with TestClient(create_app()):
            pass

    with store.conexao() as conn:
        quantos = conn.execute(
            "SELECT COUNT(*) AS n FROM principais WHERE nome = ?", ("semeado-do-ambiente",)
        ).fetchone()["n"]
    assert quantos == 1


def test_the_service_refuses_to_start_without_a_signing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Descobrir que nao ha segredo na primeira tentativa de login e descobrir tarde demais."""
    monkeypatch.delenv("XVISION_AUTH_SECRET", raising=False)

    with pytest.raises(auth.SegredoAusente), TestClient(create_app()):
        pass


@pytest.mark.parametrize("torto", ["abc.b", "A.b", "abc.bc", "a.b", ".", "..", "x."])
def test_a_malformed_session_answers_401_and_not_500(anonimo: TestClient, torto: str) -> None:
    """Ausente, malformada, expirada e revogada respondem igual, e isso e promessa escrita.

    Base64 torto escapava de `verificar` como `binascii.Error` e virava 500: com `abc.b` dando
    500 e `abc.bc` dando 401, a diferenca entre as duas contava a quem tenta que a primeira chegou
    mais longe. Alem disso enchia o log de traceback a cada tentativa.

    Nao ha caso unicode aqui de proposito: cabecalho HTTP nao carrega nao-ASCII, entao ele nao e
    alcancavel por cliente conforme.
    """
    resposta = anonimo.get(f"{PREFIXO}/auth/me", headers={"Authorization": f"Bearer {torto}"})

    assert resposta.status_code == 401, f"{torto!r} respondeu {resposta.status_code}"


def test_a_service_token_with_a_dot_appended_does_not_break_the_service(
    anonimo: TestClient,
) -> None:
    """Anexar um ponto a um token faz ele ser lido como sessao, e isso nao pode derrubar nada."""
    token = auth.novo_token()
    store.criar_principal("servico", "s", token_hash=auth.hash_de_token(token))

    assert (
        anonimo.get(
            f"{PREFIXO}/auth/me", headers={"Authorization": f"Bearer {token}.x"}
        ).status_code
        == 401
    )


@pytest.mark.parametrize("fraca", ["x", "123", "password", "curto-demais", "a" * 31])
def test_a_weak_seed_token_refuses_to_start_instead_of_becoming_a_credential(
    monkeypatch: pytest.MonkeyPatch, fraca: str
) -> None:
    """A variavel vira credencial exatamente como esta escrita.

    Medido antes do piso: com `XVISION_API_TOKEN=x`, `Bearer x` abria a API inteira. Um token
    sorteado tem 43 caracteres, entao aceitar 1 e ter senha fraca com nome de token.
    """
    monkeypatch.setenv("XVISION_API_TOKEN", fraca)

    with pytest.raises(ValueError, match="XVISION_API_TOKEN"), TestClient(create_app()):
        pass


@pytest.mark.parametrize("branco", ["", "   ", "\t"])
def test_the_seed_is_ignored_when_the_variable_is_absent_or_blank(
    monkeypatch: pytest.MonkeyPatch, branco: str
) -> None:
    """Vazio e so espaco significam a mesma coisa que ausente: nao semear, e nao recusar a subir.

    Recusar a subir por variavel em branco puniria quem exporta a variavel vazia num compose, que
    e a forma normal de dizer "nao quero semente".
    """
    monkeypatch.setenv("XVISION_API_TOKEN", branco)

    with TestClient(create_app()):
        pass

    assert store.principal_por_nome("semeado-do-ambiente") is None


def test_two_accounts_cannot_differ_only_by_letter_case(anonimo: TestClient) -> None:
    """`UNIQUE` do SQLite compara byte a byte, entao a normalizacao mora antes dele.

    Duas contas para a mesma pessoa dividem os videos dela entre as duas, e ninguem entende por
    que metade sumiu.
    """
    primeiro = anonimo.post(
        f"{PREFIXO}/auth/register", json={"email": "Mixed@Case.com", "senha": SENHA}
    )
    segundo = anonimo.post(
        f"{PREFIXO}/auth/register", json={"email": "mixed@case.com", "senha": SENHA}
    )

    assert primeiro.status_code == 201
    assert segundo.status_code == 409, "a mesma pessoa criou duas contas trocando a caixa"
    assert primeiro.json()["nome"] == "mixed@case.com"


@pytest.mark.parametrize("vazio", ["", "   ", "\t"])
def test_an_empty_email_is_not_an_identifier(anonimo: TestClient, vazio: str) -> None:
    resposta = anonimo.post(f"{PREFIXO}/auth/register", json={"email": vazio, "senha": SENHA})

    assert resposta.status_code == 422


def test_the_login_accepts_the_email_in_any_case(anonimo: TestClient) -> None:
    """A guarda da normalizacao: se ela so valesse no registro, ninguem mais entraria."""
    anonimo.post(f"{PREFIXO}/auth/register", json={"email": "Pessoa@X.com", "senha": SENHA})

    entrada = anonimo.post(
        f"{PREFIXO}/auth/login", json={"email": "  PESSOA@x.COM  ", "senha": SENHA}
    )

    assert entrada.status_code == 200, entrada.text


def test_the_login_hashes_a_password_even_when_the_account_does_not_exist() -> None:
    """A defesa de tempo do login e estrutural, porque medi-la num teste seria flaky.

    Sem o `scrypt` no caminho do email desconhecido, a rota volta antes de derivar chave nenhuma,
    e o tempo de resposta separa "essa conta existe" de "essa conta nao existe" sozinho. O ataque
    nao precisa da senha: precisa saber quem tem conta.

    O que se cobra e que a funcao lenta esteja no caminho, e nao o tempo medido.
    """
    fonte = Path(api.__file__).read_text(encoding="utf-8")
    corpo = fonte[fonte.index("def _rota_login") : fonte.index("def _rota_eu")]

    assert "conferir_senha" in corpo
    assert "SENHA_FALSA" in corpo, (
        "o login precisa conferir uma senha mesmo sem conta, senao o tempo de resposta "
        "diz quem tem conta"
    )
    assert corpo.index("conferir_senha") < corpo.index("raise HTTPException"), (
        "a conferencia precisa acontecer antes da recusa, nao depois"
    )


def test_a_flood_of_wrong_passwords_is_cut_off_before_it_becomes_a_guess_machine(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem teto, o login e um oraculo que responde quantas vezes quiserem perguntar."""
    monkeypatch.setenv(api.TENTATIVAS_ENV, "3")
    api.zerar_tentativas()
    errada = {"email": "ninguem@inteli.edu.br", "senha": "senha-errada-mas-longa"}

    codigos = [anonimo.post(f"{PREFIXO}/auth/login", json=errada).status_code for _ in range(4)]

    assert codigos == [401, 401, 401, 429]
    assert anonimo.post(f"{PREFIXO}/auth/login", json=errada).headers["Retry-After"]


def test_the_counter_resets_when_the_password_is_finally_right(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quem erra, corrige e volta a errar nao pode herdar a contagem da tentativa anterior."""
    monkeypatch.setenv(api.TENTATIVAS_ENV, "3")
    api.zerar_tentativas()
    email = "acerta@inteli.edu.br"
    anonimo.post(f"{PREFIXO}/auth/register", json={"email": email, "senha": SENHA})
    errada = {"email": email, "senha": "senha-errada-mas-longa"}

    anonimo.post(f"{PREFIXO}/auth/login", json=errada)
    anonimo.post(f"{PREFIXO}/auth/login", json=errada)
    assert (
        anonimo.post(f"{PREFIXO}/auth/login", json={"email": email, "senha": SENHA}).status_code
        == 200
    )

    codigos = [anonimo.post(f"{PREFIXO}/auth/login", json=errada).status_code for _ in range(3)]

    assert codigos == [401, 401, 401]


def test_a_neighbour_account_is_not_locked_out_by_someone_elses_wrong_password(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teto so por origem entregaria o trancamento da conta alheia a quem divide o mesmo NAT."""
    monkeypatch.setenv(api.TENTATIVAS_ENV, "2")
    api.zerar_tentativas()
    vizinho = "vizinho@inteli.edu.br"
    anonimo.post(f"{PREFIXO}/auth/register", json={"email": vizinho, "senha": SENHA})
    alvo = {"email": "alvo@inteli.edu.br", "senha": "senha-errada-mas-longa"}

    anonimo.post(f"{PREFIXO}/auth/login", json=alvo)
    anonimo.post(f"{PREFIXO}/auth/login", json=alvo)
    assert anonimo.post(f"{PREFIXO}/auth/login", json=alvo).status_code == 429

    entrada = anonimo.post(f"{PREFIXO}/auth/login", json={"email": vizinho, "senha": SENHA})

    assert entrada.status_code == 200


def test_the_window_forgets_old_attempts_instead_of_banning_forever(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teto sem janela vira banimento permanente do endereco na primeira rajada."""
    monkeypatch.setenv(api.TENTATIVAS_ENV, "2")
    monkeypatch.setenv(api.JANELA_ENV, "60")
    api.zerar_tentativas()
    errada = {"email": "esquece@inteli.edu.br", "senha": "senha-errada-mas-longa"}

    anonimo.post(f"{PREFIXO}/auth/login", json=errada)
    anonimo.post(f"{PREFIXO}/auth/login", json=errada)
    assert anonimo.post(f"{PREFIXO}/auth/login", json=errada).status_code == 429

    daqui_a_pouco = time.monotonic() + 61
    monkeypatch.setattr(api.time, "monotonic", lambda: daqui_a_pouco)

    assert anonimo.post(f"{PREFIXO}/auth/login", json=errada).status_code == 401


def test_the_registration_is_capped_too_because_it_is_the_most_expensive_open_route(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`scrypt` mais uma linha no banco, sem credencial e sem teto, e trabalho de graca.

    A chave e so a origem: cada chamada traz um email novo, entao uma chave que o incluisse
    nunca acumularia duas tentativas e o teto nao valeria nada.
    """
    monkeypatch.setenv(api.REGISTROS_ENV, "3")
    api.zerar_tentativas()

    codigos = [
        anonimo.post(
            f"{PREFIXO}/auth/register", json={"email": f"conta-{i}@inteli.edu.br", "senha": SENHA}
        ).status_code
        for i in range(4)
    ]

    assert codigos == [201, 201, 201, 429]
    recusa = anonimo.post(
        f"{PREFIXO}/auth/register", json={"email": "mais-uma@inteli.edu.br", "senha": SENHA}
    )
    assert recusa.headers["Retry-After"]


def test_the_registration_cap_does_not_lock_the_login_of_who_just_signed_up(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Os dois baldes sao separados: estourar o registro nao pode impedir de entrar."""
    monkeypatch.setenv(api.REGISTROS_ENV, "2")
    api.zerar_tentativas()
    email = "primeira@inteli.edu.br"
    assert (
        anonimo.post(f"{PREFIXO}/auth/register", json={"email": email, "senha": SENHA}).status_code
        == 201
    )
    anonimo.post(
        f"{PREFIXO}/auth/register", json={"email": "segunda@inteli.edu.br", "senha": SENHA}
    )
    assert (
        anonimo.post(
            f"{PREFIXO}/auth/register", json={"email": "terceira@inteli.edu.br", "senha": SENHA}
        ).status_code
        == 429
    )

    entrada = anonimo.post(f"{PREFIXO}/auth/login", json={"email": email, "senha": SENHA})

    assert entrada.status_code == 200


def test_the_attempt_counter_does_not_grow_without_a_ceiling(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chave carrega o email, que quem chama escolhe: sem teto ela e memoria que so cresce."""
    monkeypatch.setenv(api.JANELA_ENV, "60")
    monkeypatch.setattr(api, "CHAVES_MAXIMAS", 3)
    api.zerar_tentativas()

    for i in range(12):
        anonimo.post(
            f"{PREFIXO}/auth/login",
            json={"email": f"email-{i}@inteli.edu.br", "senha": "senha-errada-mas-longa"},
        )

    assert len(api.chaves_de_tentativa()) <= 4, "cada email novo deixava uma entrada permanente"


def test_a_key_whose_window_expired_leaves_the_counter_instead_of_living_there_forever(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Varrer o que venceu e o que devolve a memoria de quem nunca mais voltou.

    O teto e 2 e nao 0 de proposito: com 0 o despejo por tamanho esvazia o dicionario em toda
    chamada e satisfaz a assercao sozinho, e apagar a condicao de vencimento continuaria passando.
    Com 2 e duas chaves vivas, so a poda por vencimento pode tirar a terceira.
    """
    monkeypatch.setenv(api.JANELA_ENV, "60")
    monkeypatch.setattr(api, "CHAVES_MAXIMAS", 2)
    api.zerar_tentativas()
    errada = {"email": "some-daqui@inteli.edu.br", "senha": "senha-errada-mas-longa"}

    anonimo.post(f"{PREFIXO}/auth/login", json=errada)
    assert len(api.chaves_de_tentativa()) == 1

    daqui_a_pouco = time.monotonic() + 61
    monkeypatch.setattr(api.time, "monotonic", lambda: daqui_a_pouco)
    for numero in range(3):
        anonimo.post(
            f"{PREFIXO}/auth/login",
            json={"email": f"depois-{numero}@inteli.edu.br", "senha": "senha-errada-mas-longa"},
        )

    restantes = api.chaves_de_tentativa()
    assert "some-daqui@inteli.edu.br" not in " ".join(restantes)
    assert len(restantes) <= 3, "a poda por vencimento nao rodou, e so o teto de tamanho segurou"


def test_the_expiry_sweep_runs_before_the_size_eviction_and_not_instead_of_it(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uma chave vencida com muitos carimbos nao pode empurrar para fora uma viva com poucos.

    O despejo por tamanho ordena por numero de carimbos, entao sem a poda por vencimento antes
    dele a chave morta e pesada fica e a viva e leve sai. Este e o unico cenario que distingue as
    duas metades de `_varrer`, e ele existe porque a primeira versao deste teste passava com a
    poda por vencimento apagada.
    """
    monkeypatch.setenv(api.TENTATIVAS_ENV, "10")
    monkeypatch.setenv(api.JANELA_ENV, "60")
    monkeypatch.setattr(api, "CHAVES_MAXIMAS", 1)
    api.zerar_tentativas()
    errada = "senha-errada-mas-longa"

    for _ in range(3):
        anonimo.post(
            f"{PREFIXO}/auth/login", json={"email": "pesado@inteli.edu.br", "senha": errada}
        )

    daqui_a_pouco = time.monotonic() + 61
    monkeypatch.setattr(api.time, "monotonic", lambda: daqui_a_pouco)
    anonimo.post(f"{PREFIXO}/auth/login", json={"email": "leve@inteli.edu.br", "senha": errada})
    anonimo.post(f"{PREFIXO}/auth/login", json={"email": "gatilho@inteli.edu.br", "senha": errada})

    restantes = " ".join(api.chaves_de_tentativa())

    assert "pesado@inteli.edu.br" not in restantes, (
        "a chave vencida sobreviveu por ter mais carimbos"
    )
    assert "leve@inteli.edu.br" in restantes, "a chave viva foi despejada no lugar da vencida"


def test_the_flood_does_not_evict_the_key_that_is_being_counted(
    anonimo: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Despejar pela ordem do dicionario deixava quem inunda desfazer a propria tranca.

    `dict` mantem ordem de primeira insercao, e nem `_erradas[chave] = recentes` nem o `append`
    movem a chave para o fim: a chave da vitima, criada antes de tudo, era a primeira a cair.
    """
    monkeypatch.setenv(api.TENTATIVAS_ENV, "3")
    monkeypatch.setattr(api, "CHAVES_MAXIMAS", 4)
    api.zerar_tentativas()
    alvo = {"email": "alvo@inteli.edu.br", "senha": "senha-errada-mas-longa"}

    for _ in range(3):
        anonimo.post(f"{PREFIXO}/auth/login", json=alvo)
    assert anonimo.post(f"{PREFIXO}/auth/login", json=alvo).status_code == 429

    for numero in range(20):
        anonimo.post(
            f"{PREFIXO}/auth/login",
            json={"email": f"lixo-{numero}@inteli.edu.br", "senha": "senha-errada-mas-longa"},
        )

    assert anonimo.post(f"{PREFIXO}/auth/login", json=alvo).status_code == 429, (
        "a inundacao desfez a tranca da conta que estava sendo protegida"
    )
