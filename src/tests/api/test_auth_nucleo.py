"""A prova das credenciais, sem HTTP no meio.

Cada afirmacao aqui e uma coisa que, quebrada, deixa entrar quem nao deveria. Sao os testes que
eu mais quero que falhem alto, porque a falha silenciosa deles e alguem entrando.
"""

from __future__ import annotations

import ast
import base64
from pathlib import Path

import pytest

from api import auth


@pytest.fixture(autouse=True)
def segredo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(auth.SEGREDO_ENV, "segredo-de-teste-nao-usar-em-lugar-nenhum")


def test_the_service_refuses_to_sign_without_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Segredo padrao e pior que segredo ausente: funciona, ninguem repara, e vai para producao."""
    monkeypatch.delenv(auth.SEGREDO_ENV, raising=False)

    with pytest.raises(auth.SegredoAusente, match=auth.SEGREDO_ENV):
        auth.assinar(1)


def test_a_password_never_appears_in_what_is_stored() -> None:
    guardado = auth.hash_de_senha("senha-secreta-do-usuario")

    assert "senha-secreta-do-usuario" not in guardado
    assert guardado.startswith("scrypt$")


def test_the_same_password_gives_different_hashes() -> None:
    """Sem sal por senha, duas pessoas com a mesma senha ficam com o mesmo hash no banco."""
    assert auth.hash_de_senha("igual") != auth.hash_de_senha("igual")


def test_the_right_password_passes_and_the_wrong_one_does_not() -> None:
    guardado = auth.hash_de_senha("certa")

    assert auth.conferir_senha("certa", guardado)
    assert not auth.conferir_senha("errada", guardado)
    assert not auth.conferir_senha("", guardado)


@pytest.mark.parametrize(
    "corrompido",
    ["", "nao-tem-cifrao", "scrypt$1$2$3$sal", "outro$16384$8$1$c2Fs$aGFzaA", "$$$$$"],
)
def test_a_malformed_stored_hash_refuses_instead_of_exploding(corrompido: str) -> None:
    """Linha corrompida no banco nao pode virar excecao nao tratada no caminho de login."""
    assert not auth.conferir_senha("qualquer", corrompido)


def test_a_session_token_round_trips() -> None:
    assert auth.verificar(auth.assinar(42)) == 42


def test_a_tampered_token_is_refused() -> None:
    """Trocar o principal dentro do token e a tentativa obvia, e assinar de novo e o que
    quem tenta nao consegue fazer sem o segredo.
    """
    _, assinatura = auth.assinar(1).split(".")
    outro = base64.urlsafe_b64encode(b'{"exp":99999999999,"sub":999}').decode().rstrip("=")

    with pytest.raises(auth.TokenInvalido, match="assinatura"):
        auth.verificar(f"{outro}.{assinatura}")


def test_a_token_signed_with_another_secret_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    token = auth.assinar(7)
    monkeypatch.setenv(auth.SEGREDO_ENV, "outro-segredo-completamente-diferente")

    with pytest.raises(auth.TokenInvalido, match="assinatura"):
        auth.verificar(token)


def test_an_expired_session_is_refused() -> None:
    """Sessao sem expiracao e credencial permanente: um token vazado vale para sempre."""
    antigo = auth.assinar(3, agora=1_000_000)

    with pytest.raises(auth.TokenInvalido, match="expirou"):
        auth.verificar(antigo, agora=1_000_000 + auth.validade() + 1)

    assert auth.verificar(antigo, agora=1_000_000 + 1) == 3


@pytest.mark.parametrize("lixo", ["", "sem-ponto", "a.b.c", "..", "!!!.???"])
def test_garbage_is_refused_as_a_token(lixo: str) -> None:
    with pytest.raises(auth.AuthError):
        auth.verificar(lixo)


def test_two_service_tokens_are_never_the_same() -> None:
    assert len({auth.novo_token() for _ in range(200)}) == 200


def test_a_service_token_has_enough_entropy_to_not_be_guessed() -> None:
    """Nao basta serem diferentes entre si: precisam ser impossiveis de adivinhar.

    Encolher o sorteio nao quebra nenhum teste de igualdade, porque duzentas amostras de um
    espaco pequeno ainda saem distintas. O que quebra e o custo de forca bruta, e esse so aparece
    se alguem afirmar o tamanho. Um token de 3 bytes tem 16 milhoes de valores, que um script
    percorre numa tarde.
    """
    token = auth.novo_token()

    assert len(token) >= 40, f"o token tem {len(token)} caracteres, e isso e adivinhavel"


def test_the_stored_shape_of_a_service_token_is_not_the_token() -> None:
    """O banco guarda o resumo. Vazado, ele nao serve para chamar a API."""
    token = auth.novo_token()

    assert auth.hash_de_token(token) != token
    assert auth.hash_de_token(token) == auth.hash_de_token(token)


def test_every_secret_comparison_uses_a_constant_time_primitive() -> None:
    """Este e estrutural porque o defeito nao muda resultado nenhum, so o tempo.

    Trocar `hmac.compare_digest` por `==` passa em todo teste funcional: a senha certa continua
    aceita e a errada continua recusada. O que muda e que `==` volta no primeiro byte diferente,
    e o tempo de resposta passa a contar quantos bytes o palpite acertou. Medir isso num teste
    seria flaky por construcao, entao o que se cobra e a primitiva.

    A ultima asserção e a que faz o resto valer. Sem ela a varredura passava calada quando nao
    achava as funcoes, e uma revisao provou: renomeando as duas e deixando alias com o nome
    antigo, o arquivo ficou com zero `compare_digest` e os 23 testes deste arquivo passaram.
    """
    fonte = (Path(auth.__file__)).read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    alvos = {"conferir_senha", "verificar"}
    vistas: set[str] = set()

    for funcao in ast.walk(arvore):
        if not isinstance(funcao, ast.FunctionDef) or funcao.name not in alvos:
            continue
        vistas.add(funcao.name)
        comparacoes = [
            no
            for no in ast.walk(funcao)
            if isinstance(no, ast.Compare)
            and any(isinstance(op, ast.Eq | ast.NotEq) for op in no.ops)
        ]
        for comparacao in comparacoes:
            texto = ast.get_source_segment(fonte, comparacao) or ""
            assert "compare_digest" in texto or "marca" in texto or "corpo" in texto, (
                f"{funcao.name} compara com `==` em {texto!r}: use hmac.compare_digest para o "
                "que for segredo, ou o tempo de resposta conta quanto o palpite acertou"
            )
        chamadas = {
            no.func.attr
            for no in ast.walk(funcao)
            if isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute)
        }
        assert "compare_digest" in chamadas, f"{funcao.name} nao usa comparacao em tempo constante"

    assert vistas == alvos, (
        f"a varredura nao encontrou {sorted(alvos - vistas)}, entao ela nao afirmou nada sobre "
        "elas. Teste que procura e nao acha passa calado, e foi o que este fazia: renomear as "
        "duas funcoes e deixar alias com o nome antigo zerava o `compare_digest` do arquivo "
        "inteiro com a suite verde"
    )


def test_a_session_that_expires_exactly_now_is_already_invalid() -> None:
    """A fronteira exata: `exp == agora` conta como expirada, e nao como o ultimo instante valido.

    Trocar `<=` por `<` sobrevivia a suite inteira, porque nenhum teste tocava o instante exato.
    Um segundo a mais numa sessao nao derruba nada sozinho, mas a fronteira e onde o proximo
    defeito se esconde: quem mexer nesta linha depois nao tem como saber qual dos dois era.
    """
    emitido_em = 1_000_000
    token = auth.assinar(1, agora=emitido_em)
    expira_em = emitido_em + auth.validade()

    assert auth.verificar(token, agora=expira_em - 1) == 1

    with pytest.raises(auth.TokenInvalido, match="expirou"):
        auth.verificar(token, agora=expira_em)
