"""O inventario que a documentacao descreve precisa ser o que a configuracao declara.

A revisao da Sprint 3 achou tres desvios do mesmo tipo, e nenhum deles levanta excecao: a pagina de
operacao dizia que `identity` continuava stub quando ela ja era `TeamIdentity`, dizia que duas
features estavam declaradas quando eram quatro, e o comentario do `players` dizia que a bola ficava
em fp32 um dia depois de a bola passar a declarar `fp16` no mesmo arquivo.

Decisao antiga que continua parecendo atual e pior do que decisao ausente: quem le segue a pagina,
monta o comando errado e conclui que o produto e outro. E a correcao manual nao resolve, porque o
proximo commit que mudar a configuracao vai deixar a pagina para tras de novo.

O teste le a configuracao de producao como fonte da verdade e cobra que a pagina de operacao
concorde com ela. Ele nao cobra prosa: cobra que todo nome declarado apareca, e que nenhuma especie
com implementacao real seja descrita como stub.
"""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORIO = Path(__file__).resolve().parents[2]
MODELOS = REPOSITORIO / "src" / "config" / "pipeline.models.toml"
OPERACAO = REPOSITORIO / "docs" / "content" / "docs" / "sprint-3" / "produto" / "como-rodar.mdx"

SECAO_MODULO = re.compile(r"^\[modules\.(\w+)\]", re.MULTILINE)
SECAO_FEATURE = re.compile(r"^\[features\.(\w+)\]", re.MULTILINE)
IMPL = re.compile(r"^impl\s*=\s*\"([^\"]+)\"", re.MULTILINE)


def _blocos(texto: str) -> dict[str, str]:
    """Cada secao `[modules.x]` com o corpo que vem ate a proxima secao."""
    partes = re.split(r"^\[", texto, flags=re.MULTILINE)
    blocos: dict[str, str] = {}
    for parte in partes:
        nome = re.match(r"modules\.(\w+)\]", parte)
        if nome is not None:
            blocos[nome.group(1)] = parte
    return blocos


def _configuracao() -> str:
    assert (REPOSITORIO / ".gitlab-ci.yml").is_file(), f"{REPOSITORIO} nao e a raiz do repositorio"
    return MODELOS.read_text(encoding="utf-8")


def _pagina() -> str:
    return OPERACAO.read_text(encoding="utf-8")


def _stub(bloco: str) -> bool:
    achado = IMPL.search(bloco)
    return achado is not None and "stubs:" in achado.group(1)


def test_toda_feature_declarada_aparece_na_pagina_de_operacao() -> None:
    """O defeito replantado: a pagina dizia duas features quando a configuracao declarava quatro."""
    pagina = _pagina()

    faltando = [f for f in SECAO_FEATURE.findall(_configuracao()) if f"`{f}`" not in pagina]

    assert not faltando, (
        f"{OPERACAO.relative_to(REPOSITORIO)} nao cita as features {faltando}, "
        f"que {MODELOS.relative_to(REPOSITORIO)} declara"
    )


def test_especie_com_implementacao_real_nao_e_descrita_como_stub() -> None:
    """O defeito replantado: `identity` virou `TeamIdentity` e a pagina continuou dizendo stub."""
    blocos = _blocos(_configuracao())
    reais = sorted(nome for nome, bloco in blocos.items() if not _stub(bloco))

    enganosas = [
        f"{nome} (linha {numero})"
        for numero, linha in enumerate(_pagina().splitlines(), start=1)
        if "stub" in linha.lower()
        for nome in reais
        if f"`{nome}`" in linha
    ]

    assert not enganosas, (
        f"{OPERACAO.relative_to(REPOSITORIO)} descreve como stub uma especie que "
        f"{MODELOS.relative_to(REPOSITORIO)} implementa de verdade: {enganosas}"
    )


def test_especie_que_e_stub_continua_declarada_como_stub() -> None:
    """A guarda vale nos dois sentidos, senao ela empurra a pagina para o otimismo."""
    blocos = _blocos(_configuracao())
    stubs = sorted(nome for nome, bloco in blocos.items() if _stub(bloco))
    pagina = _pagina()

    assert stubs, "a configuracao de producao deixou de ter stub: este teste precisa ser revisto"
    esquecidas = [nome for nome in stubs if f"`{nome}`" not in pagina]

    assert not esquecidas, (
        f"{OPERACAO.relative_to(REPOSITORIO)} nao menciona {esquecidas}, "
        "que continuam stub na configuracao de producao"
    )


def test_o_comentario_de_uma_especie_nao_contradiz_a_precisao_que_ela_declara() -> None:
    """O defeito replantado: o bloco do `players` afirmava fp32 para a bola.

    A contradicao vivia no mesmo arquivo, 40 linhas acima da linha que declarava `fp16`, e passou
    por revisao humana duas vezes.
    """
    texto = _configuracao()
    blocos = _blocos(texto)
    meia = {nome for nome, bloco in blocos.items() if 'quantize = "fp16"' in bloco}

    assert meia, "nenhuma especie declara meia precisao: este teste precisa ser revisto"
    contradicoes = [
        f"{nome}: {linha.strip()}"
        for nome in meia
        for linha in blocos[nome].splitlines()
        if linha.lstrip().startswith("#") and re.search(rf"\b{nome}\b.*fica em fp32", linha)
    ]

    assert not contradicoes, (
        f"{MODELOS.relative_to(REPOSITORIO)} tem comentario afirmando fp32 numa especie que "
        f"declara fp16: {contradicoes}"
    )
