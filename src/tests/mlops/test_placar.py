"""O placar, e a regra de quando declarar vitoria.

O criterio nao e a media passar da referencia, e sim o limite inferior do intervalo passar. Com 14 a
38 clipes anotados a media sobe e desce por um acerto, e declarar vitoria que o proximo clipe
desfaz e pior que nao declarar nenhuma.

O outro modo de falha que estes testes fecham e a comparacao sobre conjuntos diferentes: previsao e
gabarito so podem ser comparados nos clipes que os dois tem, senao clipe sem previsao conta como
erro e a feature de menor cobertura ganha por nao ter respondido.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlops.placar import Linha, colapsa, linha_de_feature, maioria, previsoes, tabela


def test_colapsar_junta_as_duas_pernas_e_preserva_a_cabeca() -> None:
    assert colapsa("PERNA_ESQUERDA") == "PE"
    assert colapsa("PERNA_DIREITA") == "PE"
    assert colapsa("CABECA") == "CABECA"


def test_vencer_exige_o_limite_inferior_e_nao_a_media() -> None:
    """Com n pequeno a media sobe e desce por um acerto, e vitoria assim nao sobrevive."""
    media_acima = Linha("x", acertos=13, n=24, referencia=0.50, referencia_nome="maioria")
    folgada = Linha("y", acertos=95, n=100, referencia=0.50, referencia_nome="maioria")

    assert media_acima.taxa > media_acima.referencia
    assert media_acima.ganha is False
    assert folgada.ganha is True


def test_so_compara_nos_clipes_que_os_dois_tem() -> None:
    """Clipe sem previsao contado como erro faria a feature de menor cobertura ganhar."""
    previsto = {"a": "PE", "b": "CABECA"}
    verdade = {"a": "PE", "b": "PE", "c": "PE"}

    linha = linha_de_feature("t", previsto, verdade)

    assert linha is not None
    assert linha.n == 2
    assert linha.acertos == 1


def test_sem_clipe_em_comum_nao_inventa_linha() -> None:
    assert linha_de_feature("t", {"a": "PE"}, {"b": "PE"}) is None


def test_a_maioria_sai_do_gabarito_dos_clipes_comparados() -> None:
    """Sobre a populacao inteira a maioria seria outra, e a comparacao sairia errada."""
    assert maioria(["PE", "PE", "PE", "CABECA"]) == 0.75
    assert maioria([]) == 0.0


def test_previsoes_ignora_feature_indisponivel(tmp_path: Path) -> None:
    """Feature sem valor e ausencia declarada, e conta-la como previsao mediria o vazio."""
    eventos = tmp_path / "events.jsonl"
    eventos.write_text(
        json.dumps({"clip_id": "a", "features": [{"name": "body_part", "value": "CABECA"}]})
        + "\n"
        + json.dumps({"clip_id": "b", "features": [{"name": "body_part", "value": None}]})
        + "\n",
        encoding="utf-8",
    )

    assert previsoes(eventos, "body_part") == {"a": "CABECA"}


def test_execucao_inexistente_devolve_vazio_em_vez_de_estourar(tmp_path: Path) -> None:
    assert previsoes(tmp_path / "nao-existe.jsonl", "body_part") == {}


def test_a_tabela_marca_o_que_passou() -> None:
    texto = tabela(
        [
            Linha("perde", acertos=1, n=10, referencia=0.5, referencia_nome="maioria"),
            Linha("passa", acertos=95, n=100, referencia=0.5, referencia_nome="maioria"),
        ]
    )

    linhas = texto.splitlines()
    assert "PASSA" not in linhas[1]
    assert "PASSA" in linhas[2]


def test_linha_sem_clipe_nenhum_nao_divide_por_zero() -> None:
    vazia = Linha("x", acertos=0, n=0, referencia=0.5, referencia_nome="maioria")

    assert vazia.taxa == 0.0
    assert vazia.ganha is False
