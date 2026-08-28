"""O navegador de quadros: o que cada tecla faz e o que ele grava.

A janela nao entra em teste, porque nenhum servidor tem uma e o CI roda no wheel headless. O que
entra e o resto, que e onde mora o erro caro: tecla que anda para o lado errado, indice que escapa
do clipe, e o pulo virando anotacao de quadro zero. Um pulo gravado como zero poe uma linha falsa
no gabarito, e o gabarito e o que mede todo o resto.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contracts import Clip
from mlops import scrub
from mlops.instant import read

CLIP = Clip(clip_id="c1", path="c1.mp4", fps=60.0, frame_count=100, width=320, height=180)
OUTRO = Clip(clip_id="c2", path="c2.mp4", fps=60.0, frame_count=100, width=320, height=180)


class _Captura:
    """Quadro preto, e anota o indice pedido para o teste ver por onde a pessoa andou."""

    def __init__(self) -> None:
        self.pedidos: list[int] = []
        self._proximo = 0

    def isOpened(self) -> bool:  # noqa: N802 - a assinatura e a do cv2
        return True

    def set(self, _prop: int, valor: float) -> None:
        self._proximo = int(valor)
        self.pedidos.append(int(valor))

    def read(self) -> tuple[bool, Any]:
        return True, np.zeros((180, 320, 3), dtype=np.uint8)

    def release(self) -> None: ...


def _nada(*_args: object, **_kwargs: object) -> None: ...


@pytest.fixture
def janela(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Captura]:
    captura = _Captura()
    monkeypatch.setattr(scrub, "tem_janela", lambda: True)

    def abrir(_caminho: str) -> _Captura:
        return captura

    monkeypatch.setattr(scrub.cv2, "VideoCapture", abrir)
    monkeypatch.setattr(scrub.cv2, "imshow", _nada)
    monkeypatch.setattr(scrub.cv2, "destroyWindow", _nada)
    yield captura


def _teclas(monkeypatch: pytest.MonkeyPatch, sequencia: list[int]) -> None:
    restantes = iter(sequencia)

    def proxima(_espera: int) -> int:
        return next(restantes)

    monkeypatch.setattr(scrub.cv2, "waitKeyEx", proxima)


def test_espaco_marca_o_quadro_em_que_a_pessoa_parou(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch
) -> None:
    _teclas(monkeypatch, [83, 83, 83, scrub.MARCAR])

    decisao = scrub.scrub(CLIP, sugestao=40)

    assert decisao.frame_index == 43


def test_seta_para_a_esquerda_volta_um_quadro(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch
) -> None:
    _teclas(monkeypatch, [81, scrub.MARCAR])

    assert scrub.scrub(CLIP, sugestao=40).frame_index == 39


def test_a_e_d_andam_de_dez_em_dez(janela: _Captura, monkeypatch: pytest.MonkeyPatch) -> None:
    _teclas(monkeypatch, [ord("d"), ord("d"), ord("a"), scrub.MARCAR])

    assert scrub.scrub(CLIP, sugestao=40).frame_index == 50


def test_nao_passa_do_ultimo_quadro(janela: _Captura, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem o limite o `set` pede quadro inexistente, o `read` falha e o laco nunca sai."""
    _teclas(monkeypatch, [ord("d")] * 5 + [scrub.MARCAR])

    assert scrub.scrub(CLIP, sugestao=98).frame_index == CLIP.frame_count - 1


def test_nao_passa_do_primeiro_quadro(janela: _Captura, monkeypatch: pytest.MonkeyPatch) -> None:
    _teclas(monkeypatch, [ord("a")] * 3 + [scrub.MARCAR])

    assert scrub.scrub(CLIP, sugestao=5).frame_index == 0


def test_pular_nao_vira_anotacao_de_quadro_zero(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clipe sem lance visivel tem que sair do gabarito, e nao entrar nele com um palpite."""
    _teclas(monkeypatch, [scrub.PULAR])

    assert scrub.scrub(CLIP, sugestao=40).frame_index is None


def test_sair_tambem_nao_grava(janela: _Captura, monkeypatch: pytest.MonkeyPatch) -> None:
    _teclas(monkeypatch, [scrub.SAIR])

    assert scrub.scrub(CLIP, sugestao=40).frame_index is None


def test_sem_sugestao_comeca_no_meio(janela: _Captura, monkeypatch: pytest.MonkeyPatch) -> None:
    _teclas(monkeypatch, [scrub.MARCAR])

    assert scrub.scrub(CLIP).frame_index == CLIP.frame_count // 2


def test_voltar_reabre_o_clipe_anterior(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pular fica ao lado de marcar, e sem volta um toque errado tira o clipe da amostra calado."""
    monkeypatch.setattr("mlops.instant.var_dir", lambda: tmp_path)
    _teclas(monkeypatch, [scrub.PULAR, scrub.VOLTAR, scrub.MARCAR, scrub.MARCAR])
    marcados = scrub.anotar([(CLIP, 40), (OUTRO, 10)], "davi")

    assert marcados == 2
    assert sorted(a.clip_id for a in read(tmp_path)) == ["c1", "c2"]


def test_voltar_no_primeiro_clipe_fica_no_primeiro(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("mlops.instant.var_dir", lambda: tmp_path)
    _teclas(monkeypatch, [scrub.VOLTAR, scrub.VOLTAR, scrub.MARCAR])

    assert scrub.anotar([(CLIP, 40)], "davi") == 1


def test_remarcar_o_mesmo_clipe_conta_uma_vez_so(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """O log guarda as duas linhas, mas o clipe tem um quadro so, e o relatorio le o ultimo."""
    from mlops.instant import consensus

    monkeypatch.setattr("mlops.instant.var_dir", lambda: tmp_path)
    _teclas(monkeypatch, [scrub.MARCAR, scrub.VOLTAR, 63235, scrub.MARCAR, scrub.MARCAR])

    marcados = scrub.anotar([(CLIP, 40), (OUTRO, 10)], "davi")

    gravadas = read(tmp_path)
    assert marcados == 2
    assert [(a.clip_id, a.frame_index) for a in gravadas] == [("c1", 40), ("c1", 41), ("c2", 10)]
    assert consensus(gravadas) == {"c1": 41, "c2": 10}


def test_sair_no_meio_preserva_o_que_ja_foi_marcado(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("mlops.instant.var_dir", lambda: tmp_path)
    _teclas(monkeypatch, [scrub.MARCAR, scrub.SAIR])
    assert scrub.anotar([(CLIP, 40), (OUTRO, 10)], "davi") == 1
    assert [a.clip_id for a in read(tmp_path)] == ["c1"]


def test_voltar_abre_no_quadro_que_a_pessoa_escolheu(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Quem volta esta corrigindo, e reabrir no meio do clipe faria reencontrar o lance de novo."""
    monkeypatch.setattr("mlops.instant.var_dir", lambda: tmp_path)
    _teclas(monkeypatch, [scrub.MARCAR, scrub.VOLTAR, scrub.MARCAR, scrub.PULAR])

    scrub.anotar([(CLIP, 40), (OUTRO, 10)], "davi")

    assert [a.frame_index for a in read(tmp_path)] == [40, 40]


def test_marcar_o_ultimo_clipe_encerra_a_sessao(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nao ha para onde avancar, entao o unico jeito de rever o ultimo e rodar com `--refazer`.

    Documentado num teste em vez de resolvido com uma tela de fim: seria um estado a mais no laco
    para um caso que o proprio comando ja cobre ao ser rodado de novo.
    """
    monkeypatch.setattr("mlops.instant.var_dir", lambda: tmp_path)
    _teclas(monkeypatch, [scrub.MARCAR, scrub.VOLTAR, scrub.MARCAR])

    assert scrub.anotar([(CLIP, 40)], "davi") == 1
    assert [a.frame_index for a in read(tmp_path)] == [40]


def test_anotar_grava_so_o_que_foi_marcado(
    janela: _Captura, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("edge.paths.var_dir", lambda: tmp_path)
    monkeypatch.setattr("mlops.instant.var_dir", lambda: tmp_path)
    _teclas(monkeypatch, [scrub.MARCAR, scrub.PULAR])
    marcados = scrub.anotar([(CLIP, 40), (OUTRO, 10)], "davi")

    gravadas = read(tmp_path)
    assert marcados == 1
    assert [(a.clip_id, a.frame_index, a.annotator) for a in gravadas] == [("c1", 40, "davi")]


def test_sem_janela_diz_o_que_usar_no_lugar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scrub, "tem_janela", lambda: False)

    with pytest.raises(scrub.SemJanela, match="label sheet"):
        scrub.scrub(CLIP)


def test_a_marcacao_do_parceiro_nunca_diz_a_parte_do_corpo() -> None:
    """O `body_part` e o alvo do classificador, e mostra-lo entregaria a resposta a quem anota."""
    linha = scrub.contexto(
        {"body_part": "CABECA", "used": "head", "origin_of_move": "CRUZAMENTO", "meters": "10"}
    )

    assert "cabeca" not in linha.lower()
    assert "head" not in linha.lower()
    assert linha == "cruzamento  10 m"


def test_campo_falso_nao_vira_texto() -> None:
    """As colunas booleanas do parceiro sao o texto `false`, e imprimi-las poluiria a faixa."""
    linha = scrub.contexto(
        {"is_goal": "false", "is_blocked": "true", "savedGoalkeeper": "false", "meters": "20"}
    )

    assert linha == "20 m  bloqueado"


def test_o_desfecho_aparece_para_desambiguar_o_rebote() -> None:
    """Clipe com chute, defesa e segundo chute: sem o desfecho nao da para saber qual e o evento."""
    linha = scrub.contexto(
        {"origin_of_move": "PASSE", "penalty_area": "DENTRO_AREA", "is_goal": "true"}
    )

    assert linha == "passe  dentro area  GOL"


def test_marcacao_vazia_nao_quebra() -> None:
    assert scrub.contexto({}) == ""


def test_shift_anda_um_segundo(janela: _Captura, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem um passo grosso, atravessar 540 quadros de dez em dez sao 54 toques, vezes 40 clipes."""
    _teclas(monkeypatch, [ord("D"), scrub.MARCAR])

    assert scrub.scrub(CLIP, sugestao=20).frame_index == 80
