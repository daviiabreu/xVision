"""A feature `type_of_play`: a arvore de tres ramos sobre a trajetoria da bola antes do contato.

O ramo `individual` e o limite declarado da feature: o enum do parceiro colapsa padrao de jogo e
assistido contra individual, e nenhuma regra sobre a bola separa lance individual de jogada aberta
sem assistencia. Os testes fixam o que a regra promete, e nao mais que isso.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from contracts import (
    BallDetection,
    Box,
    FramePerception,
    Perception,
    PlayerDetection,
)
from edge.features.base import ShotActions
from edge.features.evidence_model import VALOR, Column, EvidenceModel
from edge.features.type_of_play import ABERTA, INDIVIDUAL, PARADA, TypeOfPlay

CONTATO = 120


class _Shot:
    """O duble padrao nao declara procedencia, igual aos outros dez dubles do projeto."""

    def __init__(self, frame_index: int = CONTATO) -> None:
        self.frame_index = frame_index
        self.shooter_track_id = None


class _ShotComProcedencia(_Shot):
    def __init__(self, source: str, frame_index: int = CONTATO) -> None:
        super().__init__(frame_index)
        self.source = source
        self.confidence = 0.05 if source == "midpoint" else 0.9
        self.ball_support = "absent" if source == "midpoint" else "detected"


def _quadro(indice: int, centro: tuple[float, float]) -> FramePerception:
    x, y = centro
    return FramePerception(
        frame_index=indice,
        timestamp_s=indice / 59.94,
        ball=BallDetection(box=Box(x - 5, y - 5, x + 5, y + 5), confidence=0.9),
    )


def _trajetoria(passos: list[tuple[int, float, float]]) -> Perception:
    return Perception(clip_id="c", frames=tuple(_quadro(i, (x, y)) for i, x, y in passos))


def _parada_ate(contato: int, quantos: int, passo: int = 12) -> list[tuple[int, float, float]]:
    return [(contato - i * passo, 500.0, 300.0) for i in range(quantos)][::-1]


def test_bola_imovel_por_um_trecho_longo_e_bola_parada() -> None:
    percepcao = _trajetoria(_parada_ate(CONTATO, 5))

    valor = TypeOfPlay().extract(percepcao, _Shot())

    assert valor.status == "ok"
    assert valor.value == PARADA


def test_bola_chegando_rapido_de_outro_lugar_e_jogada_aberta() -> None:
    passos = [(CONTATO - i * 12, 500.0 - i * 200.0, 300.0) for i in range(5)][::-1]

    assert TypeOfPlay().extract(_trajetoria(passos), _Shot()).value == ABERTA


def test_bola_vindo_devagar_junto_do_finalizador_e_lance_individual() -> None:
    passos = [(CONTATO - i * 12, 500.0 - i * 36.0, 300.0) for i in range(5)][::-1]

    assert TypeOfPlay().extract(_trajetoria(passos), _Shot()).value == INDIVIDUAL


def test_sem_bola_suficiente_na_janela_sai_unavailable() -> None:
    percepcao = _trajetoria([(CONTATO, 500.0, 300.0)])

    valor = TypeOfPlay().extract(percepcao, _Shot())

    assert valor.status == "unavailable"
    assert valor.value is None
    assert valor.reason


def test_bola_fora_da_janela_anterior_nao_conta() -> None:
    """A janela e a mesma de 1,5 s que o frameset declara: antes disso e outra jogada."""
    passos = [(CONTATO - 400 + i * 12, 500.0, 300.0) for i in range(5)]

    assert TypeOfPlay().extract(_trajetoria(passos), _Shot()).status == "unavailable"


def test_a_contagem_de_imoveis_nao_soma_trechos_separados_por_movimento() -> None:
    """Somar os dois trechos parados diria 60 quadros, e a contagem para no primeiro movimento.

    Todas as amostras cabem na janela de propósito. A versão anterior deste teste punha a
    primeira em `CONTATO - 96`, fora dos 90 quadros da janela, e ela era descartada em silêncio.
    """
    passos = [
        (CONTATO - 84, 500.0, 300.0),
        (CONTATO - 72, 500.0, 300.0),
        (CONTATO - 60, 500.0, 300.0),
        (CONTATO - 48, 100.0, 300.0),
        (CONTATO - 36, 100.0, 300.0),
        (CONTATO - 24, 100.0, 300.0),
    ]

    valor = TypeOfPlay().extract(_trajetoria(passos), _Shot())

    assert int(cast("int", valor.evidence["quadros_imoveis"])) == 24, "somou os dois trechos"
    assert valor.evidence["quadros_lidos"] == 6, "alguma amostra caiu fora da janela"


def test_trecho_imovel_curto_no_fim_nao_e_bola_parada() -> None:
    """O teste anterior media a contagem e nao a classe, e passava chamando `set piece` de outra
    coisa: com o trecho final em exatamente `PARADA_MIN_QUADROS` a classe sai bola parada mesmo.

    Aqui o trecho final tem 12 quadros, metade do limiar, e a classe precisa mudar.
    """
    passos = [
        (CONTATO - 72, 900.0, 300.0),
        (CONTATO - 60, 600.0, 300.0),
        (CONTATO - 48, 300.0, 300.0),
        (CONTATO - 12, 100.0, 300.0),
        (CONTATO, 100.0, 300.0),
    ]

    valor = TypeOfPlay().extract(_trajetoria(passos), _Shot())

    assert valor.value != PARADA
    assert int(cast("int", valor.evidence["quadros_imoveis"])) == 12


def test_amostra_fora_da_janela_e_descartada_em_silencio() -> None:
    """Comportamento real e nao desejado, fixado porque ele já enganou um teste desta suíte.

    A janela é `[contato - 90, contato]`, e amostra anterior a isso não entra nem aparece na
    evidência. Quem monta cenário de teste com passo largo tropeça nisso sem receber aviso.
    """
    dentro = [(CONTATO - i * 12, 500.0, 300.0) for i in range(6)][::-1]
    com_uma_fora = [(CONTATO - 96, 500.0, 300.0), *dentro]

    so_dentro = TypeOfPlay().extract(_trajetoria(dentro), _Shot())
    com_fora = TypeOfPlay().extract(_trajetoria(com_uma_fora), _Shot())

    assert com_fora.evidence["quadros_lidos"] == so_dentro.evidence["quadros_lidos"] == 6
    assert com_fora.evidence == so_dentro.evidence


def test_a_velocidade_sai_por_quadro_e_nao_por_amostra() -> None:
    """A especie ball roda a 5 fps: dividir pela posicao na lista daria doze vezes a velocidade."""
    espacado = [(CONTATO - i * 12, 500.0 - i * 36.0, 300.0) for i in range(5)][::-1]

    valor = TypeOfPlay().extract(_trajetoria(espacado), _Shot())

    assert valor.evidence["velocidade_de_chegada_px_por_quadro"] == 3.0


def test_a_evidencia_diz_como_a_leitura_foi_feita() -> None:
    valor = TypeOfPlay().extract(_trajetoria(_parada_ate(CONTATO, 5)), _Shot())

    assert valor.evidence["quadros_lidos"] == 5
    assert "imovel" in str(valor.evidence["leitura"])


def test_o_valor_cabe_no_dominio_que_o_rotulo_declara() -> None:
    valor = TypeOfPlay().extract(_trajetoria(_parada_ate(CONTATO, 5)), _Shot())

    assert valor.value in {ABERTA, PARADA, INDIVIDUAL}


class TestGateDoPontoMedio:
    """A feature le a trajetoria antes do contato, e no caminho `midpoint` nao existe um antes.

    Esse caminho devolve o quadro do meio do clipe quando a trajetoria nao sustenta quebra nenhuma.
    A janela anterior a ele e um trecho arbitrario do clipe, e a classe que sai dali chega ao
    consumidor com a mesma cara de uma classe medida.
    """

    def _percepcao(self) -> Perception:
        return _trajetoria(_parada_ate(CONTATO, 5))

    def test_instante_no_ponto_medio_sai_indisponivel_com_motivo(self) -> None:
        valor = TypeOfPlay().extract(self._percepcao(), _ShotComProcedencia("midpoint"))

        assert valor.status == "unavailable"
        assert valor.value is None
        assert valor.reason is not None and "midpoint" in valor.reason

    def test_instante_medido_continua_valendo(self) -> None:
        valor = TypeOfPlay().extract(self._percepcao(), _ShotComProcedencia("ball-break"))

        assert valor.status == "ok" and valor.value == PARADA

    def test_instante_sem_procedencia_nao_e_tratado_como_ponto_medio(self) -> None:
        """Recusar por falta de declaracao recusaria todo clipe de um detector que nao declara, e
        tratar ausencia como medida publicaria confianca que ninguem mediu. Ausencia segue a
        regra, e quem separa os dois casos e a procedencia no registro do instante."""
        valor = TypeOfPlay().extract(self._percepcao(), _Shot())

        assert valor.status == "ok" and valor.value == PARADA

    def test_procedencia_desconhecida_tambem_nao_e_ponto_medio(self) -> None:
        valor = TypeOfPlay().extract(self._percepcao(), _ShotComProcedencia("inventada"))

        assert valor.status == "ok" and valor.value == PARADA

    def test_o_gate_vem_antes_da_contagem_de_amostras(self) -> None:
        """Com poucas amostras e `midpoint`, o motivo precisa ser o `midpoint`: dizer "poucas
        amostras" mandaria quem le procurar o defeito no detector de bola."""
        curta = _trajetoria([(CONTATO - 12, 500.0, 300.0)])

        valor = TypeOfPlay().extract(curta, _ShotComProcedencia("midpoint"))

        assert valor.reason is not None and "midpoint" in valor.reason


ANTIGAS = (
    "quadros_lidos",
    "quadros_imoveis",
    "velocidade_de_chegada_px_por_quadro",
    "leitura",
)


class TestEvidenciaEmUnidadesReais:
    """A evidencia ganha segundos e alturas de jogador ao lado das chaves em pixel por quadro.

    Ao lado e nao no lugar: trocar unidade e trocar limiar na mesma rodada tornaria impossivel
    separar o efeito de um do efeito do outro. O golden das sete percepcoes prende as antigas.
    """

    def _com_jogadores(self, alturas: float) -> Perception:
        quadros: list[FramePerception] = []
        for i in range(6):
            indice = CONTATO - (5 - i) * 12
            x = 100.0 + i * 60.0
            quadros.append(
                FramePerception(
                    frame_index=indice,
                    timestamp_s=indice / 59.94,
                    ball=BallDetection(
                        box=Box(x - 5, 295.0, x + 5, 305.0), confidence=0.9, source="detected"
                    ),
                    players=(
                        PlayerDetection(
                            box=Box(400.0, 300.0 - alturas / 2, 440.0, 300.0 + alturas / 2),
                            confidence=0.8,
                            track_id=7,
                        ),
                    ),
                )
            )
        return Perception(clip_id="c", frames=tuple(quadros))

    def test_as_acoes_do_tdeed_entram_na_evidencia_sem_mudar_a_classe(self) -> None:
        cena = self._com_jogadores(100.0)
        chute = _Shot()
        chute.actions = ShotActions(  # pyright: ignore[reportAttributeAccessIssue]
            classes=("BACKGROUND", "CROSS", "SHOT"),
            frame_indices=(CONTATO - 60, CONTATO),
            probabilities=((0.2, 0.8, 0.0), (0.1, 0.0, 0.9)),
        )

        com_acoes = TypeOfPlay().extract(cena, chute)
        sem_acoes = TypeOfPlay().extract(cena, _Shot())

        assert com_acoes.value == sem_acoes.value
        assert com_acoes.evidence["ultima_acao"] == "CROSS"
        assert sem_acoes.evidence["acoes_antes_do_chute"] is None

    def test_publica_as_chaves_novas_sem_mexer_nas_antigas(self) -> None:
        valor = TypeOfPlay().extract(self._com_jogadores(100.0), _Shot())

        assert valor.status == "ok"
        for chave in ANTIGAS:
            assert chave in valor.evidence
        for chave in (
            "imobilidade_s",
            "caminho_alturas",
            "velocidade_de_chegada_alturas_por_s",
            "fracao_observada",
            "toque_anterior",
            "toque_anterior_do_finalizador",
            "idade_do_toque_anterior_s",
            "corte_de_camera",
            "procedencia_do_instante",
            "apoio_da_bola",
            "escala_alturas_px",
            "quadros_na_janela",
            "bola_detectada",
            "episodios_de_posse",
        ):
            assert chave in valor.evidence, chave

    def test_a_escala_muda_a_medida_em_alturas_e_nao_a_em_pixel(self) -> None:
        """A prova de que a unidade nova mede outra coisa: dobrar a altura dos jogadores divide o
        caminho em alturas pela metade e nao toca na velocidade em pixel por quadro."""
        normal = TypeOfPlay().extract(self._com_jogadores(100.0), _Shot())
        altos = TypeOfPlay().extract(self._com_jogadores(200.0), _Shot())

        caminho_normal = cast("float", normal.evidence["caminho_alturas"])
        caminho_altos = cast("float", altos.evidence["caminho_alturas"])
        assert caminho_normal == pytest.approx(caminho_altos * 2.0)
        assert normal.evidence["escala_alturas_px"] == 100.0
        assert altos.evidence["escala_alturas_px"] == 200.0
        for chave in ANTIGAS:
            assert normal.evidence[chave] == altos.evidence[chave]

    def test_jogador_na_percepcao_nao_muda_a_decisao_da_regra_legada(self) -> None:
        """A regra continua lendo so a bola. Se `players` mudasse a classe, a feature teria passado
        a depender de um modulo que `requires` nao declara."""
        com = self._com_jogadores(100.0)
        sem = Perception(
            clip_id="c",
            frames=tuple(
                FramePerception(frame_index=q.frame_index, timestamp_s=q.timestamp_s, ball=q.ball)
                for q in com.frames
            ),
        )

        a = TypeOfPlay().extract(com, _Shot())
        b = TypeOfPlay().extract(sem, _Shot())

        assert a.value == b.value
        for chave in ANTIGAS:
            assert a.evidence[chave] == b.evidence[chave]

    def test_sem_jogador_a_medida_em_alturas_sai_ausente_e_nao_zero(self) -> None:
        """Zero se le como bola parada e sem deslocamento, entao clipe sem jogador sairia igual a
        clipe com a bola quieta no chao."""
        sem = Perception(
            clip_id="c",
            frames=tuple(
                FramePerception(frame_index=q.frame_index, timestamp_s=q.timestamp_s, ball=q.ball)
                for q in self._com_jogadores(100.0).frames
            ),
        )

        valor = TypeOfPlay().extract(sem, _Shot())

        assert valor.status == "ok"
        assert valor.evidence["escala_alturas_px"] is None
        assert valor.evidence["caminho_alturas"] is None
        assert valor.evidence["imobilidade_s"] is None
        assert valor.evidence["fracao_observada"] == 1.0, "o que foi observado ainda e medido"

    def test_a_evidencia_e_exatamente_este_conjunto_de_chaves(self) -> None:
        """A evidencia sai no `ShotEvent`, e conteudo restrito do parceiro nao atravessa essa
        fronteira. Prender o conjunto exato, e nao uma lista de proibidos, e o que pega a chave
        nova que alguem acrescente: uma lista de proibidos so pega os nomes que alguem previu.
        """
        valor = TypeOfPlay().extract(self._com_jogadores(100.0), _Shot())

        assert set(valor.evidence) == {
            *ANTIGAS,
            "imobilidade_s",
            "caminho_alturas",
            "velocidade_de_chegada_alturas_por_s",
            "fracao_observada",
            "toque_anterior",
            "toque_anterior_do_finalizador",
            "toque_anterior_relacao",
            "idade_do_toque_anterior_s",
            "corte_de_camera",
            "procedencia_do_instante",
            "apoio_da_bola",
            "escala_alturas_px",
            "quadros_na_janela",
            "bola_detectada",
            "episodios_de_posse",
            "acoes_antes_do_chute",
            "ultima_acao",
            "ultima_acao_antecedencia_s",
            "reinicio_s",
            "reinicio_antecedencia_s",
            "plano_fechado_antecedencia_s",
            "ultimo_corte_antecedencia_s",
        }

    def test_a_evidencia_nao_carrega_o_track_do_jogador(self) -> None:
        """O `track_id` 7 da cena nao pode aparecer em valor nenhum: ele identifica pessoa dentro
        do clipe do parceiro, e o que o consumidor precisa e se o toque foi do finalizador."""
        cena = self._com_jogadores(100.0)
        valor = TypeOfPlay().extract(cena, _Shot())

        assert valor.evidence["toque_anterior_do_finalizador"] in (True, False, None)
        assert 7 not in [
            v for v in valor.evidence.values() if isinstance(v, int) and not isinstance(v, bool)
        ]

    def test_a_evidencia_atravessa_o_contrato_com_nulo_dentro(self) -> None:
        """`_freeze` recusa tipo que o json nao serializa, e recusa nulo derrubaria o clipe todo."""
        from contracts import FeatureValue as Contrato

        valor = TypeOfPlay().extract(self._com_jogadores(100.0), _Shot())
        congelado = Contrato(
            name=valor.name,
            version=valor.version,
            status=valor.status,
            value=valor.value,
            confidence=valor.confidence,
            reason=valor.reason,
            evidence=dict(valor.evidence),
        )

        assert congelado.evidence["quadros_lidos"] == valor.evidence["quadros_lidos"]


class TestFronteirasDosLimiares:
    """Cada limiar preso por cima e por baixo, porque a suite anterior os deixava soltos.

    Medido por mutacao antes de escrever esta classe: `PARADA_MIN_QUADROS` valia 24 e a suite
    passava inteira com qualquer valor de 13 a 48, um fator de 3,7 de liberdade nao detectada no
    proprio numero que decide bola parada. Os outros quatro limiares e as duas comparacoes de
    fronteira estavam na mesma situacao.
    """

    def _imovel_por(self, quadros: int) -> Perception:
        """Trajetoria que anda antes e fica exatamente `quadros` parada antes do contato."""
        passos = [
            (CONTATO - quadros - 24, 900.0, 300.0),
            (CONTATO - quadros - 12, 500.0, 300.0),
            (CONTATO - quadros, 100.0, 300.0),
            (CONTATO, 100.0, 300.0),
        ]
        return _trajetoria(passos)

    def test_exatamente_no_limiar_e_bola_parada(self) -> None:
        valor = TypeOfPlay().extract(self._imovel_por(24), _Shot())

        assert valor.evidence["quadros_imoveis"] == 24
        assert valor.value == PARADA, "o limiar e inclusivo, e 24 tem que bastar"

    def test_um_quadro_abaixo_do_limiar_nao_e_bola_parada(self) -> None:
        valor = TypeOfPlay().extract(self._imovel_por(23), _Shot())

        assert valor.evidence["quadros_imoveis"] == 23
        assert valor.value != PARADA, "23 nao pode bastar, senao o limiar nao e 24"

    def test_a_velocidade_de_chegada_le_os_dois_ultimos_trechos_e_nao_tres(self) -> None:
        """Tres amostras dao tres trechos, e o primeiro deles e o passe que chegou.

        Ler dois trechos ve so a conducao e sai `individual`; ler tres ve o passe e sai
        `open play`. Com quatro amostras a mutacao nao aparecia, porque o trecho rapido ficava
        fora dos tres ultimos tambem.
        """
        passos = [
            (CONTATO - 36, 1000.0, 300.0),
            (CONTATO - 24, 100.0, 300.0),
            (CONTATO - 12, 124.0, 300.0),
            (CONTATO, 148.0, 300.0),
        ]

        valor = TypeOfPlay().extract(_trajetoria(passos), _Shot())

        assert valor.evidence["velocidade_de_chegada_px_por_quadro"] == 2.0
        assert valor.value == INDIVIDUAL, "leu um trecho a mais e viu o passe que veio antes"

    def test_chegada_exatamente_no_piso_de_passe_e_jogada_aberta(self) -> None:
        """72 px em 12 quadros sao 6,0 exatos, e o piso e exclusivo: 6,0 ja e passe."""
        passos = [
            (CONTATO - 36, 1000.0, 300.0),
            (CONTATO - 24, 928.0, 300.0),
            (CONTATO - 12, 856.0, 300.0),
            (CONTATO, 784.0, 300.0),
        ]

        valor = TypeOfPlay().extract(_trajetoria(passos), _Shot())

        assert valor.evidence["velocidade_de_chegada_px_por_quadro"] == 6.0
        assert valor.value == ABERTA

    def test_velocidade_exatamente_no_limiar_de_parada_ainda_conta_como_parada(self) -> None:
        """18 px em 12 quadros sao 1,5 exatos, e o limiar de movimento e exclusivo."""
        passos = [
            (CONTATO - 60, 900.0, 300.0),
            (CONTATO - 48, 100.0, 300.0),
            (CONTATO - 36, 118.0, 300.0),
            (CONTATO - 24, 136.0, 300.0),
            (CONTATO - 12, 154.0, 300.0),
            (CONTATO, 172.0, 300.0),
        ]

        valor = TypeOfPlay().extract(_trajetoria(passos), _Shot())

        assert valor.evidence["quadros_imoveis"] == 48, "1,5 exato virou movimento"
        assert valor.value == PARADA

    def test_a_amostra_na_borda_da_janela_entra(self) -> None:
        """A janela e `[contato - 90, contato]` com as duas pontas dentro."""
        na_borda = [
            (CONTATO - 90, 500.0, 300.0),
            (CONTATO - 60, 500.0, 300.0),
            (CONTATO - 30, 500.0, 300.0),
            (CONTATO, 500.0, 300.0),
        ]
        um_antes = [(CONTATO - 91, 500.0, 300.0), *na_borda[1:]]

        assert TypeOfPlay().extract(_trajetoria(na_borda), _Shot()).evidence["quadros_lidos"] == 4
        assert TypeOfPlay().extract(_trajetoria(um_antes), _Shot()).evidence["quadros_lidos"] == 3

    def test_tres_amostras_bastam_e_duas_nao(self) -> None:
        tres = [(CONTATO - i * 12, 500.0, 300.0) for i in range(3)][::-1]
        duas = tres[1:]

        assert TypeOfPlay().extract(_trajetoria(tres), _Shot()).status == "ok"
        indisponivel = TypeOfPlay().extract(_trajetoria(duas), _Shot())
        assert indisponivel.status == "unavailable"
        assert indisponivel.reason is not None and "3" in indisponivel.reason

    def test_o_gate_compara_a_procedencia_inteira_e_nao_um_pedaco(self) -> None:
        """`midpoint` como substring de outra procedencia nao pode disparar o gate: uma
        procedencia nova que mencione o ponto medio ainda pode ser uma medida."""
        valor = TypeOfPlay().extract(
            _trajetoria(_parada_ate(CONTATO, 5)), _ShotComProcedencia("refined-from-midpoint")
        )

        assert valor.status == "ok" and valor.value == PARADA


def _modelo_que_troca_parada_por_individual(destino: Path) -> Path:
    modelo = EvidenceModel(
        feature="type_of_play",
        classes=(INDIVIDUAL, ABERTA, PARADA),
        columns=(Column((VALOR,), "equals", PARADA),),
        mean=(0.0,),
        scale=(1.0,),
        coef=((5.0,), (0.0,), (0.0,)),
        intercept=(0.0, 0.0, 0.0),
    )
    modelo.save(destino)
    return destino


def test_com_modelo_a_classe_sai_do_artefato_e_a_da_regra_fica_na_evidencia(tmp_path: Path) -> None:
    feature = TypeOfPlay(model=str(_modelo_que_troca_parada_por_individual(tmp_path / "m.json")))

    valor = feature.extract(_trajetoria(_parada_ate(CONTATO, 5)), _Shot())

    assert valor.value == INDIVIDUAL
    assert valor.evidence["classe_da_regra"] == PARADA
    assert valor.version == "0.3.0+modelo"
    assert valor.confidence is not None and valor.confidence > 0.9


def test_modelo_de_outra_feature_e_recusado(tmp_path: Path) -> None:
    destino = _modelo_que_troca_parada_por_individual(tmp_path / "m.json")
    destino.write_text(destino.read_text().replace('"type_of_play"', '"occupied_area"'))

    with pytest.raises(ValueError, match="occupied_area"):
        TypeOfPlay(model=str(destino))
