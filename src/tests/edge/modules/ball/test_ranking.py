"""O ranking de candidatos da emenda 15, e o que ele recusa em vez de chutar.

A avaliacao que sustenta o padrao esta na pagina do modelo e usa quadro real, que nao entra aqui.
O que estes testes cobram e a fronteira: artefato malformado, caixa degenerada, alinhamento que nao
se sustenta e residuo que nao vale para esta caixa. Cada um deles, se passar calado, produz numero
plausivel em vez de erro, e foi por isso que cada um virou teste.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from contracts import BallDetection, Box
from edge.modules.ball.ranking import (
    APPEARANCE_FEATURES,
    ARTIFACT,
    MAX_MOTION_GAP_FRAMES,
    MOTION_FEATURES,
    CandidateRanker,
    RankerArtifact,
    RankerError,
    appearance_features,
    estimate_motion,
    load_ranker,
    motion_features,
    motion_residual,
)

Array = np.ndarray[Any, Any]


def _artefato(size: int = 2) -> RankerArtifact:
    return RankerArtifact(
        version="teste",
        features=tuple(f"f{i}" for i in range(size)),
        center=(0.0,) * size,
        scale=(1.0,) * size,
        coefficient=(0.0,) * (size + 1),
    )


def _ruido(width: int = 320, height: int = 240, seed: int = 0) -> Array:
    """Gramado sintetico com textura: sem ela o ORB nao acha ponto nenhum e o teste mede outra
    coisa que nao a que ele diz medir."""
    generator = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 160
    image[:, :, 0] = 40
    ruido = generator.integers(0, 90, size=(height, width), dtype=np.uint8)
    image[:, :, 2] = ruido
    image[:, :, 1] = np.clip(image[:, :, 1].astype(int) + ruido // 2, 0, 255).astype(np.uint8)
    return image


class TestRankerArtifact:
    def test_recusa_tamanho_incompativel(self) -> None:
        with pytest.raises(RankerError, match="center e scale"):
            RankerArtifact(
                version="x",
                features=("a", "b"),
                center=(0.0,),
                scale=(1.0,),
                coefficient=(0.0, 0.0, 0.0),
            )

    def test_recusa_coeficiente_sem_termo_constante(self) -> None:
        with pytest.raises(RankerError, match="termo constante"):
            RankerArtifact(
                version="x", features=("a",), center=(0.0,), scale=(1.0,), coefficient=(0.0,)
            )

    def test_recusa_artefato_sem_medida_nenhuma(self) -> None:
        with pytest.raises(RankerError, match="sem nenhuma medida"):
            RankerArtifact(version="x", features=(), center=(), scale=(), coefficient=(0.0,))

    def test_recusa_escala_nao_positiva(self) -> None:
        with pytest.raises(RankerError, match="scale"):
            RankerArtifact(
                version="x", features=("a",), center=(0.0,), scale=(0.0,), coefficient=(0.0, 0.0)
            )

    def test_recusa_coeficiente_nao_finito(self) -> None:
        with pytest.raises(RankerError, match="nao finito"):
            RankerArtifact(
                version="x",
                features=("a",),
                center=(0.0,),
                scale=(1.0,),
                coefficient=(float("nan"), 0.0),
            )

    def test_recusa_vetor_de_tamanho_errado(self) -> None:
        with pytest.raises(RankerError, match="esperava 2 medidas"):
            _artefato().probability([1.0])

    def test_probabilidade_fica_no_intervalo_aberto(self) -> None:
        artefato = RankerArtifact(
            version="x", features=("a",), center=(0.0,), scale=(1.0,), coefficient=(0.0, 1000.0)
        )
        assert artefato.probability([1e6]) == pytest.approx(1.0)
        assert artefato.probability([-1e6]) == pytest.approx(0.0)

    def test_artefato_versionado_tem_as_dezenove_medidas(self) -> None:
        artefato = load_ranker()
        assert artefato.features == APPEARANCE_FEATURES + MOTION_FEATURES
        assert artefato.uses_camera_motion

    def test_ausencia_do_arquivo_e_erro_explicito(self, tmp_path: Any) -> None:
        with pytest.raises(RankerError, match="nao existe"):
            load_ranker(tmp_path / "ausente.json")

    def test_arquivo_sem_campo_obrigatorio(self, tmp_path: Any) -> None:
        caminho = tmp_path / "parcial.json"
        caminho.write_text('{"version": "1.0", "features": []}', encoding="utf-8")
        with pytest.raises(RankerError, match="sem os campos"):
            load_ranker(caminho)

    def test_o_artefato_em_disco_carrega_sem_ajuste(self) -> None:
        assert ARTIFACT.is_file()
        assert (
            len(load_ranker(ARTIFACT).coefficient)
            == len(APPEARANCE_FEATURES) + len(MOTION_FEATURES) + 1
        )


class TestAppearanceFeatures:
    def test_recusa_caixa_degenerada(self) -> None:
        with pytest.raises(RankerError, match="degenerada"):
            appearance_features(_ruido(), (10.0, 10.0, 10.0, 20.0), 0.5)

    def test_produz_catorze_medidas_finitas(self) -> None:
        valores = appearance_features(_ruido(), (100.0, 80.0, 114.0, 94.0), 0.42)
        assert len(valores) == len(APPEARANCE_FEATURES)
        assert all(np.isfinite(valores))

    def test_recorte_fora_do_quadro_nao_produz_nan(self) -> None:
        """Caixa colada na borda esquerda: o recorte de raio inteiro sai vazio de um lado, e a media
        de um recorte vazio seria `nan` silencioso dentro do vetor."""
        valores = appearance_features(_ruido(), (-40.0, -40.0, -26.0, -26.0), 0.3)
        assert all(np.isfinite(valores))
        assert valores[8:] == [0.0] * 6

    def test_confianca_zero_nao_estoura_o_logito(self) -> None:
        valores = appearance_features(_ruido(), (100.0, 80.0, 114.0, 94.0), 0.0)
        assert np.isfinite(valores[0]) and valores[0] < 0


class TestEstimateMotion:
    def test_recusa_quadros_de_tamanhos_diferentes(self) -> None:
        assert estimate_motion(_ruido(), _ruido(width=321)) is None

    def test_recusa_quadro_ausente(self) -> None:
        assert estimate_motion(None, _ruido()) is None

    def test_recusa_imagem_sem_textura(self) -> None:
        liso = np.full((240, 320, 3), (40, 160, 40), dtype=np.uint8)
        assert estimate_motion(liso, liso) is None

    def test_recusa_par_sem_correspondencia(self) -> None:
        assert estimate_motion(_ruido(seed=1), _ruido(seed=2)) is None

    def test_recusa_correspondencias_concentradas(self) -> None:
        """Textura num canto so: a homografia sai de um pedaco e descreve o quadro inteiro.

        Esta e a guarda de cobertura espacial, exigida pela emenda 15 e a unica das seis recusas
        que nao tinha teste. Sem ela, inverter a comparacao passaria a suite inteira e a homografia
        ruim viraria residuo ruim sem aviso.
        """
        canto = np.zeros((480, 640, 3), dtype=np.uint8)
        canto[:, :, 1] = 160
        ruido = np.random.default_rng(11).integers(0, 90, size=(70, 70), dtype=np.uint8)
        canto[10:80, 10:80, 2] = ruido
        canto[10:80, 10:80, 1] = np.clip(160 + ruido // 2, 0, 255).astype(np.uint8)

        assert estimate_motion(canto, np.roll(canto, 3, axis=1)) is None

    def test_recusa_quadrilatero_degenerado(self) -> None:
        """Homografia que dobra o quadro sobre si mesmo produz area negativa ou nao convexa."""
        origem = _ruido(width=640, height=480, seed=7)
        espelhado = origem[:, ::-1]

        assert estimate_motion(origem, espelhado) is None

    def test_recusa_deformacao_grande_demais(self) -> None:
        """Reescalar o conteudo tira a area do quadrilatero da faixa aceita."""
        import cv2

        origem = _ruido(width=640, height=480, seed=9)
        pequeno = cv2.resize(origem, (160, 120))
        ampliado = np.zeros_like(origem)
        ampliado[:120, :160] = pequeno

        assert estimate_motion(origem, ampliado) is None

    def test_recupera_translacao_conhecida(self) -> None:
        origem = _ruido(width=640, height=480, seed=3)
        deslocado = np.roll(origem, 7, axis=1)
        matriz = estimate_motion(origem, deslocado)
        assert matriz is not None
        assert matriz[0, 2] == pytest.approx(7.0, abs=1.5)
        assert matriz[1, 2] == pytest.approx(0.0, abs=1.5)


class TestMotionFeatures:
    def test_residuo_ausente_zera_as_cinco(self) -> None:
        assert motion_features((0.0, 0.0, 4.0, 4.0), None) == [0.0] * len(MOTION_FEATURES)

    def test_residuo_nulo_para_matriz_ausente(self) -> None:
        assert motion_residual(_ruido(), _ruido(), None) is None

    def test_caixa_fora_da_regiao_valida_zera_as_cinco(self) -> None:
        delta = np.zeros((240, 320), dtype=float)
        valido = np.zeros((240, 320), dtype=bool)
        valido[100:200, 100:200] = True
        assert motion_features((10.0, 10.0, 20.0, 20.0), (delta, valido)) == [0.0] * len(
            MOTION_FEATURES
        )

    def test_caixa_fora_do_quadro_zera_as_cinco(self) -> None:
        delta = np.zeros((240, 320), dtype=float)
        valido = np.ones((240, 320), dtype=bool)
        assert motion_features((300.0, 10.0, 340.0, 40.0), (delta, valido)) == [0.0] * len(
            MOTION_FEATURES
        )

    def test_objeto_movel_tem_contraste_positivo(self) -> None:
        delta = np.zeros((240, 320), dtype=float)
        delta[100:110, 100:110] = 0.6
        valido = np.ones((240, 320), dtype=bool)
        disponivel, media, p90, contraste, ativos = motion_features(
            (100.0, 100.0, 110.0, 110.0), (delta, valido)
        )
        assert disponivel == 1.0
        assert media == pytest.approx(0.6)
        assert p90 == pytest.approx(0.6)
        assert contraste > 0
        assert ativos == pytest.approx(1.0)


class TestCandidateRanker:
    def test_lista_vazia_devolve_nada(self) -> None:
        ranker = CandidateRanker(_artefato(len(APPEARANCE_FEATURES) + len(MOTION_FEATURES)))
        assert ranker.rank(_ruido(), 0, ()) == ()

    def test_devolve_um_unico_candidato(self) -> None:
        artefato = load_ranker()
        ranker = CandidateRanker(artefato)
        candidatos = (
            BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),
            BallDetection(Box(200.0, 150.0, 214.0, 164.0), 0.9),
        )
        escolhidos = ranker.rank(_ruido(), 0, candidatos)
        assert len(escolhidos) == 1
        assert escolhidos[0] in candidatos

    def test_preserva_a_confianca_do_detector(self) -> None:
        """A pontuacao ordena, mas o registro publicado continua dizendo o que o detector achou:
        trocar o campo mudaria o significado de `confidence` para todo consumidor."""
        candidato = BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.37)
        escolhido = CandidateRanker(load_ranker()).rank(_ruido(), 0, (candidato,))[0]
        assert escolhido.confidence == pytest.approx(0.37)

    def test_primeiro_quadro_nao_tem_movimento(self) -> None:
        ranker = CandidateRanker(load_ranker())
        ranker.rank(_ruido(width=640, height=480), 0, ())
        assert ranker.motion_frames == 0

    def test_quadro_seguinte_usa_o_movimento(self) -> None:
        """Com candidato, porque quadro sem candidato nao paga a estimativa de movimento."""
        ranker = CandidateRanker(load_ranker())
        origem = _ruido(width=640, height=480, seed=3)
        candidato = (BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),)
        ranker.rank(origem, 0, candidato)
        ranker.rank(np.roll(origem, 7, axis=1), 4, candidato)
        assert ranker.motion_frames == 1

    def test_quadro_sem_candidato_nao_paga_a_estimativa(self) -> None:
        """A estimativa custa 53,1 ms por quadro medidos, e pontuar lista vazia nao usa o
        resultado. O quadro continua guardado, entao o passo entre quadros nao muda."""
        ranker = CandidateRanker(load_ranker())
        origem = _ruido(width=640, height=480, seed=3)
        candidato = (BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),)
        ranker.rank(origem, 0, candidato)
        ranker.rank(np.roll(origem, 7, axis=1), 4, ())

        assert ranker.motion_frames == 0

        ranker.rank(np.roll(origem, 14, axis=1), 8, candidato)

        assert ranker.motion_frames == 1

    def test_salto_grande_de_indice_descarta_o_movimento(self) -> None:
        ranker = CandidateRanker(load_ranker())
        origem = _ruido(width=640, height=480, seed=3)
        candidato = (BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),)
        ranker.rank(origem, 0, candidato)
        ranker.rank(np.roll(origem, 7, axis=1), MAX_MOTION_GAP_FRAMES + 1, candidato)
        assert ranker.motion_frames == 0

    def test_indice_que_nao_avanca_descarta_o_movimento(self) -> None:
        ranker = CandidateRanker(load_ranker())
        origem = _ruido(width=640, height=480, seed=3)
        candidato = (BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),)
        ranker.rank(origem, 5, candidato)
        ranker.rank(np.roll(origem, 7, axis=1), 5, candidato)
        assert ranker.motion_frames == 0

    def test_reset_esquece_o_quadro_anterior(self) -> None:
        ranker = CandidateRanker(load_ranker())
        origem = _ruido(width=640, height=480, seed=3)
        candidato = (BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),)
        ranker.rank(origem, 0, candidato)
        ranker.reset()
        ranker.rank(np.roll(origem, 7, axis=1), 4, candidato)
        assert ranker.motion_frames == 0

    def test_primeiro_quadro_com_candidato_nao_tem_movimento(self) -> None:
        ranker = CandidateRanker(load_ranker())
        ranker.rank(
            _ruido(width=640, height=480), 0, (BallDetection(Box(10.0, 10.0, 24.0, 24.0), 0.5),)
        )
        assert ranker.motion_frames == 0

    def test_artefato_sem_movimento_nao_estima_homografia(self) -> None:
        somente_aparencia = RankerArtifact(
            version="x",
            features=APPEARANCE_FEATURES,
            center=(0.0,) * len(APPEARANCE_FEATURES),
            scale=(1.0,) * len(APPEARANCE_FEATURES),
            coefficient=(0.0,) * (len(APPEARANCE_FEATURES) + 1),
        )
        ranker = CandidateRanker(somente_aparencia)
        origem = _ruido(width=640, height=480, seed=3)
        candidato = (BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),)
        ranker.rank(origem, 0, candidato)
        ranker.rank(np.roll(origem, 7, axis=1), 4, candidato)
        assert ranker.motion_frames == 0


class TestKeep:
    def test_recusa_keep_menor_que_um(self) -> None:
        with pytest.raises(RankerError, match="pelo menos um"):
            CandidateRanker(load_ranker(), keep=0)

    def test_devolve_dois_em_ordem_decrescente(self) -> None:
        """A ordem e a decisao inteira: `candidate_order` faz o rastreador respeitar exatamente
        esta sequencia, e uma ordem errada aqui custa 16 acertos no piloto."""
        artefato = load_ranker()
        candidatos = (
            BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),
            BallDetection(Box(200.0, 150.0, 214.0, 164.0), 0.9),
            BallDetection(Box(60.0, 200.0, 74.0, 214.0), 0.2),
        )
        pontuacoes = dict(zip(candidatos, _pontuacoes(artefato, _ruido(), candidatos), strict=True))
        esperado = sorted(candidatos, key=lambda c: -pontuacoes[c])

        escolhidos = CandidateRanker(artefato, keep=2).rank(_ruido(), 0, candidatos)

        assert list(escolhidos) == esperado[:2]
        assert pontuacoes[escolhidos[0]] >= pontuacoes[escolhidos[1]]

    def test_devolve_tres_na_ordem_completa(self) -> None:
        artefato = load_ranker()
        candidatos = (
            BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),
            BallDetection(Box(200.0, 150.0, 214.0, 164.0), 0.9),
            BallDetection(Box(60.0, 200.0, 74.0, 214.0), 0.2),
        )
        pontuacoes = dict(zip(candidatos, _pontuacoes(artefato, _ruido(), candidatos), strict=True))

        escolhidos = CandidateRanker(artefato, keep=3).rank(_ruido(), 0, candidatos)

        assert [pontuacoes[c] for c in escolhidos] == sorted(pontuacoes.values(), reverse=True)

    def test_keep_maior_que_a_lista_devolve_a_lista(self) -> None:
        candidato = BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4)

        assert len(CandidateRanker(load_ranker(), keep=5).rank(_ruido(), 0, (candidato,))) == 1

    def test_empate_de_pontuacao_segue_a_ordem_do_detector(self) -> None:
        """Duas caixas identicas pontuam igual, e a saida nao pode depender da ordem de iteracao:
        sem o desempate por posicao, a mesma entrada daria saidas diferentes entre plataformas."""
        plano = RankerArtifact(
            version="x",
            features=APPEARANCE_FEATURES,
            center=(0.0,) * len(APPEARANCE_FEATURES),
            scale=(1.0,) * len(APPEARANCE_FEATURES),
            coefficient=(0.0,) * (len(APPEARANCE_FEATURES) + 1),
        )
        primeiro = BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.5)
        segundo = BallDetection(Box(200.0, 150.0, 214.0, 164.0), 0.5)

        assert CandidateRanker(plano, keep=1).rank(_ruido(), 0, (primeiro, segundo)) == (primeiro,)
        assert CandidateRanker(plano, keep=1).rank(_ruido(), 0, (segundo, primeiro)) == (segundo,)

    def test_a_pontuacao_registrada_e_a_do_primeiro_colocado(self) -> None:
        ranker = CandidateRanker(load_ranker(), keep=3)
        candidatos = (
            BallDetection(Box(100.0, 80.0, 114.0, 94.0), 0.4),
            BallDetection(Box(200.0, 150.0, 214.0, 164.0), 0.9),
        )

        escolhidos = ranker.rank(_ruido(), 0, candidatos)

        pontuacoes = dict(
            zip(candidatos, _pontuacoes(load_ranker(), _ruido(), candidatos), strict=True)
        )
        assert ranker.scores == (pytest.approx(pontuacoes[escolhidos[0]]),)


def _pontuacoes(
    artefato: RankerArtifact, imagem: Array, candidatos: tuple[BallDetection, ...]
) -> list[float]:
    valores: list[float] = []
    for candidato in candidatos:
        caixa = (candidato.box.x1, candidato.box.y1, candidato.box.x2, candidato.box.y2)
        medidas = [
            *appearance_features(imagem, caixa, candidato.confidence),
            *motion_features(caixa, None),
        ]
        valores.append(artefato.probability(medidas))
    return valores


class TestEquivalenciaComOCalculoVetorizado:
    """A classe de producao tem de escolher o mesmo candidato que a aritmetica da auditoria.

    A auditoria que sustenta os 186 acertos calcula as pontuacoes de uma vez, em matriz. Se a classe
    divergir dela, o numero publicado descreve um codigo que nao e o que roda. A emenda 15 exige
    essa amarra, e ela existia so em `tmp/`, que o git ignora e a CI nao executa: aqui ela roda com
    quadros sinteticos, que e o que falta para a propriedade valer dentro da arvore.
    """

    def _vetorizado(
        self, artefato: RankerArtifact, imagem: Array, candidatos: tuple[BallDetection, ...]
    ) -> Array:
        medidas = np.asarray(
            [
                [
                    *appearance_features(
                        imagem,
                        (c.box.x1, c.box.y1, c.box.x2, c.box.y2),
                        c.confidence,
                    ),
                    *motion_features((c.box.x1, c.box.y1, c.box.x2, c.box.y2), None),
                ]
                for c in candidatos
            ]
        )
        centro = np.asarray(artefato.center)
        escala = np.asarray(artefato.scale)
        coeficiente = np.asarray(artefato.coefficient)
        desenho = np.column_stack((np.ones(len(medidas)), (medidas - centro) / escala))
        return 1 / (1 + np.exp(-np.clip(desenho @ coeficiente, -30, 30)))

    def test_a_escolha_bate_em_todos_os_quadros_sinteticos(self) -> None:
        artefato = load_ranker()
        gerador = np.random.default_rng(17)
        divergencias: list[int] = []
        for quadro in range(12):
            imagem = _ruido(width=640, height=480, seed=quadro)
            candidatos = tuple(
                BallDetection(
                    Box(float(x), float(y), float(x) + 14.0, float(y) + 14.0),
                    float(conf),
                )
                for x, y, conf in zip(
                    gerador.integers(20, 600, 4),
                    gerador.integers(20, 440, 4),
                    gerador.uniform(0.1, 0.95, 4),
                    strict=True,
                )
            )
            probabilidades = self._vetorizado(artefato, imagem, candidatos)
            esperado = candidatos[int(np.argmax(probabilidades))]

            (escolhido,) = CandidateRanker(artefato).rank(imagem, quadro * 4, candidatos)

            if escolhido is not esperado:
                divergencias.append(quadro)

        assert not divergencias, f"a classe divergiu do calculo nos quadros {divergencias}"

    def test_a_ordem_completa_tambem_bate(self) -> None:
        artefato = load_ranker()
        imagem = _ruido(width=640, height=480, seed=5)
        candidatos = tuple(
            BallDetection(Box(x, 100.0, x + 14.0, 114.0), 0.5) for x in (50.0, 200.0, 400.0)
        )
        probabilidades = self._vetorizado(artefato, imagem, candidatos)
        ordem = [int(i) for i in np.argsort(-probabilidades).tolist()]
        esperado = [candidatos[i] for i in ordem]

        escolhidos = CandidateRanker(artefato, keep=3).rank(imagem, 0, candidatos)

        assert list(escolhidos) == esperado
