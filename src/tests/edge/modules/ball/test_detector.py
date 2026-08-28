def test_ladrilhar_devolve_caixa_no_espaco_do_quadro_inteiro() -> None:
    """Sem somar o deslocamento do pedaco, toda bola do lado direito sairia no lado esquerdo."""
    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector

    class _Modelo:
        names = {0: "ball"}

        def __call__(self, frame: object, **_kwargs: object) -> list[object]:
            return [_Resultado()]

    class _Resultado:
        boxes = _Caixas()

    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    config = DetectorConfig(weights="x", ladrilhos=2, sobreposicao_px=100, ball_class_id=0)
    detector = YoloBallDetector(_Modelo(), config)

    cruas = detector.detect_raw(quadro)

    assert len(cruas) == 4
    cantos = sorted((round(c.x1), round(c.y1)) for c, _, _ in cruas)
    assert cantos == [(10, 10), (10, 450), (870, 10), (870, 450)]


class _Caixas:
    """Uma deteccao por pedaco, sempre em (10, 10) do pedaco, para o teste ler o deslocamento."""

    def __init__(self) -> None:
        self.xyxy = _Lista([[10.0, 10.0, 22.0, 22.0]])
        self.conf = _Lista([0.9])
        self.cls = _Lista([0.0])


class _Lista:
    def __init__(self, valores: list[object]) -> None:
        self._valores = valores

    def tolist(self) -> list[object]:
        return self._valores


def test_a_mesma_bola_em_dois_pedacos_sai_uma_vez_so() -> None:
    """Na faixa de sobreposicao a bola aparece nos dois pedacos, e duas caixas quase iguais.

    Sem a supressao o rastreador receberia dois candidatos identicos e o filtro de forma aprovaria
    os dois, dobrando o peso desse quadro na escolha por confianca.
    """
    from contracts import Box
    from edge.modules.ball.detector import sem_repetidas

    quase_iguais = (
        (Box(100.0, 100.0, 112.0, 112.0), 0.8, 0),
        (Box(101.0, 101.0, 113.0, 113.0), 0.6, 0),
        (Box(900.0, 500.0, 912.0, 512.0), 0.7, 0),
    )

    mantidas = sem_repetidas(quase_iguais, 0.1)

    assert len(mantidas) == 2
    assert mantidas[0][1] == 0.8


def test_ladrilhos_desligado_e_uma_passada_so() -> None:
    import numpy as np

    from edge.modules.ball.detector import pedacos

    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert len(pedacos(quadro, 2, 100)) == 4
    assert len(pedacos(quadro, 1, 100)) == 1


def test_os_pedacos_se_sobrepoem_para_a_bola_da_juncao_nao_sair_partida() -> None:
    """Bola na divisa sairia como duas meias caixas, e o filtro de aspecto mataria as duas."""
    import numpy as np

    from edge.modules.ball.detector import pedacos

    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pedacos = pedacos(quadro, 2, 100)

    larguras = [p.shape[1] for p, _, _ in pedacos]
    assert min(larguras) == 1060
    assert sum(larguras) > 1920 * 2
