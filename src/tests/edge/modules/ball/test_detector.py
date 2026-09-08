from typing import Any

import pytest


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
    assert detector.telemetry.tiles_executed == 4
    assert detector.telemetry.raw_candidates == 4
    assert detector.telemetry.candidates_after_nms == 4
    assert detector.telemetry.inference_ms >= 0.0
    assert detector.telemetry.nms_ms >= 0.0


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


def test_tiled_baseline_has_fixed_backend_bindings() -> None:
    from edge.modules.ball.detector import DetectorConfig, backend_specs

    input_spec, output_spec = backend_specs(
        DetectorConfig(weights="x", imgsz=1600, ladrilhos=2, sobreposicao_px=100)
    )

    assert input_spec.shape == (1, 3, 992, 1600)
    assert output_spec.shape == (1, 5, 32550)


def test_backend_path_uses_nchw_and_restores_tile_coordinates() -> None:
    from types import SimpleNamespace

    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector, backend_specs
    from edge.runtime import RuntimeTensor

    config = DetectorConfig(
        weights="x", conf=0.1, imgsz=1600, ladrilhos=2, sobreposicao_px=100, ball_class_id=0
    )
    input_spec, output_spec = backend_specs(config)
    raw = np.zeros(output_spec.shape, dtype=np.float32)
    raw[0, 0, 0] = 800.0
    raw[0, 1, 0] = 496.0
    raw[0, 2, 0] = 20.0
    raw[0, 3, 0] = 20.0
    raw[0, 4, 0] = 0.9

    class _Backend:
        def __init__(self) -> None:
            self.inputs: list[dict[str, np.ndarray[Any, Any]]] = []
            self.closed = False
            self.info = SimpleNamespace(inference_count=0)

        def infer(self, inputs: dict[str, np.ndarray[Any, Any]]) -> tuple[RuntimeTensor, ...]:
            self.inputs.append(inputs)
            return (RuntimeTensor("predictions", raw),)

        def close(self) -> None:
            self.closed = True

    backend = _Backend()
    detector = YoloBallDetector(None, config, backend=backend)  # type: ignore[arg-type]
    frame = np.zeros((1088, 1920, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)

    detections = detector.detect_raw(frame)
    detector.close()

    tensor = backend.inputs[0]["images"]
    assert tensor.shape == input_spec.shape
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert len(backend.inputs) == 4
    assert len(detections) == 4
    box, confidence, class_id = detections[0]
    assert ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0) == pytest.approx(
        (530.0, 321.975), abs=0.1
    )
    assert confidence == pytest.approx(0.9)
    assert class_id == 0
    assert backend.closed is True


def test_ball_weight_must_be_a_detect_model_with_one_ball_class() -> None:
    from types import SimpleNamespace

    from edge.modules.ball.detector import BallModelIncompatible, check_ball_model

    check_ball_model(SimpleNamespace(task="detect", names={0: "ball"}))

    with pytest.raises(BallModelIncompatible, match="classe ball"):
        check_ball_model(SimpleNamespace(task="detect", names={0: "player", 1: "ball"}))
