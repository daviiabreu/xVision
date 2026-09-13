from collections.abc import Callable
from typing import Any, cast

import pytest


def test_ladrilhar_devolve_caixa_no_espaco_do_quadro_inteiro() -> None:
    """Sem somar o deslocamento do pedaco, toda bola do lado direito sairia no lado esquerdo."""
    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector

    class _Modelo:
        """Devolve um result por imagem, com lote ou sem, como o Ultralytics de verdade faz."""

        names = {0: "ball"}

        def __call__(self, frames: object, **_kwargs: object) -> list[object]:
            imagens = cast("list[object]", frames) if isinstance(frames, list) else [frames]
            return [_Resultado() for _ in imagens]

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

    assert input_spec.shape == (4, 3, 992, 1600)
    assert output_spec.shape == (4, 5, 32550)


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
    for item in range(output_spec.shape[0]):
        raw[item, 0, 0] = 800.0
        raw[item, 1, 0] = 496.0
        raw[item, 2, 0] = 20.0
        raw[item, 3, 0] = 20.0
        raw[item, 4, 0] = 0.9

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
    assert len(backend.inputs) == 1
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


class _ModeloContador:
    """Fake do Ultralytics que conta despachos e devolve um result por imagem, como o real.

    O Ultralytics aceita uma imagem so ou uma lista delas, e no segundo caso devolve uma lista de
    results na mesma ordem das entradas. O fake registra o tamanho de cada despacho para o teste
    separar quantos ladrilhos rodaram de quantas chamadas ao modelo foram feitas.
    """

    names = {0: "ball"}

    def __init__(self, roteiro: list[tuple[list[float], float]]) -> None:
        self._roteiro = list(roteiro)
        self.despachos: list[int] = []
        self.entradas: list[object] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, frames: object, **kwargs: object) -> list[object]:
        self.entradas.append(frames)
        self.kwargs.append(dict(kwargs))
        imagens = cast("list[object]", frames) if isinstance(frames, list) else [frames]
        self.despachos.append(len(imagens))
        return [_ResultadoDo(*self._roteiro.pop(0)) for _ in imagens]


class _ResultadoDo:
    def __init__(self, caixa: list[float], confianca: float) -> None:
        self.boxes = _CaixasDe(caixa, confianca)


class _CaixasDe:
    def __init__(self, caixa: list[float], confianca: float) -> None:
        self.xyxy = _Lista([list(caixa)])
        self.conf = _Lista([confianca])
        self.cls = _Lista([0.0])


_ROTEIRO: list[tuple[list[float], float]] = [
    ([10.0, 10.0, 22.0, 22.0], 0.5),
    ([30.0, 40.0, 42.0, 52.0], 0.9),
    ([50.0, 60.0, 62.0, 72.0], 0.9),
    ([70.0, 80.0, 82.0, 92.0], 0.3),
]
"""Uma caixa por ladrilho, em posicoes distantes para o NMS nao suprimir nenhuma.

As duas do meio empatam em confianca de proposito: o empate e o que expoe a ordem de insercao no
`sorted` estavel de `sem_repetidas`, e e essa ordem que vira `detection_index` rio abaixo.
"""


def test_um_2x2_despacha_os_quatro_ladrilhos_numa_chamada_so() -> None:
    """Quatro despachos sequenciais a imgsz 1600 custam 4 x 70,4 ms = 282 ms por quadro na Jetson.

    O lote nao remove trabalho visual, remove ida e volta: os mesmos quatro ladrilhos entram numa
    chamada so ao modelo.
    """
    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector

    modelo = _ModeloContador(_ROTEIRO)
    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    config = DetectorConfig(weights="x", ladrilhos=2, sobreposicao_px=100, ball_class_id=0)

    YoloBallDetector(modelo, config).detect_raw(quadro)

    assert modelo.despachos == [4]


def test_o_lote_devolve_as_mesmas_deteccoes_na_mesma_ordem_que_o_laco() -> None:
    """A ordem da saida e contrato: dela sai o `detection_index` de que o rastreador depende.

    A referencia e o que o laco sequencial produzia: uma passada por ladrilho, na ordem de
    `pedacos`, deslocada pelo canto do ladrilho e depois unida por `sem_repetidas`.
    """
    import numpy as np

    from contracts import Box
    from edge.modules.ball.detector import (
        DetectorConfig,
        YoloBallDetector,
        pedacos,
        sem_repetidas,
    )

    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    config = DetectorConfig(weights="x", ladrilhos=2, sobreposicao_px=100, ball_class_id=0)
    sequencial = sem_repetidas(
        [
            (Box(caixa[0] + x, caixa[1] + y, caixa[2] + x, caixa[3] + y), confianca, 0)
            for (_, x, y), (caixa, confianca) in zip(
                pedacos(quadro, config.ladrilhos, config.sobreposicao_px), _ROTEIRO, strict=True
            )
        ],
        config.nms_iou,
    )

    em_lote = YoloBallDetector(_ModeloContador(_ROTEIRO), config).detect_raw(quadro)

    assert em_lote == sequencial
    assert [confianca for _, confianca, _ in em_lote] == [0.9, 0.9, 0.5, 0.3]


def test_ladrilho_unico_continua_uma_passada_sem_lote() -> None:
    """Sem ladrilhar nao ha o que batelar: o quadro vai cru ao modelo, e nao dentro de lista."""
    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector

    modelo = _ModeloContador(_ROTEIRO)
    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    config = DetectorConfig(weights="x", ladrilhos=1, ball_class_id=0)

    cruas = YoloBallDetector(modelo, config).detect_raw(quadro)

    assert modelo.despachos == [1]
    assert modelo.entradas[0] is quadro
    assert len(cruas) == 1


def test_tiles_executed_conta_ladrilhos_e_nao_despachos() -> None:
    """A telemetria mede trabalho visual: com o lote os despachos caem para um, os ladrilhos nao."""
    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector

    modelo = _ModeloContador(_ROTEIRO)
    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    config = DetectorConfig(weights="x", ladrilhos=2, sobreposicao_px=100, ball_class_id=0)
    detector = YoloBallDetector(modelo, config)

    detector.detect_raw(quadro)

    assert detector.telemetry.tiles_executed == 4
    assert modelo.despachos == [4]


def test_o_lote_pode_ser_reduzido_sem_mudar_o_resultado() -> None:
    """Quem medir memoria na Jetson precisa poder descer o lote sem reescrever o detector."""
    from dataclasses import replace

    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector

    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    base = DetectorConfig(weights="x", ladrilhos=2, sobreposicao_px=100, ball_class_id=0)
    inteiro = YoloBallDetector(_ModeloContador(_ROTEIRO), base).detect_raw(quadro)

    modelo = _ModeloContador(_ROTEIRO)
    detector = YoloBallDetector(modelo, replace(base, ladrilhos_por_lote=2))

    aos_pares = detector.detect_raw(quadro)

    assert modelo.despachos == [2, 2]
    assert aos_pares == inteiro
    assert detector.telemetry.tiles_executed == 4


def test_a_precisao_declarada_no_toml_chega_ao_despacho_da_bola() -> None:
    """Ajuste que fica no TOML sem alcancar o modelo e pior que ajuste nenhum.

    O ladrilho a 1600 mediu 66,8 ms em fp32 e 53,3 ms com `quantize='fp16'` nesta Jetson, e o
    detector da bola e cerca de 70% do tempo de um clipe: e a alavanca de tempo mais cara do
    pipeline, e ela so existe se o parametro atravessar ate a chamada.
    """
    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector

    modelo = _ModeloContador(_ROTEIRO)
    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    config = DetectorConfig(weights="x", ball_class_id=0).com_ajustes({"quantize": "fp16"})

    YoloBallDetector(modelo, config).detect_raw(quadro)

    assert modelo.kwargs[0]["quantize"] == "fp16"


def test_sem_declaracao_o_despacho_da_bola_nao_recebe_quantize() -> None:
    """O padrao e inegociavel: meia precisao ainda nao foi medida contra gabarito.

    Nao mandar o parametro, em vez de mandar `fp32`, e o que prova que a chamada de hoje continua
    exatamente a de antes desta alavanca existir.
    """
    import numpy as np

    from edge.modules.ball.detector import DetectorConfig, YoloBallDetector

    modelo = _ModeloContador(_ROTEIRO)
    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)

    YoloBallDetector(modelo, DetectorConfig(weights="x", ball_class_id=0)).detect_raw(quadro)

    assert "quantize" not in modelo.kwargs[0]


@pytest.mark.parametrize("valor", ["fp8", True, 16, "half"])
def test_precisao_invalida_na_secao_da_bola_levanta_dizendo_o_que_e_aceito(valor: object) -> None:
    from edge.modules.ball.detector import AjusteDesconhecido, DetectorConfig

    with pytest.raises(AjusteDesconhecido, match="fp32.*fp16"):
        DetectorConfig(weights="x.pt").com_ajustes({"quantize": valor})


def test_o_campo_half_morreu_e_declarar_half_no_toml_e_erro() -> None:
    """Ele nunca chegou ao predict, e agora tem substituto: 62,4 ms contra 28,6 ms de fp16.

    O que muda aqui e o campo deixar de existir na dataclass: `DetectorConfig(half=True)` passava
    calado e nao ligava nada. A recusa do TOML nao e nova, porque `half` estava na lista de
    exclusao de `com_ajustes` e ja caia em `desconhecidas`; o `raises` esta aqui para garantir que
    matar o campo nao abriu por acidente uma chave aceita.
    """
    from dataclasses import fields

    from edge.modules.ball.detector import AjusteDesconhecido, DetectorConfig

    assert "half" not in {c.name for c in fields(DetectorConfig)}

    with pytest.raises(AjusteDesconhecido, match="half"):
        DetectorConfig(weights="x.pt").com_ajustes({"half": True})


def test_a_meia_precisao_atravessa_o_loader_do_backend(
    peso_de_mentira: Callable[..., Any],
) -> None:
    """O caminho de producao da bola e o backend fixo, e nao o predict do Ultralytics.

    `from_config` sempre monta o `PyTorchBackend`, entao um `quantize` que so alcancasse o predict
    seria uma alavanca desligada justamente na especie que custa 70% do clipe. O binding do
    artefato continua float32 dos dois lados (`TensorSpec` cravado, e a saida volta com `.float()`);
    quem passa a meia precisao sao o peso e o tensor, dentro do loader.
    """
    from pathlib import Path

    import numpy as np

    from edge.modules.ball.detector import carregador_ultralytics

    peso = peso_de_mentira("edge.modules.ball.detector", names={0: "ball"})

    infer = carregador_ultralytics("fp16")(Path("bola.pt"), "cuda")
    infer({"images": np.zeros((1, 3, 32, 32), dtype=np.float32)})

    assert peso.meio_peso is True
    assert peso.entradas_meias == [True]


def test_sem_declaracao_o_loader_do_backend_continua_em_fp32(
    peso_de_mentira: Callable[..., Any],
) -> None:
    """A saida volta float32 nos dois casos, porque o `TensorSpec` de saida e float32."""
    from pathlib import Path

    import numpy as np

    from edge.modules.ball.detector import carregador_ultralytics

    peso = peso_de_mentira("edge.modules.ball.detector", names={0: "ball"})

    infer = carregador_ultralytics()(Path("bola.pt"), "cuda")
    saida = infer({"images": np.zeros((1, 3, 32, 32), dtype=np.float32)})

    assert peso.meio_peso is False
    assert peso.entradas_meias == [False]
    assert saida["predictions"].dtype.name == "float32"


def test_backend_specs_declara_o_lote_dos_ladrilhos() -> None:
    """O peso e `.pt` e aceita qualquer lote: o lote 1 era restricao nossa, nao do artefato.

    Medido no forward do shape de producao (992x1600) nesta Jetson: quatro despachos de lote 1
    custam 220,6 ms e um despacho de lote 4 custa 158,3 ms, com pico de 375 MiB. O binding
    precisa declarar o lote para o `PyTorchBackend` aceitar o tensor empilhado.
    """
    from edge.modules.ball.detector import DetectorConfig, backend_specs

    config = DetectorConfig(
        weights="x", imgsz=1600, ladrilhos=2, sobreposicao_px=100, ladrilhos_por_lote=4
    )

    entrada, saida = backend_specs(config)

    assert entrada.shape[0] == 4
    assert saida.shape[0] == 4
    assert entrada.shape[1:] == (3, 992, 1600)


def test_backend_specs_com_ladrilho_unico_fica_em_lote_um() -> None:
    """Sem ladrilhar ha um pedaco so, e declarar lote 4 pediria tensor que nunca chega."""
    from edge.modules.ball.detector import DetectorConfig, backend_specs

    config = DetectorConfig(weights="x", imgsz=1600, ladrilhos=1, ladrilhos_por_lote=4)

    entrada, saida = backend_specs(config)

    assert entrada.shape[0] == 1
    assert saida.shape[0] == 1


def test_lote_que_nao_divide_os_ladrilhos_e_recusado() -> None:
    """Lote 3 sobre quatro ladrilhos daria um despacho de 3 e outro de 1, dois shapes.

    O binding do backend e fixo, entao o segundo despacho seria recusado no meio do quadro. Falhar
    na configuracao e melhor que falhar no meio da execucao.
    """
    from edge.modules.ball.detector import (
        BallModelIncompatible,
        DetectorConfig,
        backend_specs,
    )

    config = DetectorConfig(
        weights="x", imgsz=1600, ladrilhos=2, sobreposicao_px=100, ladrilhos_por_lote=3
    )

    with pytest.raises(BallModelIncompatible, match="nao divide"):
        backend_specs(config)


def test_o_letterbox_padrao_continua_no_cv2() -> None:
    """A interpolacao da GPU nao e identica a do cv2, entao ela e opt-in e nao herdada.

    Medido nesta Jetson: o desvio maximo por pixel entre `cv2.INTER_LINEAR` e o bilinear do
    `F.interpolate` e 0,003 na escala 0..1. E pouco, mas a bola tem cerca de 13 px e a divergencia
    de deteccao ainda nao foi medida contra gabarito, entao o padrao nao muda sozinho.
    """
    from edge.modules.ball.detector import DetectorConfig

    assert DetectorConfig(weights="x").letterbox == "cv2"


def test_o_letterbox_declarado_no_toml_e_aceito() -> None:
    from edge.modules.ball.detector import DetectorConfig

    ajustado = DetectorConfig(weights="x").com_ajustes({"letterbox": "cuda"})

    assert ajustado.letterbox == "cuda"


def test_letterbox_invalido_no_toml_e_recusado() -> None:
    """`gpu` em vez de `cuda` cairia no padrao e a execucao reportaria o tempo de antes."""
    from edge.modules.ball.detector import AjusteDesconhecido, DetectorConfig

    with pytest.raises(AjusteDesconhecido, match="cv2 ou cuda"):
        DetectorConfig(weights="x").com_ajustes({"letterbox": "gpu"})


def test_os_dois_letterbox_produzem_a_mesma_geometria() -> None:
    """Ganho e enchimento tem que casar, porque `_restore_box` desfaz os dois na volta.

    Se a geometria divergir, a caixa sai no espaco errado e continua sendo quatro floats
    plausiveis. Este teste roda so onde ha torch com CUDA; no venv de teste ele pula.
    """
    import numpy as np

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("o letterbox em cuda exige GPU")

    from edge.modules.ball.detector import (
        _letterbox_tensor,  # pyright: ignore[reportPrivateUsage]
    )

    pedaco = np.zeros((640, 1060, 3), dtype=np.uint8)
    pedaco[100:140, 200:240] = 200

    cv2_tensor, cv2_gain, cv2_pad = _letterbox_tensor(pedaco, 1600, "cv2")
    cuda_tensor, cuda_gain, cuda_pad = _letterbox_tensor(pedaco, 1600, "cuda")

    assert (cv2_gain, cv2_pad) == (cuda_gain, cuda_pad)
    assert cv2_tensor.shape == cuda_tensor.shape
    assert cuda_tensor.dtype == np.float32
    assert cuda_tensor.flags.c_contiguous
    assert float(np.abs(cv2_tensor - cuda_tensor).max()) < 0.01
