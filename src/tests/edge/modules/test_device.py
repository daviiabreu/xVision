"""Onde a inferencia roda, que muda entre as tres maquinas do time e a Jetson.

Cravar acelerador quebra alguem: `cuda` quebra o Mac, `mps` quebra o Linux, `cpu` desperdica a
placa. Estes testes fixam a ordem de decisao sem exigir torch instalado, porque o CI nao tem
placa nenhuma e a borda sem extra de modelo nao carrega torch.

Aqui tambem fica a guarda de que meia precisao precisa render mais que precisao cheia. Ela so tem
o que verificar na propria Jetson e pula em qualquer outra maquina, o que e a intencao: guarda de
hardware que falha onde nao ha hardware vira ruido ignorado.

Houve uma segunda guarda, que exigia `sm_87` em `torch.cuda.get_arch_list()`, e ela saiu por estar
errada duas vezes. O invariante era insatisfazivel: a tabela `PYTORCH_RELEASES_CODE_CC` do proprio
torch compila `{80, 90, 100, 110, 120}` para aarch64 em toda versao de CUDA, e nenhum wheel oficial
traz `sm_87`. E a causa que ela afirmava era falsa: trocar para um wheel `+cu132`, que faz o torch
aceitar a placa pela tabela `DEVICE_REQUIREMENT_POST_JETSON_SBSA_UNIFICATION`, mediu 41,5 ms contra
os 40,8 ms de antes, ou seja nada. Meia precisao lenta nunca foi o wheel.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from edge.modules.device import DEVICE_ENV, preferred_device


def _torch_falso(*, cuda: bool, mps: bool) -> types.ModuleType:
    modulo = types.ModuleType("torch")
    modulo.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    modulo.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return modulo


def test_a_variavel_de_ambiente_ganha_de_qualquer_placa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quem opera precisa forcar sem editar arquivo, e e assim que a suite roda igual no CI."""
    monkeypatch.setitem(sys.modules, "torch", _torch_falso(cuda=True, mps=True))
    monkeypatch.setenv(DEVICE_ENV, "cpu")

    assert preferred_device() == "cpu"


def test_sem_torch_instalado_o_piso_e_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A borda sem extra de modelo nao tem torch, e ler um caminho nao pode exigir a pilha."""
    monkeypatch.delenv(DEVICE_ENV, raising=False)

    def recusa(nome: str) -> Any:
        raise ImportError(nome)

    monkeypatch.setattr("edge.modules.device.import_module", recusa)

    assert preferred_device() == "cpu"


@pytest.mark.parametrize(
    ("cuda", "mps", "esperado"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"), (True, True, "cuda")],
)
def test_o_acelerador_so_entra_quando_o_torch_confirma(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, esperado: str
) -> None:
    """`cuda` ganha de `mps` quando os dois respondem, e a ausencia dos dois cai em `cpu`."""
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _torch_falso(cuda=cuda, mps=mps))

    assert preferred_device() == esperado


BENCH_ENV = "XVISION_BENCH"
PESO_JOGADORES = Path("models/track_players/best.pt")
BENCH_IMGSZ = 960
BENCH_REPETICOES = 5


@pytest.mark.skipif(
    os.environ.get(BENCH_ENV, "") not in {"1", "true"},
    reason=f"benchmark de precisao e lento e depende de peso em disco: ligue {BENCH_ENV}=1",
)
def test_meia_precisao_nao_pode_ser_mais_lenta_que_precisao_cheia() -> None:
    """Meia precisao mais lenta que precisao cheia denuncia tensor core ocioso, e o custo e mudo.

    A causa nao e o wheel, e isso foi medido: em torch puro nesta placa o fp16 e 5,9x mais rapido
    que o fp32 num matmul e 1,3 a 1,6x numa convolucao, com o mesmo wheel generico. O que fazia
    meia precisao perder era o parametro `half`, que o ultralytics 8.4 depreciou: no mesmo peso e
    no mesmo quadro, `half=True` mediu 62,4 ms contra 45,2 ms de fp32 e 28,6 ms de
    `quantize='fp16'`.

    Por isso a guarda mede por `quantize`. Medida por `half` ela reprovaria com os tensor core
    funcionando perfeitamente, e o defeito que ela procura passaria despercebido no falso positivo.
    E opt-in porque depende de peso em disco e de placa, nao porque seja secundaria.
    """
    torch = pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")
    if not torch.cuda.is_available():
        pytest.skip("sem CUDA: nao ha o que comparar entre fp16 e fp32")
    if not PESO_JOGADORES.is_file():
        pytest.skip(f"o peso de jogadores nao esta em {PESO_JOGADORES.resolve()}")

    quadro = np.zeros((1080, 1920, 3), dtype=np.uint8)
    fp32 = _mediana_ms(torch, ultralytics, quadro, quantize="fp32")
    fp16 = _mediana_ms(torch, ultralytics, quadro, quantize="fp16")

    assert fp16 <= fp32, (
        f"fp16 mediu {fp16:.1f} ms contra {fp32:.1f} ms de fp32. Meia precisao mais lenta que "
        "precisao cheia indica tensor core ocioso. Confira primeiro se o caminho medido usa "
        "`quantize` e nao o `half` depreciado, que foi a causa da unica vez que isso aconteceu aqui"
    )


def _mediana_ms(torch: Any, ultralytics: Any, quadro: NDArray[np.uint8], *, quantize: str) -> float:
    """Mediana, e nao media, porque uma pausa de governor de clock desloca a media inteira.

    O quadro e sintetico de proposito: os dois lados veem exatamente o mesmo tensor, entao o que
    a comparacao isola e o custo de convolucao, sem deteccao nenhuma para fazer o NMS pesar
    diferente entre as precisoes. O que precisa ser real e a forma 1920x1080, que e a que o
    pipeline entrega.

    Carrega um modelo por precisao para nao medir um backend ja montado na precisao anterior, e
    descarta a primeira passada, que paga a compilacao de PTX. O `synchronize` antes de parar o
    cronometro e o que separa inferencia de enfileiramento: sem ele a chamada volta antes da GPU
    terminar e todas as medidas ficam parecidas e falsas.
    """
    modelo = ultralytics.YOLO(str(PESO_JOGADORES))
    medidas: list[float] = []
    for passada in range(BENCH_REPETICOES + 1):
        inicio = time.perf_counter()
        modelo.predict(quadro, imgsz=BENCH_IMGSZ, quantize=quantize, device="cuda", verbose=False)
        torch.cuda.synchronize()
        decorrido = (time.perf_counter() - inicio) * 1000
        if passada:
            medidas.append(decorrido)
    return statistics.median(medidas)
