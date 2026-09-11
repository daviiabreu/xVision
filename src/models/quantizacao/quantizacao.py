"""Exporta os modelos para TensorRT FP16 na própria Jetson.

Uso:
  python3 -u quantizacao.py                    # exporta todos os modelos
  python3 -u quantizacao.py --modelo ball      # exporta apenas um modelo

Os arquivos .pt devem estar na mesma pasta que este script.
Os outputs (.onnx, .fp16.onnx, .engine) são salvos em subpastas por modelo:
  ball/ball.engine, player/player.engine, keypoints/keypoints.engine
"""

import argparse
import gc
import shutil
from pathlib import Path
from time import perf_counter

MODELOS = {"player": 960, "keypoints": 960, "ball": 1280}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--modelo", choices=MODELOS, help="Exporta apenas este modelo. Omita para exportar todos."
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=1.0,
        help="Limite do workspace TensorRT em GiB (padrão: 1).",
    )
    args = parser.parse_args()
    if args.workspace <= 0:
        parser.error("--workspace deve ser maior que zero")

    print("Iniciando exportação TensorRT FP16...", flush=True)
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível neste Python. Verifique o ambiente da Jetson.")

    pasta = Path(__file__).resolve().parent
    nomes = [args.modelo] if args.modelo else list(MODELOS)

    # Verifica se todos os .pt existem antes de começar
    for nome in nomes:
        pt = pasta / f"{nome}.pt"
        if not pt.is_file():
            raise FileNotFoundError(f"Modelo não encontrado: {pt}")

    for indice, nome in enumerate(nomes, start=1):
        inicio = perf_counter()
        print(
            f"[{indice}/{len(nomes)}] Exportando {nome}.pt (imgsz={MODELOS[nome]})...",
            flush=True,
        )

        modelo = YOLO(str(pasta / f"{nome}.pt"))
        arquivo = modelo.export(
            format="engine",
            quantize=16,
            imgsz=MODELOS[nome],
            device=0,
            batch=1,
            dynamic=False,
            workspace=args.workspace,
            # Evita a etapa opcional que solicita onnxruntime-gpu na Jetson.
            simplify=False,
        )

        if not arquivo or not Path(arquivo).is_file() or Path(arquivo).stat().st_size == 0:
            raise RuntimeError(f"A exportação de {nome} não gerou um engine válido.")

        # Move os arquivos gerados para a subpasta do modelo
        pasta_saida = pasta / nome
        pasta_saida.mkdir(exist_ok=True)

        for ext in (".onnx", ".fp16.onnx", ".engine"):
            origem = pasta / f"{nome}{ext}"
            if origem.is_file():
                destino = pasta_saida / f"{nome}{ext}"
                shutil.move(str(origem), str(destino))
                print(f"  Salvo: {destino}", flush=True)

        print(
            f"[{indice}/{len(nomes)}] {nome} concluído em {perf_counter() - inicio:.1f}s",
            flush=True,
        )
        del modelo
        gc.collect()
        torch.cuda.empty_cache()

    print(
        "\nExportações concluídas. Valide a precisão com imagens reais antes de usar em produção.",
        flush=True,
    )


if __name__ == "__main__":
    main()
