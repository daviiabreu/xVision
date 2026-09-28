"""Exporta os modelos para TensorRT INT8 com calibração local.

Uso:
  python3 -u quantizacao_int8.py                    # exporta todos os modelos
  python3 -u quantizacao_int8.py --modelo ball      # exporta apenas um modelo

Os arquivos .pt devem estar na mesma pasta que este script.
Os datasets de calibração devem ficar em dataset/<modelo>/ com um .yaml dentro.
Os outputs (.onnx, .engine) são salvos em subpastas por modelo:
  ball_int8/ball.int8.onnx, ball_int8/ball.int8.engine
"""

import argparse
import gc
import shutil
from pathlib import Path
from time import perf_counter

MODELOS = {"player": 960, "keypoints": 960, "ball": 1280}
PASTA = Path(__file__).resolve().parent


def localizar_dataset(modelo: str, dataset_root: Path) -> Path:
    pasta = dataset_root / modelo
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta de dataset não encontrada: {pasta}")

    for nome in ("dataset.yaml", "data.yaml", "players_data.yaml"):
        arquivo = pasta / nome
        if arquivo.is_file():
            return arquivo

    yamls = sorted(pasta.glob("*.yaml")) + sorted(pasta.glob("*.yml"))
    if not yamls:
        raise FileNotFoundError(f"Nenhum .yaml/.yml encontrado em: {pasta}")
    return yamls[0]


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
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Raiz dos datasets de calibração (padrão: dataset/).",
    )
    args = parser.parse_args()
    if args.workspace <= 0:
        parser.error("--workspace deve ser maior que zero")

    print("Iniciando exportação TensorRT INT8...", flush=True)
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível neste Python. Verifique o ambiente da Jetson.")

    dataset_root = (
        args.dataset_root if args.dataset_root.is_absolute() else PASTA / args.dataset_root
    )
    nomes = [args.modelo] if args.modelo else list(MODELOS)

    for nome in nomes:
        pt = PASTA / f"{nome}.pt"
        if not pt.is_file():
            raise FileNotFoundError(f"Modelo não encontrado: {pt}")

    for indice, nome in enumerate(nomes, start=1):
        inicio = perf_counter()
        data_yaml = localizar_dataset(nome, dataset_root)
        print(
            f"[{indice}/{len(nomes)}] Exportando {nome}.pt "
            f"(imgsz={MODELOS[nome]}) usando {data_yaml}...",
            flush=True,
        )

        modelo = YOLO(str(PASTA / f"{nome}.pt"))
        arquivo = modelo.export(
            format="engine",
            int8=True,
            imgsz=MODELOS[nome],
            device=0,
            batch=1,
            dynamic=False,
            workspace=args.workspace,
            data=str(data_yaml.resolve()),
            simplify=False,
        )

        if not arquivo or not Path(arquivo).is_file() or Path(arquivo).stat().st_size == 0:
            raise RuntimeError(f"A exportação de {nome} não gerou um engine válido.")

        pasta_saida = PASTA / f"{nome}_int8"
        pasta_saida.mkdir(exist_ok=True)

        nomes_saida = {".onnx": f"{nome}.int8.onnx", ".engine": f"{nome}.int8.engine"}
        for ext, destino_nome in nomes_saida.items():
            origem = PASTA / f"{nome}{ext}"
            if origem.is_file():
                destino = pasta_saida / destino_nome
                destino.unlink(missing_ok=True)
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
        "\nExportações INT8 concluídas. Valide a precisão com o dataset de "
        "calibração antes de usar em produção.",
        flush=True,
    )


if __name__ == "__main__":
    main()
