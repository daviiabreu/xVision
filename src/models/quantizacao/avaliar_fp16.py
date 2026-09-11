"""Valida o .fp16.onnx em qualquer PC com ONNX Runtime + GPU (ex: A4000).

Uso:
  # Validação com dataset (busca o .yaml em dataset/<modelo>/)
  python3 -u avaliar_fp16.py validar --modelo ball

  # Inferência em vídeo
  python3 -u avaliar_fp16.py video --modelo ball --video video.mp4

O modelo carregado é <modelo>/<modelo>.fp16.onnx (gerado na Jetson pelo quantizacao.py).
Esse arquivo contém os pesos FP16 e permite comparar métricas com o .pt original
sem precisar de TensorRT — roda via ONNX Runtime com provider CUDA.
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

PASTA = Path(__file__).resolve().parent
MODELOS = {"player": ("detect", 960), "ball": ("detect", 1280), "keypoints": ("pose", 960)}


def salvar_json(caminho, dados):
    def converter(valor):
        if hasattr(valor, "tolist"):
            return valor.tolist()
        if isinstance(valor, Path):
            return str(valor)
        raise TypeError(f"Tipo não serializável: {type(valor)}")

    caminho.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False, allow_nan=False, default=converter) + "\n",
        encoding="utf-8",
    )
    print(f"Relatório: {caminho}", flush=True)


def resumo_metrica(metrica):
    import numpy as np

    ap = np.asarray(metrica.all_ap)
    if ap.size and (ap.ndim != 2 or ap.shape[1] != 10):
        raise ValueError(f"Formato inesperado da matriz AP: {ap.shape}")
    return {
        "precision": float(metrica.mp),
        "recall": float(metrica.mr),
        "f1_medio_por_classe": float(np.mean(metrica.f1)) if len(metrica.f1) else 0.0,
        "mAP50": float(metrica.map50),
        "mAP75": float(metrica.map75),
        "mAP50-90": float(ap[:, :9].mean()) if ap.size else 0.0,
        "mAP50-95": float(metrica.map),
    }


def video(args, modelo, tamanho, saida):
    import cv2
    import numpy as np

    captura = cv2.VideoCapture(str(args.video.resolve()))
    gravador = None
    tempos = {"preprocess": [], "inference": [], "postprocess": [], "predict_parede": []}
    frames = 0
    try:
        ok, frame = captura.read()
        if not ok:
            raise RuntimeError(f"Não foi possível ler o vídeo: {args.video}")
        fps_video = captura.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps_video) or fps_video <= 0:
            raise RuntimeError("O vídeo não informa FPS válido.")
        altura, largura = frame.shape[:2]
        destino = saida / "predicoes.mp4"
        gravador = cv2.VideoWriter(
            str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (largura, altura)
        )
        if not gravador.isOpened():
            raise RuntimeError("OpenCV não conseguiu abrir o gravador MP4 (codec mp4v).")

        opcoes = dict(
            imgsz=tamanho,
            device=0,
            batch=1,
            rect=False,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            save=False,
        )
        print("Aquecendo o modelo (3 inferências fora da medição)...", flush=True)
        for _ in range(3):
            modelo.predict(frame, **opcoes)

        inicio_total = perf_counter()
        while ok and (args.max_frames == 0 or frames < args.max_frames):
            inicio = perf_counter()
            resultado = modelo.predict(frame, **opcoes)[0]
            tempos["predict_parede"].append((perf_counter() - inicio) * 1000)
            for chave in ("preprocess", "inference", "postprocess"):
                tempos[chave].append(float(resultado.speed[chave]))
            gravador.write(resultado.plot(kpt_line=False, line_width=2))
            frames += 1
            if frames % 30 == 0:
                print(f"{frames} frames processados...", flush=True)
            if args.max_frames and frames >= args.max_frames:
                break
            ok, frame = captura.read()
        gravador.release()
        gravador = None
        duracao = perf_counter() - inicio_total
    finally:
        captura.release()
        if gravador is not None:
            gravador.release()

    estatisticas = {
        chave: {
            "media": float(np.mean(v)),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
        }
        for chave, v in tempos.items()
    }
    relatorio = {
        "modelo": args.modelo,
        "formato": "fp16.onnx",
        "video": args.video.resolve(),
        "imgsz": tamanho,
        "conf": args.conf,
        "iou_nms": args.iou,
        "frames": frames,
        "fps_video_original": fps_video,
        "latencias_ms": estatisticas,
        "fps_predict": 1000 / estatisticas["predict_parede"]["media"],
        "fps_processamento_com_desenho_e_gravacao": frames / duracao,
    }
    salvar_json(saida / "velocidade.json", relatorio)
    print(
        f"Vídeo: {destino}\nFPS predict: {relatorio['fps_predict']:.2f}\n"
        f"FPS com gravação: {frames / duracao:.2f}",
        flush=True,
    )


def validar(args, modelo, tamanho, saida, data_yaml):
    metricas = modelo.val(
        data=str(data_yaml.resolve()),
        split=args.split,
        imgsz=tamanho,
        batch=1,
        device=0,
        rect=False,
        workers=0,
        plots=True,
        conf=0.001,
        iou=args.iou,
        project=str(saida),
        name="validacao",
        exist_ok=True,
        verbose=True,
    )
    if metricas.nt_per_class is None or int(metricas.nt_per_class.sum()) == 0:
        raise RuntimeError("Nenhuma anotação encontrada no split.")
    relatorio = {
        "modelo": args.modelo,
        "formato": "fp16.onnx",
        "dataset": data_yaml.resolve(),
        "split": args.split,
        "imgsz": tamanho,
        "conf_minima": 0.001,
        "iou_nms": args.iou,
        "limiares_AP": [round(0.5 + 0.05 * i, 2) for i in range(10)],
        "boxes": resumo_metrica(metricas.box),
        "classes_com_anotacoes": metricas.box.ap_class_index,
        "por_classe": metricas.summary(),
        "tempos_ms_por_imagem": metricas.speed,
    }
    if hasattr(metricas, "pose"):
        relatorio["pose"] = resumo_metrica(metricas.pose)
    salvar_json(saida / "metricas.json", relatorio)
    for tipo in ("boxes", "pose"):
        if tipo in relatorio:
            print(
                tipo
                + ": "
                + ", ".join(
                    f"{k}={v:.4f}" for k, v in relatorio[tipo].items() if isinstance(v, float)
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("modo", choices=("video", "validar"))
    parser.add_argument("--modelo", required=True, choices=MODELOS)
    parser.add_argument("--video", type=Path, help="Caminho do vídeo (modo video).")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or args.max_frames < 0:
        parser.error("conf/iou devem estar entre 0 e 1, e max-frames deve ser >= 0.")
    if args.modo == "video" and (args.video is None or not args.video.is_file()):
        parser.error("Modo 'video' requer --video com caminho válido.")

    # Modelo: sempre o .fp16.onnx na subpasta do modelo
    arquivo = PASTA / args.modelo / f"{args.modelo}.fp16.onnx"
    if not arquivo.is_file():
        parser.error(
            f"Arquivo não encontrado: {arquivo}\n"
            f"Copie o .fp16.onnx gerado na Jetson para esta pasta."
        )

    # Dataset auto-descoberto
    data_yaml = None
    if args.modo == "validar":
        pasta_dataset = PASTA / "dataset" / args.modelo
        if not pasta_dataset.is_dir():
            parser.error(f"Pasta de dataset não encontrada: {pasta_dataset}")
        yamls = list(pasta_dataset.glob("*.yaml")) + list(pasta_dataset.glob("*.yml"))
        if not yamls:
            parser.error(f"Nenhum .yaml/.yml encontrado em: {pasta_dataset}")
        data_yaml = yamls[0]
        print(f"Dataset: {data_yaml}", flush=True)

    # Verifica ONNX Runtime com GPU
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            print(
                "AVISO: CUDAExecutionProvider não disponível. Rodando em CPU (mais lento).",
                flush=True,
            )
            print(f"Providers disponíveis: {providers}", flush=True)
    except ImportError as err:
        raise RuntimeError(
            "onnxruntime não encontrado. Instale com:\n  pip install onnxruntime-gpu"
        ) from err

    print(f"Carregando {arquivo}...", flush=True)
    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path

    tarefa, tamanho = MODELOS[args.modelo]
    modelo = YOLO(str(arquivo), task=tarefa)
    saida = increment_path(PASTA / "resultados" / f"{args.modelo}_fp16_{args.modo}", mkdir=True)

    if args.modo == "video":
        video(args, modelo, tamanho, saida)
    else:
        validar(args, modelo, tamanho, saida, data_yaml)


if __name__ == "__main__":
    main()
