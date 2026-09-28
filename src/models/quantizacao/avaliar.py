"""Inferência em vídeo e validação dos modelos quantizados.

Uso:
  # Inferência em vídeo
  python3 -u avaliar.py video --modelo ball --video video.mp4

    # Inferência em vídeo com INT8
    python3 -u avaliar.py video --modelo ball --precisao int8 --video video.mp4

  # Validação com dataset (busca o .yaml automaticamente em dataset/<modelo>/)
  python3 -u avaliar.py validar --modelo ball

O engine é carregado de <modelo>/<modelo>.engine para FP16 ou
<modelo>_int8/<modelo>.int8.engine para INT8.
O dataset deve estar em dataset/<modelo>/ com um arquivo .yaml dentro.
Os resultados são salvos em resultados/<modelo>_<precisao>_<modo>/
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
        "AP_por_classe_e_limiar": ap.tolist(),
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
            raise RuntimeError(
                "O vídeo não informa FPS válido; converta-o para um vídeo com FPS definido."
            )
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
            # Pontos do campo: não desenhar conexões do esqueleto humano.
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

    if not frames:
        raise RuntimeError("Nenhum frame foi processado no vídeo.")

    estatisticas = {
        chave: {
            "media": float(np.mean(valores)),
            "p50": float(np.percentile(valores, 50)),
            "p95": float(np.percentile(valores, 95)),
        }
        for chave, valores in tempos.items()
    }
    relatorio = {
        "modelo": args.modelo,
        "video": args.video.resolve(),
        "imgsz": tamanho,
        "conf": args.conf,
        "iou_nms": args.iou,
        "frames": frames,
        "fps_video_original": fps_video,
        "latencias_ms": estatisticas,
        "fps_predict": 1000 / estatisticas["predict_parede"]["media"],
        "fps_processamento_com_desenho_e_gravacao": frames / duracao,
        "aquecimento_excluido": True,
        "nota": (
            "Tempos de um modelo por vez. Não incluem carregamento/aquecimento. "
            "FPS de processamento inclui desenho, codificação e leitura "
            "(exceto o primeiro frame)."
        ),
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
        raise RuntimeError(
            "Nenhuma anotação encontrada no split. Não há métricas de qualidade válidas."
        )
    relatorio = {
        "modelo": args.modelo,
        "dataset": data_yaml.resolve(),
        "split": args.split,
        "imgsz": tamanho,
        "conf_minima": 0.001,
        "iou_nms": args.iou,
        "limiares_AP": [round(0.5 + 0.05 * i, 2) for i in range(10)],
        "nota": (
            "Precision/recall são resumos no ponto de melhor F1 do Ultralytics; "
            "não representam necessariamente conf=0.25. mAP50-90 usa limiares "
            "0.50 a 0.90, passo 0.05. Boxes usam IoU; pose usa OKS."
        ),
        "boxes": resumo_metrica(metricas.box),
        "classes_com_anotacoes": metricas.box.ap_class_index,
        "por_classe": metricas.summary(),
        "tempos_ms_por_imagem": metricas.speed,
        "graficos": saida / "validacao",
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
    parser.add_argument(
        "modo",
        choices=("video", "validar"),
        help="'video' para inferência em vídeo, 'validar' para avaliação com dataset.",
    )
    parser.add_argument(
        "--modelo", required=True, choices=MODELOS, help="Modelo a usar: player, ball ou keypoints."
    )
    parser.add_argument(
        "--precisao",
        choices=("fp16", "int8"),
        default="fp16",
        help="Precisão do engine TensorRT (padrão: fp16).",
    )
    parser.add_argument("--video", type=Path, help="Caminho do vídeo (obrigatório no modo video).")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Split do dataset a avaliar (padrão: val).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confiança mínima no modo vídeo (padrão: 0.25)."
    )
    parser.add_argument("--iou", type=float, default=0.7, help="IoU do NMS (padrão: 0.7).")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limita frames no modo vídeo; 0 processa tudo (padrão: 0).",
    )
    args = parser.parse_args()

    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or args.max_frames < 0:
        parser.error("conf/iou devem estar entre 0 e 1, e max-frames deve ser >= 0.")

    # Valida inputs por modo
    if args.modo == "video" and (args.video is None or not args.video.is_file()):
        parser.error("Modo 'video' requer --video com caminho válido para o arquivo.")

    pasta_modelo = f"{args.modelo}" if args.precisao == "fp16" else f"{args.modelo}_int8"
    nome_engine = (
        f"{args.modelo}.engine" if args.precisao == "fp16" else f"{args.modelo}.int8.engine"
    )

    # Engine em <modelo>/<modelo>.engine para FP16 ou <modelo>_int8/<modelo>.int8.engine para INT8
    arquivo = PASTA / pasta_modelo / nome_engine
    if not arquivo.is_file():
        comando = (
            f"python3 quantizacao.py --modelo {args.modelo}"
            if args.precisao == "fp16"
            else f"python3 quantizacao_int8.py --modelo {args.modelo}"
        )
        parser.error(f"Engine não encontrado: {arquivo}\nExecute primeiro: {comando}")

    # Dataset auto-descoberto em dataset/<modelo>/
    data_yaml = None
    if args.modo == "validar":
        pasta_dataset = PASTA / "dataset" / args.modelo
        if not pasta_dataset.is_dir():
            parser.error(
                f"Pasta de dataset não encontrada: {pasta_dataset}\n"
                f"Crie a pasta e coloque o .yaml do dataset dentro."
            )
        prioridade = [
            pasta_dataset / "dataset.yaml",
            pasta_dataset / "data.yaml",
            pasta_dataset / "players_data.yaml",
        ]
        yamls = [arquivo_yaml for arquivo_yaml in prioridade if arquivo_yaml.is_file()]
        if not yamls:
            yamls = sorted(pasta_dataset.glob("*.yaml")) + sorted(pasta_dataset.glob("*.yml"))
        if not yamls:
            parser.error(f"Nenhum .yaml/.yml encontrado em: {pasta_dataset}")
        data_yaml = yamls[0]
        print(f"Dataset encontrado: {data_yaml}", flush=True)

    print(f"Carregando {arquivo}...", flush=True)
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.files import increment_path

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponível neste Python; execute no venv com acesso à GPU da Jetson."
        )

    tarefa, tamanho = MODELOS[args.modelo]
    modelo = YOLO(str(arquivo), task=tarefa)
    saida = increment_path(
        PASTA / "resultados" / f"{args.modelo}_{args.precisao}_{args.modo}", mkdir=True
    )

    if args.modo == "video":
        video(args, modelo, tamanho, saida)
    else:
        validar(args, modelo, tamanho, saida, data_yaml)


if __name__ == "__main__":
    main()
