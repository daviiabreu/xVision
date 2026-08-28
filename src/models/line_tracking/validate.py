"""
Instalação:
    pip install ultralytics --break-system-packages

Exemplos de uso:
    # uma imagem
    python 4_test_yolo_pose.py --weights ./best.pt --source foto.jpg

    # pasta de imagens
    python 4_test_yolo_pose.py --weights ./best.pt --source ./pasta_de_fotos/

    # vídeo
    python 4_test_yolo_pose.py --weights ./best.pt --source video.mp4

    # webcam
    python 4_test_yolo_pose.py --weights ./best.pt --source 0

Resultados são salvos em runs/pose/predict*/ (imagens/vídeos anotados) e,
se --save_txt, também os keypoints em formato .txt por arquivo.
"""

import argparse

from ultralytics import YOLO

KEYPOINT_NAMES = [
    "sideline_top_left",
    "big_rect_left_top_pt1",
    "big_rect_left_top_pt2",
    "big_rect_left_bottom_pt1",
    "big_rect_left_bottom_pt2",
    "small_rect_left_top_pt1",
    "small_rect_left_top_pt2",
    "small_rect_left_bottom_pt1",
    "small_rect_left_bottom_pt2",
    "sideline_bottom_left",
    "left_semicircle_right",
    "center_line_top",
    "center_line_bottom",
    "center_circle_top",
    "center_circle_bottom",
    "field_center",
    "sideline_top_right",
    "big_rect_right_top_pt1",
    "big_rect_right_top_pt2",
    "big_rect_right_bottom_pt1",
    "big_rect_right_bottom_pt2",
    "small_rect_right_top_pt1",
    "small_rect_right_top_pt2",
    "small_rect_right_bottom_pt1",
    "small_rect_right_bottom_pt2",
    "sideline_bottom_right",
    "right_semicircle_left",
    "center_circle_left",
    "center_circle_right",
]


def main():
    parser = argparse.ArgumentParser(description="Inferência do modelo SoccerNet Keypoints")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Caminho para o .pt treinado (ex: best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Imagem, pasta, vídeo, ou índice de webcam (ex: 0)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confiança mínima para detecção")
    parser.add_argument(
        "--kpt_conf",
        type=float,
        default=0.5,
        help="Confiança mínima de cada keypoint para ser considerado 'visível' no print",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--save_txt", action="store_true", help="Também salvar keypoints em .txt")
    parser.add_argument(
        "--show_labels",
        action="store_true",
        help="Imprimir nome + coordenada de cada keypoint no terminal",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/pose",
        help="Pasta onde salvar os resultados",
    )
    parser.add_argument("--name", type=str, default="predict")
    args = parser.parse_args()

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        stream=True,
    )

    n_frames = 0
    for result in results:
        n_frames += 1
        if not args.show_labels:
            continue

        if result.keypoints is None or len(result.keypoints) == 0:
            print(f"[frame {n_frames}] nenhum campo detectado")
            continue

        kpts = result.keypoints.xy[0]
        kconf = result.keypoints.conf[0] if result.keypoints.conf is not None else None

        print(f"\n[frame {n_frames}] {result.path if hasattr(result, 'path') else ''}")
        for i, name in enumerate(KEYPOINT_NAMES):
            x, y = kpts[i].tolist()
            c = kconf[i].item() if kconf is not None else None
            if c is not None and c < args.kpt_conf:
                continue
            conf_str = f" (conf={c:.2f})" if c is not None else ""
            print(f"  {i:2d} {name:30s} x={x:7.1f} y={y:7.1f}{conf_str}")

    print(f"\n{n_frames} frame(s) processado(s). Resultados salvos em: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
