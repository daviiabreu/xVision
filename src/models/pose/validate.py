"""Mede o peso treinado sobre a dobra reservada do dataset de recortes, pelo criterio do consumidor.

O `model.val()` do Ultralytics responde mAP de pose, e mAP nao distingue o erro que derruba a
entrega: os dois tornozelos ficam a 10,7 px um do outro (mediana sobre os 38 clipes anotados), e
qualquer tolerancia que aceite o caso facil aceita tambem a troca de lado.

Publica o mesmo trio de `mlops.pose_quality`, importando de la os sigmas e o piso de confianca, e
nao substitui o `xvision eval-pose`: aqui o dominio e o do dataset de treino, e la e o video do
parceiro. Um peso pode subir num e nao no outro, e e essa diferenca que diz se o dataset descreve o
problema.

Uso:

    uv run --extra training python models/pose/validate.py \\
        --weights runs/pose/xvision-pose/weights/best.pt \\
        --data ../data/pose-crops --split test --saida ../var/out/pose-validate.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from edge.features.body_part import MIN_CONFIANCA, TORNOZELO_DIREITO, TORNOZELO_ESQUERDO
from edge.modules.pose import COCO_KEYPOINTS
from edge.modules.pose.yolo import CONF, IMGSZ
from mlops.pose_quality import OKS_SIGMAS


@dataclass(frozen=True)
class Verdade:
    imagem: Path
    box: tuple[float, float, float, float]
    pontos: dict[str, tuple[float, float]]

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


def ler_verdade(imagem: Path, rotulo: Path, tamanho: tuple[int, int]) -> Verdade | None:
    """Le a primeira instancia do arquivo. `dataset.py` escreve um recorte por pessoa, e as demais
    sao ignoradas de proposito em vez de somadas calado."""
    linhas = [linha for linha in rotulo.read_text(encoding="utf-8").splitlines() if linha.strip()]
    if not linhas:
        return None
    campos = linhas[0].split()
    if len(campos) != 5 + 3 * len(COCO_KEYPOINTS):
        return None
    largura, altura = tamanho
    cx, cy, bw, bh = (float(v) for v in campos[1:5])
    pontos: dict[str, tuple[float, float]] = {}
    for indice, nome in enumerate(COCO_KEYPOINTS):
        base = 5 + indice * 3
        x, y, v = (float(campos[base]), float(campos[base + 1]), int(float(campos[base + 2])))
        if v > 0:
            pontos[nome] = (x * largura, y * altura)
    return Verdade(
        imagem=imagem,
        box=(
            (cx - bw / 2) * largura,
            (cy - bh / 2) * altura,
            (cx + bw / 2) * largura,
            (cy + bh / 2) * altura,
        ),
        pontos=pontos,
    )


def oks(verdade: Verdade, previsto: dict[str, tuple[float, float, float]]) -> float | None:
    if verdade.area <= 0 or not verdade.pontos:
        return None
    soma = 0.0
    for nome, (vx, vy) in verdade.pontos.items():
        estimado = previsto.get(nome)
        if estimado is None:
            continue
        distancia2 = (vx - estimado[0]) ** 2 + (vy - estimado[1]) ** 2
        soma += math.exp(-distancia2 / (2 * verdade.area * (2 * OKS_SIGMAS[nome]) ** 2))
    return soma / len(verdade.pontos)


def lado_correto(verdade: Verdade, previsto: dict[str, tuple[float, float, float]]) -> bool | None:
    """`None` quando nao ha lado a conferir: contar ausencia como erro misturaria recusa com
    troca."""
    esquerdo_v = verdade.pontos.get(TORNOZELO_ESQUERDO)
    direito_v = verdade.pontos.get(TORNOZELO_DIREITO)
    esquerdo_p = previsto.get(TORNOZELO_ESQUERDO)
    direito_p = previsto.get(TORNOZELO_DIREITO)
    if esquerdo_v is None or direito_v is None or esquerdo_p is None or direito_p is None:
        return None
    if min(esquerdo_p[2], direito_p[2]) < MIN_CONFIANCA:
        return None
    direto = math.dist(esquerdo_v, esquerdo_p[:2]) + math.dist(direito_v, direito_p[:2])
    trocado = math.dist(esquerdo_v, direito_p[:2]) + math.dist(direito_v, esquerdo_p[:2])
    return direto < trocado


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _mediana(valores: Sequence[float]) -> float | None:
    if not valores:
        return None
    ordenado = sorted(valores)
    meio = len(ordenado) // 2
    if len(ordenado) % 2:
        return ordenado[meio]
    return (ordenado[meio - 1] + ordenado[meio]) / 2


def _melhor(saida: object) -> dict[str, tuple[float, float, float]] | None:
    """O corpo de maior confianca do recorte, na mesma regra do adaptador de producao."""
    poses = getattr(saida, "keypoints", None)
    caixas = getattr(saida, "boxes", None)
    if poses is None or caixas is None or not len(poses.data):
        return None
    confiancas = caixas.conf.tolist()
    melhor = max(range(len(confiancas)), key=lambda i: confiancas[i]) if confiancas else 0
    linha = poses.data[melhor].tolist()
    return {
        nome: (float(ponto[0]), float(ponto[1]), float(ponto[2]) if len(ponto) > 2 else 1.0)
        for nome, ponto in zip(COCO_KEYPOINTS, linha, strict=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mede o peso pelo criterio que a feature consome.")
    parser.add_argument("--weights", type=Path, required=True, help="best.pt treinado")
    parser.add_argument("--data", type=Path, required=True, help="raiz do dataset de recortes")
    parser.add_argument("--split", default="test", help="train, valid ou test")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--device", default="0")
    parser.add_argument("--saida", type=Path, default=None, help="Grava o relatorio em JSON")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

    imagens = sorted((args.data / "images" / args.split).glob("*.jpg"))
    if not imagens:
        raise SystemExit(f"{args.data / 'images' / args.split} nao tem imagem nenhuma")
    modelo = YOLO(str(args.weights))

    okss: list[float] = []
    disponivel = 0
    com_lado = 0
    certos = 0
    respondidos = 0
    for caminho in imagens:
        rotulo = args.data / "labels" / args.split / f"{caminho.stem}.txt"
        if not rotulo.is_file():
            continue
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            continue
        verdade = ler_verdade(caminho, rotulo, (imagem.shape[1], imagem.shape[0]))
        if verdade is None:
            continue
        com_lado += TORNOZELO_ESQUERDO in verdade.pontos and TORNOZELO_DIREITO in verdade.pontos
        saida = modelo.predict(
            source=imagem, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        previsto = _melhor(saida)
        if previsto is None:
            continue
        respondidos += 1
        valor = oks(verdade, previsto)
        if valor is not None:
            okss.append(valor)
        lado = lado_correto(verdade, previsto)
        if lado is None:
            continue
        disponivel += 1
        certos += lado

    relatorio = {
        "schema_version": "1.0",
        "weights": str(args.weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "min_confidence": MIN_CONFIANCA,
        "n_crops": len(imagens),
        "n_answered": respondidos,
        "completion": round(respondidos / len(imagens), 4),
        "oks_median": None if not okss else round(_mediana(okss) or 0.0, 4),
        "oks_at_50": round(sum(v >= 0.50 for v in okss) / len(imagens), 4),
        "oks_at_75": round(sum(v >= 0.75 for v in okss) / len(imagens), 4),
        "n_ankle_crops": com_lado,
        "ankle_available": disponivel,
        "ankle_availability": round(disponivel / com_lado, 4) if com_lado else 0.0,
        "ankle_side_correct": certos,
        "ankle_side_accuracy": round(certos / disponivel, 4) if disponivel else 0.0,
        "ankle_side_wilson_95": [round(v, 4) for v in _wilson(certos, disponivel)],
        "usable_rate": round(certos / com_lado, 4) if com_lado else 0.0,
        "usable_rate_wilson_95": [round(v, 4) for v in _wilson(certos, com_lado)],
    }
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))
    if args.saida is not None:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(json.dumps(relatorio, indent=2) + "\n", encoding="utf-8")
        print(f"  {args.saida}")
    print("este e o dominio do dataset; o do parceiro sai de xvision eval-pose")


if __name__ == "__main__":
    main()
