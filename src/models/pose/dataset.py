"""Monta o dataset YOLO-pose de recortes de jogador, e nao de quadros inteiros.

Medido no proprio modulo que consome o peso: rodando o estimador uma vez por quadro num clipe do
parceiro, ele achou 2 pessoas onde o detector de jogadores achou 14, porque jogador de transmissao
tem caixa de cerca de 32 por 102 px num quadro de 1920 por 1080. Sobre o recorte ampliado do mesmo
jogador, 14 de 14. Se a producao so ve recorte, treinar em quadro inteiro ensina outro problema.

Por isso a margem vem de `edge.modules.pose.MARGEM` em vez de ser redigitada, e `train.py` treina
no mesmo `IMGSZ` do adaptador.

A entrada e qualquer arquivo de keypoints no formato COCO com os 17 pontos na ordem canonica: serve
o `person_keypoints_train2017.json` e serve um conjunto de futebol exportado no mesmo formato. O
dominio dos dois nao e o mesmo, e o script imprime a distribuicao de altura das caixas que aceitou
para quem treina comparar com os 93,4 px de mediana do finalizador nos clipes do parceiro.

Uso:

    uv run --extra training python models/pose/dataset.py \\
        --annotations ../data/coco/annotations/person_keypoints_train2017.json \\
        --images ../data/coco/train2017 \\
        --output ../data/pose-crops
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from edge.modules.pose import COCO_KEYPOINTS, MARGEM

FLIP_IDX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)
"""Para onde cada keypoint vai quando a imagem espelha horizontalmente.

E o unico motivo pelo qual `train.py` pode ligar `fliplr`. O treino de campo deste repositorio usa
`fliplr = 0.0` porque a correspondencia lateral dos 29 pontos dele nunca foi conferida; a dos 17 do
COCO e publicada, e `_flip_idx_dos_nomes` a deriva para o teste conferir a constante contra a
derivacao em vez de contra a memoria de quem a digitou.
"""

MIN_PONTOS_VISIVEIS = 8
"""Abaixo disso a instancia e um pedaco de corpo atras de outro jogador."""

MIN_ALTURA_PX = 48
"""Piso da altura da caixa. O minimo medido no conjunto do parceiro e 23,5 px, e o piso fica acima
dele de proposito: o recorte vai ser AMPLIADO para 256 px, e ampliar 23 px nao cria pixel."""

MAX_ALTURA_PX = 640
"""Teto da altura da caixa. Sem ele o COCO entra dominado por pessoa em primeiro plano, que e a
escala em que o peso generico ja acerta e a que a borda nunca ve."""

MIN_RAZAO_ALTURA_LARGURA = 1.2
"""Filtro de pose e nao de imagem: o gesto que este peso precisa descrever acontece de pe."""

SPLITS = ("train", "valid", "test")
PESOS_DO_SORTEIO = (80, 10, 10)
"""A divisao e por IMAGEM e nao por recorte: dois jogadores da mesma foto dividem iluminacao,
camera e compressao, e deixar um de cada lado viraria memorizacao com cara de generalizacao."""


class DatasetError(ValueError): ...


@dataclass(frozen=True)
class Instancia:
    image_id: int
    arquivo: str
    largura: int
    altura: int
    box: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, int], ...]

    @property
    def altura_da_caixa(self) -> float:
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class Recorte:
    origem: tuple[int, int]
    tamanho: tuple[int, int]
    box: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, int], ...]


def _flip_idx_dos_nomes(nomes: Sequence[str]) -> tuple[int, ...]:
    posicao = {nome: indice for indice, nome in enumerate(nomes)}
    saida: list[int] = []
    for nome in nomes:
        if nome.endswith("_left"):
            par = nome.removesuffix("_left") + "_right"
        elif nome.endswith("_right"):
            par = nome.removesuffix("_right") + "_left"
        else:
            par = nome
        if par not in posicao:
            raise DatasetError(f"{nome} nao tem par lateral em {list(nomes)}")
        saida.append(posicao[par])
    return tuple(saida)


def ler_instancias(annotations: Path, *, min_pontos: int = MIN_PONTOS_VISIVEIS) -> list[Instancia]:
    if not annotations.is_file():
        raise DatasetError(f"{annotations} nao existe")
    documento = json.loads(annotations.read_text(encoding="utf-8"))
    imagens = {int(imagem["id"]): imagem for imagem in documento.get("images", [])}
    categorias = {
        int(c["id"]) for c in documento.get("categories", []) if c.get("name") == "person"
    }
    if not categorias:
        raise DatasetError(f"{annotations.name} nao declara a categoria 'person'")
    aceitas: list[Instancia] = []
    for anotacao in documento.get("annotations", []):
        instancia = _instancia(anotacao, imagens, categorias, min_pontos)
        if instancia is not None:
            aceitas.append(instancia)
    aceitas.sort(key=lambda i: (i.image_id, i.box))
    return aceitas


def _instancia(
    anotacao: dict[str, object],
    imagens: dict[int, dict[str, object]],
    categorias: set[int],
    min_pontos: int,
) -> Instancia | None:
    if int(anotacao.get("category_id", -1)) not in categorias:
        return None
    if int(anotacao.get("iscrowd", 0)):
        return None
    crus = anotacao.get("keypoints")
    if not isinstance(crus, list) or len(crus) != 3 * len(COCO_KEYPOINTS):
        return None
    pontos = tuple(
        (float(crus[i * 3]), float(crus[i * 3 + 1]), int(crus[i * 3 + 2]))
        for i in range(len(COCO_KEYPOINTS))
    )
    if sum(1 for _x, _y, v in pontos if v > 0) < min_pontos:
        return None
    if not _tem_os_dois_tornozelos(pontos):
        return None
    caixa = anotacao.get("bbox")
    if not isinstance(caixa, list) or len(caixa) != 4:
        return None
    x, y, largura, altura = (float(valor) for valor in caixa)
    if largura <= 0 or altura <= 0:
        return None
    if not MIN_ALTURA_PX <= altura <= MAX_ALTURA_PX:
        return None
    if altura / largura < MIN_RAZAO_ALTURA_LARGURA:
        return None
    imagem = imagens.get(int(anotacao.get("image_id", -1)))
    if imagem is None:
        return None
    return Instancia(
        image_id=int(imagem["id"]),
        arquivo=str(imagem["file_name"]),
        largura=int(imagem["width"]),
        altura=int(imagem["height"]),
        box=(x, y, x + largura, y + altura),
        keypoints=pontos,
    )


def _tem_os_dois_tornozelos(pontos: Sequence[tuple[float, float, int]]) -> bool:
    """O filtro que decide o dataset: e o tornozelo que separa perna esquerda de direita, e o teto
    de quem nao as separa e 0,6705 sobre os 604 registros do conjunto."""
    indices = {nome: i for i, nome in enumerate(COCO_KEYPOINTS)}
    return all(pontos[indices[nome]][2] > 0 for nome in ("ankle_left", "ankle_right"))


def recortar(instancia: Instancia, *, margem: float = MARGEM) -> Recorte | None:
    """Ponto que a margem nao alcanca vira `v = 0` em vez de coordenada recortada ate a borda:
    grudar na moldura ensinaria que todo tornozelo cortado mora ali."""
    x1, y1, x2, y2 = instancia.box
    margem_x, margem_y = (x2 - x1) * margem, (y2 - y1) * margem
    esquerda = max(0, int(x1 - margem_x))
    topo = max(0, int(y1 - margem_y))
    direita = min(instancia.largura, int(x2 + margem_x))
    baixo = min(instancia.altura, int(y2 + margem_y))
    largura, altura = direita - esquerda, baixo - topo
    if largura < 2 or altura < 2:
        return None
    pontos = tuple(
        (px - esquerda, py - topo, v)
        if v > 0 and 0 <= px - esquerda < largura and 0 <= py - topo < altura
        else (0.0, 0.0, 0)
        for px, py, v in instancia.keypoints
    )
    if not _tem_os_dois_tornozelos(pontos):
        return None
    return Recorte(
        origem=(esquerda, topo),
        tamanho=(largura, altura),
        box=(
            max(0.0, x1 - esquerda),
            max(0.0, y1 - topo),
            min(float(largura), x2 - esquerda),
            min(float(altura), y2 - topo),
        ),
        keypoints=pontos,
    )


def linha_yolo(recorte: Recorte) -> str:
    largura, altura = recorte.tamanho
    x1, y1, x2, y2 = recorte.box
    campos = [
        "0",
        _n((x1 + x2) / 2 / largura),
        _n((y1 + y2) / 2 / altura),
        _n((x2 - x1) / largura),
        _n((y2 - y1) / altura),
    ]
    for px, py, v in recorte.keypoints:
        campos += [_n(px / largura), _n(py / altura), str(int(v))]
    return " ".join(campos)


def _n(valor: float) -> str:
    return f"{min(1.0, max(0.0, valor)):.6f}"


def split_de(image_id: int, *, seed: int) -> str:
    """Sorteio por hash e nao por `random`: rodar de novo com mais imagens nao pode mover para o
    treino uma imagem que ja estava no teste."""
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    ponto = int.from_bytes(digest[:8], "big") % sum(PESOS_DO_SORTEIO)
    acumulado = 0
    for nome, peso in zip(SPLITS, PESOS_DO_SORTEIO, strict=True):
        acumulado += peso
        if ponto < acumulado:
            return nome
    return SPLITS[-1]


def dataset_yaml(output: Path) -> str:
    pontos = "\n".join(f"  {i}: {nome}" for i, nome in enumerate(COCO_KEYPOINTS))
    return f"""# Gerado por models/pose/dataset.py. Recortes de jogador, nao quadros inteiros.
path: {output.resolve().as_posix()}
train: images/train
val: images/valid
test: images/test

names:
  0: person

kpt_shape: [{len(COCO_KEYPOINTS)}, 3]
flip_idx: {list(FLIP_IDX)}

keypoint_names:
{pontos}
"""


def escrever(
    instancias: Sequence[Instancia], images_root: Path, output: Path, *, seed: int
) -> Counter[str]:
    import cv2

    contagem: Counter[str] = Counter()
    for instancia in instancias:
        recorte = recortar(instancia)
        if recorte is None:
            contagem["sem_recorte"] += 1
            continue
        origem = images_root / instancia.arquivo
        imagem = cv2.imread(str(origem))
        if imagem is None:
            contagem["imagem_ausente"] += 1
            continue
        esquerda, topo = recorte.origem
        largura, altura = recorte.tamanho
        pedaco = imagem[topo : topo + altura, esquerda : esquerda + largura]
        split = split_de(instancia.image_id, seed=seed)
        nome = f"{instancia.image_id:012d}_{esquerda}_{topo}"
        destino_img = output / "images" / split / f"{nome}.jpg"
        destino_lbl = output / "labels" / split / f"{nome}.txt"
        destino_img.parent.mkdir(parents=True, exist_ok=True)
        destino_lbl.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destino_img), pedaco)
        destino_lbl.write_text(linha_yolo(recorte) + "\n", encoding="utf-8")
        contagem[split] += 1
    return contagem


def alturas(instancias: Sequence[Instancia]) -> Iterator[float]:
    for instancia in instancias:
        yield instancia.altura_da_caixa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="JSON de keypoints COCO")
    parser.add_argument("--images", type=Path, required=True, help="Pasta das imagens de origem")
    parser.add_argument("--output", type=Path, required=True, help="Pasta do dataset final")
    parser.add_argument("--seed", type=int, default=20260922, help="Semente do sorteio dos splits")
    parser.add_argument("--limit", type=int, default=0, help="Para em N instancias. 0 e sem limite")
    args = parser.parse_args()

    if _flip_idx_dos_nomes(COCO_KEYPOINTS) != FLIP_IDX:
        raise DatasetError("FLIP_IDX nao corresponde aos nomes: espelhar corromperia o rotulo")

    instancias = ler_instancias(args.annotations)
    if args.limit:
        instancias = instancias[: args.limit]
    if not instancias:
        raise DatasetError(
            "nenhuma instancia sobreviveu aos filtros: confira a fonte e os limiares"
        )

    medidas = sorted(alturas(instancias))
    meio = medidas[len(medidas) // 2]
    imagens = len({i.image_id for i in instancias})
    print(f"{len(instancias)} instancias aceitas de {imagens} imagens")
    print(
        f"altura da caixa: minimo {medidas[0]:.1f} px, "
        f"mediana {meio:.1f} px, maximo {medidas[-1]:.1f} px"
    )
    print("no conjunto do parceiro o finalizador tem 93,4 px de mediana; compare antes de treinar")

    contagem = escrever(instancias, args.images, args.output, seed=args.seed)
    for chave in (*SPLITS, "sem_recorte", "imagem_ausente"):
        if contagem[chave]:
            print(f"  {chave}: {contagem[chave]}")

    caminho = args.output / "dataset.yaml"
    caminho.write_text(dataset_yaml(args.output), encoding="utf-8")
    print(f"\ndataset.yaml em {caminho.resolve()}")


if __name__ == "__main__":
    main()
