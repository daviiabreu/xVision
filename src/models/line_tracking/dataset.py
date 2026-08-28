"""
Instalação:
    pip install SoccerNet huggingface_hub --break-system-packages

Uso básico (baixa tudo e organiza em ./dataset):
    python 1_prepare_dataset.py --output ./dataset

Uso pulando downloads (se já tiver baixado antes, só reorganiza):
    python 1_prepare_dataset.py --output ./dataset --skip_images --skip_labels \
        --images_cache ./downloads/images --labels_cache ./downloads/labels
"""

import argparse
import shutil
import zipfile
from pathlib import Path

HF_REPO_ID = "Adit-jain/Soccana_Keypoint_detection_v1"
HF_LABELS_FILENAME = "dataset_labels.zip"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

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


def extract_split_zip(images_root: Path, split: str):
    zip_path = images_root / f"{split}.zip"
    split_dir = images_root / split

    already_has_images = split_dir.exists() and any(
        p.suffix.lower() in IMAGE_EXTS for p in split_dir.rglob("*") if p.is_file()
    )
    if already_has_images:
        return

    if not zip_path.exists():
        return

    print(f"  extraindo {zip_path.name}...")
    split_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(split_dir)

    entries = list(split_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(split_dir / item.name))
        nested.rmdir()


def download_images(images_cache_dir: Path, challenge: bool) -> Path:
    from SoccerNet.Downloader import SoccerNetDownloader

    images_cache_dir.mkdir(parents=True, exist_ok=True)
    print("== Baixando imagens do SoccerNet (calibration-2023) ==")
    downloader = SoccerNetDownloader(LocalDirectory=str(images_cache_dir))

    splits = list(SPLITS)
    if challenge:
        splits.append("challenge")

    downloader.downloadDataTask(task="calibration-2023", split=splits)

    images_root = images_cache_dir / "calibration-2023"

    print("Extraindo zips de imagem...")
    for split in splits:
        extract_split_zip(images_root, split)

    print(f"Imagens salvas em: {images_root.resolve()}")
    return images_root


def download_labels(labels_cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    labels_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== Baixando labels do Hugging Face ({HF_REPO_ID}) ==")
    zip_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=HF_LABELS_FILENAME,
    )

    print(f"Extraindo {HF_LABELS_FILENAME}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(labels_cache_dir)

    print(f"Labels extraídos em: {labels_cache_dir.resolve()}")
    return labels_cache_dir


def find_yolo_labels_root(labels_root: Path) -> Path:
    candidate = labels_root / "yolo_labels"
    if candidate.exists():
        return candidate
    found = [f for f in labels_root.rglob("yolo_labels") if f.is_dir()]
    if found:
        print(f"  (pasta yolo_labels encontrada em: {found[0]})")
        return found[0]
    raise FileNotFoundError(f"Não encontrei nenhuma pasta 'yolo_labels' dentro de '{labels_root}'.")


def resolve_images_root(images_path: Path) -> Path:
    def has_split_with_images(p: Path) -> bool:
        for split in SPLITS:
            split_dir = p / split
            if split_dir.exists() and any(
                f.suffix.lower() in IMAGE_EXTS for f in split_dir.rglob("*") if f.is_file()
            ):
                return True
        return False

    if has_split_with_images(images_path):
        return images_path

    candidate = images_path / "calibration-2023"
    if candidate.exists() and has_split_with_images(candidate):
        return candidate

    for base in [images_path, candidate]:
        if base.exists():
            for split in SPLITS + ["challenge"]:
                extract_split_zip(base, split)
            if has_split_with_images(base):
                return base

    for split_dir in images_path.rglob("train"):
        if split_dir.is_dir() and any(
            f.suffix.lower() in IMAGE_EXTS for f in split_dir.rglob("*") if f.is_file()
        ):
            print(f"  (pasta de imagens encontrada em: {split_dir.parent})")
            return split_dir.parent

    raise FileNotFoundError(
        f"Não encontrei imagens (train/valid/test) dentro de '{images_path}'. "
        "Verifique se os zips do SoccerNet foram extraídos."
    )


def find_image(basename: str, images_dir: Path, split: str):
    candidates = []
    for ext in IMAGE_EXTS:
        candidates.append(images_dir / split / f"{basename}{ext}")
        candidates.append(images_dir / f"{basename}{ext}")
    for c in candidates:
        if c.exists():
            return c
    for ext in IMAGE_EXTS:
        matches = list(images_dir.rglob(f"{basename}{ext}"))
        if matches:
            return matches[0]
    return None


def organize_dataset(labels_root: Path, images_root: Path, output_dir: Path, copy_images: bool):
    yolo_labels_root = find_yolo_labels_root(labels_root)

    total_ok, total_missing = 0, 0
    missing_files = []

    for split in SPLITS:
        split_label_dir = yolo_labels_root / split
        if not split_label_dir.exists():
            print(f"  aviso: split '{split}' não encontrado em yolo_labels/, pulando")
            continue

        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        label_files = sorted(split_label_dir.glob("*.txt"))
        print(f"[{split}] {len(label_files)} labels encontrados")

        for label_path in label_files:
            basename = label_path.stem
            img_src = find_image(basename, images_root, split)

            if img_src is None:
                total_missing += 1
                missing_files.append(f"{split}/{basename}")
                continue

            img_dst = img_out / img_src.name
            lbl_dst = lbl_out / label_path.name

            if copy_images:
                shutil.copy2(img_src, img_dst)
            else:
                if img_dst.exists() or img_dst.is_symlink():
                    img_dst.unlink()
                img_dst.symlink_to(img_src.resolve())

            shutil.copy2(label_path, lbl_dst)
            total_ok += 1

    print(f"\nOK: {total_ok} pares imagem+label organizados")
    if total_missing:
        print(
            "AVISO: "
            f"{total_missing} labels sem imagem correspondente "
            "(não incluídos no dataset final)"
        )
        print("Primeiros exemplos faltando:", missing_files[:10])

    yaml_content = f"""# Gerado automaticamente por 1_prepare_dataset.py
path: {output_dir.resolve()}
train: images/train
val: images/valid
test: images/test

# 1 classe: o campo (pitch) como objeto de bounding box
names:
  0: pitch

# 29 keypoints do campo
kpt_shape: [{len(KEYPOINT_NAMES)}, 3]  # [num_keypoints, dims (x,y,visibility)]

# Sem mapeamento de flip verificado (pt1/pt2 esquerda-direita não confirmado) -
# por segurança, treine com fliplr=0.0 para não corromper os keypoints ao
# espelhar a imagem horizontalmente.
flip_idx: {list(range(len(KEYPOINT_NAMES)))}

# nomes dos keypoints, na ordem (apenas para referência/visualização)
keypoint_names:
{chr(10).join(f"  {i}: {name}" for i, name in enumerate(KEYPOINT_NAMES))}
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\ndataset.yaml criado em: {yaml_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Baixa imagens (SoccerNet) + labels (Hugging Face) e organiza pro treino YOLO-pose"
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./dataset",
        help="Pasta final organizada (images/ + labels/ + dataset.yaml)",
    )
    parser.add_argument(
        "--downloads_dir",
        type=str,
        default="./downloads",
        help="Pasta onde ficam os downloads brutos (cache)",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Também baixar o split 'challenge' de imagens (sem labels)",
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Não baixar as imagens do SoccerNet (usa --images_cache)",
    )
    parser.add_argument(
        "--skip_labels",
        action="store_true",
        help="Não baixar as labels do Hugging Face (usa --labels_cache)",
    )
    parser.add_argument(
        "--images_cache",
        type=str,
        default=None,
        help=(
            "Pasta com imagens já baixadas (ex: .../calibration-2023). "
            "Só necessário com --skip_images"
        ),
    )
    parser.add_argument(
        "--labels_cache",
        type=str,
        default=None,
        help=(
            "Pasta com labels já extraídos (contendo yolo_labels/). Só necessário com --skip_labels"
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copiar imagens no dataset final (padrão: link simbólico)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = Path(args.downloads_dir)

    if args.skip_images:
        if not args.images_cache:
            parser.error("--skip_images requer --images_cache apontando pras imagens já baixadas")
        images_root = resolve_images_root(Path(args.images_cache))
        print(f"Pulando download de imagens, usando: {images_root.resolve()}")
    else:
        images_root = download_images(downloads_dir / "images", args.challenge)

    if args.skip_labels:
        if not args.labels_cache:
            parser.error("--skip_labels requer --labels_cache apontando pras labels já extraídas")
        labels_root = Path(args.labels_cache)
        print(f"Pulando download de labels, usando: {labels_root.resolve()}")
    else:
        labels_root = download_labels(downloads_dir / "labels")

    print("\n== Organizando dataset final ==")
    organize_dataset(labels_root, images_root, output_dir, args.copy)

    print(f"\nPronto! Use este dataset.yaml pra treinar: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
