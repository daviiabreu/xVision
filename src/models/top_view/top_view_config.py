from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]


def get_homography_model_path():
    paths = [
        REPO_ROOT / "homografia" / "best.pt",
        REPO_ROOT / "src" / "models" / "homografia" / "best.pt",
    ]

    for path in paths:
        if path.exists():
            return path

    return paths[0]
