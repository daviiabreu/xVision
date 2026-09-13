from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]

DATASET_DIR = REPO_ROOT / "datasets" / "football_players_v20"
DATA_YAML = DATASET_DIR / "players_data.yaml"

RUN_NAME = "players_yolov8s"
RUNS_DIR = MODULE_DIR / "runs"
RUN_DIR = RUNS_DIR / RUN_NAME
MODEL_PATH = RUN_DIR / "weights" / "best.pt"


def get_model_path():
    return MODULE_DIR / "best.pt"
