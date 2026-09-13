from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODULE_DIR.parent


def get_keypoint_model_path():
    return MODELS_DIR / "field_keypoints" / "best.pt"
