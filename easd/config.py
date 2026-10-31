"""Shared model constants and configuration-file readers."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DEFAULT_ANCHOR_MASKS = ((6, 7, 8), (3, 4, 5), (0, 1, 2))
INPUT_FRAMES = 41


def class_file(input_type="dsec"):
    return CONFIG_DIR / ("pku_classes.txt" if input_type == "spike" else "dsec_classes.txt")


def load_class_names(path):
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not names or len(set(names)) != len(names):
        raise ValueError("Class names must be nonempty and unique")
    return names


def load_anchors(path=None):
    path = Path(path) if path else CONFIG_DIR / "anchors.txt"
    anchors = np.asarray(
        [float(value) for value in path.read_text().strip().split(",")], dtype=np.float32
    ).reshape(-1, 2)
    if anchors.shape != (9, 2) or not np.isfinite(anchors).all() or (anchors <= 0).any():
        raise ValueError("Expected nine finite positive width,height anchors")
    return anchors
