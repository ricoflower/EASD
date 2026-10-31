"""Explicit, offline checkpoint loading for EASD and CSPDarknet."""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def read_state_dict(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{path}: expected a tensor state_dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_backbone(backbone, path=None, phi="s"):
    if path is None:
        candidates = [ROOT / "weights" / f"cspdarknet_{phi}_backbone.pth",
                      ROOT / "model_data" / f"cspdarknet_{phi}_backbone.pth"]
        if phi == "s":
            candidates.extend((ROOT / "weights/yolov5_s.pth", ROOT / "yolov5_s.pth"))
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No local CSPDarknet-{phi} weights; supply --backbone-weights")
    state = read_state_dict(path)
    if any(key.startswith("backbone.") for key in state):
        state = {key.removeprefix("backbone."): value for key, value in state.items()
                 if key.startswith("backbone.")}
    backbone.load_state_dict(state, strict=True)
    return str(path)
