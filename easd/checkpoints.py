"""Strict checkpoint loading and explicit migration of the original EASD key schema."""

from dataclasses import dataclass
import re

import torch

from easd.config import ROOT


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    migrated: bool
    removed_unused_keys: tuple[str, ...] = ()


_LEGACY_UNUSED = {"alpha"} | {
    f"sp_attn.group.body.0.mixer.route.out_mask.{layer}.{parameter}"
    for layer in (0, 2)
    for parameter in ("weight", "bias")
}


def read_state_dict(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected a tensor state_dict")
    if "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError(f"{path}: expected a tensor state_dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def migrate_legacy_state_dict(state):
    """Rename known old modules; drop only five explicitly unused tensor entries."""
    legacy = any(
        key.startswith(("extractor.", "sp_attn.", "blockfuse4.", "backbone.")) for key in state
    )
    if not legacy:
        return dict(state), False, ()
    migrated, removed = {}, []
    for key, value in state.items():
        if key in _LEGACY_UNUSED:
            removed.append(key)
            continue
        name = key
        name = re.sub(
            r"^extractor\.conv0_(\d)\.", r"temporal_encoder.branches.\1.input_conv.", name
        )
        name = re.sub(
            r"^extractor\.conv1_(\d)\.", r"temporal_encoder.branches.\1.output_conv.", name
        )
        for old, new in (
            ("extractor.ca.", "temporal_encoder.channel_attention."),
            ("extractor.conv.", "temporal_encoder.projection."),
            ("blockfuse4.global_att_fusion.mlp.", "temporal_fusion.attention."),
            ("sp_attn.conv1.", "spatial_encoder.intensity_embedding."),
            ("sp_attn.group.body_tail.", "spatial_encoder.attention_group.projection."),
            ("sp_attn.group.body.", "spatial_encoder.attention_group.blocks."),
            ("conv4attn.", "spatial_projection."),
            ("backbone.", "decoder."),
            ("yolo_head_P3.", "head_fine."),
            ("yolo_head_P4.", "head_medium."),
            ("yolo_head_P5.", "head_coarse."),
        ):
            if name.startswith(old):
                name = new + name[len(old) :]
                break
        name = name.replace(".mixer.route.", ".mixer.predictor.")
        name = name.replace(".mixer.conv_sptial.", ".mixer.spatial_conv.")
        if name in migrated:
            raise ValueError(f"Checkpoint keys collide after migration: {name}")
        migrated[name] = value
    return migrated, True, tuple(sorted(removed))


def load_easd_weights(model, path):
    state, migrated, removed = migrate_legacy_state_dict(read_state_dict(path))
    model.load_state_dict(state, strict=True)
    return CheckpointInfo(str(path), migrated, removed)


def load_decoder_weights(decoder, path=None, variant="s"):
    if path is None:
        candidates = [ROOT / "weights" / f"cspdarknet_{variant}_backbone.pth"]
        if variant == "s":
            candidates.append(ROOT / "weights/yolov5_s.pth")
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"No local CSPDarknet-{variant} weights; supply --backbone-weights or --scratch"
            )
    state = read_state_dict(path)
    for prefix in ("decoder.", "backbone."):
        if any(key.startswith(prefix) for key in state):
            state = {
                key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
            }
            break
    decoder.load_state_dict(state, strict=True)
    return str(path)
