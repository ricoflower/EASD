"""Spike I/O and the shared raw-image/padded-image coordinate convention."""

from pathlib import Path

import numpy as np
import torch


def parse_annotation(line):
    """Read trailing xywh,class fields while allowing spaces in the file path."""
    fields = line.strip().split()
    labels = []
    while fields and fields[-1].count(",") == 4:
        labels.append([float(value) for value in fields.pop().split(",")])
    if not fields:
        raise ValueError("Annotation has no spike file path")
    return Path(" ".join(fields)), np.asarray(labels[::-1], dtype=np.float32).reshape(-1, 5)


def padded_shape(shape, multiple=16):
    """8-pixel attention windows merged in 2x2 groups require multiples of 16."""
    return tuple((int(size) + multiple - 1) // multiple * multiple for size in shape)


def padding_for(shape, target_shape=None):
    height, width = shape
    target_h, target_w = target_shape or padded_shape(shape)
    if target_h < height or target_w < width:
        raise ValueError(f"Cannot pad {tuple(shape)} to {(target_h, target_w)}")
    top, left = (target_h - height) // 2, (target_w - width) // 2
    return left, target_w - width - left, top, target_h - height - top


def pad_spikes(spikes, target_shape=None):
    """Symmetric padding, identical to np.pad(mode='symmetric'), on any device."""
    left, right, top, bottom = padding_for(spikes.shape[-2:], target_shape)
    if not any((left, right, top, bottom)):
        return spikes
    height, width = spikes.shape[-2:]

    def indices(size, before, after):
        index = torch.arange(-before, size + after, device=spikes.device) % (2 * size)
        return torch.where(index < size, index, 2 * size - 1 - index)

    return spikes.index_select(-2, indices(height, top, bottom)).index_select(
        -1, indices(width, left, right))


def raw_to_spike(video_seq, height=250, width=400, flipud=True):
    """Decode the little-endian, bit-packed PKU-Vidar .dat format."""
    data = np.frombuffer(video_seq, dtype=np.uint8) if isinstance(
        video_seq, (bytes, bytearray)) else np.asarray(video_seq, dtype=np.uint8)
    pixels = height * width
    if pixels % 8 or data.size % (pixels // 8):
        raise ValueError("Incomplete bit-packed spike frames")
    spikes = np.unpackbits(data, bitorder="little").reshape(-1, height, width)
    return spikes[:, ::-1].copy() if flipud else spikes


def load_spikes(path, input_type="dsec", dat_start=180):
    path = Path(path)
    if input_type == "spike":
        spikes = raw_to_spike(path.read_bytes())[dat_start:dat_start + 41]
    elif input_type == "dsec":
        spikes = np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    if spikes.ndim != 3 or spikes.shape[0] != 41:
        raise ValueError(f"{path}: expected (41, H, W), got {spikes.shape}")
    return torch.from_numpy(np.ascontiguousarray(spikes, dtype=np.float32))


def boxes_to_padded(boxes, source_shape, target_shape):
    """Convert normalized xywh labels to the padded canvas without resizing."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5).copy()
    left, _, top, _ = padding_for(source_shape, target_shape)
    height, width = source_shape
    target_h, target_w = target_shape
    boxes[:, 0] = (boxes[:, 0] * width + left) / target_w
    boxes[:, 1] = (boxes[:, 1] * height + top) / target_h
    boxes[:, 2] *= width / target_w
    boxes[:, 3] *= height / target_h
    return boxes


def clip_boxes(boxes, num_classes):
    """Clip partially visible boxes and discard boxes entirely outside the sensor."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5).copy()
    if not len(boxes):
        return boxes
    if (not np.isfinite(boxes).all() or (boxes[:, 2:4] <= 0).any()
            or (boxes[:, 4] < 0).any() or (boxes[:, 4] >= num_classes).any()
            or (boxes[:, 4] != np.floor(boxes[:, 4])).any()):
        raise ValueError("Expected finite xywh labels, positive sizes, and valid integer class IDs")
    corners_min = (boxes[:, :2] - boxes[:, 2:4] / 2).clip(0, 1)
    corners_max = (boxes[:, :2] + boxes[:, 2:4] / 2).clip(0, 1)
    boxes[:, :2] = (corners_min + corners_max) / 2
    boxes[:, 2:4] = corners_max - corners_min
    return boxes[(boxes[:, 2:4] > 0).all(axis=1)]
