"""Complete EASD inference: weights, network, decoding, NMS, and sensor coordinates."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from easd.checkpoints import load_easd_weights
from easd.config import ROOT, class_file, load_anchors, load_class_names
from easd.model import EASD
from easd.postprocess import DetectionPostProcessor


def resolve_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


class EASDDetector:
    """Predict one Nx6 [xmin,ymin,xmax,ymax,score,class_id] array per image."""

    def __init__(
        self,
        weights=None,
        classes=None,
        anchors=None,
        variant="s",
        device="auto",
        confidence=0.5,
        nms_iou=0.5,
    ):
        weights = Path(weights) if weights else ROOT / "weights/easd_dsec_s.pth"
        if not weights.is_file():
            raise FileNotFoundError(f"Missing EASD checkpoint: {weights}; see weights/README.md")
        names = load_class_names(classes or class_file())
        model = EASD(num_classes=len(names), variant=variant)
        self.checkpoint_info = load_easd_weights(model, weights)
        self._configure(model, names, load_anchors(anchors), device, confidence, nms_iou)

    def _configure(self, model, class_names, anchors, device, confidence, nms_iou):
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.class_names = list(class_names)
        if len(self.class_names) != model.num_classes:
            raise ValueError("The number of class names does not match the model")
        self.anchors = np.asarray(anchors)
        self.postprocessor = DetectionPostProcessor(
            self.anchors, model.num_classes, model.anchor_masks, confidence, nms_iou
        )

    @classmethod
    def from_model(cls, model, class_names, anchors, device="auto", confidence=0.5, nms_iou=0.5):
        """Wrap an in-memory EASD model for validation without reloading a checkpoint."""
        detector = cls.__new__(cls)
        detector.checkpoint_info = None
        detector._configure(model, class_names, anchors, device, confidence, nms_iou)
        return detector

    @torch.no_grad()
    def predict(self, spikes):
        spikes = torch.as_tensor(spikes, dtype=torch.float32, device=self.device)
        if spikes.ndim == 3:
            spikes = spikes.unsqueeze(0)
        if spikes.ndim != 4 or spikes.shape[1] != 41:
            raise ValueError("Expected (41,H,W) or (B,41,H,W) spike input")
        sensor_shape = tuple(spikes.shape[-2:])
        self.model.eval()
        return self.postprocessor(self.model(spikes).head_logits, sensor_shape)

    def render(self, spikes, detections):
        """Draw detections on a temporal-mean display image; this is not model input."""
        spikes = torch.as_tensor(spikes)
        if spikes.ndim != 3:
            raise ValueError("Rendering requires one (41,H,W) recording")
        intensity = spikes.float().mean(0).cpu().numpy()
        image = Image.fromarray((intensity.clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for xmin, ymin, xmax, ymax, score, category in detections:
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)
            draw.text((xmin, ymin), f"{self.class_names[int(category)]} {score:.2f}", fill="yellow")
        return image
