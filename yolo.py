"""Spike-native inference, with detections mapped back to the raw sensor canvas."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from nets.yolo import YoloBody
from utils.checkpoints import ROOT, read_state_dict
from utils.spikes import padded_shape, padding_for
from utils.utils import get_anchors, get_classes
from utils.utils_bbox import DecodeBox

ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


class YOLO:
    def __init__(self, model_path=None, classes_path=None, anchors_path=None,
                 phi="s", device="auto", confidence=0.5, nms_iou=0.5,
                 input_shape=(250, 400), cuda=None):
        if cuda is not None:
            device = "cuda" if cuda else "cpu"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_path = Path(model_path) if model_path else ROOT / "weights/easd_dsec_s.pth"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing EASD checkpoint: {self.model_path}. Pass --weights or see weights/README.md")
        classes_path = classes_path or ROOT / "model_data/dsec.txt"
        anchors_path = anchors_path or ROOT / "model_data/yolo_anchors.txt"
        self.class_names, self.num_classes = get_classes(classes_path)
        self.anchors, _ = get_anchors(anchors_path)
        self.confidence, self.nms_iou = confidence, nms_iou
        self.input_shape = tuple(input_shape)
        self.net = YoloBody(ANCHORS_MASK, self.num_classes, phi, input_shape=input_shape)
        self.net.load_state_dict(read_state_dict(self.model_path), strict=True)
        self.net.to(self.device).eval()

    @torch.no_grad()
    def predict_tensor(self, spikes):
        """Return one Nx6 array [xmin,ymin,xmax,ymax,score,class] per sample."""
        # Keep raw dimensions so padding can be removed after decoding.
        spikes = torch.as_tensor(spikes, dtype=torch.float32, device=self.device)
        if spikes.ndim == 3:
            spikes = spikes.unsqueeze(0)
        if spikes.ndim != 4 or spikes.shape[1] != 41:
            raise ValueError("Expected (41,H,W) or (B,41,H,W) spike input")
        raw_shape = tuple(spikes.shape[-2:])
        canvas = padded_shape(raw_shape)
        decoder = DecodeBox(self.anchors, self.num_classes, canvas, ANCHORS_MASK)
        outputs = self.net(spikes)
        predictions = torch.cat(decoder.decode_box(outputs[:3]), dim=1)
        results = decoder.non_max_suppression(
            predictions, self.num_classes, canvas, canvas, False,
            conf_thres=self.confidence, nms_thres=self.nms_iou)
        left, _, top, _ = padding_for(raw_shape, canvas)
        detections = []
        for result in results:
            if result is None:
                detections.append(np.empty((0, 6), dtype=np.float32))
                continue
            boxes = result[:, [1, 0, 3, 2]].copy()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left).clip(0, raw_shape[1])
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top).clip(0, raw_shape[0])
            scores = result[:, 4:5] * result[:, 5:6]
            output = np.concatenate((boxes, scores, result[:, 6:7]), axis=1)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            detections.append(output[valid].astype(np.float32))
        return detections

    def visualize(self, spikes, detections):
        intensity = torch.as_tensor(spikes).float().mean(0).cpu().numpy()
        image = Image.fromarray((intensity.clip(0, 1) * 255).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for xmin, ymin, xmax, ymax, score, category in detections:
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)
            draw.text((xmin, ymin), f"{self.class_names[int(category)]} {score:.2f}", fill="yellow")
        return image

    def detect_spike(self, spikes, dir_save_path, image_id):
        detections = self.predict_tensor(spikes)[0]
        image = self.visualize(spikes, detections)
        directory = Path(dir_save_path)
        directory.mkdir(parents=True, exist_ok=True)
        image.save(directory / f"{image_id}.png")
        return image
