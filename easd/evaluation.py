"""Evaluate the same EASDDetector pipeline used for prediction."""

import torch
from tqdm import tqdm

from easd.metrics import coco_ap, normalized_xywh_to_xyxy, voc_ap50
from easd.spikes import clip_boxes, load_spikes, parse_annotation


@torch.no_grad()
def evaluate_detector(detector, annotation_lines, input_type="dsec", metric="voc"):
    if metric not in ("voc", "coco"):
        raise ValueError("metric must be 'voc' or 'coco'")
    predictions, ground_truth, shapes = [], [], []
    for line in tqdm(annotation_lines, desc="evaluate"):
        path, labels = parse_annotation(line)
        spikes = load_spikes(path, input_type)
        shape = tuple(spikes.shape[-2:])
        labels = clip_boxes(labels, len(detector.class_names))
        predictions.append(detector.predict(spikes)[0])
        ground_truth.append(normalized_xywh_to_xyxy(labels, shape))
        shapes.append(shape)
    if metric == "coco":
        return coco_ap(predictions, ground_truth, detector.class_names, shapes)
    return voc_ap50(predictions, ground_truth, detector.class_names)
