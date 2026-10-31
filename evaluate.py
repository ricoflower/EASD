"""Evaluate trained EASD detections with VOC AP50 or official COCO AP50:95."""

import argparse
import json
from pathlib import Path

import torch

from easd import EASDDetector
from easd.config import class_file
from easd.data import read_annotations
from easd.evaluation import evaluate_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--input-type", choices=("dsec", "spike"), default="dsec")
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
    parser.add_argument("--metric", choices=("voc", "coco"), default="voc")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.05,
        help="Low score cutoff used to build an evaluation precision/recall curve",
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("map_out/validation"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    detector = EASDDetector(
        weights=args.weights,
        classes=args.classes or class_file(args.input_type),
        anchors=args.anchors,
        variant=args.variant,
        device=args.device,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
    )
    lines = read_annotations(args.annotations, args.data_root)
    metrics = evaluate_detector(detector, lines, args.input_type, args.metric)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
