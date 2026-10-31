"""Evaluate a spike annotation list with VOC AP50 or COCO AP50:95."""

import argparse
import json
from pathlib import Path

import torch

from train import ANCHORS_MASK, read_annotations
from utils.callbacks import EvalCallback
from utils.spikes import padded_shape
from yolo import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--input-type", choices=("dsec", "spike"), default="dsec")
    parser.add_argument("--input-shape", type=int, nargs=2, default=(250, 400))
    parser.add_argument("--phi", choices=("s", "m", "l", "x"), default="s")
    parser.add_argument("--metric", choices=("voc", "coco"), default="voc")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("map_out"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.classes is None and args.input_type == "spike":
        args.classes = Path(__file__).resolve().parent / "model_data/spike.txt"
    detector = YOLO(model_path=args.weights, classes_path=args.classes,
                    phi=args.phi, device=args.device, input_shape=args.input_shape)
    lines = read_annotations(args.annotations, args.data_root)
    args.output.mkdir(parents=True, exist_ok=False)
    evaluator = EvalCallback(args.input_type, detector.net, padded_shape(args.input_shape),
                             detector.anchors, ANCHORS_MASK, detector.class_names,
                             detector.num_classes, lines, str(args.output),
                             detector.device.type == "cuda", metric=args.metric)
    evaluator.on_epoch_end(1, detector.net)
    report = dict(metric="VOC AP50" if args.metric == "voc" else "COCO AP50:95",
                  value=float(evaluator.maps[-1]), samples=len(lines))
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
