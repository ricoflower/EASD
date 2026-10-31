"""Predict objects in a .npy spike tensor or a PKU-Vidar .dat recording."""

import argparse
import json
from pathlib import Path

import torch

from easd import EASDDetector
from easd.config import class_file
from easd.spikes import load_spikes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-type", choices=("dsec", "spike"), default="dsec")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.classes is None and args.input_type == "spike":
        args.classes = class_file("spike")
    detector = EASDDetector(
        weights=args.weights,
        classes=args.classes,
        variant=args.variant,
        device=args.device,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
    )
    spikes = load_spikes(args.input, args.input_type)
    detections = detector.predict(spikes)[0]
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    records = [
        dict(
            box=box[:4].tolist(),
            score=float(box[4]),
            class_id=int(box[5]),
            class_name=detector.class_names[int(box[5])],
        )
        for box in detections
    ]
    (args.output / f"{stem}.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    detector.render(spikes, detections).save(args.output / f"{stem}.png")
    print(f"{len(records)} detections; results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
