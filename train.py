"""Single-device EASD training, with an offline end-to-end smoke test."""

import argparse
import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from easd import EASD, EASDDetector
from easd.checkpoints import load_decoder_weights, load_easd_weights
from easd.config import (
    CONFIG_DIR,
    ROOT,
    DEFAULT_ANCHOR_MASKS,
    class_file,
    load_anchors,
    load_class_names,
)
from easd.data import SpikeDetectionDataset, collate_spikes, read_annotations
from easd.ema import ExponentialMovingAverage
from easd.loss import DetectionLoss
from easd.spikes import padded_shape
from easd.training import run_epoch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-type", choices=("dsec", "spike"), default="dsec")
    parser.add_argument("--train-annotations", type=Path)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--data-root", type=Path, help="Root for relative paths in annotations")
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--anchors", type=Path, default=CONFIG_DIR / "anchors.txt")
    parser.add_argument("--input-shape", type=int, nargs=2, default=(250, 400), metavar=("H", "W"))
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument(
        "--weights", type=Path, help="EASD state_dict, loaded strictly; starts a new optimizer"
    )
    weights.add_argument(
        "--backbone-weights", type=Path, help="Local CSPDarknet or YOLO state_dict"
    )
    weights.add_argument("--scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true", help="CUDA mixed precision")
    parser.add_argument("--eval-map", action="store_true", help="VOC AP50 after each epoch")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, help="Bound train and validation steps per epoch")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Synthetic full-resolution train/val/checkpoint/decode",
    )
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads, *args.input_shape) < 1:
        parser.error("epochs, batch size, threads, and dimensions must be positive")
    if args.workers < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("workers must be nonnegative and max-steps must be positive")
    return args


def synthetic_annotations(directory, shape):
    rng = np.random.default_rng(42)
    lines = []
    for index in range(2):
        path = directory / f"sample_{index}.npy"
        np.save(path, rng.binomial(1, 0.15 + index * 0.2, (41, *shape)).astype(np.uint8))
        # Include both a positive sample and an empty image.
        label = " 0.5,0.6,0.2,0.3,0" if index == 0 else ""
        lines.append(str(path) + label)
    return lines, lines[:1]


def main(argv=None):
    args = parse_args(argv)
    if args.smoke_test:
        args.input_type = "dsec"
        args.epochs, args.batch_size, args.max_steps = 1, 1, 1
    if args.smoke_test and not args.weights and not args.backbone_weights:
        args.scratch = True
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")
    classes_path = args.classes or class_file(args.input_type)
    class_names = load_class_names(classes_path)
    num_classes = len(class_names)
    anchors = load_anchors(args.anchors)
    canvas = padded_shape(args.input_shape)
    output = args.output or ROOT / "logs" / datetime.now().strftime("easd_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(device=str(device), canvas=canvas, class_names=class_names)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="easd-smoke-") as temporary:
        if args.smoke_test:
            train_lines, val_lines = synthetic_annotations(Path(temporary), args.input_shape)
        else:
            train_path = args.train_annotations or ROOT / "data/train.txt"
            val_path = args.val_annotations or ROOT / "data/val.txt"
            train_lines = read_annotations(train_path, args.data_root)
            val_lines = read_annotations(val_path, args.data_root)
        loaders = []
        for lines, training in ((train_lines, True), (val_lines, False)):
            dataset = SpikeDetectionDataset(
                args.input_type, lines, args.input_shape, num_classes, anchors, DEFAULT_ANCHOR_MASKS
            )
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=training,
                    num_workers=args.workers,
                    pin_memory=device.type == "cuda",
                    collate_fn=collate_spikes,
                )
            )
        model = EASD(num_classes=num_classes, variant=args.variant)
        if args.weights:
            load_easd_weights(model, args.weights)
            print(f"Loaded EASD weights: {args.weights}")
        elif not args.scratch:
            print(
                f"Loaded pretrained CSPDarknet: {load_decoder_weights(model.decoder, args.backbone_weights, args.variant)}"
            )
        model.to(device)
        criterion = DetectionLoss(anchors, num_classes, canvas, DEFAULT_ANCHOR_MASKS)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        ema = ExponentialMovingAverage(model)
        print(
            f"Device: {device}; raw: (41, {args.input_shape[0]}, {args.input_shape[1]}); canvas: {canvas}"
        )
        best_loss = float("inf")
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("epoch", "lr", "train_loss", "val_loss"))
            for epoch in range(args.epochs):
                train_loss = run_epoch(
                    model, loaders[0], criterion, device, optimizer, scaler, ema, args.max_steps
                )
                val_loss = run_epoch(
                    ema.model, loaders[1], criterion, device, max_steps=args.max_steps
                )
                writer.writerow((epoch + 1, optimizer.param_groups[0]["lr"], train_loss, val_loss))
                stream.flush()
                state = ema.model.state_dict()
                torch.save(state, output / "last_epoch_weights.pth")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(state, output / "best_epoch_weights.pth")
                if args.eval_map:
                    from easd.evaluation import evaluate_detector

                    detector = EASDDetector.from_model(
                        ema.model, class_names, anchors, device=device, confidence=0.05
                    )
                    metrics = evaluate_detector(detector, val_lines, args.input_type)
                    with (output / "evaluation.jsonl").open("a") as evaluation_stream:
                        evaluation_stream.write(json.dumps(dict(epoch=epoch + 1, **metrics)) + "\n")
                print(
                    f"Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )
                scheduler.step()
        if args.smoke_test:
            restored = EASD(num_classes=num_classes, variant=args.variant).to(device).eval()
            load_easd_weights(restored, output / "last_epoch_weights.pth")
            detector = EASDDetector.from_model(restored, class_names, anchors, device=device)
            sample = torch.randint(0, 2, (1, 41, *args.input_shape), device=device).float()
            with torch.no_grad():
                outputs = restored(sample)
                decoded = detector.postprocessor.decode(outputs.head_logits, canvas)
                if not torch.isfinite(decoded).all():
                    raise FloatingPointError("Non-finite decoded predictions")
                detections = detector.predict(sample)
            report = dict(
                input=list(sample.shape),
                head_logits=[list(t.shape) for t in outputs.head_logits],
                representation=list(outputs.representation.shape),
                decoded=list(decoded.shape),
                detections=[list(t.shape) for t in detections],
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint="strict reload passed",
                nms="passed",
            )
            (output / "smoke_test.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
