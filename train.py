"""Single-device YOLO training for EASD, with an offline end-to-end smoke test."""

import argparse
import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import ModelEMA, YOLOLoss
from utils.checkpoints import ROOT, load_backbone, read_state_dict
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.spikes import padded_shape, parse_annotation
from utils.utils import get_anchors, get_classes
from utils.utils_fit import run_epoch

ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-type", choices=("dsec", "spike"), default="dsec")
    parser.add_argument("--train-annotations", type=Path)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--data-root", type=Path, help="Root for relative paths in annotations")
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--anchors", type=Path, default=ROOT / "model_data/yolo_anchors.txt")
    parser.add_argument("--input-shape", type=int, nargs=2, default=(250, 400), metavar=("H", "W"))
    parser.add_argument("--phi", choices=("s", "m", "l", "x"), default="s")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--weights", type=Path, help="EASD state_dict, loaded strictly; starts a new optimizer")
    weights.add_argument("--backbone-weights", type=Path, help="Local CSPDarknet or YOLO state_dict")
    weights.add_argument("--scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true", help="CUDA mixed precision")
    parser.add_argument("--eval-map", action="store_true", help="Legacy VOC mAP@0.5 after each epoch")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, help="Bound train and validation steps per epoch")
    parser.add_argument("--smoke-test", action="store_true", help="Synthetic full-resolution train/val/checkpoint/decode")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads, *args.input_shape) < 1:
        parser.error("epochs, batch size, threads, and dimensions must be positive")
    if args.workers < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("workers must be nonnegative and max-steps must be positive")
    return args


def read_annotations(path, data_root=None):
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sample, boxes = parse_annotation(line)
        if not sample.is_absolute():
            sample = (data_root or path.resolve().parent) / sample
        if not sample.is_file():
            raise FileNotFoundError(f"{path}: spike data not found: {sample}")
        labels = " ".join(",".join(str(float(value)) for value in box) for box in boxes)
        lines.append(f"{sample} {labels}".strip())
    if not lines:
        raise ValueError(f"Empty annotation list: {path}")
    return lines


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
    classes_path = args.classes or ROOT / f"model_data/{args.input_type}.txt"
    class_names, num_classes = get_classes(classes_path)
    anchors, _ = get_anchors(args.anchors)
    canvas = padded_shape(args.input_shape)
    output = args.output or ROOT / "logs" / datetime.now().strftime("easd_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
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
            dataset = YoloDataset(args.input_type, lines, args.input_shape, num_classes, anchors, ANCHORS_MASK)
            loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                                      num_workers=args.workers, pin_memory=device.type == "cuda",
                                      collate_fn=yolo_dataset_collate))
        model = YoloBody(ANCHORS_MASK, num_classes, args.phi, input_shape=args.input_shape)
        if args.weights:
            model.load_state_dict(read_state_dict(args.weights), strict=True)
            print(f"Loaded EASD weights: {args.weights}")
        elif not args.scratch:
            print(f"Loaded pretrained CSPDarknet: {load_backbone(model.backbone, args.backbone_weights, args.phi)}")
        model.to(device)
        criterion = YOLOLoss(anchors, num_classes, canvas, device.type == "cuda", ANCHORS_MASK)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        ema = ModelEMA(model)
        evaluator = None
        if args.eval_map:
            from utils.callbacks import EvalCallback
            evaluator = EvalCallback(args.input_type, model, canvas, anchors, ANCHORS_MASK,
                                     class_names, num_classes, val_lines, str(output),
                                     device.type == "cuda", eval_flag=True, period=1)
        print(f"Device: {device}; raw: (41, {args.input_shape[0]}, {args.input_shape[1]}); canvas: {canvas}")
        best_loss = float("inf")
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("epoch", "lr", "train_loss", "val_loss"))
            for epoch in range(args.epochs):
                train_loss = run_epoch(model, loaders[0], criterion, device, optimizer,
                                       scaler, ema, args.max_steps)
                val_loss = run_epoch(ema.ema, loaders[1], criterion, device, max_steps=args.max_steps)
                writer.writerow((epoch + 1, optimizer.param_groups[0]["lr"], train_loss, val_loss))
                stream.flush()
                state = ema.ema.state_dict()
                torch.save(state, output / "last_epoch_weights.pth")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(state, output / "best_epoch_weights.pth")
                if evaluator:
                    evaluator.on_epoch_end(epoch + 1, ema.ema)
                print(f"Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                scheduler.step()
        if args.smoke_test:
            from utils.utils_bbox import DecodeBox
            restored = YoloBody(ANCHORS_MASK, num_classes, args.phi).to(device).eval()
            restored.load_state_dict(read_state_dict(output / "last_epoch_weights.pth"), strict=True)
            with torch.no_grad():
                sample = torch.randint(0, 2, (1, 41, *args.input_shape), device=device).float()
                outputs = restored(sample)
                decoder = DecodeBox(anchors, num_classes, canvas, ANCHORS_MASK)
                decoded = torch.cat(decoder.decode_box(outputs[:3]), dim=1)
                if not torch.isfinite(decoded).all():
                    raise FloatingPointError("Non-finite decoded predictions")
                decoder.non_max_suppression(decoded, num_classes, canvas, canvas, False,
                                            conf_thres=0.5, nms_thres=0.5)
            report = dict(input=list(sample.shape), outputs=[list(t.shape) for t in outputs],
                          decoded=list(decoded.shape), train_loss=train_loss, val_loss=val_loss,
                          checkpoint="strict reload passed", nms="passed")
            (output / "smoke_test.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
