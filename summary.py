"""Inspect EASD shapes, parameter counts, and optional THOP FLOPs."""

import argparse

import torch

from nets.yolo import YoloBody


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi", choices=("s", "m", "l", "x"), default="s")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--input-shape", type=int, nargs=2, default=(250, 400))
    parser.add_argument("--flops", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], args.num_classes, args.phi).eval()
    sample = torch.zeros(1, 41, *args.input_shape)
    with torch.no_grad():
        outputs = model(sample)
    print(f"Input: {tuple(sample.shape)}")
    print(f"Outputs: {[tuple(value.shape) for value in outputs]}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if args.flops:
        from thop import profile
        with torch.no_grad():
            macs, _ = profile(model, inputs=(sample,), verbose=False)
        print(f"THOP: {macs / 1e9:.3f} GMAC; {macs * 2 / 1e9:.3f} GFLOPs (2 operations/MAC)")
        print("Only operators counted by THOP are included; window attention matmul/routing may be omitted.")


if __name__ == "__main__":
    main()
