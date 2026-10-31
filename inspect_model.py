"""Inspect EASD's named forward outputs and parameter count."""

import argparse

import torch

from easd import EASD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("s", "m", "l", "x"), default="s")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--input-shape", type=int, nargs=2, default=(250, 400))
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = EASD(num_classes=args.num_classes, variant=args.variant).eval()
    sample = torch.zeros(1, 41, *args.input_shape)
    with torch.no_grad():
        output = model(sample)
    print(f"Input: {tuple(sample.shape)}")
    print(f"Head logits: {[tuple(value.shape) for value in output.head_logits]}")
    print(f"Representation: {tuple(output.representation.shape)}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")


if __name__ == "__main__":
    main()
