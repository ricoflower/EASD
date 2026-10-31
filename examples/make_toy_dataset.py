"""Generate deterministic binary spike tensors for testing data loading, not accuracy."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/toy"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    lines = []
    for index in range(3):
        name = f"sample_{index}.npy"
        spikes = rng.binomial(1, 0.1 + index * 0.1, (41, 250, 400)).astype(np.uint8)
        np.save(args.output / name, spikes)
        # Include an empty-label image as a data-loader regression example.
        label = " 0.5,0.6,0.2,0.3,0" if index != 1 else ""
        lines.append(name + label)
    (args.output / "train.txt").write_text("\n".join(lines[:2]) + "\n")
    (args.output / "val.txt").write_text(lines[2] + "\n")
    print(f"Wrote synthetic train/val data: {args.output.resolve()}")


if __name__ == "__main__":
    main()
