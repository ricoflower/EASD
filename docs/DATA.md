# Data preparation

EASD consumes binary spike streams. DSEC-Spike and PKU-Vidar-DVS are described in [the paper](https://arxiv.org/abs/2511.15459). This repository contains configuration files and synthetic data generation code; it does not distribute the full recordings, benchmark splits, or corrected PKU labels. A public download link for the author's prepared dataset has not been supplied in this release.

## NumPy input: DSEC-Spike

Each `.npy` file must contain a `(41,H,W)` array in temporal-height-width order. The standard raw resolution is `(41,250,400)`, with values 0 or 1 and a uint8 or floating-point dtype. Pickled/object arrays are not accepted.

```text
data/
  spikes/
    frame_000001.npy
    frame_000002.npy
  train.txt
  val.txt
```

Each annotation line contains a path followed by zero or more boxes:

```text
spikes/frame_000001.npy 0.5,0.6,0.2,0.3,2
spikes/frame_000002.npy
```

The box fields are `x_center,y_center,width,height,class_id`, normalized to the **raw sensor canvas**, before padding. An empty image has only its path. Relative paths are resolved against the annotation file's directory, or against `--data-root` when supplied. File paths containing spaces are supported.

DSEC class indices, from `model_data/dsec.txt`:

| Index | Class |
| --- | --- |
| 0 | person |
| 1 | large_vehicle |
| 2 | car |

Partially visible boxes are clipped to the raw sensor canvas; boxes entirely outside it are discarded. Invalid class indices, non-finite labels, and nonpositive sizes produce errors.

## Bit-packed input: PKU-Vidar-DVS

Use `--input-type spike` with `.dat` recordings. Decoding uses:

- `250 × 400` pixels per frame;
- little-endian bit order within each byte;
- a vertical flip after unpacking;
- the temporal slice `[180:221]`, yielding 41 frames.

A recording must contain the complete requested slice. Change the loader explicitly if using a different sensor format or timestamp convention.

The nine class indices are defined by `model_data/spike.txt`. A 9-class checkpoint must be used with that mapping; the 3-class DSEC checkpoint is incompatible with the 9-class detection heads.

## Padding convention

The raw `250 × 400` input receives three symmetric rows at the top and bottom. For raw normalized `y` and `h`, the internal padded labels are:

```text
y_padded = (250 * y + 3) / 256
h_padded = 250 * h / 256
```

Inference subtracts the three-row top padding before returning boxes. Already-padded input is supported, but its labels must already use the padded canvas. Do not pad the arrays manually while retaining raw normalized labels.

## Synthetic example

```bash
python examples/make_toy_dataset.py --output data/toy
python train.py --scratch \
  --train-annotations data/toy/train.txt --val-annotations data/toy/val.txt \
  --epochs 1 --batch-size 1 --max-steps 1 --device cpu
```

Synthetic samples check I/O and training behavior. They are not a substitute for either benchmark.
