# Validation scope

## Execution checks

The cleaned implementation has been exercised locally on CPU with Python 3.12, PyTorch 2.3.1, torchvision 0.18.1, and NumPy 2.0. A fresh copy of the public source tree is checked independently of the author's original repository, private checkpoints, and dataset directories.

The regression suite checks:

- symmetric padding and raw/padded label coordinates;
- clipping partially visible annotations;
- little-endian PKU bit-packed spike decoding;
- empty labels, paths with spaces, and target-grid sizes;
- per-sample entropy decisions;
- all-background, all-foreground, and unequal foreground counts in a batch;
- training mode preservation and gradients in both branches and the decoder;
- full-resolution output shapes and strict state_dict round trips;
- loading pretrained backbones from `backbone.*` checkpoint keys;
- mapping predictions back to sensor coordinates.

The full-resolution smoke test performs a training update, validation, EMA checkpoint save, strict reload, decoding and NMS, using generated binary spikes. The CI workflow is provided in `.github/workflows/ci.yml`; no remote CI success is claimed before it has actually run on GitHub.

## Additional checks against author files

Before packaging the public tree, the author's original 3-class checkpoint loaded strictly with 288 matching state_dict entries. On one real DSEC-Spike example and the same padded input, the original and cleaned batch-one model logits differed by at most approximately `1.15e-5`; the fused representation differed by approximately `2.4e-7`.

The same real example and its matching annotation completed one training update, validation, inference, and VOC evaluation. These are execution checks on one sample, not an estimate of generalization or a replacement for evaluation on a held-out test split.

## Remaining experimental scope

A complete six-epoch benchmark training run, evaluation over the full DSEC-Spike/PKU-Vidar-DVS splits, CUDA mixed precision, and the COCO evaluation path have not been validated during this cleanup. The paper's `52.9/60.1` values are reported mAP@50 values from the original paper, not newly reproduced results.
