# Validation scope

## Execution checks

The `easd` package has been exercised locally with Python 3.12, PyTorch 2.3.1, torchvision 0.18.1 and NumPy 2.0 on CPU. The public source is also checked from an independent temporary copy, without the old `nets`, `utils` or `model_data` directories.

The 17 regression tests cover symmetric padding and label coordinates, visible-box clipping, little-endian PKU spike decoding, paths with spaces, empty annotations, effective target strides, per-sample entropy decisions, empty/full/mixed routing, gradients through both branches and the decoder, full-resolution named outputs, native checkpoint round trips, pretrained decoder loading, legacy key migration, rejection of unknown checkpoint keys, sensor-coordinate postprocessing/NMS, and VOC/COCO metric fixtures.

The full-resolution smoke test performs a training update, validation, EMA checkpoint saving, strict reload, decoding and NMS. Epoch-end VOC evaluation is also exercised. Its synthetic execution report is [validation_smoke.json](validation_smoke.json). Synthetic losses and detections have no benchmark meaning.

The official pycocotools COCO evaluator has been checked with a perfect detection fixture, an empty prediction fixture, and one real recording. CI installs the optional evaluation dependency and runs the same regression suite and full-resolution training smoke test.

## Checkpoint and output parity

The author's original 3-class checkpoint contains 288 tensor entries. The renamed model loads it by explicitly migrating its active keys and dropping only five unused entries from `alpha` and `out_mask`; the new small-model state_dict has 283 entries.

On the same real DSEC-Spike recording, the renamed package and the immediately preceding cleaned implementation produced exactly equal outputs: maximum absolute difference **0** for all three head logits, the fused representation, and all nine final detections. This checks that this package reorganization preserves computation and postprocessing. Earlier fixes to the original experimental code are described in [IMPLEMENTATION](IMPLEMENTATION.md).

The real recording and its matching annotation also completed prediction, a bounded training/validation run and standalone VOC/COCO evaluation through the new entry points. These are execution checks on one sample, not measurements of generalization.

## Remaining experimental scope

A complete six-epoch benchmark run, full DSEC-Spike/PKU-Vidar-DVS evaluation and CUDA mixed precision have not been validated during this cleanup. The paper's `52.9/60.1` results are reported mAP@50 values, not newly reproduced measurements.
