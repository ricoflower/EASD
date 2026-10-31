# Implementation and compatibility

This source release reorganizes and fixes the author's existing implementation. It preserves the learned operators while giving modules explicit EASD names. Original checkpoint keys are migrated by one dedicated loader. It should not be interpreted as a new model variant or a new benchmark measurement.

## Architecture

The temporal branch splits the 41-frame stream into five overlapping 13-frame blocks at offsets `0,7,14,21,28`. One shared texture module processes each block through four nested crops of length `13,11,9,7`, convolution, residual squeeze-and-excitation, and a 3-channel projection. A sigmoid MLP weights and sums the five block features.

The spatial branch sums the spike stream along time, embeds it into 60 channels, and applies entropy selective attention. The entropy selector merges adjacent `2×2` groups of `8×8` windows. A merged group's mean entropy must be between half the image's mean window entropy and the mean itself to be selected as foreground. Thresholds and selections are computed separately for each image. Selected windows receive offset-warped attention; the remaining windows use the lightweight background path.

A projected, channel-softmax spatial feature is added to the temporal feature. The CSPDarknet decoder upsamples the representation by two before its Focus stem, yielding effective detection strides `4/8/16` on the padded spike canvas. The model returns `EASDOutput(head_logits, representation)`, with three head logits in coarse-to-fine order (`16/8/4`).

## Network forward and complete inference

`easd/model.py` defines `EASD`: temporal texture extraction and fusion, entropy selective spatial attention, branch fusion, CSPDarknet and the detection heads. `forward` returns raw `head_logits` and `representation`.

`easd/inference.py` defines `EASDDetector`. Construction loads trained weights strictly and switches to evaluation mode. `predict` runs the model, decodes its heads, filters confidence, applies class-wise NMS, removes padding and clips boxes to sensor dimensions. It returns one `N × 6` array per image: `[xmin,ymin,xmax,ymax,score,class_id]`.

`predict.py` reads recordings and writes JSON and a visualization. The temporal mean provides a display background only. Evaluation uses the same detector; `easd/metrics.py` calculates VOC AP50 or calls the official pycocotools evaluator directly.

## Corrections relative to the original code

- Forward passes no longer force modules into evaluation mode.
- Entropy thresholds and foreground window counts are independent across samples.
- Empty foreground sets and unequal foreground counts are supported.
- Window layout is derived from the input shape rather than fixed `32×50` constants.
- Raw-input padding, training labels, anchor scaling, and inference coordinates share one convention.
- The CPU evaluation path and input-type handling are corrected.
- Paths are configurable; tests and synthetic data do not depend on the author's machine.

## Differences between paper notation and the supplied code

These original implementation choices are retained to preserve checkpoint behavior:

| Component | Paper notation / diagram | Supplied implementation retained here |
| --- | --- | --- |
| Spatial embedding | `1×1` convolution in the text | `3×3` convolution |
| Deformable features | Deformable convolution for Q/K | `grid_sample` offset warping for keys; query from the input feature |
| Attention logits | `QKᵀ / sqrt(d)` | Unscaled `QKᵀ` |
| SE | SE reweighting | Residual SE reweighting |

Changing these operators requires a separately identified model variant and retraining/evaluation. The entropy selection illustrations use the paper's conceptual diagram; the executable operators are defined by the code.

The unused scalar `alpha` and unused predictor `out_mask` MLP have been removed from the model. The 3-class small model now has 283 state_dict entries. `easd/checkpoints.py` migrates the original 288-entry layout, drops exactly those five unused tensor entries, and requires a strict match for everything else. Unknown parameters and incompatible class counts fail explicitly. See [migration notes](MIGRATION.md).

## Scope

The supported interface is single-device CPU/CUDA training and spike inference/evaluation. Legacy RGB detection, webcam/ONNX examples, unused plotting and profiling utilities, old module aliases, optimizer-state resume and distributed training are outside this release.
