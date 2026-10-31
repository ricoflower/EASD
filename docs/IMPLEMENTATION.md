# Implementation and compatibility

This source release reorganizes and fixes the author's existing implementation. It preserves the learned operators and parameter names used by the available EASD checkpoints. It should not be interpreted as a new model variant or a new benchmark measurement.

## Architecture

The temporal branch splits the 41-frame stream into five overlapping 13-frame blocks at offsets `0,7,14,21,28`. One shared texture module processes each block through four nested crops of length `13,11,9,7`, convolution, residual squeeze-and-excitation, and a 3-channel projection. A sigmoid MLP weights and sums the five block features.

The spatial branch sums the spike stream along time, embeds it into 60 channels, and applies entropy selective attention. The entropy selector merges adjacent `2×2` groups of `8×8` windows. A merged group's mean entropy must be between half the image's mean window entropy and the mean itself to be selected as foreground. Thresholds and selections are computed separately for each image. Selected windows receive offset-warped attention; the remaining windows use the lightweight background path.

A projected, channel-softmax spatial feature is added to the temporal feature. The CSPDarknet decoder upsamples the representation by two before its Focus stem, yielding effective detection strides `4/8/16` on the padded spike canvas. The model returns detection heads in coarse-to-fine order (`16/8/4`) and a fourth tensor containing the fused representation.

## Network forward and complete inference

The name `YoloBody` is inherited from the YOLO training framework. In this repository it contains the whole EASD network, including both the shared temporal branch and the entropy selective spatial branch, their fusion, the upsampled CSPDarknet decoder, and all three detection heads. It is not an unmodified RGB YOLO model.

`YoloBody.forward` returns `(coarse_logits, medium_logits, fine_logits, representation)`. The fourth tensor is an intermediate feature representation; none of these tensors is the final list of detections.

The complete prediction API is `YOLO.predict_tensor` in `yolo.py`. Construction loads a full EASD state_dict and switches the network to evaluation mode. Prediction calls the network, decodes only the first three outputs, applies confidence filtering and class-wise NMS, removes the symmetric padding offset, and clips boxes to the original sensor dimensions. It returns one `N × 6` array per image, ordered as `[xmin, ymin, xmax, ymax, score, class_id]`.

`predict.py` is the command-line wrapper: it reads the spike recording, calls `YOLO.predict_tensor`, and writes JSON and a visualization. `load_spikes` does not reconstruct RGB images; the detection visualization uses the temporal mean only as a display background.

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

Two unused parameter groups from the old checkpoint schema, `alpha` and the predictor's `out_mask`, are retained for strict loading and excluded from optimization. The DSEC-small schema contains 288 state_dict entries.

## Scope

The supported interface is single-device CPU/CUDA training and spike-native inference/evaluation. It does not include RGB webcam inference, RGB ONNX examples, optimizer-state resume, or distributed training. Optional THOP profiling may omit attention matrix multiplications and routing; its estimate is not a complete theoretical FLOP count.
