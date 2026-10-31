# Checkpoints

Model checkpoints are not stored in Git. No public EASD checkpoint download URL has been supplied for this source release.

## Pretrained CSPDarknet

Training normally initializes the CSPDarknet decoder from an image-pretrained checkpoint. Provide one of:

```text
weights/cspdarknet_s_backbone.pth   # CSPDarknet-only state_dict
weights/yolov5_s.pth               # YOLO state_dict containing backbone.* keys
```

Or pass `--backbone-weights /path/to/checkpoint.pth`. The loader extracts `backbone.*` keys when necessary and requires a strict match. A checkpoint pickled as an arbitrary Python model object, such as some `yolov5s.pt` files, is not interchangeable with a tensor state_dict.

Upstream backbone checkpoints are distributed by [yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch). Match the model size (`s/m/l/x`) to `--phi`. The code does not silently download weights or fall back to random initialization.

## Trained EASD

```text
weights/easd_dsec_s.pth            # 3-class EASD-small, if provided by the authors
```

Use `--weights /path/to/easd_state_dict.pth` for inference, evaluation, or fine-tuning. Loading is strict. A training checkpoint in this release contains the EMA model's tensor state_dict, rather than optimizer or scheduler state.

## No weights available

`python train.py --smoke-test --device cpu` runs from random initialization automatically. For normal training from scratch, specify `--scratch` explicitly. Regression tests create their own temporary checkpoints and do not require private files.
