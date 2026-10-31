# Contributing

For a bug report, include the command, Python/PyTorch/torchvision versions, device, input shape and dtype, and a minimal traceback. Prefer synthetic binary spikes to uploading dataset recordings.

Run the following before submitting a change:

```bash
python -m unittest discover -s tests -v
python train.py --smoke-test --device cpu
```

Model or loss changes should include a small regression case and a description of checkpoint compatibility. Claims about detection accuracy should specify the dataset split, class mapping, checkpoint, coordinate convention, and evaluation metric. Preserve upstream attribution and licenses.
