# Publishing this prepared repository

Target: **https://github.com/ricoflower/EASD**

Description:

```text
EASD: Entropy-guided object detection directly from spike streams. PyTorch implementation of Driving in Spikes.
```

Suggested topics: `spike-camera`, `object-detection`, `neuromorphic-vision`, `pytorch`, `autonomous-driving`.

The local Git repository is initialized on `main`, with `origin` pointing to that URL. Create an **empty public repository** named `EASD` under `ricoflower`, without generating another README or license, then run:

```bash
git push -u origin main
```

With an authenticated GitHub CLI, creating the empty repository can instead be done with:

```bash
gh repo create ricoflower/EASD --public \
  --description "EASD: Entropy-guided object detection directly from spike streams"
git push -u origin main
```

The source repository intentionally excludes local recordings, large checkpoints, training logs, and legacy experiments. Checkpoint and dataset release links can be added when the corresponding artifacts are published. The GitHub Actions workflow will run after the initial push; its success must be checked on GitHub.
