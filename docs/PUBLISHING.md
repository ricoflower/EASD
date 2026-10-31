# Maintaining the public repository

Repository: [ricoflower/EASD](https://github.com/ricoflower/EASD).

Before publishing changes, run the checks in [CONTRIBUTING](../CONTRIBUTING.md), inspect the diff and ensure the latest GitHub Actions run passes. Preserve upstream licenses and figure provenance.

```bash
git add <changed-source-files>
git commit -m "Describe the change"
git push origin main
```

Local recordings, large checkpoints, training logs and legacy experiments are excluded. Add artifact download links when corresponding datasets/checkpoints are published. Accuracy claims must distinguish results reported in the paper from reproduced measurements.
