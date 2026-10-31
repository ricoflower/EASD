# Interface migration

The public package is `easd`. Old module files and class aliases have been removed to make the EASD architecture and complete inference boundary explicit.

| Previous interface | Current interface |
| --- | --- |
| `nets.yolo.YoloBody` | `easd.EASD` |
| Four positional forward outputs | `EASDOutput.head_logits` (three tensors), `.representation` |
| `yolo.YOLO(model_path=...)` | `easd.EASDDetector(weights=...)` |
| `predict_tensor(spikes)` | `predict(spikes)` |
| `net` on the detector | `model` |
| `nets/CSPdarknet.py` | `easd/backbone.py` |
| `utils/dataloader.py` | `easd/data.py` |
| `get_map.py` | `evaluate.py` |
| `summary.py` | `inspect_model.py` |
| `flops.py` | Removed; incomplete estimates are no longer presented |
| `--phi` | `--variant` |
| `model_data/dsec.txt`, `spike.txt`, `yolo_anchors.txt` | `configs/dsec_classes.txt`, `pku_classes.txt`, `anchors.txt` |

Model attributes now describe their roles: `temporal_encoder`, `temporal_fusion`, `spatial_encoder`, `spatial_projection`, `decoder`, and `head_fine/head_medium/head_coarse`. Detection decoding and NMS live in `easd/postprocess.py`; `easd/inference.py` combines them with the model and checkpoint loader.

## Existing checkpoints

`load_easd_weights` and `EASDDetector` automatically recognize the original author state_dict layout. They rename known temporal, spatial, decoder and head keys, discard only the unused `alpha` and four unused `out_mask` tensor entries, then perform strict loading. The model's forward operators are unchanged by renaming. New checkpoints contain only the 283 active state_dict entries for the 3-class small model, rather than 288 entries.

This is a one-way source-interface migration. Original checkpoints remain readable; use this package to read newly saved checkpoints. The old experimental directory has been preserved separately and is not part of the public source tree.

Pretrained CSPDarknet accepts a backbone-only state_dict or extracts `backbone.*`/`decoder.*` from a larger tensor state_dict. Upstream YOLO attribution, licenses, anchor-head formulas and actual upstream checkpoint filenames are retained.
