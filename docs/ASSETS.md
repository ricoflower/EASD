# Figure provenance

The README uses the original composed figures from the paper/manuscript. Panel arrangements, RGB–entropy overlays, mask overlays, and figure labels are retained from their sources.

## Figures displayed in the README

| Asset | Source | Description |
| --- | --- | --- |
| `assets/architecture.png` | arXiv:2511.15459v1, Figure 3, page 3 | Dual-branch overview |
| `assets/attention.png` | arXiv:2511.15459v1, Figure 4, page 4 | Entropy selective attention |
| `assets/qualitative.png` | arXiv:2511.15459v1, Figure 6, page 6 | Published qualitative comparison |
| `assets/feature_entropy_visualization.png` | Author-provided 10-page EASD manuscript, Figure 7, page 8 | Three paired RGB references and RGB–entropy overlays |
| `assets/foreground_mask_comparison.png` | Author-provided 10-page EASD manuscript, Figure 8, page 8 | Foreground-mask comparison across window/merge settings |

The author-provided manuscript is titled *End-to-End Object Detection from Spike Streams with Entropy Attention*. Its figure numbering refers to that supplied version; it is distinct from the linked arXiv version of *Driving in Spikes*.

The arXiv figures were rendered at 3× scale. The supplied manuscript's Figures 7 and 8 were rendered at 8× scale and cropped to their complete figure boundaries. The source panel composition, entropy overlays, printed values, setting labels, and ground-truth boxes were preserved. Manuscript headers, review line numbers, and surrounding body text were excluded. The README explanations follow the original figure captions. No new heatmaps, overlays, or masks were synthesized.

## Interpreting Figures 7 and 8

- **Figure 7:** entropy is computed for feature windows. Higher entropy appears brighter and lower entropy appears darker. RGB reference images and the entropy overlays help relate feature entropy to scene content.
- **Figure 8:** unmasked regions denote selected foreground and masked regions denote background. Red boxes indicate ground-truth objects. The row labels are the settings from the original comparison. The RGB image is supplied for visualization; it is not used to compute the masks.

These are figures from the author's experiments, not newly generated runtime outputs or fresh benchmark measurements of this source release. Implementation choices retained for checkpoint compatibility are described in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Additional source assets retained for reference

The following individual experiment images are retained as source material and are **not displayed as the README's entropy visualization**.

| Asset | Source | Description |
| --- | --- | --- |
| `assets/entropy_block.png` | arXiv:2511.15459v1, Figure 5, page 5 | Window entropy selection diagram |
| `assets/day_scene.png` | Author's `vis/entropy/origin/000009.png` | Reference daytime frame |
| `assets/day_entropy.png` | Author's `vis/entropy/entropy/000009.png` | Individual daytime window entropy map |
| `assets/night_scene.png` | Author's `vis/entropy/origin/000022.png` | Reference nighttime frame |
| `assets/night_entropy.png` | Author's `vis/entropy/entropy/000022.png` | Individual nighttime window entropy map |
| `assets/bus_scene.png` | Author's `vis/entropy/origin/000068.png` | Reference large-vehicle frame |
| `assets/bus_entropy.png` | Author's `vis/entropy/entropy/000068.png` | Individual large-vehicle window entropy map |

[arXiv paper](https://arxiv.org/abs/2511.15459). Rights in figures and underlying dataset imagery remain with their respective authors/providers; the source-code license does not replace dataset access or redistribution terms.
