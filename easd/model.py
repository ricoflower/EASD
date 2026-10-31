"""The complete EASD network: temporal branch, spatial branch, and detection decoder."""

from typing import NamedTuple

import torch
from torch import nn

from easd.backbone import CSPDarknet, Conv
from easd.config import DEFAULT_ANCHOR_MASKS, INPUT_FRAMES
from easd.spatial import EntropySelectiveAttention
from easd.spikes import pad_spikes
from easd.temporal import TemporalFeatureFusion, TemporalTextureModule


class EASDOutput(NamedTuple):
    """Raw network outputs; final bounding boxes are produced by EASDDetector."""

    head_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    representation: torch.Tensor


class EASD(nn.Module):
    """Accept (B,41,H,W) or (41,H,W), pad, and return three heads plus representation."""

    def __init__(
        self, num_classes=3, variant="s", anchor_masks=DEFAULT_ANCHOR_MASKS, pretrained=False
    ):
        super().__init__()
        depth = {"s": 0.33, "m": 0.67, "l": 1.0, "x": 1.33}
        width = {"s": 0.50, "m": 0.75, "l": 1.0, "x": 1.25}
        if variant not in depth:
            raise ValueError(f"Unknown model variant: {variant}")
        if num_classes < 1 or len(anchor_masks) != 3:
            raise ValueError("Expected positive num_classes and three anchor groups")
        channels = int(width[variant] * 64)
        blocks = max(round(depth[variant] * 3), 1)
        self.num_classes = num_classes
        self.variant = variant
        self.anchor_masks = tuple(tuple(mask) for mask in anchor_masks)
        self.temporal_block_size = 13
        self.temporal_block_starts = (0, 7, 14, 21, 28)
        self.temporal_encoder = TemporalTextureModule()
        self.temporal_fusion = TemporalFeatureFusion()
        self.spatial_encoder = EntropySelectiveAttention(dim=60)
        self.spatial_projection = Conv(60, 3, 1, 1)
        self.spatial_softmax = nn.Softmax(dim=1)
        self.decoder = CSPDarknet(channels, blocks)
        self.head_fine = nn.Conv2d(channels * 4, len(anchor_masks[2]) * (5 + num_classes), 1)
        self.head_medium = nn.Conv2d(channels * 8, len(anchor_masks[1]) * (5 + num_classes), 1)
        self.head_coarse = nn.Conv2d(channels * 16, len(anchor_masks[0]) * (5 + num_classes), 1)
        if pretrained:
            from easd.checkpoints import load_decoder_weights

            load_decoder_weights(self.decoder, variant=variant)

    def forward(self, spikes):
        if spikes.ndim == 3:
            spikes = spikes.unsqueeze(0)
        if spikes.ndim != 4 or spikes.shape[1] != INPUT_FRAMES:
            raise ValueError(f"Expected (B,41,H,W) or (41,H,W), got {tuple(spikes.shape)}")
        if not spikes.is_floating_point():
            spikes = spikes.float()
        spikes = pad_spikes(spikes)
        blocks = [
            self.temporal_encoder(spikes[:, start : start + self.temporal_block_size])
            for start in self.temporal_block_starts
        ]
        temporal = self.temporal_fusion(blocks)
        spatial = self.spatial_softmax(self.spatial_projection(self.spatial_encoder(spikes)))
        representation = temporal + spatial
        fine, medium, coarse = self.decoder(representation)
        return EASDOutput(
            (self.head_coarse(coarse), self.head_medium(medium), self.head_fine(fine)),
            representation,
        )
