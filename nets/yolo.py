"""EASD: temporal texture + entropy selective attention + CSPDarknet/YOLO."""

import torch
from torch import nn

from nets.CSPdarknet import CSPDarknet, Conv
from nets.spatial import EntropySelectiveAttention
from nets.temporal import FeatureFusionModule, TemporalTextureModule
from utils.spikes import pad_spikes


class YoloBody(nn.Module):
    """Accept (B, 41, H, W) or one (41, H, W) spike tensor.

    Raw 250x400 inputs are symmetrically padded to 256x400 internally. Return
    coarse/medium/fine YOLO logits and the fused 3-channel padded representation.
    Detection coordinates and training targets refer to that padded canvas.
    """

    def __init__(self, anchors_mask, num_classes, phi="s", backbone="cspdarknet",
                 pretrained=False, input_shape=(250, 400)):
        super().__init__()
        if backbone != "cspdarknet":
            raise ValueError("EASD supports backbone='cspdarknet'")
        depth = {"s": 0.33, "m": 0.67, "l": 1.0, "x": 1.33}
        width = {"s": 0.50, "m": 0.75, "l": 1.0, "x": 1.25}
        if phi not in depth:
            raise ValueError(f"Unknown model size: {phi}")
        if num_classes < 1 or len(anchors_mask) != 3:
            raise ValueError("Expected positive num_classes and three anchor groups")
        channels = int(width[phi] * 64)
        blocks = max(round(depth[phi] * 3), 1)
        self.input_shape = tuple(input_shape)
        self.win_r = 6
        self.win_step = 7
        self.extractor = TemporalTextureModule()
        self.blockfuse4 = FeatureFusionModule()
        self.sp_attn = EntropySelectiveAttention(dim=60)
        self.conv4attn = Conv(60, 3, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        # Unused in the original forward; retained for old checkpoint loading.
        self.alpha = nn.Parameter(torch.full((1,), 0.1), requires_grad=False)
        self.backbone = CSPDarknet(channels, blocks, phi, pretrained)
        self.yolo_head_P3 = nn.Conv2d(channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, spikes):
        if spikes.ndim == 3:
            spikes = spikes.unsqueeze(0)
        if spikes.ndim != 4 or spikes.shape[1] != 41:
            raise ValueError(f"Expected (B, 41, H, W) or (41, H, W), got {tuple(spikes.shape)}")
        if not spikes.is_floating_point():
            spikes = spikes.float()
        spikes = pad_spikes(spikes)
        features = [self.extractor(spikes[:, start:start + 13])
                    for start in range(0, 29, self.win_step)]
        texture = self.blockfuse4(features)
        spatial = self.softmax(self.conv4attn(self.sp_attn(spikes)))
        representation = texture + spatial
        fine, medium, coarse = self.backbone(representation)
        return (self.yolo_head_P5(coarse), self.yolo_head_P4(medium),
                self.yolo_head_P3(fine), representation)
