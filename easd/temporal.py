"""Temporal-Based Texture Module and cross-block feature fusion."""

import torch
from torch import nn


class ResidualChannelAttention(nn.Module):
    """The residual squeeze-and-excitation operator used by EASD checkpoints."""

    def __init__(self, channels, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.avg_pool(x).flatten(1)
        weights = self.sigmoid(self.fc2(self.relu(self.fc1(weights))))
        return x + x * weights[:, :, None, None]


class TemporalCropBranch(nn.Module):
    def __init__(self, frames):
        super().__init__()
        self.input_conv = nn.Conv2d(frames, 16, 3, padding=1)
        self.activation = nn.LeakyReLU(0.1)
        self.output_conv = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        return self.output_conv(self.activation(self.input_conv(x)))


class TemporalTextureModule(nn.Module):
    """Four nested 13/11/9/7-frame crops processed with shared block weights."""

    def __init__(self, frames=13, output_channels=3, crop_step=1):
        super().__init__()
        if crop_step < 1 or frames <= 6 * crop_step:
            raise ValueError("Temporal crops must contain at least one frame")
        self.crop_step = crop_step
        self.branches = nn.ModuleList(
            [TemporalCropBranch(frames - 2 * index * crop_step) for index in range(4)]
        )
        self.channel_attention = ResidualChannelAttention(4)
        self.projection = nn.Conv2d(4, output_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        features = []
        for index, branch in enumerate(self.branches):
            trim = index * self.crop_step
            crop = x if trim == 0 else x[:, trim:-trim]
            features.append(branch(crop))
        return self.activation(self.projection(self.channel_attention(torch.cat(features, dim=1))))


class TemporalFeatureFusion(nn.Module):
    """Global descriptors predict one sigmoid weight for each temporal block."""

    def __init__(self, channels=3, num_blocks=5, reduction=4):
        super().__init__()
        self.num_blocks = num_blocks
        hidden = max(channels // reduction, 1)
        self.attention = nn.Sequential(
            nn.Linear(channels * num_blocks, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_blocks),
            nn.Sigmoid(),
        )

    def forward(self, features):
        if len(features) != self.num_blocks:
            raise ValueError(f"Expected {self.num_blocks} temporal blocks")
        descriptors = torch.cat([feature.mean((2, 3)) for feature in features], dim=1)
        weights = self.attention(descriptors)
        return (torch.stack(features, dim=1) * weights[:, :, None, None, None]).sum(1)
