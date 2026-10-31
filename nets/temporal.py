"""Shared temporal texture extraction and cross-slice feature fusion (Fig. 3)."""

import torch
from torch import nn


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.avg_pool(x).flatten(1)
        weights = self.sigmoid(self.fc2(self.relu(self.fc1(weights))))
        # Retain the residual SE used by the original EASD checkpoints.
        return x + x * weights[:, :, None, None]


class TemporalTextureModule(nn.Module):
    """Four nested temporal crops, convolution, residual SE, and projection.

    Attribute names match the original FeatureExtractor checkpoint keys.
    Legacy unused constructor arguments are accepted for compatibility.
    """

    def __init__(self, in_channels=13, features=3, out_channels=3,
                 channel_step=1, num_of_layers=12):
        super().__init__()
        if in_channels <= 6 * channel_step or channel_step < 1:
            raise ValueError("Temporal crops must contain at least one frame")
        self.channel_step = channel_step
        for branch in range(4):
            setattr(self, f"conv0_{branch}", nn.Conv2d(
                in_channels - 2 * branch * channel_step, 16, 3, padding=1))
            setattr(self, f"conv1_{branch}", nn.Conv2d(16, 1, 3, padding=1))
        self.ca = SEModule(4)
        self.conv = nn.Conv2d(4, features, 3, padding=1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        features = []
        for branch in range(4):
            trim = branch * self.channel_step
            crop = x if trim == 0 else x[:, trim:-trim]
            hidden = self.relu(getattr(self, f"conv0_{branch}")(crop))
            features.append(getattr(self, f"conv1_{branch}")(hidden))
        return self.relu(self.conv(self.ca(torch.cat(features, dim=1))))


class ChannelAttentionFusion(nn.Module):
    def __init__(self, in_channels, num_branches=5, reduction=4):
        super().__init__()
        self.num_branches = num_branches
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * num_branches, max(in_channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 1), num_branches),
            nn.Sigmoid(),
        )

    def forward(self, features):
        if len(features) != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} temporal blocks")
        descriptors = torch.cat([feature.mean((2, 3)) for feature in features], dim=1)
        weights = self.mlp(descriptors)
        fused = (torch.stack(features, dim=1) * weights[:, :, None, None, None]).sum(1)
        return fused, weights


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_branches=5):
        super().__init__()
        self.global_att_fusion = ChannelAttentionFusion(in_channels, num_branches)

    def forward(self, blocks):
        return self.global_att_fusion(blocks)[0]


FeatureExtractor = TemporalTextureModule
ModelResAttentionHybridNoBranch = FeatureFusionModule
