# Portions adapted from CAMixerSR, Copyright 2024 CAMixerSR Authors.
# Apache-2.0 terms: LICENSES/CAMixerSR-Apache-2.0.txt.
# Modified for EASD entropy routing and checkpoint compatibility; see NOTICE.

"""Entropy selective window attention (Figs. 4-5).

The original checkpoint names and learned operators are preserved. Routing is
independent for every sample, including samples with no foreground windows.
"""

import torch
from torch import nn
from torch.nn import functional as F


def window_partition(x, size):
    batch, channels, height, width = x.shape
    return x.reshape(batch, channels, height // size, size, width // size, size).permute(
        0, 2, 4, 3, 5, 1).reshape(batch, -1, size * size, channels)


def window_reverse(windows, size, height, width):
    batch, _, _, channels = windows.shape
    return windows.reshape(batch, height // size, width // size, size, size, channels).permute(
        0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)


def flow_warp(x, flow):
    """Retain the original bilinear offset warping used to construct keys."""
    batch, _, height, width = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=x.device, dtype=x.dtype),
        torch.arange(width, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0) + flow
    grid = torch.stack((2 * grid[..., 0] / max(width - 1, 1) - 1,
                        2 * grid[..., 1] / max(height - 1, 1) - 1), dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps) * self.weight[:, None, None] + self.bias[:, None, None]


class EntropyMask(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        size = self.window_size
        height, width = x.shape[-2:]
        rows, cols = height // size, width // size
        if height % (2 * size) or width % (2 * size):
            raise ValueError("Entropy masking requires spatial dimensions divisible by 2 * window_size")
        # Hard entropy routing is a selection rule, not a learned mask.
        with torch.no_grad():
            windows = window_partition(x.float(), size).squeeze(-1)
            probabilities = windows.softmax(dim=-1)
            entropy = -(probabilities * (probabilities + 1e-8).log2()).sum(-1)
            average = entropy.mean(dim=1, keepdim=True)
            merged = F.avg_pool2d(entropy.reshape(-1, 1, rows, cols), 2, 2)
            foreground = (merged < average[:, :, None, None]) & (
                merged > 0.5 * average[:, :, None, None])
            return foreground.repeat_interleave(2, -2).repeat_interleave(2, -1).flatten(1)


class PredictorLG(nn.Module):
    def __init__(self, dim, window_size=8):
        super().__init__()
        channels = dim + 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1), LayerNorm(channels // 4),
            nn.LeakyReLU(0.1, inplace=True))
        self.out_offsets = nn.Sequential(
            nn.Conv2d(channels // 4, channels // 8, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 8, 2, 1))
        # Kept solely for strict loading of the original checkpoints.
        self.out_mask = nn.Sequential(
            nn.Linear(window_size ** 2, window_size), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(window_size, 2), nn.Softmax(dim=-1))
        self.out_mask.requires_grad_(False)
        self.entropy_mask = EntropyMask(window_size)
        self.out_CA = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels // 4, dim, 1), nn.Sigmoid())
        self.out_SA = nn.Sequential(nn.Conv2d(channels // 4, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.in_conv(x)
        return (self.entropy_mask(x.sum(1, keepdim=True)),
                self.out_offsets(x).tanh() * 8.0, self.out_CA(x), self.out_SA(x))


class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.project_v = nn.Conv2d(dim, dim, 1)
        self.project_q = nn.Linear(dim, dim)
        self.project_k = nn.Linear(dim, dim)
        # Original spelling retained for checkpoint compatibility.
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=2, groups=dim, dilation=2))
        self.project_out = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.route = PredictorLG(dim, window_size)

    def forward(self, x):
        batch, _, height, width = x.shape
        size = self.window_size
        values = self.project_v(x)
        positions = torch.linspace(-1, 1, size, device=x.device, dtype=x.dtype)
        position = torch.stack(torch.meshgrid(positions, positions, indexing="ij"))
        position = position.unsqueeze(0).repeat(batch, 1, height // size, width // size)
        foreground, offsets, channel_attention, spatial_attention = self.route(
            torch.cat((values, position), dim=1))
        queries = window_partition(x, size)
        keys = window_partition(x + flow_warp(x, offsets.permute(0, 2, 3, 1)), size)
        value_windows = window_partition(values, size)
        output = window_partition(values * spatial_attention, size)
        # Boolean indexing allows a different foreground count for each sample.
        if foreground.any():
            query = self.project_q(queries[foreground])
            key = self.project_k(keys[foreground])
            # Preserve original (unscaled) logits; see README for paper differences.
            attention = (query @ key.transpose(-2, -1)).softmax(dim=-1)
            output = output.clone()
            output[foreground] = attention @ value_windows[foreground]
        output = window_reverse(output, size, height, width)
        output = output + self.act(self.conv_sptial(output)) * channel_attention
        return self.project_out(output)


class GatedFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        first, second = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class Block(nn.Module):
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.mixer = CAMixer(dim, window_size)
        self.norm2 = LayerNorm(dim)
        self.ffn = GatedFeedForward(dim)

    def forward(self, x):
        x = self.norm1(x + self.mixer(x))
        return self.norm2(x + self.ffn(x))


class Group(nn.Module):
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.body = nn.ModuleList([Block(dim, window_size)])
        self.body_tail = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x
        for block in self.body:
            x = block(x)
        return self.body_tail(x) + shortcut


class EntropySelectiveAttention(nn.Module):
    def __init__(self, dim=60, window_size=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, dim, 3, padding=1)
        self.group = Group(dim, window_size)

    def forward(self, spikes):
        return self.group(self.conv1(spikes.sum(dim=1, keepdim=True)))
