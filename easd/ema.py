"""Exponential moving average of trainable model state for validation checkpoints."""

from copy import deepcopy
import math

import torch


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.9999, tau=2000):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.updates = 0
        self.decay = decay
        self.tau = tau

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        weight = self.decay * (1 - math.exp(-self.updates / self.tau))
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                value *= weight
                value += (1 - weight) * source[name].detach()
            else:
                value.copy_(source[name])
