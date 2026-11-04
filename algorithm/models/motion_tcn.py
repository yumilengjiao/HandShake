"""Experimental causal hand-motion contrasts; the original T13 remains unchanged.

Only current/past geometry is used. Means exclude zero-padded/missing-hand
frames using the schema's keypoint masks; no truth/readiness label is consulted.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .tcn import GestureTCN


class MotionGestureTCN(GestureTCN):
    """Original last-frame TCN plus contrasts at fixed 4/8/16-frame scales.

    The branch sees 69 geometry channels, not normalized timing/quality/masks.
    Its support is at most 16 frames, so the full model's receptive field stays
    31. A zero-initialized class residual starts from the baseline logits.
    """

    scales = (4, 8, 16)
    geometry_channels = 69

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.input_dim != 94 or self.window_size != 32:
            raise ValueError("motion variant requires the fixed 32x94 feature schema")
        probability = self.blocks[0].dropout.p
        self.motion_encoder = nn.Sequential(
            nn.Linear(len(self.scales) * self.geometry_channels, self.channels),
            nn.LayerNorm(self.channels),
            nn.ReLU(),
            nn.Dropout(probability),
        )
        self.motion_classifier = nn.Linear(self.channels, self.num_classes)
        nn.init.zeros_(self.motion_classifier.weight)
        nn.init.zeros_(self.motion_classifier.bias)

    def motion_contrasts(self, x):
        geometry = x[..., : self.geometry_channels].transpose(1, 2)
        present = (x[..., 73:94] > 0).any(dim=-1).to(x.dtype).unsqueeze(1)
        contrasts = []
        for scale in self.scales:
            count = F.avg_pool1d(F.pad(present, (scale - 1, 0)), scale, stride=1)
            mean = F.avg_pool1d(
                F.pad(geometry * present, (scale - 1, 0)), scale, stride=1
            ) / count.clamp_min(1 / scale)
            contrasts.append((geometry - mean) * present)
        return torch.cat(contrasts, dim=1).transpose(1, 2), present.transpose(1, 2)

    def forward_all(self, x):
        baseline = super().forward_all(x)
        contrasts, present = self.motion_contrasts(x)
        residual = self.motion_classifier(self.motion_encoder(contrasts))
        return baseline + residual * present
