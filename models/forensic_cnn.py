"""Small residual CNN for low-level image-forensics evidence."""

from __future__ import annotations

import torch
from torch import nn

from .srm_filters import SRMFilterBank


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two convolutions followed by spatial downsampling."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ForensicCNN(nn.Module):
    """Classify raw 32x32 simplest patches using fixed SRM residuals.

    The output is one uncalibrated synthetic-class logit per patch. Train it
    with ``BCEWithLogitsLoss``; apply ``sigmoid`` only when probabilities are
    needed.
    """

    patch_size = 32

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.srm = SRMFilterBank()
        self.features = nn.Sequential(
            nn.BatchNorm2d(3),
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Return synthetic-class logits shaped ``(B,)``."""
        if patches.ndim != 4 or tuple(patches.shape[1:]) != (3, 32, 32):
            raise ValueError("Expected patches shaped (B, 3, 32, 32)")
        return self.classifier(self.features(self.srm(patches))).squeeze(1)
