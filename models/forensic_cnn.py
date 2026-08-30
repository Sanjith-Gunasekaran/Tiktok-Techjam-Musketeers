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

    Inputs are the pipeline's float pixels in ``[0, 255]``. They are scaled
    and ImageNet-normalized before SRM filtering, matching SSP preprocessing.
    The output is one uncalibrated synthetic-class logit per patch.
    """

    patch_size = 32

    def __init__(self, dropout: float = 0.2, *, srm_clip_value: float | None = 3.0) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.srm = SRMFilterBank(clip_value=srm_clip_value)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.features = nn.Sequential(
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
        expected_shape = (3, self.patch_size, self.patch_size)
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected_shape:
            raise ValueError(f"Expected patches shaped (B, {', '.join(map(str, expected_shape))})")
        if not torch.is_floating_point(patches):
            raise TypeError("Forensic patches must be floating-point tensors")
        patches = patches / 255.0
        patches = (patches - self.mean.to(dtype=patches.dtype)) / self.std.to(
            dtype=patches.dtype
        )
        return self.classifier(self.features(self.srm(patches))).squeeze(1)
