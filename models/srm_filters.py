"""Fixed Spatial Rich Model filters for the forensics branch."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _kernels() -> torch.Tensor:
    """Return three standard 5x5 high-pass SRM kernels."""
    return torch.tensor(
        [
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(1) / torch.tensor((4.0, 12.0, 2.0)).view(3, 1, 1, 1)


class SRMFilterBank(nn.Module):
    """Apply fixed high-pass filters to an RGB patch.

    Input is model-normalized RGB data. Each kernel is shared across RGB,
    yielding three residual maps. Kernels are buffers, so optimizers never
    update them.
    """

    def __init__(self, clip_value: float | None = 3.0) -> None:
        super().__init__()
        if clip_value is not None and (not math.isfinite(clip_value) or clip_value <= 0):
            raise ValueError("clip_value must be positive or None")
        self.register_buffer("kernels", _kernels(), persistent=True)
        self.clip_value = clip_value

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Return clipped valid-convolution residual maps."""
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError("Expected patches shaped (B, 3, H, W)")
        if not torch.is_floating_point(patches):
            raise TypeError("SRM patches must be floating-point tensors")
        kernel_size = self.kernels.shape[-1]
        if patches.shape[-2] < kernel_size or patches.shape[-1] < kernel_size:
            raise ValueError(f"SRM patches must be at least {kernel_size}x{kernel_size}")

        kernels = self.kernels.to(dtype=patches.dtype).repeat(1, 3, 1, 1)
        residuals = F.conv2d(patches, kernels)
        if self.clip_value is None:
            return residuals
        return F.hardtanh(residuals, min_val=-self.clip_value, max_val=self.clip_value)

    @torch.no_grad()
    def clipping_fraction(self, patches: torch.Tensor) -> float:
        """Return the fraction of residual values affected by clipping."""
        if self.clip_value is None:
            return 0.0
        kernels = self.kernels.to(dtype=patches.dtype).repeat(1, 3, 1, 1)
        residuals = F.conv2d(patches, kernels)
        return (residuals.abs() >= self.clip_value).float().mean().item()
