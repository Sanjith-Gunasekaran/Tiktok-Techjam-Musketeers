"""Fixed score-level fusion of the semantic and forensic branches."""

from __future__ import annotations

import math

import torch
from torch import nn


class TwoBranchDetector(nn.Module):
    """Fuse DINO and forensic synthetic-class logits with a fixed weight."""

    def __init__(
        self, dino: nn.Module, forensic: nn.Module, *, dino_weight: float = 0.8
    ) -> None:
        super().__init__()
        self.dino = dino
        self.forensic = forensic
        self.register_buffer("dino_weight", torch.tensor(self._valid_weight(dino_weight)))
        self._branches_frozen = False

    @staticmethod
    def _valid_weight(weight: float) -> float:
        if not isinstance(weight, (float, int)) or isinstance(weight, bool):
            raise TypeError("dino_weight must be a number")
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("dino_weight must be between 0 and 1")
        return float(weight)

    def set_dino_weight(self, weight: float) -> None:
        """Set the validation-selected DINO logit weight."""
        self.dino_weight.fill_(self._valid_weight(weight))

    def freeze_branches(self) -> None:
        """Freeze both branches when fitting a later fusion-only module."""
        self._branches_frozen = True
        for branch in (self.dino, self.forensic):
            for parameter in branch.parameters():
                parameter.requires_grad = False
            branch.eval()

    def train(self, mode: bool = True) -> "TwoBranchDetector":
        """Keep explicitly frozen branches in evaluation mode."""
        super().train(mode)
        if self._branches_frozen:
            self.dino.eval()
            self.forensic.eval()
        return self

    def branch_logits(
        self, dino_images: torch.Tensor, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the individual branch logits, each shaped ``(B,)``."""
        dino_logits = self.dino(dino_images)
        forensic_logits = self.forensic(patches)
        for name, logits in (("dino", dino_logits), ("forensic", forensic_logits)):
            if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
                raise ValueError(f"{name} branch must return logits shaped (B,)")
        if len(dino_logits) != len(forensic_logits):
            raise ValueError("DINO and forensic branches returned different batch sizes")
        return dino_logits, forensic_logits

    def forward(self, dino_images: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """Return fixed-weight fused synthetic-class logits shaped ``(B,)``."""
        dino_logits, forensic_logits = self.branch_logits(dino_images, patches)
        weight = self.dino_weight.to(dtype=dino_logits.dtype)
        return weight * dino_logits + (1.0 - weight) * forensic_logits

    @torch.no_grad()
    def predict_proba(self, dino_images: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """Return fused synthetic-class probabilities shaped ``(B,)``."""
        return torch.sigmoid(self(dino_images, patches))
