"""Fixed or learned score-level fusion of detector branches."""

from __future__ import annotations

import math

import torch
from torch import nn


class TwoBranchDetector(nn.Module):
    """Fuse DINO and forensic synthetic-class logits."""

    def __init__(
        self,
        dino: nn.Module,
        forensic: nn.Module,
        *,
        fusion_mode: str = "fixed",
        dino_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"fixed", "learned"}:
            raise ValueError("fusion_mode must be 'fixed' or 'learned'")
        if not isinstance(dino_weight, (float, int)) or isinstance(dino_weight, bool):
            raise TypeError("dino_weight must be a number")
        if not math.isfinite(dino_weight) or not 0.0 <= dino_weight <= 1.0:
            raise ValueError("dino_weight must be between 0 and 1")
        self.dino, self.forensic, self.fusion_mode = dino, forensic, fusion_mode
        self.register_buffer("dino_weight", torch.tensor(float(dino_weight)))
        self.fusion = nn.Linear(2, 1) if fusion_mode == "learned" else None
        if self.fusion is not None:
            with torch.no_grad():
                self.fusion.weight.copy_(torch.tensor([[dino_weight, 1.0 - dino_weight]]))
                self.fusion.bias.zero_()
        self._branches_frozen = False

    def set_dino_weight(self, weight: float) -> None:
        if self.fusion_mode != "fixed":
            raise RuntimeError("dino_weight applies only to fixed fusion")
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("dino_weight must be between 0 and 1")
        self.dino_weight.fill_(weight)

    def freeze_branches(self) -> None:
        self._branches_frozen = True
        for branch in (self.dino, self.forensic):
            for parameter in branch.parameters():
                parameter.requires_grad = False
            branch.eval()

    def unfreeze_branches(self) -> None:
        self._branches_frozen = False
        for branch in (self.dino, self.forensic):
            for parameter in branch.parameters():
                parameter.requires_grad = True
            branch.train(self.training)

    def train(self, mode: bool = True) -> "TwoBranchDetector":
        super().train(mode)
        if self._branches_frozen:
            self.dino.eval()
            self.forensic.eval()
        return self

    def branch_logits(self, dino_images: torch.Tensor, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dino_logits, forensic_logits = self.dino(dino_images), self.forensic(patches)
        if not all(isinstance(logits, torch.Tensor) and logits.ndim == 1 for logits in (dino_logits, forensic_logits)):
            raise ValueError("branches must return logits shaped (B,)")
        if len(dino_logits) != len(forensic_logits):
            raise ValueError("DINO and forensic branches returned different batch sizes")
        return dino_logits, forensic_logits

    def forward(self, dino_images: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        dino_logits, forensic_logits = self.branch_logits(dino_images, patches)
        if self.fusion is not None:
            return self.fusion(torch.stack((dino_logits, forensic_logits), dim=1)).squeeze(1)
        weight = self.dino_weight.to(dtype=dino_logits.dtype)
        return weight * dino_logits + (1.0 - weight) * forensic_logits

    @torch.no_grad()
    def predict_proba(self, dino_images: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self(dino_images, patches))
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict(self, dino_images: torch.Tensor, patches: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(dino_images, patches) >= threshold).to(torch.int64)

    def checkpoint_config(self) -> dict[str, object]:
        """Configuration training code should save beside ``state_dict``."""
        return {"fusion_mode": self.fusion_mode, "dino_weight": self.dino_weight.item(), "branches_frozen": self._branches_frozen, "dino": getattr(self.dino, "checkpoint_config", lambda: {})(), "forensic": getattr(self.forensic, "checkpoint_config", lambda: {})()}
