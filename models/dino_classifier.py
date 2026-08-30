"""DINOv2 binary classifier with explicit backbone-training states."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

DEFAULT_BACKBONE = "facebook/dinov2-small"


class DINOClassifier(nn.Module):
    """Return one synthetic-class logit from normalized DINOv2 inputs."""

    def __init__(
        self,
        model_name: str = DEFAULT_BACKBONE,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        revision: str | None = None,
        local_files_only: bool = False,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("hidden_dim must be positive and dropout must be in [0, 1)")
        if backbone is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError("DINOClassifier requires transformers") from exc
            backbone = AutoModel.from_pretrained(
                model_name, revision=revision, local_files_only=local_files_only
            )
        feature_dim = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("backbone.config.hidden_size must be a positive integer")

        self.backbone = backbone
        self.model_name, self.revision = model_name, revision
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._backbone_state = "frozen"
        self._unfrozen_blocks = 0
        self.set_backbone_trainable(not freeze_backbone)

    def _layers(self) -> nn.ModuleList:
        layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError("backbone must expose encoder.layer as a ModuleList")
        return layers

    def _apply_backbone_state(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = self._backbone_state == "full"
        if self._backbone_state == "frozen":
            self.backbone.eval()
            return
        if self._backbone_state == "full":
            self.backbone.train(self.training)
            return
        layers = self._layers()
        self.backbone.eval()
        for block in layers[-self._unfrozen_blocks :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
            block.train(self.training)
        layernorm = getattr(self.backbone, "layernorm", None)
        if isinstance(layernorm, nn.Module):
            for parameter in layernorm.parameters():
                parameter.requires_grad = True
            layernorm.train(self.training)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Fully freeze or fully unfreeze DINOv2."""
        self._backbone_state = "full" if trainable else "frozen"
        self._unfrozen_blocks = 0
        self._apply_backbone_state()

    def unfreeze_last_blocks(self, count: int) -> None:
        """Train only DINOv2's final blocks and final layer norm."""
        if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= len(self._layers()):
            raise ValueError("count must be between 1 and the number of encoder blocks")
        self._backbone_state, self._unfrozen_blocks = "partial", count
        self._apply_backbone_state()

    def train(self, mode: bool = True) -> "DINOClassifier":
        super().train(mode)
        self._apply_backbone_state()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return synthetic-class logits shaped ``(B,)``."""
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape (B, 3, H, W)")
        return self.classifier(self._class_token(self.backbone(pixel_values=pixel_values))).squeeze(-1)

    @staticmethod
    def _class_token(outputs: Any) -> torch.Tensor:
        hidden = getattr(outputs, "last_hidden_state", outputs[0] if isinstance(outputs, (tuple, list)) and outputs else None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("backbone must return last_hidden_state shaped (B, tokens, hidden)")
        return hidden[:, 0]

    @torch.no_grad()
    def predict_proba(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return probabilities while restoring the prior train/eval mode."""
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self(pixel_values))
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(pixel_values) >= threshold).to(torch.int64)

    def checkpoint_config(self) -> dict[str, object]:
        """Configuration training code should save beside ``state_dict``."""
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "backbone_state": self._backbone_state,
            "unfrozen_blocks": self._unfrozen_blocks,
        }
