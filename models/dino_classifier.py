"""Binary image classifier built on top of a pretrained DINOv2 backbone.

The pipeline already creates ImageNet-normalized ``(3, 224, 224)`` tensors for
this model.  ``forward`` returns one logit per image so the training loss can be
``torch.nn.BCEWithLogitsLoss`` and the evaluator can use ``from_logits=True``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


DEFAULT_BACKBONE = "facebook/dinov2-small"


class DINOClassifier(nn.Module):
    """Classify an image as real (0) or synthetic (1) using DINOv2 features.

    Parameters
    ----------
    model_name:
        Hugging Face model ID used when ``backbone`` is not supplied.
    hidden_dim:
        Width of the small trainable classification head.
    dropout:
        Dropout probability in the classification head.
    freeze_backbone:
        Keep DINOv2 fixed and train only the classification head. This is the
        inexpensive default intended by the project architecture.
    backbone:
        Optional preconstructed backbone. Primarily useful for tests or for
        supplying an already-downloaded model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BACKBONE,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if backbone is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "DINOClassifier requires transformers; install requirements.txt"
                ) from exc
            backbone = AutoModel.from_pretrained(model_name)

        feature_dim = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("backbone.config.hidden_size must be a positive integer")

        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the pretrained feature extractor."""
        self.freeze_backbone = not trainable
        for parameter in self.backbone.parameters(): # pyright: ignore
            parameter.requires_grad = trainable
        if self.freeze_backbone:
            self.backbone.eval() # pyright: ignore

    def unfreeze_last_blocks(self, count: int) -> None:
        """Train the final ``count`` DINOv2 blocks and final layer norm only."""
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("count must be a positive integer")
        layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError("backbone must expose encoder.layer as a ModuleList")
        if count > len(layers):
            raise ValueError("count cannot exceed the number of encoder blocks")

        self.set_backbone_trainable(False)
        for block in layers[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        layernorm = getattr(self.backbone, "layernorm", None)
        if isinstance(layernorm, nn.Module):
            for parameter in layernorm.parameters():
                parameter.requires_grad = True
        self.freeze_backbone = False
        self.backbone.train(self.training) # pyright: ignore

    def train(self, mode: bool = True) -> "DINOClassifier":
        """Keep a frozen backbone in evaluation mode while training the head."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval() # pyright: ignore
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return synthetic-class logits with shape ``(batch_size,)``."""
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape (batch_size, 3, height, width)")

        outputs = self.backbone(pixel_values=pixel_values) # pyright: ignore
        features = self._class_token(outputs)
        return self.classifier(features).squeeze(-1)

    @staticmethod
    def _class_token(outputs: Any) -> torch.Tensor:
        """Extract DINOv2's CLS token from a Transformers model output."""
        hidden_state = getattr(outputs, "last_hidden_state", None)
        if hidden_state is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden_state = outputs[0]
        if not isinstance(hidden_state, torch.Tensor) or hidden_state.ndim != 3:
            raise ValueError("backbone must return last_hidden_state shaped (B, tokens, hidden)")
        return hidden_state[:, 0]

    @torch.no_grad()
    def predict_proba(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return the probability that each image is synthetic, shape ``(B,)``."""
        return torch.sigmoid(self(pixel_values))

    @torch.no_grad()
    def predict(
        self, pixel_values: torch.Tensor, *, threshold: float = 0.5
    ) -> torch.Tensor:
        """Return integer labels where 0 is real and 1 is synthetic."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(pixel_values) >= threshold).to(torch.int64)
