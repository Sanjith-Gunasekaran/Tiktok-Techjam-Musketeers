"""DINOv2 binary classifier shared by training notebooks and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from transformers import Dinov2Config, Dinov2Model


DINO_MODEL_ID = "facebook/dinov2-base"
AI_CLASS_NAMES = {"ai", "aigc", "fake", "synthetic", "ai-generated"}


class DINOv2BinaryClassifier(nn.Module):
    """Frozen DINOv2 backbone with the notebook's two-class linear head."""

    def __init__(
        self,
        model_id: str = DINO_MODEL_ID,
        dropout: float = 0.20,
        *,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        if pretrained_backbone:
            self.backbone = Dinov2Model.from_pretrained(model_id)
        else:
            # Inference checkpoints contain the entire backbone. Download only
            # its small configuration before restoring the saved parameters.
            config = Dinov2Config.from_pretrained(model_id)
            self.backbone = Dinov2Model(config)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)


def load_detector_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    model_id: str | None = None,
    ai_class_index: int | None = None,
) -> tuple[DINOv2BinaryClassifier, str, int]:
    """Load a checkpoint produced by ``DINOv2_ML.ipynb``.

    Returns the restored model, processor/model ID, and the logit index whose
    softmax probability represents AI-generated content.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        # ``weights_only`` is unavailable on older supported PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state dictionary or checkpoint mapping")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_model_id = checkpoint.get("model_id")
        class_to_idx = checkpoint.get("class_to_idx")
    else:
        state_dict = checkpoint
        saved_model_id = None
        class_to_idx = None

    resolved_model_id = model_id or saved_model_id or DINO_MODEL_ID
    resolved_ai_index = _resolve_ai_class_index(class_to_idx, ai_class_index)

    model = DINOv2BinaryClassifier(
        model_id=resolved_model_id, pretrained_backbone=False
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, resolved_model_id, resolved_ai_index


def _resolve_ai_class_index(
    class_to_idx: Any, requested_index: int | None
) -> int:
    if requested_index is not None:
        if requested_index not in (0, 1):
            raise ValueError("ai_class_index must be 0 or 1")
        return requested_index

    if isinstance(class_to_idx, Mapping):
        for class_name, index in class_to_idx.items():
            if str(class_name).strip().lower() in AI_CLASS_NAMES:
                index = int(index)
                if index not in (0, 1):
                    raise ValueError("AI class index in checkpoint must be 0 or 1")
                return index

    raise ValueError(
        "Checkpoint does not identify the AI class. Pass --ai-class-index 0 or 1."
    )
