"""Tests for the DINOv2 binary classification head without network access."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.dino_classifier import DINOClassifier


class FakeDINO(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.projection = nn.Linear(3, hidden_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(4)])
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, *, pixel_values: torch.Tensor):
        pooled_rgb = pixel_values.mean(dim=(2, 3))
        class_token = self.projection(pooled_rgb).unsqueeze(1)
        patch_tokens = torch.zeros(
            len(pixel_values), 4, self.config.hidden_size, device=pixel_values.device
        )
        return SimpleNamespace(
            last_hidden_state=torch.cat((class_token, patch_tokens), dim=1)
        )


def test_output_probability_and_prediction_shapes():
    model = DINOClassifier(backbone=FakeDINO(), hidden_dim=8, dropout=0.0)
    images = torch.randn(3, 3, 224, 224)

    logits = model(images)
    probabilities = model.predict_proba(images)
    predictions = model.predict(images)

    assert logits.shape == probabilities.shape == predictions.shape == (3,)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert predictions.dtype == torch.int64
    assert set(predictions.tolist()) <= {0, 1}


def test_backbone_is_frozen_and_stays_in_eval_mode_by_default():
    model = DINOClassifier(backbone=FakeDINO())
    model.train()

    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_backbone_can_be_unfrozen():
    model = DINOClassifier(backbone=FakeDINO(), freeze_backbone=False)
    model.train()

    assert model.backbone.training
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_backbone_can_unfreeze_only_its_final_blocks():
    model = DINOClassifier(backbone=FakeDINO())
    model.unfreeze_last_blocks(2)
    model.train()

    assert model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.projection.parameters())
    assert all(
        not parameter.requires_grad for block in model.backbone.encoder.layer[:2] for parameter in block.parameters()
    )
    assert all(
        parameter.requires_grad for block in model.backbone.encoder.layer[2:] for parameter in block.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.backbone.layernorm.parameters())


@pytest.mark.parametrize("count", [0, -1, 5])
def test_unfreeze_last_blocks_rejects_invalid_counts(count: int):
    with pytest.raises(ValueError):
        DINOClassifier(backbone=FakeDINO()).unfreeze_last_blocks(count)


def test_invalid_input_shape_is_rejected():
    model = DINOClassifier(backbone=FakeDINO())
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(3, 224, 224))
