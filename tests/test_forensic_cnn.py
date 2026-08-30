"""Tests for the fixed-SRM forensic branch."""

from __future__ import annotations

import pytest
import torch

from models.forensic_cnn import ForensicCNN
from models.srm_filters import SRMFilterBank
from pipeline import simplest_patch


def test_srm_filters_are_fixed_and_preserve_shape() -> None:
    bank = SRMFilterBank()
    patches = torch.randn(2, 3, 32, 32, requires_grad=True)

    residuals = bank(patches)
    residuals.square().mean().backward()

    assert residuals.shape == (2, 3, 32, 32)
    assert list(bank.parameters()) == []
    assert bank.kernels.requires_grad is False
    assert patches.grad is not None


def test_forensic_cnn_returns_logits_and_trains_cnn_only() -> None:
    model = ForensicCNN(dropout=0.0)
    patches = torch.randn(4, 3, 32, 32)

    logits = model(patches)
    torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0, 0.0, 1.0])).backward()

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    assert model.features[1][0].weight.grad is not None
    assert model.srm.kernels.grad is None


@pytest.mark.parametrize("shape", [(2, 3, 31, 32), (2, 1, 32, 32), (2, 3, 32)])
def test_forensic_cnn_rejects_wrong_patch_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="B, 3, 32, 32"):
        ForensicCNN()(torch.randn(shape))


def test_srm_filters_reject_integer_patches() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        SRMFilterBank()(torch.ones(1, 3, 32, 32, dtype=torch.uint8))


def test_forensic_cnn_accepts_pipeline_patch(textured_image) -> None:
    """The raw simplest-patch output is directly usable by this branch."""
    patch = simplest_patch(textured_image).unsqueeze(0)
    model = ForensicCNN().eval()

    with torch.inference_mode():
        logits = model(patch)

    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()
