"""Tests for the fixed-SRM forensic branch."""

from __future__ import annotations

import pytest
import torch

from models import ForensicCNN, SRMFilterBank
from pipeline import simplest_patch


def test_srm_filters_are_fixed_and_preserve_shape() -> None:
    bank = SRMFilterBank()
    patches = torch.rand(2, 3, 32, 32, requires_grad=True)

    residuals = bank(patches)
    residuals.square().mean().backward()

    assert residuals.shape == (2, 3, 28, 28)
    assert list(bank.parameters()) == []
    assert bank.kernels.requires_grad is False
    assert patches.grad is not None


def test_forensic_cnn_returns_logits_and_keeps_srm_fixed() -> None:
    model = ForensicCNN(dropout=0.0)
    patches = torch.rand(4, 3, 32, 32) * 255
    before = model.srm.kernels.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = torch.nn.BCEWithLogitsLoss()(
        model(patches), torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert model.features[0][0].weight.grad is not None
    assert torch.equal(model.srm.kernels, before)


@pytest.mark.parametrize("shape", [(2, 3, 31, 32), (2, 1, 32, 32), (2, 3, 32)])
def test_forensic_cnn_rejects_wrong_patch_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="B, 3, 32, 32"):
        ForensicCNN()(torch.randn(shape))


def test_srm_filters_reject_integer_patches() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        SRMFilterBank()(torch.ones(1, 3, 32, 32, dtype=torch.uint8))


def test_srm_filters_reject_too_small_patches() -> None:
    with pytest.raises(ValueError, match="at least 5x5"):
        SRMFilterBank()(torch.rand(1, 3, 4, 4))


def test_srm_filters_match_ssp_kernel_and_do_not_create_borders() -> None:
    bank = SRMFilterBank()
    constant = torch.full((1, 3, 32, 32), 0.5)

    assert torch.equal(
        bank.kernels[2, 0, 2], torch.tensor([0.0, 0.5, -1.0, 0.5, 0.0])
    )
    assert torch.allclose(bank(constant), torch.zeros(1, 3, 28, 28), atol=1e-6)


def test_srm_filters_clip_strong_residuals() -> None:
    bank = SRMFilterBank()
    patches = torch.zeros(1, 3, 9, 9)
    patches[:, :, 2:7, 2:7] = (bank.kernels[1, 0] > 0).to(torch.float32)

    assert bank(patches)[0, 1, 2, 2] == 3.0


def test_forensic_cnn_accepts_pipeline_patch(textured_image) -> None:
    """The raw simplest-patch output is directly usable by this branch."""
    patch = simplest_patch(textured_image).unsqueeze(0)
    model = ForensicCNN().eval()

    with torch.inference_mode():
        logits = model(patch)

    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()
