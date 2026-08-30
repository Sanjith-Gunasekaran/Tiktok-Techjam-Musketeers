"""Tests for fixed score-level branch fusion."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models import TwoBranchDetector


class ConstantBranch(nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.register_buffer("logit", torch.tensor(logit))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(len(inputs))


class TrainableBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.mean(dim=tuple(range(1, inputs.ndim)), keepdim=True)).flatten()


def test_branch_logits_and_fixed_fusion() -> None:
    model = TwoBranchDetector(ConstantBranch(2.0), ConstantBranch(-1.0), dino_weight=0.8)
    images, patches = torch.randn(3, 3, 224, 224), torch.randn(3, 3, 32, 32)

    dino_logits, forensic_logits = model.branch_logits(images, patches)

    assert dino_logits.shape == forensic_logits.shape == (3,)
    assert torch.allclose(model(images, patches), torch.full((3,), 1.4))
    assert torch.all((model.predict_proba(images, patches) >= 0) & (model.predict_proba(images, patches) <= 1))


@pytest.mark.parametrize(("weight", "expected"), [(0.0, -1.0), (1.0, 2.0)])
def test_extreme_fusion_weights_match_one_branch(weight: float, expected: float) -> None:
    model = TwoBranchDetector(ConstantBranch(2.0), ConstantBranch(-1.0), dino_weight=weight)

    assert torch.equal(model(torch.randn(2, 3, 224, 224), torch.randn(2, 3, 32, 32)), torch.full((2,), expected))


def test_fusion_rejects_mismatched_batch_sizes() -> None:
    model = TwoBranchDetector(ConstantBranch(0.0), ConstantBranch(0.0))
    with pytest.raises(ValueError, match="different batch sizes"):
        model(torch.randn(2, 3, 224, 224), torch.randn(3, 3, 32, 32))


def test_frozen_branches_receive_no_parameter_gradients() -> None:
    model = TwoBranchDetector(TrainableBranch(), TrainableBranch())
    model.freeze_branches()
    images = torch.randn(2, 3, 224, 224, requires_grad=True)
    patches = torch.randn(2, 3, 32, 32, requires_grad=True)

    model(images, patches).sum().backward()

    assert not model.dino.training and not model.forensic.training
    assert all(parameter.grad is None for parameter in model.dino.parameters())
    assert all(parameter.grad is None for parameter in model.forensic.parameters())


def test_fusion_weight_is_saved_and_restored() -> None:
    model = TwoBranchDetector(ConstantBranch(0.0), ConstantBranch(0.0), dino_weight=0.3)
    restored = TwoBranchDetector(ConstantBranch(0.0), ConstantBranch(0.0), dino_weight=0.8)
    restored.load_state_dict(model.state_dict())

    assert restored.dino_weight.item() == pytest.approx(0.3)
