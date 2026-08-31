"""Tests for fixed score-level fusion calibration."""

import pytest
import torch

from model_runs.calibrate_fusion import fused_logits, select_threshold, select_weight


def test_fixed_fusion_selects_the_better_branch() -> None:
    scores = torch.tensor([[-3.0, 100.0], [-2.0, -100.0], [2.0, 100.0], [3.0, -100.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

    assert select_weight(scores, labels, steps=11) == 1.0
    assert torch.equal(fused_logits(scores, 1.0), scores[:, 0])


def test_threshold_is_a_probability_and_requires_both_classes() -> None:
    logits = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

    assert 0.0 <= select_threshold(logits, labels) <= 1.0
    with pytest.raises(RuntimeError, match="both classes"):
        select_threshold(logits[:2], labels[:2])
