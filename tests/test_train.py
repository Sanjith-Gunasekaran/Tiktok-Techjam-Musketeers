"""Tests for the stage-aware training helpers."""

import argparse

import pytest
import torch
from torch import nn

from model_runs.train import (
    forensic_from_checkpoint,
    load_state,
    parameter_groups,
    run_epoch,
    run_fusion_epoch,
    save_checkpoint,
    split_fusion_cache,
)
from models import ForensicCNN, TwoBranchDetector


class MeanLogit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.scale * inputs.mean(dim=tuple(range(1, inputs.ndim)))


def test_run_epoch_trains_one_branch_on_synthetic_batches() -> None:
    model = MeanLogit()
    batch = (
        torch.randn(4, 3, 224, 224),
        torch.randn(4, 3, 32, 32),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 1, 0, 1]),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = run_epoch(model, "dino_head", [batch], torch.device("cpu"), optimizer)

    assert set(metrics) == {"loss", "accuracy", "auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_single_class_epoch_reports_undefined_auc() -> None:
    batch = (
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 3, 32, 32),
        torch.tensor([1, 1]),
        torch.tensor([1, 1]),
    )

    assert run_epoch(MeanLogit(), "dino_head", [batch], torch.device("cpu"))["auc"] is None


def test_run_epoch_accepts_a_single_dino_view_batch() -> None:
    batch = {
        "dino": torch.randn(4, 3, 224, 224),
        "label": torch.tensor([0, 1, 0, 1]),
        "original_label": torch.tensor([0, 1, 0, 1]),
    }

    metrics = run_epoch(MeanLogit(), "dino_head", [batch], torch.device("cpu"))

    assert metrics["auc"] is not None


def test_cached_fusion_uses_stratified_calibration_and_selection() -> None:
    scores = torch.tensor([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    calibration, selection = split_fusion_cache(scores, labels, 0.5, seed=4)
    model = TwoBranchDetector(nn.Identity(), nn.Identity(), fusion_mode="learned")
    model.freeze_branches()
    optimizer = torch.optim.SGD(model.fusion.parameters(), lr=0.1)

    train = run_fusion_epoch(model, *calibration, torch.device("cpu"), 2, optimizer)
    validation = run_fusion_epoch(model, *selection, torch.device("cpu"), 2)

    assert train["auc"] is not None
    assert validation["auc"] is not None


def test_cached_fusion_requires_two_examples_per_class() -> None:
    with pytest.raises(ValueError, match="two examples"):
        split_fusion_cache(torch.randn(3, 2), torch.tensor([0.0, 1.0, 1.0]), 0.5, seed=1)


def test_cached_fusion_keeps_image_families_together() -> None:
    scores = torch.tensor([[-2.0, -1.0], [2.0, 1.0], [-3.0, -1.0], [3.0, 1.0]])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    calibration, selection = split_fusion_cache(
        scores,
        labels,
        0.5,
        seed=5,
        image_ids=("real_a", "synthetic_a", "real_b", "synthetic_b"),
    )

    calibration_families = {abs(int(row[0].item())) for row in calibration[0]}
    selection_families = {abs(int(row[0].item())) for row in selection[0]}
    assert calibration_families.isdisjoint(selection_families)


def test_parameter_groups_exclude_bias_and_normalization_from_decay() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    groups = parameter_groups(model, [{"params": model.parameters(), "lr": 0.1}], 0.01)

    decayed = {id(parameter) for group in groups if group["weight_decay"] for parameter in group["params"]}
    assert id(model[0].weight) in decayed
    assert id(model[0].bias) not in decayed
    assert id(model[1].weight) not in decayed


def test_safe_checkpoint_round_trip_and_stage_validation(tmp_path) -> None:
    model = MeanLogit()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "checkpoint.pt"
    args = argparse.Namespace(stage="forensic", output_dir=tmp_path)

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        1,
        {"loss": 0.5, "accuracy": 0.5, "auc": 0.5},
        0.5,
        args,
    )

    loaded = load_state(MeanLogit(), checkpoint, torch.device("cpu"), ("forensic",))
    assert loaded["format_version"] == 2
    with pytest.raises(ValueError, match="expected dino_head"):
        load_state(MeanLogit(), checkpoint, torch.device("cpu"), ("dino_head",))


def test_forensic_checkpoint_restores_saved_srm_configuration(tmp_path) -> None:
    model = ForensicCNN(dropout=0.0, srm_clip_value=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = tmp_path / "forensic.pt"

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        1,
        {"loss": 0.5, "accuracy": 0.5, "auc": 0.5},
        0.5,
        argparse.Namespace(stage="forensic"),
    )

    restored = forensic_from_checkpoint(checkpoint, torch.device("cpu"))
    assert restored.dropout == 0.0
    assert restored.srm.clip_value is None
