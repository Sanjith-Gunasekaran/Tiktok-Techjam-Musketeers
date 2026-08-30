"""Tests for the stage-aware training helpers."""

import torch
from torch import nn

from model_runs.train import run_epoch


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
