"""Select fixed DINO/forensic fusion on held-out development partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import TwoBranchDetector
from pipeline import binary_auc, create_dataloaders

from .train import (
    cache_fusion_logits,
    choose_device,
    dino_from_checkpoint,
    forensic_from_checkpoint,
    require_binary_labels,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--forensic-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("model_runs/checkpoints/fusion/fixed.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--weight-steps", type=int, default=101)
    return parser.parse_args()


def fused_logits(scores: torch.Tensor, dino_weight: float) -> torch.Tensor:
    return dino_weight * scores[:, 0] + (1.0 - dino_weight) * scores[:, 1]


def select_weight(scores: torch.Tensor, labels: torch.Tensor, steps: int) -> float:
    """Choose the validation-AUC-maximising DINO weight from an even grid."""
    if steps < 2:
        raise ValueError("weight steps must be at least 2")
    require_binary_labels(labels, "selection")
    weights = torch.linspace(0.0, 1.0, steps).tolist()
    ranked = [
        (binary_auc(labels.tolist(), fused_logits(scores, weight).tolist()), -abs(weight - 0.8), weight)
        for weight in weights
    ]
    return max(ranked)[2]


def select_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calibrate an accuracy-maximising probability threshold with stable ties."""
    require_binary_labels(labels, "calibration")
    probabilities = torch.sigmoid(logits)
    candidates = torch.unique(torch.cat((torch.tensor([0.0, 0.5, 1.0]), probabilities))).tolist()
    ranked = [
        (((probabilities >= threshold).to(labels.dtype) == labels).float().mean().item(), -abs(threshold - 0.5), threshold)
        for threshold in candidates
    ]
    return max(ranked)[2]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    seed_everything(args.seed)
    device = choose_device()
    loaders = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    dino = dino_from_checkpoint(args.dino_checkpoint, device, ("dino_head", "dino_finetune"))
    forensic = forensic_from_checkpoint(args.forensic_checkpoint, device)
    model = TwoBranchDetector(dino, forensic, fusion_mode="fixed").to(device)
    model.freeze_branches()
    calibration_scores, calibration_labels = cache_fusion_logits(model, loaders.calibration, device, None, amp=False)
    selection_scores, selection_labels = cache_fusion_logits(model, loaders.validation, device, None, amp=False)
    require_binary_labels(calibration_labels, "calibration")
    weight = select_weight(selection_scores, selection_labels, args.weight_steps)
    threshold = select_threshold(fused_logits(calibration_scores, weight), calibration_labels)
    selection_logits = fused_logits(selection_scores, weight)
    report = {
        "format_version": 1,
        "dino_checkpoint": str(args.dino_checkpoint),
        "forensic_checkpoint": str(args.forensic_checkpoint),
        "dino_weight": weight,
        "threshold": threshold,
        "selection_auc": binary_auc(selection_labels.tolist(), selection_logits.tolist()),
        "calibration_accuracy": float(
            ((torch.sigmoid(fused_logits(calibration_scores, weight)) >= threshold) == calibration_labels).float().mean()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output} | dino weight {weight:.2f} | threshold {threshold:.3f}")


if __name__ == "__main__":
    main()
