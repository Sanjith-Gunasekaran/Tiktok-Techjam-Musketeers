"""Train one detector stage without touching the frozen test loader.

Run as ``python -m model_runs.train --stage dino_head``.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import DINOClassifier, ForensicCNN, TwoBranchDetector
from pipeline import binary_auc, create_dataloaders

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
STAGES = ("dino_head", "dino_finetune", "forensic", "fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_runs/checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--forensic-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--forensic-lr", type=float, default=1e-3)
    parser.add_argument("--fusion-lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--augmentation-probability", type=float, default=0.0)
    parser.add_argument("--second-augmentation-probability", type=float, default=0.0)
    parser.add_argument("--srm-clip-value", type=float, default=3.0)
    parser.add_argument("--fusion-mode", choices=("fixed", "learned"), default="learned")
    parser.add_argument("--dino-weight", type=float, default=0.5)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def forward_stage(model: nn.Module, stage: str, dino: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
    if stage.startswith("dino"):
        return model(dino)
    if stage == "forensic":
        return model(patch)
    return model(dino, patch)


def run_epoch(model: nn.Module, stage: str, batches: Iterable[Batch], device: torch.device, optimizer: torch.optim.Optimizer | None = None, max_batches: int | None = None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses, labels_all, logits_all = [], [], []
    with torch.set_grad_enabled(training):
        for batch_index, (dino, patch, labels, _) in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            labels = labels.to(device, torch.float32)
            dino = dino.to(device) if stage != "forensic" else None
            patch = patch.to(device) if not stage.startswith("dino") else None
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = forward_stage(model, stage, dino, patch)  # type: ignore[arg-type]
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.detach().item() * len(labels))
            labels_all.extend(labels.detach().cpu().tolist())
            logits_all.extend(logits.detach().cpu().tolist())
    if not labels_all:
        raise ValueError("Cannot run an epoch with an empty DataLoader")
    labels_np, logits_np = np.asarray(labels_all), np.asarray(logits_all)
    predicted = logits_np >= 0
    auc = binary_auc(labels_np, logits_np) if set(labels_np) == {0, 1} else float("nan")
    return {"loss": sum(losses) / len(labels_np), "accuracy": float((predicted == labels_np).mean()), "auc": auc}


def load_state(model: nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, list[dict[str, Any]]]:
    if args.stage == "dino_head":
        dino = DINOClassifier(freeze_backbone=True)
        return dino, [{"params": dino.classifier.parameters(), "lr": args.head_lr}]
    if args.stage == "dino_finetune":
        dino = DINOClassifier(freeze_backbone=True)
        if args.dino_checkpoint is None:
            raise ValueError("--dino-checkpoint is required for dino_finetune")
        load_state(dino, args.dino_checkpoint, device)
        dino.unfreeze_last_blocks(args.unfreeze_blocks)
        backbone = [p for p in dino.backbone.parameters() if p.requires_grad]
        return dino, [{"params": dino.classifier.parameters(), "lr": args.head_lr}, {"params": backbone, "lr": args.backbone_lr}]
    if args.stage == "forensic":
        forensic = ForensicCNN(srm_clip_value=args.srm_clip_value)
        return forensic, [{"params": forensic.parameters(), "lr": args.forensic_lr}]
    if args.dino_checkpoint is None or args.forensic_checkpoint is None:
        raise ValueError("--dino-checkpoint and --forensic-checkpoint are required for fusion")
    dino = DINOClassifier(freeze_backbone=True)
    forensic = ForensicCNN(srm_clip_value=args.srm_clip_value)
    load_state(dino, args.dino_checkpoint, device)
    load_state(forensic, args.forensic_checkpoint, device)
    model = TwoBranchDetector(dino, forensic, fusion_mode=args.fusion_mode, dino_weight=args.dino_weight)
    model.freeze_branches()
    if args.fusion_mode == "fixed":
        raise ValueError("fixed fusion is validation-selected, not trainable")
    return model, [{"params": model.fusion.parameters(), "lr": args.fusion_lr}]


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float], best_auc: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "stage": args.stage, "best_auc": best_auc, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics, "args": vars(args), "model_config": getattr(model, "checkpoint_config", lambda: {})()}, path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or min(args.head_lr, args.backbone_lr, args.forensic_lr, args.fusion_lr) <= 0:
        raise ValueError("epochs and learning rates must be positive")
    if any(value is not None and value <= 0 for value in (args.max_train_batches, args.max_validation_batches)):
        raise ValueError("max batch counts must be positive")
    seed_everything(args.seed)
    device = choose_device()
    loaders = create_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed, pin_memory=device.type == "cuda", augmentation_probability=args.augmentation_probability, second_augmentation_probability=args.second_augmentation_probability)
    model, groups = build_model(args, device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    start_epoch, best_auc = 1, float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["stage"] != args.stage:
            raise ValueError("resume checkpoint stage does not match --stage")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch, best_auc = checkpoint["epoch"] + 1, checkpoint.get("best_auc", checkpoint["metrics"]["auc"])
    checkpoint_dir = args.output_dir / args.stage
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, args.stage, loaders.train, device, optimizer, args.max_train_batches)
        validation = run_epoch(model, args.stage, loaders.validation, device, max_batches=args.max_validation_batches)
        print(f"epoch {epoch:02d} | train auc {train_metrics['auc']:.4f} | validation auc {validation['auc']:.4f}")
        improved = validation["auc"] > best_auc
        if improved:
            best_auc = validation["auc"]
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, validation, best_auc, args)
        if improved:
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, validation, best_auc, args)


if __name__ == "__main__":
    main()
