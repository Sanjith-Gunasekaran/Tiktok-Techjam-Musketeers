"""Train one detector stage without touching the frozen test loader."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import DINOClassifier, ForensicCNN, TwoBranchDetector
from pipeline import binary_auc, create_dataloaders

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor]
Metrics = dict[str, float | None]
STAGES = ("dino_head", "dino_finetune", "forensic", "fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--data-dir", default="data")
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
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--augmentation-probability", type=float, default=0.5)
    parser.add_argument("--second-augmentation-probability", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--srm-clip-value", type=float, default=3.0)
    parser.add_argument("--fusion-mode", choices=("learned",), default="learned")
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def forward_stage(
    model: nn.Module, stage: str, dino: torch.Tensor | None, patch: torch.Tensor | None
) -> torch.Tensor:
    if stage.startswith("dino"):
        if dino is None:
            raise ValueError("DINO stage requires DINO inputs")
        return model(dino)
    if stage == "forensic":
        if patch is None:
            raise ValueError("Forensic stage requires patch inputs")
        return model(patch)
    if dino is None or patch is None:
        raise ValueError("Fusion stage requires both inputs")
    return model(dino, patch)


def unpack_batch(batch: Batch) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Accept the legacy two-view tuple and single-view dictionary batches."""
    if isinstance(batch, Mapping):
        label = batch.get("label")
        dino, patch = batch.get("dino"), batch.get("patch")
        if not isinstance(label, torch.Tensor):
            raise TypeError("Dictionary batch requires a tensor label")
        return dino, patch, label
    dino, patch, label, _ = batch
    return dino, patch, label


def _metrics(losses: list[float], labels: list[float], logits: list[float]) -> Metrics:
    if not labels:
        raise ValueError("Cannot run an epoch with an empty DataLoader")
    labels_np, logits_np = np.asarray(labels), np.asarray(logits)
    auc = binary_auc(labels_np, logits_np) if set(labels_np) == {0, 1} else None
    return {
        "loss": sum(losses) / len(labels_np),
        "accuracy": float(((logits_np >= 0) == labels_np).mean()),
        "auc": auc,
    }


def _optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip_norm: float | None,
    parameters: Iterable[nn.Parameter],
) -> None:
    if scaler is None:
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
        optimizer.step()
        return
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()


def run_epoch(
    model: nn.Module,
    stage: str,
    batches: Iterable[Batch],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip_norm: float | None = 1.0,
    amp: bool = False,
) -> Metrics:
    """Run one branch-training or validation epoch."""
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels_all: list[float] = []
    logits_all: list[float] = []
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            dino, patch, labels = unpack_batch(batch)
            labels = labels.to(device, torch.float32, non_blocking=True)
            dino = dino.to(device, non_blocking=True) if dino is not None else None
            patch = patch.to(device, non_blocking=True) if patch is not None else None
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                logits = forward_stage(model, stage, dino, patch)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if training:
                _optimizer_step(loss, optimizer, scaler, grad_clip_norm, model.parameters())
            losses.append(loss.detach().float().item() * len(labels))
            labels_all.extend(labels.detach().cpu().tolist())
            logits_all.extend(logits.detach().float().cpu().tolist())
    return _metrics(losses, labels_all, logits_all)


@torch.inference_mode()
def cache_fusion_logits(
    model: TwoBranchDetector,
    batches: Iterable[Batch],
    device: torch.device,
    max_batches: int | None,
    *,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen branches once over validation and keep their two logits."""
    model.eval()
    scores, labels_all = [], []
    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        dino, patch, labels = unpack_batch(batch)
        if dino is None or patch is None:
            raise ValueError("Fusion calibration requires both branch views")
        with _autocast(device, amp):
            dino_logits, forensic_logits = model.branch_logits(
                dino.to(device, non_blocking=True), patch.to(device, non_blocking=True)
            )
        scores.append(torch.stack((dino_logits.float().cpu(), forensic_logits.float().cpu()), dim=1))
        labels_all.append(labels.to(torch.float32).cpu())
    if not scores:
        raise ValueError("Cannot fit fusion with an empty validation DataLoader")
    return torch.cat(scores), torch.cat(labels_all)


def require_binary_labels(labels: torch.Tensor, partition: str) -> None:
    if set(labels.tolist()) != {0.0, 1.0}:
        raise RuntimeError(f"Fusion {partition} partition must contain both classes")


def run_fusion_epoch(
    model: TwoBranchDetector,
    scores: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip_norm: float | None = 1.0,
    amp: bool = False,
) -> Metrics:
    """Fit or score the tiny fusion layer from cached validation logits."""
    if model.fusion is None:
        raise ValueError("Cached fusion requires learned fusion")
    training = optimizer is not None
    model.train(training)
    indices = torch.randperm(len(labels)) if training else torch.arange(len(labels))
    losses: list[float] = []
    labels_all: list[float] = []
    logits_all: list[float] = []
    with torch.set_grad_enabled(training):
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            features = scores[batch_indices].to(device, non_blocking=True)
            target = labels[batch_indices].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                logits = model.fusion(features).squeeze(1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if training:
                _optimizer_step(loss, optimizer, scaler, grad_clip_norm, model.fusion.parameters())
            losses.append(loss.detach().float().item() * len(target))
            labels_all.extend(target.detach().cpu().tolist())
            logits_all.extend(logits.detach().float().cpu().tolist())
    return _metrics(losses, labels_all, logits_all)


def _checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"{path} is not a supported training checkpoint")
    return checkpoint


def _verify_model_config(model: nn.Module, checkpoint: dict[str, Any], path: Path) -> None:
    saved = checkpoint.get("model_config")
    expected = getattr(model, "checkpoint_config", lambda: {})()
    if not isinstance(saved, dict) or not isinstance(expected, dict):
        return
    keys = ("model_name", "revision", "hidden_dim", "dropout", "model_type", "srm_clip_value")
    for key in keys:
        if key in saved and key in expected and saved[key] != expected[key]:
            raise ValueError(f"{path} has {key}={saved[key]!r}, expected {expected[key]!r}")


def load_state(
    model: nn.Module,
    path: Path,
    device: torch.device,
    expected_stages: tuple[str, ...] = STAGES,
) -> dict[str, Any]:
    checkpoint = _checkpoint(path, device)
    if checkpoint.get("stage") not in expected_stages:
        allowed = ", ".join(expected_stages)
        raise ValueError(f"{path} is a {checkpoint.get('stage')!r} checkpoint; expected {allowed}")
    _verify_model_config(model, checkpoint, path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def dino_from_checkpoint(
    path: Path, device: torch.device, expected_stages: tuple[str, ...]
) -> DINOClassifier:
    """Rebuild DINO from the checkpoint's head configuration before loading."""
    checkpoint = _checkpoint(path, device)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError(f"{path} has no DINO model configuration")
    dino = DINOClassifier(
        model_name=config.get("model_name", "facebook/dinov2-small"),
        revision=config.get("revision"),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.2),
        freeze_backbone=True,
    )
    load_state(dino, path, device, expected_stages)
    return dino


def forensic_from_checkpoint(path: Path, device: torch.device) -> ForensicCNN:
    """Rebuild the forensic model with its saved SRM clipping configuration."""
    checkpoint = _checkpoint(path, device)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or config.get("model_type") != "forensic_cnn":
        raise ValueError(f"{path} has no forensic model configuration")
    forensic = ForensicCNN(
        dropout=config.get("dropout", 0.2), srm_clip_value=config.get("srm_clip_value")
    )
    load_state(forensic, path, device, ("forensic",))
    return forensic


def display_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, list[dict[str, Any]]]:
    if args.stage == "dino_head":
        dino = DINOClassifier(freeze_backbone=True)
        return dino, [{"params": dino.classifier.parameters(), "lr": args.head_lr}]
    if args.stage == "dino_finetune":
        if args.dino_checkpoint is None:
            raise ValueError("--dino-checkpoint is required for dino_finetune")
        dino = dino_from_checkpoint(args.dino_checkpoint, device, ("dino_head", "dino_finetune"))
        dino.unfreeze_last_blocks(args.unfreeze_blocks)
        backbone = [parameter for parameter in dino.backbone.parameters() if parameter.requires_grad]
        return dino, [
            {"params": dino.classifier.parameters(), "lr": args.head_lr},
            {"params": backbone, "lr": args.backbone_lr},
        ]
    if args.stage == "forensic":
        forensic = ForensicCNN(srm_clip_value=args.srm_clip_value)
        return forensic, [{"params": forensic.parameters(), "lr": args.forensic_lr}]
    if args.dino_checkpoint is None or args.forensic_checkpoint is None:
        raise ValueError("--dino-checkpoint and --forensic-checkpoint are required for fusion")
    dino = dino_from_checkpoint(args.dino_checkpoint, device, ("dino_head", "dino_finetune"))
    forensic = forensic_from_checkpoint(args.forensic_checkpoint, device)
    model = TwoBranchDetector(dino, forensic, fusion_mode="learned", dino_weight=args.dino_weight)
    model.freeze_branches()
    return model, [{"params": model.fusion.parameters(), "lr": args.fusion_lr}]


def parameter_groups(
    model: nn.Module, groups: list[dict[str, Any]], weight_decay: float
) -> list[dict[str, Any]]:
    """Exclude bias and one-dimensional normalization parameters from decay."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result: list[dict[str, Any]] = []
    for group in groups:
        decay, no_decay = [], []
        for parameter in group["params"]:
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.ndim == 1 or names[id(parameter)].endswith(".bias") else decay).append(parameter)
        options = {key: value for key, value in group.items() if key != "params"}
        if decay:
            result.append({**options, "params": decay, "weight_decay": weight_decay})
        if no_decay:
            result.append({**options, "params": no_decay, "weight_decay": 0.0})
    if not result:
        raise ValueError("No trainable parameters for this stage")
    return result


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.no_scheduler:
        return None
    warmup = args.warmup_epochs
    duration = max(1, args.epochs - warmup)

    def scale(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, max(0.0, (epoch - warmup) / duration))
        return args.min_lr_scale + (1.0 - args.min_lr_scale) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _rng_state(train_loader: Iterable[Batch]) -> dict[str, Any]:
    generator = getattr(train_loader, "generator", None)
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if generator is not None:
        state["loader"] = generator.get_state()
    augmentation = getattr(getattr(train_loader, "dataset", None), "augmentation", None)
    if getattr(train_loader, "num_workers", 0) == 0 and augmentation is not None:
        state["augmentation"] = {
            "python": augmentation._rng.getstate(),
            "numpy": augmentation._np_rng.bit_generator.state,
        }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Any, train_loader: Iterable[Batch]) -> None:
    if not isinstance(state, dict):
        return
    if isinstance(state.get("torch"), torch.Tensor):
        torch.set_rng_state(state["torch"])
    if isinstance(state.get("python"), tuple):
        random.setstate(state["python"])
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, dict) and isinstance(numpy_state.get("state"), torch.Tensor):
        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["state"].cpu().numpy(),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
    generator = getattr(train_loader, "generator", None)
    if generator is not None and isinstance(state.get("loader"), torch.Tensor):
        generator.set_state(state["loader"])
    augmentation = getattr(getattr(train_loader, "dataset", None), "augmentation", None)
    augmentation_state = state.get("augmentation")
    if augmentation is not None and isinstance(augmentation_state, dict):
        augmentation._rng.setstate(augmentation_state["python"])
        augmentation._np_rng.bit_generator.state = augmentation_state["numpy"]
    if torch.cuda.is_available() and isinstance(state.get("cuda"), list):
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Metrics,
    best_auc: float,
    args: argparse.Namespace,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    train_loader: Iterable[Batch] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "stage": args.stage,
            "best_auc": best_auc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "rng_state": _rng_state(train_loader),
            "metrics": metrics,
            "args": config,
            "model_config": getattr(model, "checkpoint_config", lambda: {})(),
        },
        path,
    )


def write_history(
    path: Path,
    epoch: int,
    train: Metrics,
    validation: Metrics,
    optimizer: torch.optim.Optimizer,
    duration_seconds: float,
) -> None:
    row = {
        "epoch": epoch,
        "online_train": train,
        "validation": validation,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "duration_seconds": duration_seconds,
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or min(args.head_lr, args.backbone_lr, args.forensic_lr, args.fusion_lr) <= 0:
        raise ValueError("epochs and learning rates must be positive")
    if args.warmup_epochs < 0 or not 0.0 <= args.min_lr_scale <= 1.0:
        raise ValueError("warmup epochs must be non-negative and min LR scale must be in [0, 1]")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0:
        raise ValueError("grad clip norm must be positive")
    if any(value is not None and value <= 0 for value in (args.max_train_batches, args.max_validation_batches)):
        raise ValueError("max batch counts must be positive")
    if args.no_augment:
        args.augmentation_probability = args.second_augmentation_probability = 0.0
    seed_everything(args.seed)
    device = choose_device()
    amp = device.type == "cuda" and not args.no_amp
    loaders = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        augmentation_probability=args.augmentation_probability,
        second_augmentation_probability=args.second_augmentation_probability,
        view="forensic" if args.stage == "forensic" else "dino" if args.stage.startswith("dino") else "both",
    )
    model, groups = build_model(args, device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, groups, args.weight_decay))
    scheduler = make_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if amp else None
    start_epoch, best_auc = 1, float("-inf")
    if args.resume:
        checkpoint = load_state(model, args.resume, device, (args.stage,))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"), loaders.train)
        start_epoch = checkpoint["epoch"] + 1
        best_auc = checkpoint.get("best_auc", checkpoint["metrics"].get("auc", float("-inf")))
    checkpoint_dir = args.output_dir / args.stage
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = checkpoint_dir / "history.jsonl"
    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}; --epochs is {args.epochs}.")
        return

    cached_fusion: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None
    if args.stage == "fusion":
        calibration = cache_fusion_logits(
            model, loaders.calibration, device, args.max_validation_batches, amp=amp
        )
        selection = cache_fusion_logits(
            model, loaders.validation, device, args.max_validation_batches, amp=amp
        )
        require_binary_labels(calibration[1], "calibration")
        require_binary_labels(selection[1], "selection")
        cached_fusion = calibration, selection

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        if cached_fusion is None:
            train_metrics = run_epoch(model, args.stage, loaders.train, device, optimizer, args.max_train_batches, scaler=scaler, grad_clip_norm=args.grad_clip_norm, amp=amp)
            validation = run_epoch(model, args.stage, loaders.validation, device, max_batches=args.max_validation_batches, amp=amp)
        else:
            calibration, selection = cached_fusion
            train_metrics = run_fusion_epoch(model, *calibration, device, args.batch_size, optimizer, scaler=scaler, grad_clip_norm=args.grad_clip_norm, amp=amp)
            validation = run_fusion_epoch(model, *selection, device, args.batch_size, amp=amp)
        if validation["auc"] is None:
            raise RuntimeError("Validation contains only one class; increase --max-validation-batches or inspect the split.")
        duration = time.perf_counter() - epoch_start
        learning_rates = ", ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        print(
            f"epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} | "
            f"online train auc {display_metric(train_metrics['auc'])} | "
            f"validation loss {validation['loss']:.4f} | "
            f"validation accuracy {validation['accuracy']:.4f} | "
            f"validation auc {display_metric(validation['auc'])} | "
            f"lr {learning_rates} | {duration:.1f}s"
        )
        improved = validation["auc"] > best_auc
        if improved:
            best_auc = validation["auc"]
        if scheduler is not None:
            scheduler.step()
        write_history(history, epoch, train_metrics, validation, optimizer, duration)
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, epoch, validation, best_auc, args,
            scheduler=scheduler, scaler=scaler, train_loader=loaders.train,
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch, validation, best_auc, args,
                scheduler=scheduler, scaler=scaler, train_loader=loaders.train,
            )


if __name__ == "__main__":
    main()
