"""Train the detector heads while keeping the DINOv2 backbone frozen."""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models import DINOClassifier, ForensicCNN, TwoBranchDetector
from pipeline import create_dataloaders


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("model_runs/best_model.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--dino-weight", type=float, default=0.8)
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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: Batch, device: torch.device) -> tuple[torch.Tensor, ...]:
    dino_images, patches, labels, _ = batch
    return (
        dino_images.to(device, non_blocking=True),
        patches.to(device, non_blocking=True),
        labels.to(device, dtype=torch.float32, non_blocking=True),
    )


def run_epoch(
    model: TwoBranchDetector,
    batches: Iterable[Batch],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for batch in batches:
            dino_images, patches, labels = move_batch(batch, device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(dino_images, patches)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total_correct += ((logits >= 0) == labels.bool()).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty DataLoader")
    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    path: Path,
    *,
    model: TwoBranchDetector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero")

    seed_everything(args.seed)
    device = choose_device()
    print(f"device: {device}")

    loaders = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    dino = DINOClassifier(freeze_backbone=True)
    model = TwoBranchDetector(
        dino=dino,
        forensic=ForensicCNN(),
        dino_weight=args.dino_weight,
    ).to(device)

    # Only the DINO classification head and forensic branch enter the optimizer.
    assert all(not parameter.requires_grad for parameter in dino.backbone.parameters())
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()

    best_validation_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders.train, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, loaders.validation, loss_function, device
        )

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.2%} | "
            f"validation loss {validation_loss:.4f}, "
            f"accuracy {validation_accuracy:.2%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                args.output,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                args=args,
            )
            print(f"saved best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
