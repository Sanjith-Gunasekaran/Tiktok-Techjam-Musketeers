"""Train the repository's DINOv2 AIGC classifier.

Expected local layout::

    dataset/
      train/REAL/*.jpg
      train/FAKE/*.jpg
      validation/REAL/*.jpg  # optional; derived from train when absent
      validation/FAKE/*.jpg
      test/REAL/*.jpg        # optional
      test/FAKE/*.jpg
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoImageProcessor

from aigc_detector import DINO_MODEL_ID, DINOv2BinaryClassifier
from aigc_detector.augment import DINOProcessorTransform, RobustAugment
from data_loader import ImageDatasetLoader


HUMAN_LABEL = 0
AI_LABEL = 1


def train(
    source: str | Path,
    *,
    output_path: str | Path,
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: str = "test",
    human_class: str = "REAL",
    ai_class: str = "FAKE",
    validation_fraction: float = 0.10,
    model_id: str = DINO_MODEL_ID,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    backbone_learning_rate: float = 1e-5,
    unfreeze_last_blocks: int = 0,
    weight_decay: float = 1e-4,
    clean_probability: float = 0.20,
    device_name: str = "auto",
    num_workers: int = 0,
    seed: int = 42,
    max_train_images: int | None = None,
    max_validation_images: int | None = None,
    max_test_images: int | None = None,
) -> dict[str, Any]:
    """Train, select on validation macro-F1, and optionally evaluate on test."""
    _validate_arguments(
        epochs, patience, batch_size, validation_fraction, unfreeze_last_blocks
    )
    _seed_everything(seed)
    device = _resolve_device(device_name)
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_id)
    label_map = {human_class: HUMAN_LABEL, ai_class: AI_LABEL}
    train_transform = DINOProcessorTransform(
        processor, RobustAugment(clean_probability=clean_probability)
    )
    evaluation_transform = DINOProcessorTransform(processor)

    augmented_train_loader = _open_loader(
        source, train_split, label_map, train_transform
    )
    validation_is_available = _split_is_available(source, validation_split)
    if validation_split.lower() != "none" and validation_is_available:
        training_dataset: Dataset = augmented_train_loader
        validation_dataset: Dataset = _open_loader(
            source, validation_split, label_map, evaluation_transform
        )
    else:
        evaluation_train_loader = _open_loader(
            source, train_split, label_map, evaluation_transform
        )
        training_indices, validation_indices = _split_indices(
            len(augmented_train_loader), validation_fraction, seed
        )
        training_dataset = Subset(augmented_train_loader, training_indices)
        validation_dataset = Subset(evaluation_train_loader, validation_indices)
        print(
            f"No validation split used; reserved {len(validation_indices)} "
            "training images for validation"
        )

    test_dataset: Dataset | None = None
    if test_split.lower() != "none" and _split_is_available(source, test_split):
        test_dataset = _open_loader(
            source, test_split, label_map, evaluation_transform
        )

    training_dataset = _limit_dataset(
        training_dataset, max_train_images, seed
    )
    validation_dataset = _limit_dataset(
        validation_dataset, max_validation_images, seed + 1
    )
    if test_dataset is not None:
        test_dataset = _limit_dataset(test_dataset, max_test_images, seed + 2)

    generator = torch.Generator().manual_seed(seed)
    data_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": _collate_samples,
        "worker_init_fn": _seed_worker,
    }
    training_batches = DataLoader(
        training_dataset, shuffle=True, generator=generator, **data_loader_kwargs
    )
    validation_batches = DataLoader(
        validation_dataset, shuffle=False, **data_loader_kwargs
    )
    test_batches = (
        DataLoader(test_dataset, shuffle=False, **data_loader_kwargs)
        if test_dataset is not None
        else None
    )

    labels = np.asarray(_labels_for_dataset(training_dataset), dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(
            f"Both classes need training images; class counts are {class_counts.tolist()}"
        )
    class_weights = torch.tensor(
        len(labels) / (2 * class_counts), dtype=torch.float32, device=device
    )
    print(
        f"Training/validation/test images: {len(training_dataset)}/"
        f"{len(validation_dataset)}/{len(test_dataset) if test_dataset else 0}"
    )
    print(f"Training class counts [human, AI]: {class_counts.tolist()}")

    model = DINOv2BinaryClassifier(model_id=model_id).to(device)
    backbone_parameters = _unfreeze_last_blocks(model, unfreeze_last_blocks)
    parameter_groups = [
        {"params": list(model.classifier.parameters()), "lr": learning_rate}
    ]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": backbone_learning_rate}
        )
    optimizer = AdamW(parameter_groups, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        if unfreeze_last_blocks == 0:
            model.backbone.eval()
        training_losses: list[float] = []
        for pixel_values, batch_labels in training_batches:
            pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
            batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for group in parameter_groups for parameter in group["params"]),
                max_norm=1.0,
            )
            optimizer.step()
            training_losses.append(float(loss.item()))
        scheduler.step()

        validation = evaluate(model, validation_batches, criterion, device)
        training_loss = float(np.mean(training_losses))
        print(
            f"Epoch {epoch + 1}/{epochs} | train loss={training_loss:.4f} | "
            f"validation loss={validation['loss']:.4f} | "
            f"balanced accuracy={validation['balanced_accuracy']:.4f} | "
            f"macro F1={validation['macro_f1']:.4f}"
        )

        if validation["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_f1": best_validation_f1,
                    "class_to_idx": {"human": HUMAN_LABEL, "ai": AI_LABEL},
                    "source_class_to_idx": label_map,
                    "model_id": model_id,
                    "epoch": epoch + 1,
                },
                output_path,
            )
            print(f"Saved new best checkpoint to {output_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

    checkpoint = _safe_torch_load(output_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    result: dict[str, Any] = {
        "best_validation_f1": float(checkpoint["validation_f1"]),
        "validation": evaluate(model, validation_batches, criterion, device),
    }
    if test_batches is not None:
        test_result = evaluate(model, test_batches, criterion, device)
        result["test"] = test_result
        print("Test confusion matrix [human, AI]:")
        print(np.asarray(test_result["confusion_matrix"]))
        print(test_result["classification_report"])

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")
    return result


def evaluate(
    model: nn.Module,
    batches: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    true_labels: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for pixel_values, labels in batches:
            pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            logits = model(pixel_values)
            losses.append(float(criterion(logits, labels).item()))
            true_labels.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predictions)
        ),
        "macro_f1": float(f1_score(true_labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            true_labels, predictions, labels=[HUMAN_LABEL, AI_LABEL]
        ).tolist(),
        "classification_report": classification_report(
            true_labels,
            predictions,
            labels=[HUMAN_LABEL, AI_LABEL],
            target_names=["Human", "AI-generated"],
            digits=4,
            zero_division=0,
        ),
    }


def _open_loader(
    source: str | Path,
    split: str,
    label_map: dict[str, int],
    transform: Any,
) -> ImageDatasetLoader:
    return ImageDatasetLoader(
        source,
        split=split,
        streaming=False,
        label_map=label_map,
        transform=transform,
    )


def _split_is_available(source: str | Path, split: str) -> bool:
    if split.lower() == "none":
        return False
    source_path = Path(str(source)).expanduser()
    if source_path.exists():
        return (source_path / split).is_dir() or bool(
            list(source_path.rglob(f"{split}-*.parquet"))
        )
    # Remote datasets are expected to publish the requested split; loading it
    # will provide the authoritative error if they do not.
    return True


def _split_indices(
    size: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    validation_size = max(1, round(size * validation_fraction))
    if validation_size >= size:
        raise ValueError("validation split leaves no training images")
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[validation_size:], indices[:validation_size]


def _labels_for_dataset(dataset: Dataset) -> list[int]:
    return [_label_at(dataset, index) for index in range(len(dataset))]


def _label_at(dataset: Dataset, index: int) -> int:
    if isinstance(dataset, Subset):
        return _label_at(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ImageDatasetLoader):
        return int(dataset.get_label(index))
    raise TypeError("Unsupported training dataset")


def _limit_dataset(
    dataset: Dataset, maximum: int | None, seed: int
) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum <= 1:
        raise ValueError("A limited dataset must contain at least two images")

    labels = _labels_for_dataset(dataset)
    indices_by_label = {
        label: [index for index, value in enumerate(labels) if value == label]
        for label in (HUMAN_LABEL, AI_LABEL)
    }
    rng = random.Random(seed)
    selected: list[int] = []
    per_class = maximum // 2
    for label in (HUMAN_LABEL, AI_LABEL):
        available = indices_by_label[label]
        selected.extend(rng.sample(available, min(per_class, len(available))))

    remaining = maximum - len(selected)
    if remaining:
        selected_set = set(selected)
        unused = [
            index for index in range(len(dataset)) if index not in selected_set
        ]
        selected.extend(rng.sample(unused, min(remaining, len(unused))))
    rng.shuffle(selected)
    print(f"Limited a split from {len(dataset)} to {len(selected)} images")
    return Subset(dataset, selected)


def _collate_samples(samples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = torch.stack([sample["image"] for sample in samples])
    labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
    return pixel_values, labels


def _unfreeze_last_blocks(
    model: DINOv2BinaryClassifier, count: int
) -> list[nn.Parameter]:
    blocks = model.backbone.encoder.layer
    if count > len(blocks):
        raise ValueError(f"Cannot unfreeze {count} blocks; model has {len(blocks)}")
    parameters: list[nn.Parameter] = []
    for block in blocks[-count:] if count else []:
        for parameter in block.parameters():
            parameter.requires_grad = True
            parameters.append(parameter)
    return parameters


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_arguments(
    epochs: int,
    patience: int,
    batch_size: int,
    validation_fraction: float,
    unfreeze_last_blocks: int,
) -> None:
    if epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("epochs, patience, and batch_size must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if unfreeze_last_blocks < 0:
        raise ValueError("unfreeze_last_blocks cannot be negative")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local dataset root, HF ID, or kaggle:// URI")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/best_dinov2_model.pt"),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--human-class", default="REAL")
    parser.add_argument("--ai-class", default="FAKE")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--model-id", default=DINO_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clean-probability", type=float, default=0.20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-validation-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    args = parser.parse_args()

    train(
        args.source,
        output_path=args.output,
        train_split=args.train_split,
        validation_split=args.validation_split,
        test_split=args.test_split,
        human_class=args.human_class,
        ai_class=args.ai_class,
        validation_fraction=args.validation_fraction,
        model_id=args.model_id,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        weight_decay=args.weight_decay,
        clean_probability=args.clean_probability,
        device_name=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_images=args.max_train_images,
        max_validation_images=args.max_validation_images,
        max_test_images=args.max_test_images,
    )


if __name__ == "__main__":
    main()
