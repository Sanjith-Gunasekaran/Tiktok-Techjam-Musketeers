"""Evaluate a two-branch model on the frozen transform grid."""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .augmentations import EVAL_GRID
from .torch_dataset import BranchViewDataset, DataLoaderBundle


@dataclass(frozen=True)
class EvaluationRow:
    transform: str
    samples: int
    real: int
    synthetic: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    accuracy_delta_vs_clean: float
    auc_delta_vs_clean: float


class EvaluationReport(NamedTuple):
    rows: tuple[EvaluationRow, ...]
    csv_path: Path
    markdown_path: Path
    error_path: Path
    predictions_path: Path | None


@dataclass(frozen=True)
class CheckedTransform:
    """Picklable adapter that validates each transform's output."""

    transform: Callable[[Image.Image], Image.Image]

    def __call__(self, image: Image.Image) -> Image.Image:
        transformed = self.transform(image)
        if not isinstance(transformed, Image.Image):
            raise TypeError("Evaluation transforms must return PIL images")
        return transformed


def binary_auc(labels: Any, scores: Any) -> float:
    """Return binary ROC AUC using average ranks for tied scores."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be equal-length 1D arrays")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUC requires both binary classes")

    # Rank every score low-to-high; scores tied for a place share the average
    # of the ranks they span (standard tie-breaking for the Mann-Whitney form
    # of AUC below).
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2
        start = end

    # AUC = P(a random positive outscores a random negative), computed from
    # rank sums (Mann-Whitney U) rather than a full pairwise comparison.
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    rank_sum = ranks[positives].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def evaluate_model(
    model: Any,
    loaders: DataLoaderBundle,
    *,
    output_dir: str | Path,
    threshold: float = 0.5,
    from_logits: bool = False,
    device: str | torch.device | None = None,
    transforms: Mapping[str, Any] = EVAL_GRID,
    save_predictions: bool = False,
) -> EvaluationReport:
    """Evaluate every transform and write summary and error reports.

    ``model(dino_batch, patch_batch)`` must return one synthetic-class score
    per image. Probabilities use the fixed ``threshold``; pass
    ``from_logits=True`` for logits. Select thresholds on validation, never
    here on the frozen test set.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if "clean" not in transforms:
        raise ValueError("transforms must include a clean cell")
    test_loader = loaders.test
    test_dataset = test_loader.dataset
    if not isinstance(test_dataset, BranchViewDataset):
        raise TypeError("loaders.test must contain a BranchViewDataset")
    if test_dataset.partition != "test" or test_dataset.augmentation is not None:
        raise ValueError("Evaluation requires the unaugmented frozen test dataset")
    if not test_loader.batch_size:
        raise ValueError("The test DataLoader must define batch_size")

    # "clean" goes first so every row's delta-vs-clean can be computed in one pass.
    conditions = [("clean", transforms["clean"])] + [
        item for item in transforms.items() if item[0] != "clean"
    ]
    target_device = _model_device(model) if device is None else torch.device(device)
    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()

    image_ids = _image_ids(test_dataset)
    metrics: list[tuple[str, np.ndarray, np.ndarray, tuple[str, ...]]] = []
    try:
        with torch.inference_mode():
            for name, transform in conditions:
                labels, scores = _evaluate_condition(
                    model,
                    test_loader,
                    test_dataset,
                    transform,
                    target_device,
                    from_logits,
                )
                metrics.append((name, labels, scores, image_ids))
    finally:
        if was_training is True:
            train_method = getattr(model, "train", None)
            if callable(train_method):
                train_method()

    rows = _metric_rows(metrics, threshold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "clean_vs_transformed.csv"
    markdown_path = output_dir / "clean_vs_transformed.md"
    error_path = output_dir / "error_analysis.csv"
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)
    _write_errors(metrics, threshold, error_path)
    predictions_path = None
    if save_predictions:
        predictions_path = output_dir / "predictions.jsonl.gz"
        _write_predictions(metrics, threshold, predictions_path)
    return EvaluationReport(
        tuple(rows), csv_path, markdown_path, error_path, predictions_path
    )


def _evaluate_condition(
    model: Any,
    base_loader: DataLoader,
    base_dataset: BranchViewDataset,
    transform: Any,
    device: torch.device,
    from_logits: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not callable(transform):
        raise TypeError("Every evaluation transform must be callable")
    dataset = BranchViewDataset(
        base_dataset.source,
        base_dataset.rows,
        augmentation=CheckedTransform(transform),
        partition="test",
        id_column=base_dataset.id_column,
    )
    loader = DataLoader(
        dataset,
        batch_size=base_loader.batch_size,
        shuffle=False,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
        worker_init_fn=base_loader.worker_init_fn,
    )

    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for dino, patch, label, _original_label in loader:
        output = model(dino.to(device), patch.to(device))
        batch_scores = _probabilities(output, len(label), from_logits)
        labels.append(label.detach().cpu().numpy().astype(np.int64))
        scores.append(batch_scores)
    if not labels:
        raise ValueError("Frozen test set is empty")
    return np.concatenate(labels), np.concatenate(scores)


def _probabilities(output: Any, batch_size: int, from_logits: bool) -> np.ndarray:
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    scores = np.asarray(output, dtype=np.float64)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.ndim != 1 or len(scores) != batch_size:
        raise ValueError("Model must return one score per image")
    if not np.isfinite(scores).all():
        raise ValueError("Model scores must be finite")
    if from_logits:
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -709, 709)))
    elif ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Probability scores must be between 0 and 1")
    return scores


def _metric_rows(
    metrics: list[tuple[str, np.ndarray, np.ndarray, tuple[str, ...]]],
    threshold: float,
) -> list[EvaluationRow]:
    measured = []
    for name, labels, scores, _image_ids in metrics:
        predictions = (scores >= threshold).astype(np.int64)
        true_positive = int(((labels == 1) & (predictions == 1)).sum())
        true_negative = int(((labels == 0) & (predictions == 0)).sum())
        false_positive = int(((labels == 0) & (predictions == 1)).sum())
        false_negative = int(((labels == 1) & (predictions == 0)).sum())
        accuracy = (true_positive + true_negative) / len(labels)
        precision = true_positive / (true_positive + false_positive) if (
            true_positive + false_positive
        ) else 0.0
        recall = true_positive / (true_positive + false_negative) if (
            true_positive + false_negative
        ) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        measured.append(
            (
                name,
                labels,
                true_positive,
                true_negative,
                false_positive,
                false_negative,
                accuracy,
                precision,
                recall,
                f1,
                binary_auc(labels, scores),
            )
        )
    clean_accuracy, clean_auc = measured[0][6], measured[0][10]
    return [
        EvaluationRow(
            transform=name,
            samples=len(labels),
            real=int((labels == 0).sum()),
            synthetic=int((labels == 1).sum()),
            true_positive=true_positive,
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            accuracy_delta_vs_clean=accuracy - clean_accuracy,
            auc_delta_vs_clean=auc - clean_auc,
        )
        for (
            name,
            labels,
            true_positive,
            true_negative,
            false_positive,
            false_negative,
            accuracy,
            precision,
            recall,
            f1,
            auc,
        ) in measured
    ]


def _image_ids(dataset: BranchViewDataset) -> tuple[str, ...]:
    column = dataset.id_column
    if column is not None and column in dataset.rows.column_names:
        return tuple(str(value) for value in dataset.rows[column])
    return tuple(f"index_{index:06d}" for index in range(len(dataset)))


def _error_records(
    metrics: list[tuple[str, np.ndarray, np.ndarray, tuple[str, ...]]],
    threshold: float,
):
    for name, labels, scores, image_ids in metrics:
        predictions = (scores >= threshold).astype(np.int64)
        for image_id, label, score, prediction in zip(
            image_ids, labels, scores, predictions
        ):
            if label == prediction:
                continue
            yield {
                "transform": name,
                "image_id": image_id,
                "label": int(label),
                "score": float(score),
                "predicted_label": int(prediction),
                "error_type": "false_positive" if prediction else "false_negative",
            }


def _write_errors(
    metrics: list[tuple[str, np.ndarray, np.ndarray, tuple[str, ...]]],
    threshold: float,
    path: Path,
) -> None:
    fields = (
        "transform",
        "image_id",
        "label",
        "score",
        "predicted_label",
        "error_type",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_error_records(metrics, threshold))


def _write_predictions(
    metrics: list[tuple[str, np.ndarray, np.ndarray, tuple[str, ...]]],
    threshold: float,
    path: Path,
) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for name, labels, scores, image_ids in metrics:
            predictions = (scores >= threshold).astype(np.int64)
            for image_id, label, score, prediction in zip(
                image_ids, labels, scores, predictions
            ):
                record = {
                    "transform": name,
                    "image_id": image_id,
                    "label": int(label),
                    "score": float(score),
                    "predicted_label": int(prediction),
                }
                handle.write(json.dumps(record) + "\n")


def _model_device(model: Any) -> torch.device:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _write_csv(rows: list[EvaluationRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_markdown(rows: list[EvaluationRow], path: Path) -> None:
    lines = [
        (
            "| Transform | N | TP | TN | FP | FN | Accuracy | Precision | "
            "Recall | F1 | AUC | Δ Accuracy | Δ AUC |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    lines.extend(
        f"| {row.transform} | {row.samples} | {row.true_positive} | "
        f"{row.true_negative} | {row.false_positive} | {row.false_negative} | "
        f"{row.accuracy:.4f} | {row.precision:.4f} | {row.recall:.4f} | "
        f"{row.f1:.4f} | {row.auc:.4f} | "
        f"{row.accuracy_delta_vs_clean:+.4f} | {row.auc_delta_vs_clean:+.4f} |"
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
