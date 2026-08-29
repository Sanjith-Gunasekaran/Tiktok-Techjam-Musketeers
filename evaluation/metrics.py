"""Metrics and error analysis for the AI-image detector."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def _validate_inputs(
    labels: Iterable[int],
    scores: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert labels and scores into validated one-dimensional arrays."""
    label_array = np.asarray(list(labels))
    score_array = np.asarray(list(scores), dtype=float)

    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if len(label_array) == 0:
        raise ValueError("labels and scores cannot be empty")
    if len(label_array) != len(score_array):
        raise ValueError("labels and scores must have the same length")
    if not np.all(np.isin(label_array, [0, 1])):
        raise ValueError("labels must contain only 0 (real) or 1 (AI)")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite numbers")

    return label_array.astype(int), score_array


def roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Calculate binary ROC-AUC, correctly handling tied scores."""
    label_array, score_array = _validate_inputs(labels, scores)

    positive_count = int(np.sum(label_array == 1))
    negative_count = int(np.sum(label_array == 0))

    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both real and AI examples")

    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(len(score_array), dtype=float)

    start = 0
    while start < len(sorted_scores):
        end = start + 1

        while (
            end < len(sorted_scores)
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        average_rank = (start + end + 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(np.sum(ranks[label_array == 1]))

    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)

    return float(auc)


def classification_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, int | float]:
    """Calculate confusion counts and common classification metrics."""
    label_array, score_array = _validate_inputs(labels, scores)

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    predictions = (score_array >= threshold).astype(int)

    true_positive = int(np.sum((label_array == 1) & (predictions == 1)))
    true_negative = int(np.sum((label_array == 0) & (predictions == 0)))
    false_positive = int(np.sum((label_array == 0) & (predictions == 1)))
    false_negative = int(np.sum((label_array == 1) & (predictions == 0)))

    total = len(label_array)
    accuracy = (true_positive + true_negative) / total

    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall_denominator = true_positive + false_negative
    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 0.0
    )

    f1_denominator = precision + recall
    f1 = (
        2.0 * precision * recall / f1_denominator
        if f1_denominator
        else 0.0
    )

    return {
        "count": total,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_condition(
    condition: str,
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Create one row for the clean-or-transformed robustness table."""
    label_array, score_array = _validate_inputs(labels, scores)

    return {
        "condition": condition,
        "roc_auc": roc_auc(label_array, score_array),
        **classification_metrics(label_array, score_array, threshold),
    }


def find_errors(
    image_paths: Sequence[str],
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, list[dict[str, Any]]]:
    """Return the false-positive and false-negative image examples."""
    label_array, score_array = _validate_inputs(labels, scores)

    if len(image_paths) != len(label_array):
        raise ValueError(
            "image_paths, labels and scores must have the same length"
        )

    predictions = (score_array >= threshold).astype(int)
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []

    for image_path, label, score, prediction in zip(
        image_paths,
        label_array,
        score_array,
        predictions,
    ):
        example = {
            "image_path": str(image_path),
            "label": int(label),
            "pred": float(score),
            "predicted_label": int(prediction),
        }

        if label == 0 and prediction == 1:
            false_positives.append(example)
        elif label == 1 and prediction == 0:
            false_negatives.append(example)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def competition_score(clean_auc: float, robust_auc: float) -> float:
    """Calculate 0.5 × clean ROC-AUC + 0.5 × robust ROC-AUC."""
    if not 0.0 <= clean_auc <= 1.0:
        raise ValueError("clean_auc must be between 0 and 1")
    if not 0.0 <= robust_auc <= 1.0:
        raise ValueError("robust_auc must be between 0 and 1")

    return 0.5 * clean_auc + 0.5 * robust_auc
