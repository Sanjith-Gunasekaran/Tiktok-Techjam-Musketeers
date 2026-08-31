"""Aggregate clean and transformed evaluation rows into final scores."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OverallScores:
    """Clean, robust and equally weighted combined ROC-AUC scores."""

    clean_auc: float
    robust_auc: float
    combined_auc: float
    transformed_conditions: tuple[str, ...]


def _row_value(row: Any, field: str) -> Any:
    if isinstance(row, Mapping):
        if field not in row:
            raise ValueError(f"evaluation row is missing {field!r}")
        return row[field]

    if not hasattr(row, field):
        raise ValueError(f"evaluation row is missing {field!r}")
    return getattr(row, field)


def _validated_auc(value: Any, transform: str) -> float:
    try:
        auc = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{transform!r} has an invalid AUC: {value!r}"
        ) from error

    if not math.isfinite(auc) or not 0.0 <= auc <= 1.0:
        raise ValueError(
            f"{transform!r} AUC must be finite and between 0 and 1"
        )
    return auc


def calculate_overall_scores(
    rows: Sequence[Any],
    *,
    exclude_transforms: Collection[str] = (),
) -> OverallScores:
    """Calculate clean, mean-transformed and 50/50 combined ROC-AUC.

    ``rows`` may contain dictionaries or ``EvaluationRow`` objects.
    Experimental conditions can be omitted from the robust mean with
    ``exclude_transforms``.
    """
    if not rows:
        raise ValueError("evaluation rows cannot be empty")
    if isinstance(exclude_transforms, str):
        raise TypeError(
            "exclude_transforms must be a collection of names"
        )

    excluded = set(exclude_transforms)
    named_rows: list[tuple[str, float]] = []

    for row in rows:
        transform = str(_row_value(row, "transform"))
        auc = _validated_auc(_row_value(row, "auc"), transform)
        named_rows.append((transform, auc))

    clean_rows = [
        auc for transform, auc in named_rows
        if transform == "clean"
    ]
    if len(clean_rows) != 1:
        raise ValueError(
            "evaluation rows must contain exactly one clean condition"
        )

    transformed_rows = [
        (transform, auc)
        for transform, auc in named_rows
        if transform != "clean" and transform not in excluded
    ]
    if not transformed_rows:
        raise ValueError(
            "at least one transformed condition is required"
        )

    clean_auc = clean_rows[0]
    robust_auc = sum(
        auc for _, auc in transformed_rows
    ) / len(transformed_rows)

    return OverallScores(
        clean_auc=clean_auc,
        robust_auc=robust_auc,
        combined_auc=0.5 * clean_auc + 0.5 * robust_auc,
        transformed_conditions=tuple(
            transform for transform, _ in transformed_rows
        ),
    )


def write_overall_summary(
    rows: Sequence[Any],
    output_path: str | Path,
    *,
    exclude_transforms: Collection[str] = (),
) -> Path:
    """Write aggregate scores as readable UTF-8 JSON."""
    scores = calculate_overall_scores(
        rows,
        exclude_transforms=exclude_transforms,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            asdict(scores),
            output_file,
            indent=2,
            ensure_ascii=False,
        )
        output_file.write("\n")

    return output_path