"""Export robustness results as submission-ready CSV and JSON files."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import competition_score


SUMMARY_FIELDS = (
    "condition",
    "count",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)


def calculate_overall_scores(
    summary: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Calculate clean, mean transformed and combined competition scores."""
    if not summary:
        raise ValueError("summary cannot be empty")

    clean_rows = [
        row for row in summary
        if row.get("condition") == "clean"
    ]

    if len(clean_rows) != 1:
        raise ValueError("summary must contain exactly one clean condition")

    transformed_rows = [
        row for row in summary
        if row.get("condition") != "clean"
    ]

    if not transformed_rows:
        raise ValueError(
            "summary must contain at least one transformed condition"
        )

    clean_auc = float(clean_rows[0]["roc_auc"])
    transformed_aucs = [
        float(row["roc_auc"])
        for row in transformed_rows
    ]
    robust_auc = sum(transformed_aucs) / len(transformed_aucs)

    return {
        "clean_auc": clean_auc,
        "robust_auc": robust_auc,
        "competition_score": competition_score(
            clean_auc,
            robust_auc,
        ),
    }


def write_summary_csv(
    summary: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write the clean-versus-transformed robustness table."""
    if not summary:
        raise ValueError("summary cannot be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in summary:
            writer.writerow(row)

    return output_path


def write_json(
    value: Any,
    output_path: str | Path,
) -> Path:
    """Write a value as readable UTF-8 JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            value,
            json_file,
            indent=2,
            ensure_ascii=False,
        )
        json_file.write("\n")

    return output_path


def save_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save the robustness table, predictions and FP/FN analysis."""
    required_keys = {"summary", "predictions", "errors"}
    missing_keys = required_keys.difference(report)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"report is missing required keys: {missing}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = report["summary"]
    predictions = report["predictions"]
    errors = report["errors"]
    overall = calculate_overall_scores(summary)

    summary_path = write_summary_csv(
        summary,
        output_dir / "robustness_summary.csv",
    )
    predictions_path = write_json(
        predictions,
        output_dir / "robustness_predictions.json",
    )
    errors_path = write_json(
        errors,
        output_dir / "error_analysis.json",
    )
    report_path = write_json(
        {
            "overall": overall,
            "summary": summary,
            "errors": errors,
        },
        output_dir / "robustness_report.json",
    )

    return {
        "summary": summary_path,
        "predictions": predictions_path,
        "errors": errors_path,
        "report": report_path,
    }