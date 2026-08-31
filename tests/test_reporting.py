"""Tests for aggregate robustness evaluation reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pipeline.reporting import (
    calculate_overall_scores,
    write_overall_summary,
)


class ReportingTests(unittest.TestCase):
    def test_calculate_overall_scores(self) -> None:
        rows = [
            {"transform": "clean", "auc": 0.9},
            {"transform": "jpeg_q50", "auc": 0.7},
            {"transform": "blur_s20", "auc": 0.5},
        ]

        scores = calculate_overall_scores(rows)

        self.assertAlmostEqual(scores.clean_auc, 0.9)
        self.assertAlmostEqual(scores.robust_auc, 0.6)
        self.assertAlmostEqual(scores.combined_auc, 0.75)
        self.assertEqual(
            scores.transformed_conditions,
            ("jpeg_q50", "blur_s20"),
        )

    def test_attribute_rows_and_exclusions_are_supported(self) -> None:
        rows = [
            SimpleNamespace(transform="clean", auc=0.8),
            SimpleNamespace(transform="jpeg_q30", auc=0.6),
            SimpleNamespace(
                transform="chain_crop_resize_jpeg",
                auc=0.2,
            ),
        ]

        scores = calculate_overall_scores(
            rows,
            exclude_transforms={"chain_crop_resize_jpeg"},
        )

        self.assertAlmostEqual(scores.robust_auc, 0.6)
        self.assertAlmostEqual(scores.combined_auc, 0.7)
        self.assertEqual(
            scores.transformed_conditions,
            ("jpeg_q30",),
        )

    def test_write_overall_summary_creates_json(self) -> None:
        rows = [
            {"transform": "clean", "auc": 0.85},
            {"transform": "noise_s010", "auc": 0.65},
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = (
                Path(temporary_dir)
                / "reports"
                / "overall_summary.json"
            )
            saved_path = write_overall_summary(
                rows,
                output_path,
            )

            self.assertEqual(saved_path, output_path)
            with output_path.open(encoding="utf-8") as output_file:
                saved = json.load(output_file)

            self.assertAlmostEqual(saved["clean_auc"], 0.85)
            self.assertAlmostEqual(saved["robust_auc"], 0.65)
            self.assertAlmostEqual(saved["combined_auc"], 0.75)
            self.assertEqual(
                saved["transformed_conditions"],
                ["noise_s010"],
            )

    def test_invalid_evaluation_rows_are_rejected(self) -> None:
        invalid_cases = [
            [],
            [{"transform": "clean", "auc": 0.8}],
            [
                {"transform": "clean", "auc": 0.8},
                {"transform": "clean", "auc": 0.7},
                {"transform": "blur", "auc": 0.6},
            ],
            [
                {"transform": "clean", "auc": 0.8},
                {"transform": "blur", "auc": float("nan")},
            ],
            [
                {"transform": "clean", "auc": 0.8},
                {"transform": "blur", "auc": 1.1},
            ],
        ]

        for rows in invalid_cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    calculate_overall_scores(rows)


if __name__ == "__main__":
    unittest.main()
