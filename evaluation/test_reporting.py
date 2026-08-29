"""Automated tests for robustness report exporting."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.reporting import (
    calculate_overall_scores,
    save_report,
)


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = [
            {
                "condition": "clean",
                "count": 4,
                "roc_auc": 0.9,
                "accuracy": 0.75,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.7467,
                "true_positive": 2,
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 0,
            },
            {
                "condition": "jpeg_quality_50",
                "count": 4,
                "roc_auc": 0.7,
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "true_positive": 1,
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
            },
            {
                "condition": "blur_sigma_2.0",
                "count": 4,
                "roc_auc": 0.5,
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "true_positive": 1,
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
            },
        ]

        self.predictions = [
            {
                "condition": "clean",
                "image_path": "image-1.jpg",
                "label": 0,
                "pred": 0.1,
            }
        ]

        self.errors = {
            "clean": {
                "false_positives": [],
                "false_negatives": [],
            }
        }

    def test_calculate_overall_scores(self) -> None:
        scores = calculate_overall_scores(self.summary)

        self.assertAlmostEqual(scores["clean_auc"], 0.9)
        self.assertAlmostEqual(scores["robust_auc"], 0.6)
        self.assertAlmostEqual(scores["competition_score"], 0.75)

    def test_save_report_creates_all_files(self) -> None:
        report = {
            "summary": self.summary,
            "predictions": self.predictions,
            "errors": self.errors,
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            paths = save_report(report, temporary_dir)

            self.assertEqual(
                set(paths),
                {"summary", "predictions", "errors", "report"},
            )

            for path in paths.values():
                self.assertTrue(path.is_file())

            with paths["summary"].open(
                newline="",
                encoding="utf-8",
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["condition"], "clean")
            self.assertEqual(
                rows[1]["condition"],
                "jpeg_quality_50",
            )

            with paths["predictions"].open(
                encoding="utf-8",
            ) as predictions_file:
                saved_predictions = json.load(predictions_file)

            self.assertEqual(saved_predictions, self.predictions)

            with paths["report"].open(
                encoding="utf-8",
            ) as report_file:
                saved_report = json.load(report_file)

            self.assertAlmostEqual(
                saved_report["overall"]["competition_score"],
                0.75,
            )

    def test_missing_report_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(ValueError):
                save_report(
                    {"summary": self.summary},
                    Path(temporary_dir),
                )


if __name__ == "__main__":
    unittest.main()
    