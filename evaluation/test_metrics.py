"""Automated tests for the evaluation metrics."""

import unittest

from evaluation.metrics import (
    classification_metrics,
    competition_score,
    evaluate_condition,
    find_errors,
    roc_auc,
)


class MetricsTests(unittest.TestCase):
    def test_perfect_roc_auc(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]

        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)

    def test_reversed_roc_auc(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.8, 0.9, 0.1, 0.2]

        self.assertAlmostEqual(roc_auc(labels, scores), 0.0)

    def test_tied_scores_roc_auc(self) -> None:
        labels = [0, 1]
        scores = [0.5, 0.5]

        self.assertAlmostEqual(roc_auc(labels, scores), 0.5)

    def test_classification_metrics(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.8, 0.7, 0.2]

        metrics = classification_metrics(labels, scores)

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_evaluate_condition(self) -> None:
        result = evaluate_condition(
            "JPEG quality 50",
            labels=[0, 0, 1, 1],
            scores=[0.1, 0.2, 0.8, 0.9],
        )

        self.assertEqual(result["condition"], "JPEG quality 50")
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertAlmostEqual(result["accuracy"], 1.0)

    def test_find_errors(self) -> None:
        errors = find_errors(
            image_paths=[
                "real-correct.jpg",
                "real-wrong.jpg",
                "ai-correct.jpg",
                "ai-wrong.jpg",
            ],
            labels=[0, 0, 1, 1],
            scores=[0.1, 0.8, 0.9, 0.2],
        )

        self.assertEqual(len(errors["false_positives"]), 1)
        self.assertEqual(
            errors["false_positives"][0]["image_path"],
            "real-wrong.jpg",
        )

        self.assertEqual(len(errors["false_negatives"]), 1)
        self.assertEqual(
            errors["false_negatives"][0]["image_path"],
            "ai-wrong.jpg",
        )

    def test_competition_score(self) -> None:
        self.assertAlmostEqual(competition_score(0.9, 0.7), 0.8)

    def test_invalid_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            roc_auc([0, 2], [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
    