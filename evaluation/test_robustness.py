"""Automated tests for the robustness evaluation runner."""

import unittest
from collections.abc import Sequence

import numpy as np
from PIL import Image

from evaluation.robustness import (
    DEFAULT_CONDITIONS,
    RobustnessCondition,
    evaluate_robustness,
)


def brightness_predict(images: Sequence[Image.Image]) -> list[float]:
    """Simple fake model: brighter images receive higher AI scores."""
    return [
        float(np.asarray(image, dtype=float).mean() / 255.0)
        for image in images
    ]


def solid_image(value: int) -> Image.Image:
    return Image.new("RGB", (24, 24), (value, value, value))


class RobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = [
            solid_image(20),
            solid_image(40),
            solid_image(210),
            solid_image(240),
        ]
        self.labels = [0, 0, 1, 1]
        self.paths = [
            "real-1.jpg",
            "real-2.jpg",
            "ai-1.jpg",
            "ai-2.jpg",
        ]

    def test_all_required_conditions_are_present(self) -> None:
        names = {condition.name for condition in DEFAULT_CONDITIONS}

        self.assertEqual(len(DEFAULT_CONDITIONS), 18)
        self.assertIn("clean", names)
        self.assertIn("jpeg_quality_30", names)
        self.assertIn("blur_sigma_2.0", names)
        self.assertIn("resize_0.25x", names)
        self.assertIn("noise_sigma_0.10", names)
        self.assertIn("brightness_-20_percent", names)
        self.assertIn("contrast_+20_percent", names)
        self.assertIn("center_crop_80_percent", names)

    def test_clean_and_jpeg_evaluation(self) -> None:
        report = evaluate_robustness(
            self.images,
            self.labels,
            brightness_predict,
            image_paths=self.paths,
            conditions=DEFAULT_CONDITIONS[:2],
            batch_size=3,
        )

        self.assertEqual(len(report["summary"]), 2)
        self.assertEqual(len(report["predictions"]), 8)
        self.assertEqual(
            report["summary"][0]["condition"],
            "clean",
        )
        self.assertAlmostEqual(
            report["summary"][0]["roc_auc"],
            1.0,
        )
        self.assertEqual(
            report["errors"]["clean"]["false_positives"],
            [],
        )
        self.assertEqual(
            report["errors"]["clean"]["false_negatives"],
            [],
        )

    def test_prediction_is_batched(self) -> None:
        batch_sizes: list[int] = []

        def recording_predict(
            images: Sequence[Image.Image],
        ) -> list[float]:
            batch_sizes.append(len(images))
            return brightness_predict(images)

        clean_only = (
            RobustnessCondition(
                "clean",
                lambda image, seed: image.copy(),
            ),
        )

        evaluate_robustness(
            self.images,
            self.labels,
            recording_predict,
            conditions=clean_only,
            batch_size=3,
        )

        self.assertEqual(batch_sizes, [3, 1])

    def test_wrong_number_of_scores_is_rejected(self) -> None:
        def broken_predict(
            images: Sequence[Image.Image],
        ) -> list[float]:
            del images
            return []

        with self.assertRaises(ValueError):
            evaluate_robustness(
                self.images,
                self.labels,
                broken_predict,
                conditions=DEFAULT_CONDITIONS[:1],
            )

    def test_noise_evaluation_is_repeatable(self) -> None:
        noise_only = (
            next(
                condition
                for condition in DEFAULT_CONDITIONS
                if condition.name == "noise_sigma_0.05"
            ),
        )

        first = evaluate_robustness(
            self.images,
            self.labels,
            brightness_predict,
            conditions=noise_only,
            seed=42,
        )
        second = evaluate_robustness(
            self.images,
            self.labels,
            brightness_predict,
            conditions=noise_only,
            seed=42,
        )

        self.assertEqual(
            first["predictions"],
            second["predictions"],
        )


if __name__ == "__main__":
    unittest.main()