"""Tests for the directory-to-JSON inference script."""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image
from pipeline.canonicalize import canonicalize_encoding

from predict import (
    choose_device,
    find_images,
    load_model,
    predict_paths,
    write_predictions,
)


class FakeModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def eval(self) -> "FakeModel":
        return self

    def predict_proba(
        self,
        dino_images: torch.Tensor,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        if dino_images.shape[1:] != (3, 224, 224):
            raise AssertionError("Unexpected DINO input shape")
        if patches.shape[1:] != (3, 32, 32):
            raise AssertionError("Unexpected forensic input shape")
        if len(dino_images) != len(patches):
            raise AssertionError("Branch batch sizes differ")

        self.batch_sizes.append(len(dino_images))
        return torch.full(
            (len(dino_images),),
            0.75,
            device=dino_images.device,
        )


class InferenceTests(unittest.TestCase):
    @staticmethod
    def create_image(path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 240), color).save(path)

    def test_find_images_is_recursive_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = root / "a.png"
            second = root / "nested" / "B.JPG"
            self.create_image(first, (10, 20, 30))
            self.create_image(second, (40, 50, 60))
            (root / "notes.txt").write_text(
                "not an image",
                encoding="utf-8",
            )

            self.assertEqual(find_images(root), [first, second])

    def test_empty_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(ValueError):
                find_images(temporary_dir)

    def test_predictions_are_batched_and_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for index in range(3):
                self.create_image(
                    root / f"image-{index}.png",
                    (index * 20, 30, 40),
                )

            image_paths = find_images(root)
            model = FakeModel()
            with patch(
                "predict.canonicalize_encoding",
                wraps=canonicalize_encoding,
            ) as canonicalize:
                predictions = predict_paths(
                    model,
                    image_paths,
                    torch.device("cpu"),
                    batch_size=2,
                )

            self.assertEqual(canonicalize.call_count, 3)

            self.assertEqual(model.batch_sizes, [2, 1])
            self.assertEqual(len(predictions), 3)
            self.assertEqual(
                [row["image_path"] for row in predictions],
                [str(path) for path in image_paths],
            )
            for row in predictions:
                self.assertAlmostEqual(row["pred"], 0.75)

            output_path = root / "results" / "predictions.json"
            saved_path = write_predictions(
                predictions,
                output_path,
            )
            self.assertEqual(saved_path, output_path)

            with output_path.open(encoding="utf-8") as output_file:
                saved_predictions = json.load(output_file)

            self.assertEqual(saved_predictions, predictions)

    def test_invalid_batch_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            predict_paths(
                FakeModel(),
                [],
                torch.device("cpu"),
                batch_size=0,
            )

        with self.assertRaises(TypeError):
            predict_paths(
                FakeModel(),
                [],
                torch.device("cpu"),
                batch_size=True,
            )

    def test_non_fusion_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "wrong.pt"
            torch.save(
                {
                    "stage": "forensic",
                    "model_state_dict": {},
                    "model_config": {},
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(
                ValueError,
                "expected a 'fusion' checkpoint",
            ):
                load_model(
                    checkpoint_path,
                    torch.device("cpu"),
                )

    def test_cpu_device_can_be_selected(self) -> None:
        self.assertEqual(
            choose_device("cpu"),
            torch.device("cpu"),
        )


if __name__ == "__main__":
    unittest.main()