"""Tests for the labelled robustness-transformation preview."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pipeline.augmentations import EVAL_GRID
from pipeline.preview_augmentations import create_preview


class PreviewAugmentationTests(unittest.TestCase):
    @staticmethod
    def create_source(path: Path) -> None:
        image = Image.new("RGB", (160, 96), (80, 120, 160))
        image.save(path)

    def test_preview_contains_the_complete_eval_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_path = root / "source.png"
            output_path = root / "results" / "preview.png"
            self.create_source(source_path)

            saved_path = create_preview(
                source_path,
                output_path,
                columns=4,
                cell_size=64,
                label_height=20,
            )

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.is_file())

            expected_rows = math.ceil(len(EVAL_GRID) / 4)
            with Image.open(output_path) as preview:
                self.assertEqual(
                    preview.size,
                    (4 * 64, expected_rows * (64 + 20)),
                )
                self.assertEqual(preview.mode, "RGB")

    def test_preview_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_path = root / "source.png"
            first_path = root / "first.png"
            second_path = root / "second.png"
            self.create_source(source_path)

            create_preview(
                source_path,
                first_path,
                cell_size=48,
                label_height=16,
            )
            create_preview(
                source_path,
                second_path,
                cell_size=48,
                label_height=16,
            )

            self.assertEqual(
                first_path.read_bytes(),
                second_path.read_bytes(),
            )

    def test_invalid_layout_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_path = root / "source.png"
            self.create_source(source_path)

            with self.assertRaises(ValueError):
                create_preview(
                    source_path,
                    root / "preview.png",
                    columns=0,
                )

            with self.assertRaises(TypeError):
                create_preview(
                    source_path,
                    root / "preview.png",
                    columns=True,
                )

    def test_empty_or_invalid_transforms_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_path = root / "source.png"
            self.create_source(source_path)

            with self.assertRaises(ValueError):
                create_preview(
                    source_path,
                    root / "empty.png",
                    transforms={},
                )

            with self.assertRaises(TypeError):
                create_preview(
                    source_path,
                    root / "invalid.png",
                    transforms={
                        "invalid": lambda image: "not an image",
                    },
                )


if __name__ == "__main__":
    unittest.main()