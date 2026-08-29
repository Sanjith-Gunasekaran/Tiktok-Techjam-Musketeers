"""Tests for the robustness transformations."""

import unittest

from PIL import Image

from evaluation.augmentations import (
    center_crop_restore,
    color_jitter,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    resize_restore,
)


class AugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (100, 80), (120, 100, 80))

    def check_output(self, output: Image.Image) -> None:
        self.assertIsInstance(output, Image.Image)
        self.assertEqual(output.size, self.image.size)
        self.assertEqual(output.mode, "RGB")

    def test_jpeg_compress(self) -> None:
        self.check_output(jpeg_compress(self.image, quality=70))

    def test_gaussian_blur(self) -> None:
        self.check_output(gaussian_blur(self.image, sigma=1.0))

    def test_resize_restore(self) -> None:
        self.check_output(resize_restore(self.image, scale=0.5))

    def test_gaussian_noise(self) -> None:
        self.check_output(gaussian_noise(self.image, sigma=0.05, seed=42))

    def test_color_jitter(self) -> None:
        self.check_output(
            color_jitter(self.image, brightness=1.2, contrast=0.8)
        )

    def test_center_crop_restore(self) -> None:
        self.check_output(center_crop_restore(self.image, retain=0.8))

    def test_noise_is_repeatable(self) -> None:
        first = gaussian_noise(self.image, sigma=0.05, seed=42)
        second = gaussian_noise(self.image, sigma=0.05, seed=42)
        self.assertEqual(first.tobytes(), second.tobytes())


if __name__ == "__main__":
    unittest.main()