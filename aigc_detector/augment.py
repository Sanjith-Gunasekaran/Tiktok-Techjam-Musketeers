"""Image augmentations used by the DINOv2 training pipeline."""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def resize_degradation(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    reduced = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return reduced.resize((width, height), Image.Resampling.LANCZOS)


def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    array = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, array.shape)
    noisy = (np.clip(array + noise, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(noisy).convert("RGB")


def color_jitter(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))


def center_crop_80(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_width, crop_height = round(width * 0.8), round(height * 0.8)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


class RobustAugment:
    """Apply one realistic degradation while retaining some clean images."""

    def __init__(self, clean_probability: float = 0.20) -> None:
        if not 0 <= clean_probability <= 1:
            raise ValueError("clean_probability must be between zero and one")
        self.clean_probability = clean_probability

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if random.random() < self.clean_probability:
            return image

        selected = random.choice(("jpeg", "blur", "resize", "noise", "color", "crop"))
        if selected == "jpeg":
            return jpeg_compress(image, random.choice((90, 70, 50, 30)))
        if selected == "blur":
            return image.filter(
                ImageFilter.GaussianBlur(radius=random.choice((0.5, 1.0, 2.0)))
            )
        if selected == "resize":
            return resize_degradation(image, random.choice((0.5, 0.25)))
        if selected == "noise":
            return gaussian_noise(image, random.choice((0.02, 0.05, 0.10)))
        if selected == "color":
            return color_jitter(image)
        return center_crop_80(image)


class DINOProcessorTransform:
    """Apply optional augmentation and convert a PIL image to pixel values."""

    def __init__(self, processor: Any, augmenter: Any | None = None) -> None:
        self.processor = processor
        self.augmenter = augmenter

    def __call__(self, image: Image.Image) -> Any:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.augmenter is not None:
            image = self.augmenter(image)
        return self.processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0)
