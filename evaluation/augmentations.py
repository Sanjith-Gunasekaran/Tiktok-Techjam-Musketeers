"""Image transformations for robustness training and evaluation."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _as_rgb(image: Image.Image) -> Image.Image:
    """Return an independent RGB copy of an image."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    return image.convert("RGB")


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Compress an image as JPEG and decode it again."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")

    buffer = BytesIO()
    _as_rgb(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        compressed.load()
        return compressed.convert("RGB").copy()


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur."""
    if sigma < 0:
        raise ValueError("sigma cannot be negative")

    return _as_rgb(image).filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_restore(image: Image.Image, scale: float) -> Image.Image:
    """Shrink an image and restore it to its original dimensions."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be greater than 0 and at most 1")

    image = _as_rgb(image)
    original_size = image.size
    reduced_size = (
        max(1, round(original_size[0] * scale)),
        max(1, round(original_size[1] * scale)),
    )

    reduced = image.resize(reduced_size, Image.Resampling.LANCZOS)
    return reduced.resize(original_size, Image.Resampling.LANCZOS)


def gaussian_noise(
    image: Image.Image,
    sigma: float,
    seed: int = 0,
) -> Image.Image:
    """Add repeatable Gaussian noise to an image."""
    if sigma < 0:
        raise ValueError("sigma cannot be negative")

    pixels = np.asarray(_as_rgb(image), dtype=np.float32) / 255.0
    generator = np.random.default_rng(seed)
    noise = generator.normal(0.0, sigma, pixels.shape)

    noisy_pixels = np.clip(pixels + noise, 0.0, 1.0)
    return Image.fromarray(
        (noisy_pixels * 255).round().astype(np.uint8),
        mode="RGB",
    )


def color_jitter(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    """Change image brightness and contrast."""
    if brightness <= 0 or contrast <= 0:
        raise ValueError("brightness and contrast must be positive")

    image = ImageEnhance.Brightness(_as_rgb(image)).enhance(brightness)
    return ImageEnhance.Contrast(image).enhance(contrast)


def center_crop_restore(
    image: Image.Image,
    retain: float = 0.8,
) -> Image.Image:
    """Center-crop an image and restore its original dimensions."""
    if not 0 < retain <= 1:
        raise ValueError("retain must be greater than 0 and at most 1")

    image = _as_rgb(image)
    width, height = image.size
    crop_width = max(1, round(width * retain))
    crop_height = max(1, round(height * retain))

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))

    return cropped.resize((width, height), Image.Resampling.LANCZOS)