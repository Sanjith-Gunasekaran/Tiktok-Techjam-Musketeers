"""Image degradations for training and evaluation.

The hackathon brief lists six real-world transformations (JPEG re-encoding,
blur, thumbnail resize, sensor noise, color filters, cropping). This module
implements each one exactly once, then exposes them the two ways the pipeline
needs them:

* ``EVAL_GRID``     -- every transform at the brief's exact parameters,
                       deterministic, for the clean-vs-transformed table.
* ``RandomAugment`` -- one randomly chosen transform at a random strength,
                       applied during training so the model learns to cope.

Both modes share the same six functions below, so training and evaluation can
never drift apart. PIL images in, PIL images out; only PIL and numpy needed.
"""

from __future__ import annotations

import random
from functools import partial
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# The six base transforms.
# ---------------------------------------------------------------------------

def jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Really encode and decode as JPEG (no approximation), like a
    social-media re-upload. Lower quality = more damage."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur: the out-of-focus look."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_cycle(image: Image.Image, scale: float) -> Image.Image:
    """Shrink to ``scale`` then blow back up to the original size, like a
    thumbnail that got re-posted. Detail is lost on the way down."""
    width, height = image.size
    small = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BILINEAR,
    )
    return small.resize((width, height), Image.BILINEAR)


def gaussian_noise(
    image: Image.Image, sigma: float, rng: np.random.Generator | None = None
) -> Image.Image:
    """Add random pixel noise (``sigma`` on a 0-1 scale), like a low-light
    photo. Pass ``rng`` for reproducible noise."""
    if rng is None:
        rng = np.random.default_rng()
    array = np.asarray(image, dtype=np.float32)
    array = array + rng.normal(0.0, sigma * 255.0, array.shape)
    return Image.fromarray(np.clip(array, 0.0, 255.0).astype(np.uint8))


def color_jitter(
    image: Image.Image, brightness: float, contrast: float, saturation: float
) -> Image.Image:
    """Scale brightness/contrast/saturation (1.0 = unchanged), like a filter
    app or auto-enhance."""
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageEnhance.Color(image).enhance(saturation)


def center_crop(image: Image.Image, fraction: float) -> Image.Image:
    """Keep the middle ``fraction`` of each side, like profile-picture
    framing. Note: the output is SMALLER than the input on purpose; the
    branch preprocessing resizes later where needed."""
    width, height = image.size
    new_width, new_height = int(width * fraction), int(height * fraction)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height))


# ---------------------------------------------------------------------------
# Evaluation mode: the brief's exact parameter grid, fixed and reproducible.
# The noise entries use a fixed seed so every evaluation run sees identical
# noise. Color jitter is made deterministic as one "all down 20%" and one
# "all up 20%" variant.
# ---------------------------------------------------------------------------

def _seeded_noise(image: Image.Image, sigma: float) -> Image.Image:
    return gaussian_noise(image, sigma, rng=np.random.default_rng(0))


EVAL_GRID: dict[str, callable] = {
    "clean": lambda image: image,
    "jpeg_q90": partial(jpeg, quality=90),
    "jpeg_q70": partial(jpeg, quality=70),
    "jpeg_q50": partial(jpeg, quality=50),
    "jpeg_q30": partial(jpeg, quality=30),
    "blur_s05": partial(blur, sigma=0.5),
    "blur_s10": partial(blur, sigma=1.0),
    "blur_s20": partial(blur, sigma=2.0),
    "resize_050": partial(resize_cycle, scale=0.50),
    "resize_025": partial(resize_cycle, scale=0.25),
    "noise_s002": partial(_seeded_noise, sigma=0.02),
    "noise_s005": partial(_seeded_noise, sigma=0.05),
    "noise_s010": partial(_seeded_noise, sigma=0.10),
    "jitter_down": partial(color_jitter, brightness=0.8, contrast=0.8, saturation=0.8),
    "jitter_up": partial(color_jitter, brightness=1.2, contrast=1.2, saturation=1.2),
    "crop_80": partial(center_crop, fraction=0.8),
}


# ---------------------------------------------------------------------------
# Training mode: one random transform at a random strength.
# ---------------------------------------------------------------------------

class RandomAugment:
    """Randomly degrade a training image.

    With probability ``probability`` one of the six transforms is applied at
    a random strength inside the brief's ranges; otherwise the image passes
    through untouched, so the model keeps seeing clean images too.

    Reproducible when given a ``seed``. When using DataLoader workers, give
    each worker its own instance (or its own seed) so they don't all produce
    the same "random" choices.
    """

    _CHOICES = ("jpeg", "blur", "resize", "noise", "jitter", "crop")

    def __init__(self, probability: float = 0.5, seed: int | None = None):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self._rng.random() >= self.probability:
            return image
        choice = self._rng.choice(self._CHOICES)
        if choice == "jpeg":
            return jpeg(image, quality=self._rng.randint(30, 90))
        if choice == "blur":
            return blur(image, sigma=self._rng.uniform(0.3, 2.0))
        if choice == "resize":
            return resize_cycle(image, scale=self._rng.choice((0.25, 0.5)))
        if choice == "noise":
            return gaussian_noise(
                image, sigma=self._rng.uniform(0.02, 0.10), rng=self._np_rng
            )
        if choice == "jitter":
            return color_jitter(
                image,
                brightness=self._rng.uniform(0.8, 1.2),
                contrast=self._rng.uniform(0.8, 1.2),
                saturation=self._rng.uniform(0.8, 1.2),
            )
        return center_crop(image, fraction=0.8)
