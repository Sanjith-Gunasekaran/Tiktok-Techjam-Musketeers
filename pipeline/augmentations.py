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
import zlib
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
# Color jitter is made deterministic as one "all down 20%" and one "all up
# 20%" variant. A compound entry mimics a realistic re-upload (crop +
# thumbnail + JPEG stacked), since real posts rarely suffer one degradation
# alone.
# ---------------------------------------------------------------------------

def _content_seeded_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Deterministic noise whose seed comes from the image's own pixels:
    runs are repeatable, but no two images share a noise pattern (a shared
    pattern would be a giveaway to a noise-forensics branch)."""
    seed = zlib.crc32(image.tobytes()[:4096])
    return gaussian_noise(image, sigma, rng=np.random.default_rng(seed))


def _realistic_chain(image: Image.Image) -> Image.Image:
    """Crop, thumbnail, and JPEG together, like a typical repost pipeline."""
    return jpeg(resize_cycle(center_crop(image, 0.8), 0.5), 50)


def identity(image: Image.Image) -> Image.Image:
    """Return an unchanged image for the clean evaluation cell."""
    return image


EVAL_GRID: dict[str, callable] = {
    "clean": identity,
    "jpeg_q90": partial(jpeg, quality=90),
    "jpeg_q70": partial(jpeg, quality=70),
    "jpeg_q50": partial(jpeg, quality=50),
    "jpeg_q30": partial(jpeg, quality=30),
    "blur_s05": partial(blur, sigma=0.5),
    "blur_s10": partial(blur, sigma=1.0),
    "blur_s20": partial(blur, sigma=2.0),
    "resize_050": partial(resize_cycle, scale=0.50),
    "resize_025": partial(resize_cycle, scale=0.25),
    "noise_s002": partial(_content_seeded_noise, sigma=0.02),
    "noise_s005": partial(_content_seeded_noise, sigma=0.05),
    "noise_s010": partial(_content_seeded_noise, sigma=0.10),
    "jitter_down": partial(color_jitter, brightness=0.8, contrast=0.8, saturation=0.8),
    "jitter_up": partial(color_jitter, brightness=1.2, contrast=1.2, saturation=1.2),
    "crop_80": partial(center_crop, fraction=0.8),
    "chain_crop_resize_jpeg": _realistic_chain,
}


# ---------------------------------------------------------------------------
# Training mode: one random transform at a random strength.
# ---------------------------------------------------------------------------

class RandomAugment:
    """Randomly degrade a training image.

    With probability ``probability`` one of the six transforms is applied at
    a random strength inside the brief's ranges; otherwise the image passes
    through untouched, so the model keeps seeing clean images too. Because
    real uploads often stack degradations (crop + resize + JPEG), a second,
    different transform is chained on top with ``second_probability``.

    Reproducible when given a ``seed``. IMPORTANT with DataLoader workers:
    forked workers start with identical copies of this object's random
    state and would emit duplicate augmentation streams -- call ``reseed``
    from a ``worker_init_fn`` (e.g. ``reseed(base_seed + worker_id)``).
    """

    _CHOICES = ("jpeg", "blur", "resize", "noise", "jitter", "crop")

    def __init__(
        self,
        probability: float = 0.5,
        seed: int | None = None,
        second_probability: float = 0.3,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if not 0.0 <= second_probability <= 1.0:
            raise ValueError("second_probability must be between 0 and 1")
        self.probability = probability
        self.second_probability = second_probability
        self.reseed(seed)

    def reseed(self, seed: int | None) -> None:
        """Give this instance a fresh random state (see class docstring)."""
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self._rng.random() >= self.probability:
            return image
        first = self._rng.choice(self._CHOICES)
        image = self._apply(first, image)
        if self._rng.random() < self.second_probability:
            second = self._rng.choice([c for c in self._CHOICES if c != first])
            image = self._apply(second, image)
        return image

    def _apply(self, choice: str, image: Image.Image) -> Image.Image:
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
