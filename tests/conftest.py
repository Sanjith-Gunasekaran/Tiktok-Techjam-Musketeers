"""Make the repo root importable and share small fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def textured_image() -> Image.Image:
    """A deterministic 128x128 noisy image with one flat 32px tile at
    (row 2, col 1) — the known 'simplest patch'."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, (128, 128, 3)).astype(np.uint8)
    array[64:96, 32:64] = 128
    return Image.fromarray(array)
