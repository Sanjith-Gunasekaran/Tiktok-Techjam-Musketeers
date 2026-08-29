"""Tests for pipeline/preprocess.py."""

import numpy as np
import torch
from PIL import Image

from pipeline import RandomAugment, dino_view, srm_view, two_views
from pipeline.preprocess import IMAGENET_MEAN, IMAGENET_STD


def test_dino_view_shape_dtype_and_exact_normalization():
    tensor = dino_view(Image.new("RGB", (50, 50), (128, 128, 128)))
    assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
    expected = (128 / 255 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(tensor.numpy()[:, 0, 0], expected, atol=1e-5)


def test_srm_view_selects_known_flat_tile_untouched(textured_image):
    patch = srm_view(textured_image)
    assert patch.shape == (3, 32, 32) and patch.dtype == torch.float32
    assert torch.all(patch == 128.0)  # the flat tile, raw 0-255, byte-perfect


def test_srm_view_handles_odd_and_tiny_sizes():
    rng = np.random.default_rng(1)
    odd = Image.fromarray(rng.integers(0, 256, (127, 200, 3)).astype(np.uint8))
    assert srm_view(odd).shape == (3, 32, 32)
    assert srm_view(Image.new("RGB", (20, 10), "blue")).shape == (3, 32, 32)


def test_views_are_deterministic(textured_image):
    assert torch.equal(srm_view(textured_image), srm_view(textured_image))
    assert torch.equal(dino_view(textured_image), dino_view(textured_image))


def test_two_views_after_augmentation(textured_image):
    degraded = RandomAugment(1.0, seed=5)(textured_image)
    dino_tensor, srm_tensor = two_views(degraded)
    assert dino_tensor.shape == (3, 224, 224) and srm_tensor.shape == (3, 32, 32)


def test_noise_survives_into_raw_patch(textured_image):
    array = np.asarray(textured_image).copy()
    rng = np.random.default_rng(2)
    noisy_flat = np.clip(128 + rng.normal(0, 3, (32, 32, 3)), 0, 255).astype(np.uint8)
    array[64:96, 32:64] = noisy_flat
    patch = srm_view(Image.fromarray(array))
    assert patch.std() > 0  # the faint noise is preserved, not smoothed away
