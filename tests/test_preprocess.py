"""Tests for pipeline/preprocess.py."""

import numpy as np
import pytest
import torch
from PIL import Image

from pipeline import RandomAugment, canonicalize_encoding, dino_view, simplest_patch, two_views
from pipeline.preprocess import IMAGENET_MEAN, IMAGENET_STD, STANDARD_SIZE


def test_dino_view_shape_dtype_and_exact_normalization():
    tensor = dino_view(Image.new("RGB", (50, 50), (128, 128, 128)))
    assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
    expected = (128 / 255 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(tensor.numpy()[:, 0, 0], expected, atol=1e-5)


def test_encoding_canonicalization_is_deterministic_and_lossy():
    values = np.arange(64, dtype=np.uint8)
    array = np.stack(np.meshgrid(values, values, indexing="ij"), axis=-1)
    array = np.concatenate((array, array[..., :1] * 3), axis=-1)
    image = Image.fromarray(array)

    first = canonicalize_encoding(image)
    second = canonicalize_encoding(image)

    assert first.mode == "RGB" and first.size == image.size
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(image))


def test_encoding_canonicalization_rejects_invalid_quality():
    with pytest.raises(ValueError, match="quality"):
        canonicalize_encoding(Image.new("RGB", (8, 8)), quality=0)


def test_dino_view_preserves_aspect_ratio_no_stretching():
    """A wide image must be center-cropped, not squashed: a circle stays a
    circle. Build a tall red bar on grey; after shortest-edge-256 + crop the
    bar's width:height ratio must be preserved, not stretched to square."""
    array = np.full((256, 1024, 3), 200, dtype=np.uint8)
    array[:, 480:544] = [255, 0, 0]  # 64px-wide bar, centred
    tensor = dino_view(Image.fromarray(array))
    # Shortest edge is already 256, so no scaling; crop keeps the middle 224.
    red_columns = (tensor[0] > tensor[1]).float().sum(dim=1).max().item()
    assert 60 <= red_columns <= 68, red_columns  # ~64px wide, unstretched


def test_dino_view_handles_extreme_and_small_shapes():
    for size in [(10, 400), (400, 10), (32, 32), (1, 1)]:
        assert dino_view(Image.new("RGB", size, "green")).shape == (3, 224, 224)


def test_simplest_patch_selects_known_flat_tile(textured_image):
    patch = simplest_patch(textured_image)
    assert patch.shape == (3, 32, 32) and patch.dtype == torch.float32
    # The fixture is already 256x256, so the flat tile survives untouched.
    assert torch.all(patch == 128.0)


def test_simplest_patch_candidate_count_is_resolution_independent():
    """The whole point of standardizing first: a big image and a small image
    must both be scanned over the same 8x8 candidate grid, so resolution
    cannot leak into the forensics branch."""
    rng = np.random.default_rng(3)
    base = rng.integers(0, 256, (256, 256, 3)).astype(np.uint8)
    base[96:128, 160:192] = 90  # one flat tile
    small = Image.fromarray(base)
    large = small.resize((2048, 2048), Image.NEAREST)  # same content, 64x the pixels
    # Both must pick the SAME region (the flat one), not a spuriously smoother
    # tile that only exists because a big image offers more candidates.
    # Compare region identity by mean value, not pixels: resampling the
    # upscaled copy back to 256 leaves ringing around the flat block.
    assert abs(simplest_patch(small).mean().item() - 90) < 1
    assert abs(simplest_patch(large).mean().item() - 90) < 3
    # Sanity: the surrounding noise averages far from 90, so matching means
    # really does mean "same region".
    assert abs(float(np.asarray(small, np.float32).mean()) - 90) > 30


def test_simplest_patch_handles_odd_and_tiny_sizes():
    rng = np.random.default_rng(1)
    odd = Image.fromarray(rng.integers(0, 256, (127, 200, 3)).astype(np.uint8))
    assert simplest_patch(odd).shape == (3, 32, 32)
    assert simplest_patch(Image.new("RGB", (20, 10), "blue")).shape == (3, 32, 32)


def test_views_are_deterministic(textured_image):
    assert torch.equal(simplest_patch(textured_image), simplest_patch(textured_image))
    assert torch.equal(dino_view(textured_image), dino_view(textured_image))


def test_two_views_after_augmentation(textured_image):
    degraded = RandomAugment(1.0, seed=5)(textured_image)
    dino_tensor, patch_tensor = two_views(degraded)
    assert dino_tensor.shape == (3, 224, 224) and patch_tensor.shape == (3, 32, 32)


def test_noise_survives_into_raw_patch(textured_image):
    array = np.asarray(textured_image).copy()
    rng = np.random.default_rng(2)
    array[64:96, 32:64] = np.clip(128 + rng.normal(0, 3, (32, 32, 3)), 0, 255).astype(
        np.uint8
    )
    patch = simplest_patch(Image.fromarray(array))
    assert patch.std() > 0  # the faint noise is preserved, not smoothed away


def test_patch_is_raw_pixels_not_normalized(textured_image):
    """simplest_patch must NOT normalize: SRM filters run on 0-255 values."""
    patch = simplest_patch(textured_image)
    assert patch.min() >= 0.0 and patch.max() <= 255.0
    assert patch.max() > 1.5  # would be <=1 if it had been scaled to 0-1
