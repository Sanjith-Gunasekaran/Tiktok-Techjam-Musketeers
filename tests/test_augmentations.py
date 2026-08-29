"""Tests for pipeline/augmentations.py."""

import numpy as np
from PIL import Image

from pipeline import EVAL_GRID, RandomAugment
from pipeline.augmentations import center_crop, jpeg

# Entries that deliberately change the image size (cropping is part of them).
_SIZE_CHANGING = {"crop_80": (204, 204), "chain_crop_resize_jpeg": (204, 204)}


def test_eval_grid_entries_run_and_keep_expected_sizes(textured_image):
    assert len(EVAL_GRID) == 17  # 16 single transforms + 1 realistic chain
    for name, transform in EVAL_GRID.items():
        out = transform(textured_image)
        assert out.mode == "RGB", name
        assert out.size == _SIZE_CHANGING.get(name, textured_image.size), name


def test_eval_grid_is_deterministic(textured_image):
    for name, transform in EVAL_GRID.items():
        assert (
            transform(textured_image).tobytes() == transform(textured_image).tobytes()
        ), name


def test_every_transform_changes_the_image_except_clean(textured_image):
    for name, transform in EVAL_GRID.items():
        changed = transform(textured_image).tobytes() != textured_image.tobytes()
        assert changed == (name != "clean"), name


def test_eval_noise_differs_between_images():
    """Eval noise is seeded from image content, so two same-sized images must
    NOT receive the identical noise pattern — a shared pattern would be a
    giveaway to the noise-forensics branch."""
    rng = np.random.default_rng(11)
    first = Image.fromarray(rng.integers(0, 256, (64, 64, 3)).astype(np.uint8))
    second = Image.fromarray(rng.integers(0, 256, (64, 64, 3)).astype(np.uint8))
    noise = EVAL_GRID["noise_s005"]
    residual_a = np.asarray(noise(first), np.int16) - np.asarray(first, np.int16)
    residual_b = np.asarray(noise(second), np.int16) - np.asarray(second, np.int16)
    assert not np.array_equal(residual_a, residual_b)
    # ...but still repeatable for the same image.
    assert np.array_equal(
        residual_a, np.asarray(noise(first), np.int16) - np.asarray(first, np.int16)
    )


def test_realistic_chain_stacks_degradations(textured_image):
    """The chain entry must damage more than any single step it contains."""
    original = np.asarray(center_crop(textured_image, 0.8), dtype=np.int16)
    chained = np.asarray(EVAL_GRID["chain_crop_resize_jpeg"](textured_image), np.int16)
    only_jpeg = np.asarray(
        jpeg(center_crop(textured_image, 0.8), 50), dtype=np.int16
    )
    assert np.abs(original - chained).mean() > np.abs(original - only_jpeg).mean()


def test_random_augment_seeding(textured_image):
    def sequence(seed):
        augment = RandomAugment(1.0, seed=seed)
        return [augment(textured_image).tobytes() for _ in range(6)]

    assert sequence(7) == sequence(7), "same seed must reproduce"
    assert sequence(7) != sequence(8), "different seed must differ"


def test_reseed_diverges_worker_copies(textured_image):
    """Forked DataLoader workers share state; reseed must separate them."""
    worker_a, worker_b = RandomAugment(1.0, seed=4), RandomAugment(1.0, seed=4)
    worker_a.reseed(100 + 0)
    worker_b.reseed(100 + 1)
    assert [worker_a(textured_image).tobytes() for _ in range(6)] != [
        worker_b(textured_image).tobytes() for _ in range(6)
    ]


def test_second_transform_is_sometimes_chained(textured_image):
    """With second_probability=1 every augmented image gets two transforms,
    so results must differ from the single-transform stream."""
    single = RandomAugment(1.0, seed=9, second_probability=0.0)
    double = RandomAugment(1.0, seed=9, second_probability=1.0)
    assert [single(textured_image).tobytes() for _ in range(4)] != [
        double(textured_image).tobytes() for _ in range(4)
    ]


def test_random_augment_probability_edges(textured_image):
    assert RandomAugment(0.0, seed=1)(textured_image) is textured_image
    always = RandomAugment(1.0, seed=3)
    assert all(
        always(textured_image).tobytes() != textured_image.tobytes() for _ in range(12)
    )


def test_heavier_jpeg_damages_more(textured_image):
    original = np.asarray(textured_image, dtype=np.int16)

    def damage(quality):
        return np.abs(
            original - np.asarray(jpeg(textured_image, quality), dtype=np.int16)
        ).mean()

    assert damage(30) > damage(90)


def test_center_crop_handles_tiny_odd_sizes():
    assert center_crop(Image.new("RGB", (5, 7), "red"), 0.8).size == (4, 5)
