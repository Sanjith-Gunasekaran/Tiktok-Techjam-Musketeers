"""Tests for pipeline/augmentations.py."""

import numpy as np
from PIL import Image

from pipeline import EVAL_GRID, RandomAugment
from pipeline.augmentations import center_crop, jpeg


def test_eval_grid_entries_run_and_keep_expected_sizes(textured_image):
    assert len(EVAL_GRID) == 16
    for name, transform in EVAL_GRID.items():
        out = transform(textured_image)
        assert out.mode == "RGB", name
        if name == "crop_80":
            assert out.size == (102, 102)  # 80% of 128, rounded down
        else:
            assert out.size == textured_image.size, name


def test_eval_grid_is_deterministic(textured_image):
    for name, transform in EVAL_GRID.items():
        assert transform(textured_image).tobytes() == transform(textured_image).tobytes(), name


def test_every_transform_changes_the_image_except_clean(textured_image):
    for name, transform in EVAL_GRID.items():
        changed = transform(textured_image).tobytes() != textured_image.tobytes()
        assert changed == (name != "clean"), name


def test_random_augment_seeding(textured_image):
    def sequence(seed):
        augment = RandomAugment(1.0, seed=seed)
        return [augment(textured_image).tobytes() for _ in range(6)]

    assert sequence(7) == sequence(7), "same seed must reproduce"
    assert sequence(7) != sequence(8), "different seed must differ"


def test_random_augment_probability_edges(textured_image):
    assert RandomAugment(0.0, seed=1)(textured_image) is textured_image
    always = RandomAugment(1.0, seed=3)
    assert all(always(textured_image).tobytes() != textured_image.tobytes() for _ in range(12))


def test_heavier_jpeg_damages_more(textured_image):
    original = np.asarray(textured_image, dtype=np.int16)

    def damage(quality):
        return np.abs(original - np.asarray(jpeg(textured_image, quality), dtype=np.int16)).mean()

    assert damage(30) > damage(90)


def test_center_crop_handles_tiny_odd_sizes():
    assert center_crop(Image.new("RGB", (5, 7), "red"), 0.8).size == (4, 5)
