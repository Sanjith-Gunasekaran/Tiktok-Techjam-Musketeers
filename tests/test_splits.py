"""Tests for pipeline/splits.py."""

from collections import Counter

import pytest

from pipeline import exclude_heldout_families, family_id, is_internal_test, split_dataset
from pipeline.splits import TEST_FRACTION

IDS = [f"img_{i}" for i in range(20000)]


def test_membership_is_deterministic():
    assert [is_internal_test(i) for i in IDS[:2000]] == [
        is_internal_test(i) for i in IDS[:2000]
    ]


def test_fraction_is_respected():
    assert abs(sum(map(is_internal_test, IDS)) / len(IDS) - TEST_FRACTION) < 0.01
    assert abs(sum(is_internal_test(i, 0.1) for i in IDS) / len(IDS) - 0.1) < 0.01


def test_family_id_strips_class_prefixes():
    assert family_id("real_0a1b") == "0a1b"
    assert family_id("tampered_0a1b") == "0a1b"
    assert family_id("full_synthetic_0a1b") == "0a1b"
    assert family_id("0a1b") == "0a1b"  # unprefixed IDs pass through


def test_derived_images_never_straddle_the_split():
    """A tampered image built from a real photo must land on the same side as
    that photo — otherwise we would test on an edit of a training image."""
    for stem in [f"{i:05x}" for i in range(3000)]:
        verdict = is_internal_test(f"real_{stem}")
        assert is_internal_test(f"tampered_{stem}") == verdict
        assert is_internal_test(f"full_synthetic_{stem}") == verdict


def test_membership_is_frozen():
    # Pinned values: if the hashing rule ever changes, this fails loudly
    # instead of silently reshuffling the test set mid-project.
    assert not is_internal_test("img_0")
    assert not is_internal_test("coco_123")
    assert is_internal_test("img_13") == is_internal_test("img_13")


def test_split_dataset_disjoint_covering_and_balanced():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict(
        {"img_id": IDS[:5000], "label": [i % 3 for i in range(5000)]}
    )
    dev, test = split_dataset(ds)
    assert len(dev) + len(test) == len(ds)
    assert set(dev["img_id"]).isdisjoint(set(test["img_id"]))
    dev2, test2 = split_dataset(ds)
    assert test["img_id"] == test2["img_id"], "carve-out must be identical every run"
    # hash is label-blind, so class balance should carry over roughly
    assert abs(Counter(test["label"])[0] / len(test) - 1 / 3) < 0.05


def test_split_dataset_keeps_families_together():
    datasets = pytest.importorskip("datasets")
    stems = [f"{i:04d}" for i in range(1500)]
    ds = datasets.Dataset.from_dict(
        {
            "img_id": [f"real_{s}" for s in stems] + [f"tampered_{s}" for s in stems],
            "label": [0] * len(stems) + [2] * len(stems),
        }
    )
    _, test = split_dataset(ds)
    test_stems = Counter(family_id(i) for i in test["img_id"])
    # every family present in test must contribute BOTH of its images
    assert set(test_stems.values()) == {2}, sorted(test_stems.values())[:5]


def test_exclude_heldout_families_removes_prefixed_relatives():
    datasets = pytest.importorskip("datasets")
    source = datasets.Dataset.from_dict(
        {"img_id": ["real_a", "full_synthetic_a", "real_b"], "label": [0, 1, 0]}
    )

    filtered = exclude_heldout_families(source, {"a"})

    assert filtered["img_id"] == ["real_b"]


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        is_internal_test("x", 0.0)
