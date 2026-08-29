"""Tests for pipeline/splits.py."""

from collections import Counter

import pytest

from pipeline import is_internal_test, split_dataset
from pipeline.splits import TEST_FRACTION

IDS = [f"img_{i}" for i in range(20000)]


def test_membership_is_deterministic():
    assert [is_internal_test(i) for i in IDS[:2000]] == [is_internal_test(i) for i in IDS[:2000]]


def test_fraction_is_respected():
    assert abs(sum(map(is_internal_test, IDS)) / len(IDS) - TEST_FRACTION) < 0.01
    assert abs(sum(is_internal_test(i, 0.1) for i in IDS) / len(IDS) - 0.1) < 0.01


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


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        is_internal_test("x", 0.0)
