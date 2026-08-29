"""Deterministic internal test-set carve-out.

SID-Set ships with train and validation splits, but we also need our own
frozen test set for the final robustness table -- images that no training
decision ever touches. Rather than shuffling once and saving a file of IDs,
membership is decided by *hashing each image's ID*: the same image lands in
the same split on every machine, every run, forever, with nothing to store
or keep in sync.

Note: Python's built-in ``hash()`` is randomized per process on purpose, so
it would NOT be stable across runs -- hence md5.
"""

from __future__ import annotations

import hashlib

# Fraction of the source split reserved as our frozen internal test set.
TEST_FRACTION = 0.2


def is_internal_test(image_id: str, fraction: float = TEST_FRACTION) -> bool:
    """True if this image belongs to the frozen internal test set.

    The ID is hashed to a number spread evenly across 0-1; IDs landing below
    ``fraction`` are test. Deterministic: depends only on the ID string.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    digest = hashlib.md5(str(image_id).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def split_dataset(dataset, id_column: str = "img_id", fraction: float = TEST_FRACTION):
    """Split one dataset into (dev, internal_test) by the hash rule.

    ``dev`` is what day-to-day work may use (validation during training,
    threshold tuning); ``internal_test`` is only for the final evaluation.
    Works on anything with a ``.filter(fn)`` method, e.g. Hugging Face
    datasets.
    """
    test = dataset.filter(lambda row: is_internal_test(row[id_column], fraction))
    dev = dataset.filter(lambda row: not is_internal_test(row[id_column], fraction))
    return dev, test
