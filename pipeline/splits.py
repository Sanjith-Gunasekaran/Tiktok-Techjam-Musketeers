"""Deterministic internal test-set carve-out.

SID-Set publishes train/validation splits plus a 60K test split that is NOT
public (it is gated behind a request to the authors), so we carve our own
frozen test set from the validation split: images no training decision ever
touches, reserved for the final robustness table.

Membership is decided by *hashing each image's family ID*: the same image
lands in the same split on every machine, every run, with nothing to store
or keep in sync. The family ID strips the class prefix from the raw ID
(``real_12ab`` / ``tampered_12ab`` -> ``12ab``) so images derived from the
same source photo -- SID-Set builds tampered images out of real ones -- stay
on the same side of the split and the model is never tested on an edit of a
picture it trained on. (If tampered IDs turn out not to share source stems,
this degrades gracefully to plain per-ID hashing.)

What this split measures and what it does not: it supports "held-out images,
same generators" robustness claims. Unseen-generator generalization is
measured by the organizers' external benchmark (DALL·E Advanced + COCO,
never trained on), not by this carve-out.

Note: Python's built-in ``hash()`` is randomized per process on purpose, so
it would NOT be stable across runs -- hence md5.
"""

from __future__ import annotations

import hashlib

# Fraction of the source split reserved as our frozen internal test set.
TEST_FRACTION = 0.2

# Known SID-Set class prefixes, longest first so the right one is stripped.
_CLASS_PREFIXES = ("full_synthetic_", "tampered_", "synthetic_", "real_")


def family_id(image_id: str) -> str:
    """The ID with any class prefix removed: images derived from the same
    source photo share a family and must land in the same split."""
    text = str(image_id)
    for prefix in _CLASS_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def is_internal_test(image_id: str, fraction: float = TEST_FRACTION) -> bool:
    """True if this image belongs to the frozen internal test set.

    The family ID is hashed to a number spread evenly across 0-1; families
    landing below ``fraction`` are test. Deterministic: depends only on the
    ID string.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    digest = hashlib.md5(family_id(image_id).encode("utf-8")).digest()
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
