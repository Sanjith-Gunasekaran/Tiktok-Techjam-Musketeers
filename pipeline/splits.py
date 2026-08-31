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
CALIBRATION_FRACTION = 0.25

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
    Uses only the ID column, so Hugging Face does not decode images or masks
    while splitting.
    """
    filter_kwargs = {
        "input_columns": id_column,
        "fn_kwargs": {"fraction": fraction},
    }
    test = dataset.filter(is_internal_test, **filter_kwargs)
    dev = dataset.filter(_is_internal_dev, **filter_kwargs)
    return dev, test


def is_fusion_calibration(image_id: str, fraction: float = CALIBRATION_FRACTION) -> bool:
    """True if an internal-development family is reserved for fusion fitting."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    digest = hashlib.md5(f"fusion:{family_id(image_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def split_calibration_dataset(
    dataset, id_column: str = "img_id", fraction: float = CALIBRATION_FRACTION
):
    """Split internal development into selection and fusion-calibration families."""
    options = {"input_columns": id_column, "fn_kwargs": {"fraction": fraction}}
    calibration = dataset.filter(is_fusion_calibration, **options)
    selection = dataset.filter(_is_selection, **options)
    return selection, calibration


def exclude_heldout_families(
    dataset,
    heldout_families: frozenset[str] | set[str],
    id_column: str = "img_id",
):
    """Remove training rows whose ID family occurs in any held-out partition.

    Uses only the ID column, so filtering does not decode the large image and
    mask payloads.  The held-out set should be made from the complete published
    validation split before it is divided into validation/calibration/test.
    """
    families = frozenset(map(str, heldout_families))
    return dataset.filter(
        _family_is_not_heldout,
        input_columns=id_column,
        fn_kwargs={"heldout_families": families},
    )


def _is_internal_dev(image_id: str, fraction: float) -> bool:
    return not is_internal_test(image_id, fraction)


def _is_selection(image_id: str, fraction: float) -> bool:
    return not is_fusion_calibration(image_id, fraction)


def _family_is_not_heldout(
    image_id: str, heldout_families: frozenset[str]
) -> bool:
    return family_id(image_id) not in heldout_families
