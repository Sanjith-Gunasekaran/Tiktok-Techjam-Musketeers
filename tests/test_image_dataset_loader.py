"""Tests for data_loader/image_dataset_loader.py against SID-Set's REAL
schema: label is a bare int64 (no ClassLabel names) and rows carry a heavy
unused mask column."""

import io
import shutil

import numpy as np
import pytest
from PIL import Image

datasets = pytest.importorskip("datasets")
from datasets import Dataset, Features, Value
from datasets import Image as HFImage

from data_loader import SID_SET_BINARY_LABEL_MAP, SID_SET_LABEL_NAMES, ImageDatasetLoader


def _png_bytes(seed):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (48, 48, 3)).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sid_like_dir(tmp_path):
    """A folder of parquet shards mimicking a real SID-Set download."""
    root = tmp_path / "data" / "data"
    root.mkdir(parents=True)

    def build(n, offset):
        return Dataset.from_dict(
            {
                "img_id": [f"id_{offset + i}" for i in range(n)],
                "image": [{"bytes": _png_bytes(offset + i), "path": None} for i in range(n)],
                "mask": [{"bytes": _png_bytes(1000 + offset + i), "path": None} for i in range(n)],
                "width": [48] * n,
                "height": [48] * n,
                "label": [(offset + i) % 3 for i in range(n)],
            },
            features=Features(
                {
                    "img_id": Value("string"),
                    "image": HFImage(),
                    "mask": HFImage(),
                    "width": Value("int64"),
                    "height": Value("int64"),
                    "label": Value("int64"),  # bare int, like the real dataset
                }
            ),
        )

    build(6, 0).to_parquet(root / "validation-00000-of-00002.parquet")
    build(6, 6).to_parquet(root / "validation-00001-of-00002.parquet")
    return tmp_path / "data"


def _loader(sid_like_dir, **kwargs):
    return ImageDatasetLoader(
        sid_like_dir,
        split="validation",
        label_map=SID_SET_BINARY_LABEL_MAP,
        label_names=SID_SET_LABEL_NAMES,
        **kwargs,
    )


def test_int64_labels_map_and_get_names_from_fallback(sid_like_dir):
    batch = _loader(sid_like_dir).get_batch(12, seed=0)
    for sample in batch:
        assert sample["label"] == (0 if sample["original_label"] == 0 else 1)
        assert sample["original_label_name"] == SID_SET_LABEL_NAMES[sample["original_label"]]


def test_metadata_is_opt_in_and_mask_is_dropped(sid_like_dir):
    assert all(s["metadata"] == {} for s in _loader(sid_like_dir).get_batch(4, seed=0))
    kept = _loader(sid_like_dir, metadata_columns=("width",)).get_batch(4, seed=0)
    assert all(set(s["metadata"]) == {"width"} for s in kept)


def test_duplicate_shards_raise(sid_like_dir):
    nested = sid_like_dir / "data" / "copy"
    nested.mkdir()
    shutil.copy(
        sid_like_dir / "data" / "validation-00000-of-00002.parquet",
        nested / "validation-00000-of-00002.parquet",
    )
    with pytest.raises(ValueError, match="Duplicate shard"):
        _loader(sid_like_dir)


def test_partial_download_warns(sid_like_dir):
    (sid_like_dir / "data" / "validation-00001-of-00002.parquet").unlink()
    with pytest.warns(UserWarning, match="partial"):
        _loader(sid_like_dir)


def test_missing_split_names_available_ones(sid_like_dir):
    with pytest.raises(FileNotFoundError, match="validation"):
        ImageDatasetLoader(sid_like_dir, split="test")
