"""Tests for data_loader/image_dataset_loader.py against SID-Set's REAL
schema: label is a bare int64 (no ClassLabel names) and rows carry a heavy
unused mask column."""

import io
import csv
import gzip
import json
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

datasets = pytest.importorskip("datasets")
from datasets import Dataset, Features, Value
from datasets import Image as HFImage

from data_loader import (
    SID_SET_BINARY_LABEL_MAP,
    SID_SET_LABEL_NAMES,
    ImageDatasetLoader,
)
from data_loader.local_image_batch_loader import SIDDataset
from pipeline import (
    EVAL_GRID,
    BranchViewDataset,
    binary_auc,
    create_dataloaders,
    evaluate_model,
    two_views,
)


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
    build(12, 100).to_parquet(root / "train-00000-of-00001.parquet")
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
    loader = _loader(sid_like_dir)
    assert len(loader.dataset) == 8  # four tampered rows were removed
    batch = loader.get_batch(8, seed=0)
    for sample in batch:
        assert sample["original_label"] in (0, 1)
        assert sample["label"] == sample["original_label"]
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


def test_forgotten_label_map_warns_once(sid_like_dir):
    """Without a label_map, SID label 2 would reach a binary trainer."""
    loader = ImageDatasetLoader(sid_like_dir, split="validation")
    with pytest.warns(UserWarning, match="label_map"):
        loader.get_batch(12, seed=0)


def test_mapped_labels_are_binary_and_silent(sid_like_dir):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any label warning becomes a failure
        batch = _loader(sid_like_dir).get_batch(8, seed=0)
    assert {sample["label"] for sample in batch} <= {0, 1}
    assert {sample["original_label"] for sample in batch} <= {0, 1}


def test_local_preview_loader_excludes_tampered(sid_like_dir):
    batch = SIDDataset().get_random_batch(
        8, "validation", sid_like_dir / "data", seed=0
    )
    assert {sample["label"] for sample in batch} <= {0, 1}
    assert {sample["binary_label"] for sample in batch} <= {0, 1}


def test_branch_views_share_one_augmented_image(sid_like_dir):
    class SolidAugmentation:
        calls = 0

        def __call__(self, image):
            self.calls += 1
            return Image.new("RGB", image.size, (40, 80, 120))

    source = _loader(sid_like_dir)
    augmentation = SolidAugmentation()
    dataset = BranchViewDataset(source, augmentation=augmentation)
    actual = dataset[0][:2]
    expected = two_views(Image.new("RGB", (48, 48), (40, 80, 120)))
    assert augmentation.calls == 1
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_single_view_dataset_avoids_the_other_branch(sid_like_dir):
    source = _loader(sid_like_dir)

    dino = BranchViewDataset(source, view="dino")[0]
    patch = BranchViewDataset(source, view="forensic")[0]

    assert set(dino) == {"dino", "label", "original_label"}
    assert dino["dino"].shape == (3, 224, 224)
    assert set(patch) == {"patch", "label", "original_label"}
    assert patch["patch"].shape == (3, 32, 32)


def test_single_view_dataloader_collates_dictionary_batches(sid_like_dir):
    loaders = create_dataloaders(sid_like_dir, batch_size=2, view="forensic")

    batch = next(iter(loaders.train))

    assert set(batch) == {"patch", "label", "original_label"}
    assert batch["patch"].shape == (2, 3, 32, 32)


def test_dataloader_factory_builds_fixed_binary_splits(sid_like_dir):
    loaders = create_dataloaders(
        sid_like_dir,
        batch_size=4,
        num_workers=1,
        seed=7,
        test_fraction=0.5,
    )
    assert len(loaders.train.dataset) == 8
    assert len(loaders.validation.dataset) + len(loaders.test.dataset) == 8
    validation_ids = set(loaders.validation.dataset.rows["img_id"])
    test_ids = set(loaders.test.dataset.rows["img_id"])
    assert validation_ids.isdisjoint(test_ids)

    dino, patch, label, original_label = next(iter(loaders.train))
    assert dino.shape == (4, 3, 224, 224)
    assert patch.shape == (4, 3, 32, 32)
    assert set(label.tolist()) <= {0, 1}
    assert set(original_label.tolist()) <= {0, 1}


def test_binary_auc_handles_order_and_ties():
    labels = [0, 0, 1, 1]
    assert binary_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert binary_auc(labels, [0.5, 0.5, 0.5, 0.5]) == 0.5
    with pytest.raises(ValueError, match="both binary classes"):
        binary_auc([0, 0], [0.1, 0.2])


def test_evaluator_runs_full_pipeline_and_writes_reports(sid_like_dir, tmp_path):
    class FakeTwoBranchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = 0

        def forward(self, dino, patch):
            assert not torch.is_grad_enabled()
            assert dino.shape[1:] == (3, 224, 224)
            assert patch.shape[1:] == (3, 32, 32)
            self.seen += len(dino)
            signal = dino.mean(dim=(1, 2, 3)) + patch.mean(dim=(1, 2, 3)) / 255
            return torch.sigmoid(signal)

    loaders = create_dataloaders(
        sid_like_dir,
        batch_size=2,
        num_workers=1,
        seed=7,
        test_fraction=0.5,
    )
    model = FakeTwoBranchModel()
    model.train()
    report = evaluate_model(model, loaders, output_dir=tmp_path / "evaluation")

    assert model.training  # the evaluator restored the prior mode
    assert len(report.rows) == len(EVAL_GRID) == 17
    assert [row.transform for row in report.rows] == list(EVAL_GRID)
    assert all(row.samples == len(loaders.test.dataset) for row in report.rows)
    assert all(
        (
            row.true_positive
            + row.true_negative
            + row.false_positive
            + row.false_negative
            == row.samples
        )
        for row in report.rows
    )
    assert all(np.isfinite([row.accuracy, row.auc]).all() for row in report.rows)
    assert model.seen == len(EVAL_GRID) * len(loaders.test.dataset)

    with report.csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["transform"] for row in csv_rows] == list(EVAL_GRID)
    markdown = report.markdown_path.read_text(encoding="utf-8")
    assert all(name in markdown for name in EVAL_GRID)
    assert report.error_path.is_file()
    assert report.predictions_path is None


def test_evaluator_accuracy_matches_class_counts_for_a_constant_model(
    sid_like_dir, tmp_path
):
    """A model that always predicts one class must score exactly that class's
    share of the test set — the most direct check that accuracy is computed
    correctly, without needing to hand-derive numbers from the hashed split."""

    class ConstantModel(torch.nn.Module):
        def __init__(self, score):
            super().__init__()
            self.score = score

        def forward(self, dino, patch):
            return torch.full((len(dino),), self.score)

    loaders = create_dataloaders(sid_like_dir, batch_size=4, test_fraction=0.5)
    always_synthetic = evaluate_model(
        ConstantModel(0.9), loaders, output_dir=tmp_path / "a"
    ).rows[0]
    always_real = evaluate_model(
        ConstantModel(0.1), loaders, output_dir=tmp_path / "b"
    ).rows[0]
    assert always_synthetic.accuracy == always_synthetic.synthetic / always_synthetic.samples
    assert always_real.accuracy == always_real.real / always_real.samples
    assert always_synthetic.recall == 1.0
    assert always_synthetic.precision == always_synthetic.synthetic / always_synthetic.samples
    assert always_real.precision == always_real.recall == always_real.f1 == 0.0


def test_evaluator_exports_error_only_and_optional_predictions(sid_like_dir, tmp_path):
    class AlwaysSynthetic(torch.nn.Module):
        def forward(self, dino, patch):
            return torch.full((len(dino),), 0.9)

    loaders = create_dataloaders(sid_like_dir, batch_size=4, test_fraction=0.5)
    report = evaluate_model(
        AlwaysSynthetic(),
        loaders,
        output_dir=tmp_path,
        transforms={"clean": EVAL_GRID["clean"]},
        save_predictions=True,
    )
    row = report.rows[0]
    assert row.true_positive == row.synthetic
    assert row.false_positive == row.real
    assert row.true_negative == row.false_negative == 0
    assert report.predictions_path is not None

    with report.error_path.open(newline="", encoding="utf-8") as handle:
        errors = list(csv.DictReader(handle))
    assert len(errors) == row.false_positive
    assert {error["error_type"] for error in errors} == {"false_positive"}
    assert {error["image_id"] for error in errors} <= set(
        loaders.test.dataset.rows["img_id"]
    )

    with gzip.open(report.predictions_path, "rt", encoding="utf-8") as handle:
        predictions = [json.loads(line) for line in handle]
    assert len(predictions) == row.samples
    assert {prediction["image_id"] for prediction in predictions} == set(
        loaders.test.dataset.rows["img_id"]
    )


def test_evaluator_requires_a_clean_cell(sid_like_dir, tmp_path):
    loaders = create_dataloaders(sid_like_dir, batch_size=4, test_fraction=0.5)
    grid_without_clean = {k: v for k, v in EVAL_GRID.items() if k != "clean"}
    with pytest.raises(ValueError, match="clean"):
        evaluate_model(
            torch.nn.Linear(1, 1),
            loaders,
            output_dir=tmp_path,
            transforms=grid_without_clean,
        )


def test_evaluator_rejects_an_already_augmented_test_dataset(sid_like_dir, tmp_path):
    from pipeline import RandomAugment

    loaders = create_dataloaders(sid_like_dir, batch_size=4, test_fraction=0.5)
    loaders.test.dataset.augmentation = RandomAugment(1.0, seed=0)
    with pytest.raises(ValueError, match="frozen"):
        evaluate_model(lambda d, p: torch.zeros(len(d)), loaders, output_dir=tmp_path)


def test_evaluator_rejects_out_of_range_probabilities(sid_like_dir, tmp_path):
    model = lambda dino, patch: torch.full((len(dino),), 1.5)  # not a probability
    loaders = create_dataloaders(sid_like_dir, batch_size=4, test_fraction=0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_model(model, loaders, output_dir=tmp_path)
