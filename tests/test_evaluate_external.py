"""Tests for external benchmark discovery, labels, preprocessing, and reports."""

from __future__ import annotations

import csv
import json

import pytest
import torch
from PIL import Image
from torch import nn

from model_runs.evaluate_external import (
    ExternalImageDataset,
    parse_label,
    predict_records,
    records_from_directory,
    records_from_manifest,
    summarise,
    write_report,
)
from models import TwoBranchDetector


def _image(path, color=(20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 40), color).save(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (0, 0),
        ("0.0", 0),
        ("REAL", 0),
        ("real_images", 0),
        (1, 1),
        ("1.0", 1),
        ("fake", 1),
        ("AI-generated", 1),
        ("", None),
    ),
)
def test_parse_common_external_labels(raw, expected) -> None:
    assert parse_label(raw) == expected


def test_unknown_external_label_is_not_silently_guessed() -> None:
    with pytest.raises(ValueError, match="Unsupported label"):
        parse_label("tampered")


def test_directory_discovery_is_sorted_and_infers_class_directories(tmp_path) -> None:
    _image(tmp_path / "real" / "b.jpg")
    _image(tmp_path / "1_fake" / "a.png")
    _image(tmp_path / "unlabelled" / "c.webp")

    records = records_from_directory(tmp_path)

    assert [record.image_id for record in records] == [
        "1_fake/a.png",
        "real/b.jpg",
        "unlabelled/c.webp",
    ]
    assert [record.label for record in records] == [1, 0, None]


def test_csv_manifest_supports_relative_paths_and_optional_ids(tmp_path) -> None:
    _image(tmp_path / "images" / "one.jpg")
    _image(tmp_path / "images" / "two.jpg")
    manifest = tmp_path / "benchmark.csv"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("path", "label", "image_id"))
        writer.writeheader()
        writer.writerow({"path": "images/one.jpg", "label": "real", "image_id": "one"})
        writer.writerow({"path": "images/two.jpg", "label": "1", "image_id": ""})

    records = records_from_manifest(manifest)

    assert [record.image_id for record in records] == ["one", "images/two.jpg"]
    assert [record.label for record in records] == [0, 1]


def test_external_dataset_reuses_canonical_two_view_preprocessing(tmp_path) -> None:
    _image(tmp_path / "real" / "sample.png")
    record = records_from_directory(tmp_path)[0]

    dino, patch, label, image_id, _path = ExternalImageDataset([record])[0]

    assert dino.shape == (3, 224, 224)
    assert patch.shape == (3, 32, 32)
    assert label == 0
    assert image_id == "real/sample.png"


def test_prediction_path_runs_both_branches_and_fixed_fusion(tmp_path) -> None:
    class MeanBranch(nn.Module):
        def forward(self, images):
            return images.mean(dim=tuple(range(1, images.ndim)))

    _image(tmp_path / "real" / "one.png", color=(10, 10, 10))
    _image(tmp_path / "fake" / "two.png", color=(240, 240, 240))
    records = records_from_directory(tmp_path)
    model = TwoBranchDetector(
        MeanBranch(), MeanBranch(), fusion_mode="fixed", dino_weight=0.75
    )

    predictions = predict_records(
        model,
        records,
        device=torch.device("cpu"),
        threshold=0.5,
        batch_size=2,
        num_workers=0,
        amp=False,
    )

    assert [row["image_id"] for row in predictions] == ["fake/two.png", "real/one.png"]
    assert all(0.0 <= row["synthetic_probability"] <= 1.0 for row in predictions)
    assert all(isinstance(row["predicted_label"], int) for row in predictions)


def test_summary_and_reports_use_only_supplied_labels(tmp_path) -> None:
    predictions = [
        {
            "image_id": "r",
            "path": "r.jpg",
            "true_label": 0,
            "predicted_label": 0,
            "synthetic_probability": 0.1,
            "dino_logit": -2.0,
            "forensic_logit": -1.0,
            "correct": True,
        },
        {
            "image_id": "f",
            "path": "f.jpg",
            "true_label": 1,
            "predicted_label": 1,
            "synthetic_probability": 0.9,
            "dino_logit": 2.0,
            "forensic_logit": 1.0,
            "correct": True,
        },
        {
            "image_id": "u",
            "path": "u.jpg",
            "true_label": None,
            "predicted_label": 1,
            "synthetic_probability": 0.8,
            "dino_logit": 1.0,
            "forensic_logit": 1.0,
            "correct": None,
        },
    ]

    summary = summarise(predictions, 0.5)
    write_report(tmp_path, predictions, summary, {"format_version": 1})

    assert summary["samples"] == 3
    assert summary["labelled_samples"] == 2
    assert summary["accuracy"] == 1.0
    assert summary["auc"] == 1.0
    assert (tmp_path / "predictions.csv").is_file()
    submission = json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))
    assert submission == [
        {"image_path": "r", "pred": 0.1},
        {"image_path": "f", "pred": 0.9},
        {"image_path": "u", "pred": 0.8},
    ]
    assert all(set(row) == {"image_path", "pred"} for row in submission)
    assert (tmp_path / "errors.csv").read_text(encoding="utf-8").count("\n") == 1
    assert (tmp_path / "summary.md").is_file()
