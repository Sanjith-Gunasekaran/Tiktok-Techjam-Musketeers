"""Evaluate the frozen two-branch detector on an external image benchmark.

The benchmark may be a directory tree or a CSV manifest. Directory labels are
inferred from standard class-directory names such as ``real`` and ``fake``;
images without an inferable label are still scored and written to predictions.
No threshold, fusion weight, or model parameter is fitted by this command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_runs.train import choose_device, dino_from_checkpoint, forensic_from_checkpoint
from models import TwoBranchDetector
from pipeline import binary_auc, canonicalize_encoding, two_views

DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)
REAL_LABELS = frozenset(
    {
        "0",
        "0real",
        "real",
        "reals",
        "realimage",
        "realimages",
        "authentic",
        "authenticimages",
        "natural",
        "camera",
        "human",
        "nonsynthetic",
    }
)
SYNTHETIC_LABELS = frozenset(
    {
        "1",
        "1fake",
        "fake",
        "fakes",
        "fakeimage",
        "fakeimages",
        "synthetic",
        "synthetics",
        "syntheticimages",
        "generated",
        "generatedimages",
        "ai",
        "aiimages",
        "aigenerated",
        "fullsynthetic",
    }
)


@dataclass(frozen=True)
class ExternalRecord:
    image_id: str
    path: Path
    label: int | None = None


class ExternalImageDataset(Dataset):
    """Decode external files and create the exact two training-time views."""

    def __init__(self, records: list[ExternalRecord]) -> None:
        if not records:
            raise ValueError("External benchmark contains no images")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record.path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Could not decode external image {record.path}") from error
        dino, patch = two_views(canonicalize_encoding(image))
        label = -1 if record.label is None else record.label
        return dino, patch, label, record.image_id, str(record.path)


def _normalise_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def parse_label(value: Any) -> int | None:
    """Parse common external-benchmark labels without guessing unknown names."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    raw = str(value).strip().lower()
    if re.fullmatch(r"0\.0+", raw):
        return 0
    if re.fullmatch(r"1\.0+", raw):
        return 1
    normalised = _normalise_label(value)
    if not normalised:
        return None
    if normalised in REAL_LABELS:
        return 0
    if normalised in SYNTHETIC_LABELS:
        return 1
    raise ValueError(
        f"Unsupported label {value!r}; expected 0/1 or a standard real/fake class name"
    )


def infer_label(path: Path, root: Path) -> int | None:
    """Infer a label from the nearest recognised class directory."""
    relative = path.relative_to(root)
    for part in reversed(relative.parts[:-1]):
        try:
            label = parse_label(part)
        except ValueError:
            continue
        if label is not None:
            return label
    return None


def records_from_directory(
    root: Path,
    *,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    infer_labels: bool = True,
) -> list[ExternalRecord]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"External input directory does not exist: {root}")
    allowed = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )
    if not paths:
        raise ValueError(f"No supported images found below {root}")
    return [
        ExternalRecord(
            image_id=path.relative_to(root).as_posix(),
            path=path,
            label=infer_label(path, root) if infer_labels else None,
        )
        for path in paths
    ]


def records_from_manifest(
    manifest: Path,
    *,
    image_root: Path | None = None,
    path_column: str = "path",
    label_column: str = "label",
    id_column: str = "image_id",
) -> list[ExternalRecord]:
    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    root = manifest.parent if image_root is None else image_root.expanduser().resolve()
    records: list[ExternalRecord] = []
    with manifest.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or path_column not in reader.fieldnames:
            raise ValueError(f"Manifest must contain a {path_column!r} column")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get(path_column) or "").strip()
            if not raw_path:
                raise ValueError(f"Manifest row {row_number} has no image path")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Manifest row {row_number} image does not exist: {path}")
            raw_id = (row.get(id_column) or "").strip()
            image_id = raw_id or raw_path
            raw_label = row.get(label_column) if label_column in row else None
            try:
                label = parse_label(raw_label)
            except ValueError as error:
                raise ValueError(f"Manifest row {row_number}: {error}") from error
            records.append(ExternalRecord(image_id=image_id, path=path, label=label))
    if not records:
        raise ValueError("Manifest contains no image records")
    if len({record.image_id for record in records}) != len(records):
        raise ValueError("Manifest image IDs must be unique")
    return records


def _resolve_checkpoint(
    explicit: Path | None,
    stored: Any,
    fusion_path: Path,
    stage: str,
) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    if isinstance(stored, str) and stored:
        stored_path = Path(stored).expanduser()
        candidates.extend((stored_path, fusion_path.parent / stored_path))
        candidates.extend(
            (
                fusion_path.parent / stage / stored_path.name,
                fusion_path.parent.parent / stage / stored_path.name,
            )
        )
    candidates.extend(
        (
            fusion_path.parent / stage / "best.pt",
            fusion_path.parent.parent / stage / "best.pt",
        )
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    flag = "--dino-checkpoint" if stage == "dino_finetune" else "--forensic-checkpoint"
    raise FileNotFoundError(
        f"Could not find the {stage} checkpoint. Supply its current location with {flag}."
    )


def load_frozen_model(
    fusion_path: Path,
    device: torch.device,
    *,
    dino_checkpoint: Path | None = None,
    forensic_checkpoint: Path | None = None,
) -> tuple[TwoBranchDetector, float, Path, Path, dict[str, Any]]:
    fusion_path = fusion_path.expanduser().resolve()
    try:
        config = json.loads(fusion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read fusion configuration {fusion_path}") from error
    weight, threshold = config.get("dino_weight"), config.get("threshold")
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Fusion configuration has no valid dino_weight")
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Fusion configuration has no valid probability threshold")
    dino_path = _resolve_checkpoint(
        dino_checkpoint, config.get("dino_checkpoint"), fusion_path, "dino_finetune"
    )
    forensic_path = _resolve_checkpoint(
        forensic_checkpoint, config.get("forensic_checkpoint"), fusion_path, "forensic"
    )
    dino = dino_from_checkpoint(dino_path, device, ("dino_head", "dino_finetune"))
    forensic = forensic_from_checkpoint(forensic_path, device)
    model = TwoBranchDetector(
        dino, forensic, fusion_mode="fixed", dino_weight=float(weight)
    ).to(device)
    model.freeze_branches()
    model.eval()
    return model, float(threshold), dino_path, forensic_path, config


def predict_records(
    model: TwoBranchDetector,
    records: list[ExternalRecord],
    *,
    device: torch.device,
    threshold: float,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        ExternalImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for dino, patch, labels, image_ids, paths in loader:
            dino = dino.to(device, non_blocking=True)
            patch = patch.to(device, non_blocking=True)
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if amp and device.type == "cuda"
                else nullcontext()
            )
            with context:
                dino_logits, forensic_logits = model.branch_logits(dino, patch)
                weight = model.dino_weight.to(dino_logits.dtype)
                logits = weight * dino_logits + (1.0 - weight) * forensic_logits
            probabilities = torch.sigmoid(logits.float()).cpu().tolist()
            dino_values = dino_logits.float().cpu().tolist()
            forensic_values = forensic_logits.float().cpu().tolist()
            for image_id, path, label, probability, dino_logit, forensic_logit in zip(
                image_ids,
                paths,
                labels.tolist(),
                probabilities,
                dino_values,
                forensic_values,
            ):
                true_label = None if label < 0 else int(label)
                predicted_label = int(probability >= threshold)
                predictions.append(
                    {
                        "image_id": image_id,
                        "path": path,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "synthetic_probability": probability,
                        "dino_logit": dino_logit,
                        "forensic_logit": forensic_logit,
                        "correct": None if true_label is None else predicted_label == true_label,
                    }
                )
    return predictions


def summarise(predictions: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labelled = [row for row in predictions if row["true_label"] is not None]
    summary: dict[str, Any] = {
        "samples": len(predictions),
        "labelled_samples": len(labelled),
        "unlabelled_samples": len(predictions) - len(labelled),
        "threshold": threshold,
    }
    if not labelled:
        return summary
    labels = np.asarray([row["true_label"] for row in labelled], dtype=np.int64)
    predicted = np.asarray([row["predicted_label"] for row in labelled], dtype=np.int64)
    scores = np.asarray([row["synthetic_probability"] for row in labelled])
    tp = int(((labels == 1) & (predicted == 1)).sum())
    tn = int(((labels == 0) & (predicted == 0)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    summary.update(
        {
            "real_samples": int((labels == 0).sum()),
            "synthetic_samples": int((labels == 1).sum()),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "accuracy": (tp + tn) / len(labels),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "auc": binary_auc(labels, scores) if set(labels.tolist()) == {0, 1} else None,
        }
    )
    return summary


def write_report(
    output_dir: Path,
    predictions: list[dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "image_id",
        "path",
        "true_label",
        "predicted_label",
        "synthetic_probability",
        "dino_logit",
        "forensic_logit",
        "correct",
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    with (output_dir / "errors.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in predictions if row["correct"] is False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# External benchmark",
        "",
        f"- Samples: {summary['samples']}",
        f"- Labelled: {summary['labelled_samples']}",
    ]
    if "accuracy" in summary:
        auc = "n/a" if summary["auc"] is None else f"{summary['auc']:.4f}"
        lines.extend(
            (
                f"- Accuracy: {summary['accuracy']:.4f}",
                f"- Precision: {summary['precision']:.4f}",
                f"- Recall: {summary['recall']:.4f}",
                f"- F1: {summary['f1']:.4f}",
                f"- AUC: {auc}",
                "",
                "| TP | TN | FP | FN |",
                "| --: | --: | --: | --: |",
                f"| {summary['true_positive']} | {summary['true_negative']} | "
                f"{summary['false_positive']} | {summary['false_negative']} |",
            )
        )
    else:
        lines.extend(("", "No labels were supplied; predictions were generated without metrics."))
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Recursively scan an image directory")
    source.add_argument(
        "--manifest", type=Path, help="CSV containing image paths and optional labels"
    )
    parser.add_argument(
        "--image-root", type=Path, help="Base directory for relative manifest paths"
    )
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column", default="image_id")
    parser.add_argument("--no-infer-labels", action="store_true")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--fusion-config", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--forensic-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers cannot be negative")
    if args.manifest is not None:
        records = records_from_manifest(
            args.manifest,
            image_root=args.image_root,
            path_column=args.path_column,
            label_column=args.label_column,
            id_column=args.id_column,
        )
    else:
        extensions = tuple(item.strip() for item in args.extensions.split(",") if item.strip())
        records = records_from_directory(
            args.input_dir, extensions=extensions, infer_labels=not args.no_infer_labels
        )
    device = choose_device()
    model, threshold, dino_path, forensic_path, fusion = load_frozen_model(
        args.fusion_config,
        device,
        dino_checkpoint=args.dino_checkpoint,
        forensic_checkpoint=args.forensic_checkpoint,
    )
    start = time.perf_counter()
    predictions = predict_records(
        model,
        records,
        device=device,
        threshold=threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=device.type == "cuda" and not args.no_amp,
    )
    summary = summarise(predictions, threshold)
    summary["duration_seconds"] = time.perf_counter() - start
    metadata = {
        "format_version": 1,
        "benchmark_source": str(
            (args.manifest if args.manifest is not None else args.input_dir).resolve()
        ),
        "fusion_config": str(args.fusion_config.resolve()),
        "dino_checkpoint": str(dino_path),
        "forensic_checkpoint": str(forensic_path),
        "dino_weight": fusion["dino_weight"],
        "threshold": threshold,
        "device": str(device),
        "canonical_jpeg": True,
        "augmentations": False,
    }
    write_report(args.output_dir, predictions, summary, metadata)
    print(
        f"evaluated {summary['samples']} image(s) on {device} | "
        f"accuracy {summary.get('accuracy', 'n/a')} | auc {summary.get('auc', 'n/a')} | "
        f"report {args.output_dir / 'summary.md'}"
    )


if __name__ == "__main__":
    main()
