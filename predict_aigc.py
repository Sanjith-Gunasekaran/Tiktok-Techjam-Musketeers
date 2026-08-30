"""Score every image in a directory with a trained DINOv2 AIGC detector.

Example:
    python predict_aigc.py ./images --checkpoint best_dinov2_model.pt \
        --output predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from aigc_detector import load_detector_checkpoint


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class InferenceImageDataset(Dataset):
    """Load an unlabeled directory while retaining paths for JSON output."""

    def __init__(
        self,
        root: Path,
        processor: Any,
        recursive: bool = True,
        max_images: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = root.resolve()
        self.processor = processor
        iterator = self.root.rglob("*") if recursive else self.root.glob("*")
        paths = sorted(
            path.resolve()
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            scope = " or its subdirectories" if recursive else ""
            raise FileNotFoundError(
                f"No supported images found in {self.root}{scope}"
            )
        if max_images is not None:
            if max_images <= 0:
                raise ValueError("max_images must be greater than zero")
            if max_images < len(paths):
                paths = sorted(random.Random(seed).sample(paths, max_images))
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        path = self.paths[index]
        try:
            with Image.open(path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not decode image: {path}") from exc

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0)
        relative_path = path.relative_to(self.root).as_posix()
        return relative_path, pixel_values


def predict_directory(
    input_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 16,
    device_name: str = "auto",
    num_workers: int = 0,
    recursive: bool = True,
    max_images: int | None = None,
    seed: int = 42,
    model_id: str | None = None,
    ai_class_index: int | None = None,
) -> list[dict[str, str | float]]:
    """Run inference and write ``[{image_path, pred}, ...]`` as JSON."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    device = _resolve_device(device_name)
    print(f"Using device: {device}")
    model, resolved_model_id, resolved_ai_index = load_detector_checkpoint(
        checkpoint_path,
        device,
        model_id=model_id,
        ai_class_index=ai_class_index,
    )
    processor = AutoImageProcessor.from_pretrained(resolved_model_id)
    dataset = InferenceImageDataset(
        input_dir,
        processor,
        recursive=recursive,
        max_images=max_images,
        seed=seed,
    )
    print(f"Found {len(dataset)} images to score")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    predictions: list[dict[str, str | float]] = []
    started_at = time.perf_counter()
    with torch.inference_mode():
        for paths, pixel_values in loader:
            pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
            probabilities = torch.softmax(model(pixel_values), dim=1)
            ai_probabilities = probabilities[:, resolved_ai_index].cpu().tolist()
            predictions.extend(
                {"image_path": path, "pred": float(probability)}
                for path, probability in zip(paths, ai_probabilities)
            )
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            rate = len(predictions) / elapsed
            remaining = (len(dataset) - len(predictions)) / max(rate, 1e-9)
            print(
                f"Scored {len(predictions)}/{len(dataset)} images "
                f"({rate:.1f} images/s, ETA {_format_duration(remaining)})",
                end="\r",
            )
    print()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, output_path)
    return predictions


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory containing images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-images",
        type=int,
        help="score a deterministic random subset instead of the full directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed used when selecting a --max-images subset",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="do not scan image subdirectories",
    )
    parser.add_argument(
        "--model-id",
        help="override the DINOv2 model ID stored in the checkpoint",
    )
    parser.add_argument(
        "--ai-class-index",
        type=int,
        choices=(0, 1),
        help="override which output logit represents AI-generated images",
    )
    args = parser.parse_args()

    predictions = predict_directory(
        args.input_dir,
        args.checkpoint,
        args.output,
        batch_size=args.batch_size,
        device_name=args.device,
        num_workers=args.num_workers,
        recursive=not args.no_recursive,
        max_images=args.max_images,
        seed=args.seed,
        model_id=args.model_id,
        ai_class_index=args.ai_class_index,
    )
    print(f"Wrote {len(predictions)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
