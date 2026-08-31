"""Run the final detector on an image directory and write JSON predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError

from models.dino_classifier import DINOClassifier
from models.forensic_cnn import ForensicCNN
from models.two_branch_detector import TwoBranchDetector
from pipeline.preprocess import two_views
from pipeline.canonicalize import canonicalize_encoding


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_CHECKPOINT = Path("model_runs/checkpoints/fusion/best.pt")
DEFAULT_OUTPUT = Path("predictions.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing images; subdirectories are included.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Final fusion checkpoint produced by model_runs.train.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON file.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def choose_device(name: str = "auto") -> torch.device:
    """Select an available inference device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if name == "mps" and not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")

    return torch.device(name)


def _read_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a supported checkpoint")

    if checkpoint.get("stage") != "fusion":
        raise ValueError(
            f"{path} is a {checkpoint.get('stage')!r} checkpoint; "
            "expected a 'fusion' checkpoint"
        )

    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"{path} has no model_state_dict")

    if not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError(f"{path} has no model_config")

    return checkpoint


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> TwoBranchDetector:
    """Rebuild and load the final two-branch fusion model."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _read_checkpoint(checkpoint_path, device)
    config = checkpoint["model_config"]

    dino_config = config.get("dino")
    forensic_config = config.get("forensic")

    if not isinstance(dino_config, dict):
        raise ValueError(
            f"{checkpoint_path} has no DINO branch configuration"
        )
    if not isinstance(forensic_config, dict):
        raise ValueError(
            f"{checkpoint_path} has no forensic branch configuration"
        )

    dino = DINOClassifier(
        model_name=dino_config.get(
            "model_name",
            "facebook/dinov2-small",
        ),
        revision=dino_config.get("revision"),
        hidden_dim=dino_config.get("hidden_dim", 256),
        dropout=dino_config.get("dropout", 0.2),
        freeze_backbone=True,
    )
    forensic = ForensicCNN(
        dropout=forensic_config.get("dropout", 0.2),
        srm_clip_value=forensic_config.get("srm_clip_value"),
    )
    model = TwoBranchDetector(
        dino,
        forensic,
        fusion_mode=config.get("fusion_mode", "learned"),
        dino_weight=float(config.get("dino_weight", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def find_images(image_dir: str | Path) -> list[Path]:
    """Return supported images recursively in deterministic order."""
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"Expected an image directory: {image_dir}"
        )

    paths = sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )

    if not paths:
        raise ValueError(f"No supported images found in {image_dir}")

    return paths


def predict_paths(
    model: TwoBranchDetector,
    image_paths: Sequence[Path],
    device: torch.device,
    *,
    batch_size: int = 16,
) -> list[dict[str, str | float]]:
    """Return one AI-confidence score for every image path."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    model.eval()
    predictions: list[dict[str, str | float]] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        dino_images: list[torch.Tensor] = []
        patches: list[torch.Tensor] = []

        for path in batch_paths:
            try:
                with Image.open(path) as image:
                    image.load()
                    canonical_image = canonicalize_encoding(image)
                    dino_image, patch = two_views(canonical_image)

            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(
                    f"Could not decode image: {path}"
                ) from error

            dino_images.append(dino_image)
            patches.append(patch)

        dino_batch = torch.stack(dino_images).to(device)
        patch_batch = torch.stack(patches).to(device)

        with torch.inference_mode():
            probabilities = model.predict_proba(
                dino_batch,
                patch_batch,
            )

        if probabilities.ndim != 1:
            raise ValueError(
                "Model probabilities must have shape (batch_size,)"
            )
        if len(probabilities) != len(batch_paths):
            raise ValueError(
                "Model returned the wrong number of predictions"
            )

        for path, probability in zip(
            batch_paths,
            probabilities.detach().cpu().tolist(),
        ):
            score = float(probability)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Invalid probability {score!r} for {path}"
                )
            predictions.append(
                {
                    "image_path": str(path),
                    "pred": score,
                }
            )

    return predictions


def write_predictions(
    predictions: Sequence[dict[str, str | float]],
    output_path: str | Path,
) -> Path:
    """Write predictions as a readable JSON array."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            list(predictions),
            output_file,
            indent=2,
            ensure_ascii=False,
        )
        output_file.write("\n")

    return output_path


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    image_paths = find_images(args.image_dir)
    predictions = predict_paths(
        model,
        image_paths,
        device,
        batch_size=args.batch_size,
    )
    output_path = write_predictions(predictions, args.output)
    print(
        f"Wrote {len(predictions)} predictions to "
        f"{output_path} using {device}."
    )


if __name__ == "__main__":
    main()