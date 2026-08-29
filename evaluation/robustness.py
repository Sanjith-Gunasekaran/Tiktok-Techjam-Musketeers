"""Run clean and transformed robustness evaluation for any image model."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .augmentations import (
    center_crop_restore,
    color_jitter,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    resize_restore,
)
from .metrics import evaluate_condition, find_errors


PredictFunction = Callable[[Sequence[Image.Image]], Sequence[float]]
TransformFunction = Callable[[Image.Image, int], Image.Image]


@dataclass(frozen=True)
class RobustnessCondition:
    """One named image condition used during robustness evaluation."""

    name: str
    transform: TransformFunction


def _clean(image: Image.Image, seed: int) -> Image.Image:
    del seed
    return image.convert("RGB").copy()


def _jpeg(quality: int) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        del seed
        return jpeg_compress(image, quality)

    return apply


def _blur(sigma: float) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        del seed
        return gaussian_blur(image, sigma)

    return apply


def _resize(scale: float) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        del seed
        return resize_restore(image, scale)

    return apply


def _noise(sigma: float) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        return gaussian_noise(image, sigma, seed=seed)

    return apply


def _brightness(factor: float) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        del seed
        return color_jitter(image, brightness=factor)

    return apply


def _contrast(factor: float) -> TransformFunction:
    def apply(image: Image.Image, seed: int) -> Image.Image:
        del seed
        return color_jitter(image, contrast=factor)

    return apply


def _crop(image: Image.Image, seed: int) -> Image.Image:
    del seed
    return center_crop_restore(image, retain=0.8)


DEFAULT_CONDITIONS = (
    RobustnessCondition("clean", _clean),
    RobustnessCondition("jpeg_quality_90", _jpeg(90)),
    RobustnessCondition("jpeg_quality_70", _jpeg(70)),
    RobustnessCondition("jpeg_quality_50", _jpeg(50)),
    RobustnessCondition("jpeg_quality_30", _jpeg(30)),
    RobustnessCondition("blur_sigma_0.5", _blur(0.5)),
    RobustnessCondition("blur_sigma_1.0", _blur(1.0)),
    RobustnessCondition("blur_sigma_2.0", _blur(2.0)),
    RobustnessCondition("resize_0.5x", _resize(0.5)),
    RobustnessCondition("resize_0.25x", _resize(0.25)),
    RobustnessCondition("noise_sigma_0.02", _noise(0.02)),
    RobustnessCondition("noise_sigma_0.05", _noise(0.05)),
    RobustnessCondition("noise_sigma_0.10", _noise(0.10)),
    RobustnessCondition("brightness_-20_percent", _brightness(0.8)),
    RobustnessCondition("brightness_+20_percent", _brightness(1.2)),
    RobustnessCondition("contrast_-20_percent", _contrast(0.8)),
    RobustnessCondition("contrast_+20_percent", _contrast(1.2)),
    RobustnessCondition("center_crop_80_percent", _crop),
)


def _predict_in_batches(
    images: Sequence[Image.Image],
    predict: PredictFunction,
    batch_size: int,
) -> list[float]:
    """Predict scores without sending every image to the model at once."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    all_scores: list[float] = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        batch_scores = np.asarray(list(predict(batch)), dtype=float)

        if batch_scores.ndim != 1:
            raise ValueError("predict must return one score per image")
        if len(batch_scores) != len(batch):
            raise ValueError("predict returned the wrong number of scores")
        if not np.all(np.isfinite(batch_scores)):
            raise ValueError("predict returned a non-finite score")

        all_scores.extend(float(score) for score in batch_scores)

    return all_scores


def evaluate_robustness(
    images: Sequence[Image.Image],
    labels: Iterable[int],
    predict: PredictFunction,
    *,
    image_paths: Sequence[str] | None = None,
    conditions: Sequence[RobustnessCondition] = DEFAULT_CONDITIONS,
    batch_size: int = 32,
    threshold: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a prediction function on clean and transformed images.

    ``predict`` receives a batch of PIL images and returns one AI-confidence
    score for each image. A higher score means the image is more likely AI.
    """
    image_list = list(images)
    label_list = list(labels)

    if not image_list:
        raise ValueError("images cannot be empty")
    if len(image_list) != len(label_list):
        raise ValueError("images and labels must have the same length")
    if not conditions:
        raise ValueError("at least one robustness condition is required")

    if image_paths is None:
        path_list = [
            f"sample_{index:06d}"
            for index in range(len(image_list))
        ]
    else:
        path_list = [str(path) for path in image_paths]
        if len(path_list) != len(image_list):
            raise ValueError(
                "image_paths, images and labels must have the same length"
            )

    summary: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    errors: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for condition in conditions:
        transformed_images = [
            condition.transform(image, seed + index)
            for index, image in enumerate(image_list)
        ]

        scores = _predict_in_batches(
            transformed_images,
            predict,
            batch_size,
        )

        summary.append(
            evaluate_condition(
                condition.name,
                label_list,
                scores,
                threshold,
            )
        )

        errors[condition.name] = find_errors(
            path_list,
            label_list,
            scores,
            threshold,
        )

        predictions.extend(
            {
                "condition": condition.name,
                "image_path": image_path,
                "label": int(label),
                "pred": float(score),
            }
            for image_path, label, score in zip(
                path_list,
                label_list,
                scores,
            )
        )

    return {
        "summary": summary,
        "predictions": predictions,
        "errors": errors,
    }