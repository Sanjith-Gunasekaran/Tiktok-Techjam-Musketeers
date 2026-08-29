"""Create a labelled preview of every robustness transformation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageOps

from evaluation.augmentations import (
    center_crop_restore,
    color_jitter,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    resize_restore,
)


Transform = tuple[str, Callable[[Image.Image], Image.Image]]


def build_transforms() -> list[Transform]:
    """Return the required evaluation transformations and severities."""
    return [
        ("Clean", lambda image: image.copy()),
        ("JPEG 90", lambda image: jpeg_compress(image, 90)),
        ("JPEG 70", lambda image: jpeg_compress(image, 70)),
        ("JPEG 50", lambda image: jpeg_compress(image, 50)),
        ("JPEG 30", lambda image: jpeg_compress(image, 30)),
        ("Blur 0.5", lambda image: gaussian_blur(image, 0.5)),
        ("Blur 1.0", lambda image: gaussian_blur(image, 1.0)),
        ("Blur 2.0", lambda image: gaussian_blur(image, 2.0)),
        ("Resize 0.5x", lambda image: resize_restore(image, 0.5)),
        ("Resize 0.25x", lambda image: resize_restore(image, 0.25)),
        ("Noise 0.02", lambda image: gaussian_noise(image, 0.02, seed=42)),
        ("Noise 0.05", lambda image: gaussian_noise(image, 0.05, seed=42)),
        ("Noise 0.10", lambda image: gaussian_noise(image, 0.10, seed=42)),
        (
            "Brightness -20%",
            lambda image: color_jitter(image, brightness=0.8),
        ),
        (
            "Brightness +20%",
            lambda image: color_jitter(image, brightness=1.2),
        ),
        (
            "Contrast -20%",
            lambda image: color_jitter(image, contrast=0.8),
        ),
        (
            "Contrast +20%",
            lambda image: color_jitter(image, contrast=1.2),
        ),
        ("Center crop 80%", lambda image: center_crop_restore(image, 0.8)),
    ]


def create_preview(
    image_path: Path,
    output_path: Path,
    columns: int = 3,
) -> None:
    """Save a labelled grid containing the clean and transformed images."""
    with Image.open(image_path) as opened_image:
        source = opened_image.convert("RGB")

    transformed_images = [
        (name, transform(source))
        for name, transform in build_transforms()
    ]

    image_width = 256
    image_height = 256
    label_height = 32
    cell_width = image_width
    cell_height = image_height + label_height
    rows = math.ceil(len(transformed_images) / columns)

    grid = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(grid)

    for index, (name, transformed) in enumerate(transformed_images):
        column = index % columns
        row = index // columns
        x = column * cell_width
        y = row * cell_height

        displayed = ImageOps.contain(
            transformed,
            (image_width, image_height),
            Image.Resampling.LANCZOS,
        )
        image_x = x + (image_width - displayed.width) // 2
        image_y = y + (image_height - displayed.height) // 2
        grid.paste(displayed, (image_x, image_y))

        draw.text(
            (x + 8, y + image_height + 8),
            name,
            fill="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=95)
    print(f"Preview saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("augmentation_preview.jpg"),
        help="Output preview path",
    )
    args = parser.parse_args()

    create_preview(args.image, args.output)


if __name__ == "__main__":
    main()
    