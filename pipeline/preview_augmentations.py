"""Create a labelled visual preview of the robustness evaluation grid."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .augmentations import EVAL_GRID


Transform = Callable[[Image.Image], Image.Image]


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def create_preview(
    image_path: str | Path,
    output_path: str | Path,
    *,
    columns: int = 4,
    cell_size: int = 256,
    label_height: int = 32,
    transforms: Mapping[str, Transform] = EVAL_GRID,
) -> Path:
    """Save a labelled grid containing every requested transformation."""
    columns = _positive_integer(columns, "columns")
    cell_size = _positive_integer(cell_size, "cell_size")
    label_height = _positive_integer(label_height, "label_height")

    if not transforms:
        raise ValueError("transforms cannot be empty")

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    with Image.open(image_path) as opened_image:
        source = ImageOps.exif_transpose(opened_image).convert("RGB")
        source.load()

    transformed_images: list[tuple[str, Image.Image]] = []
    for name, transform in transforms.items():
        transformed = transform(source.copy())
        if not isinstance(transformed, Image.Image):
            raise TypeError(
                f"Transform {name!r} did not return a PIL image"
            )
        transformed_images.append(
            (str(name), transformed.convert("RGB"))
        )

    rows = math.ceil(len(transformed_images) / columns)
    cell_height = cell_size + label_height
    grid = Image.new(
        "RGB",
        (columns * cell_size, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(grid)

    for index, (name, transformed) in enumerate(
        transformed_images
    ):
        column = index % columns
        row = index // columns
        x = column * cell_size
        y = row * cell_height

        displayed = ImageOps.contain(
            transformed,
            (cell_size, cell_size),
            Image.Resampling.LANCZOS,
        )
        image_x = x + (cell_size - displayed.width) // 2
        image_y = y + (cell_size - displayed.height) // 2
        grid.paste(displayed, (image_x, image_y))
        draw.text(
            (x + 8, y + cell_size + 8),
            name,
            fill="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=95)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/augmentation_preview.jpg"),
        help="Output preview image path",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = create_preview(
        args.image,
        args.output,
        columns=args.columns,
        cell_size=args.cell_size,
    )
    print(f"Preview saved to {output_path}")


if __name__ == "__main__":
    main()