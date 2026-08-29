"""Randomly load image batches from the local SID-Set Parquet shards."""

from __future__ import annotations

import argparse
import random
from bisect import bisect_right
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

# Works when imported as part of the data_loader package (the dotted form) and
# when this file is run directly as a script from inside this folder.
try:
    from .batch_loader import BatchLoader
except ImportError:
    from batch_loader import BatchLoader
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "data"
VALID_SPLITS = {"train", "validation", "all"}
IMAGE_COLUMNS = ["img_id", "image", "width", "height", "label"]


class SIDDataset(BatchLoader):
    """Load random SID-Set samples without loading the whole dataset."""

    def get_random_batch(
        self, batch_size: int, split: str, data_dir: Path, seed: int
    ) -> list[dict[str, Any]]:
        """Return ``batch_size`` randomly selected images and their metadata.

        The returned images are fully-loaded :class:`PIL.Image.Image` objects,
        so they remain usable after the underlying Parquet file is closed.
        Sampling is without replacement and is repeatable for a given seed.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if split not in VALID_SPLITS:
            choices = ", ".join(sorted(VALID_SPLITS))
            raise ValueError(f"split must be one of: {choices}")

        data_dir = Path(data_dir)
        split_names = ("train", "validation") if split == "all" else (split,)
        shard_paths = [
            path
            for split_name in split_names
            for path in sorted(data_dir.glob(f"{split_name}-*.parquet"))
        ]
        if not shard_paths:
            raise FileNotFoundError(
                f"No Parquet shards for split {split!r} were found in {data_dir}"
            )

        # The dataset is stored as several Parquet files ("shards"). Count the
        # rows in each shard so one global index (0..total) can locate any row.
        parquet_files = [pq.ParquetFile(path) for path in shard_paths]
        shard_ends: list[int] = []
        total_rows = 0
        for parquet_file in parquet_files:
            total_rows += parquet_file.metadata.num_rows
            shard_ends.append(total_rows)

        if batch_size > total_rows:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds the {total_rows} available images"
            )

        # Pick random positions, then group them by shard so each file is
        # opened and read only once.
        selected_indices = random.Random(seed).sample(range(total_rows), batch_size)
        rows_by_shard: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_position, dataset_index in enumerate(selected_indices):
            shard_index = bisect_right(shard_ends, dataset_index)
            shard_start = 0 if shard_index == 0 else shard_ends[shard_index - 1]
            rows_by_shard[shard_index].append(
                (dataset_index - shard_start, output_position)
            )

        batch: list[dict[str, Any] | None] = [None] * batch_size
        for shard_index, requested_rows in rows_by_shard.items():
            parquet_file = parquet_files[shard_index]
            # Parquet stores rows in blocks called "row groups". Work out which
            # block each wanted row lives in so only those blocks are read.
            row_group_ends: list[int] = []
            row_count = 0
            for group_index in range(parquet_file.metadata.num_row_groups):
                row_count += parquet_file.metadata.row_group(group_index).num_rows
                row_group_ends.append(row_count)

            rows_by_group: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for row_index, output_position in requested_rows:
                group_index = bisect_right(row_group_ends, row_index)
                group_start = 0 if group_index == 0 else row_group_ends[group_index - 1]
                rows_by_group[group_index].append(
                    (row_index - group_start, output_position)
                )

            for group_index, group_rows in rows_by_group.items():
                table = parquet_file.read_row_group(
                    group_index, columns=IMAGE_COLUMNS
                )
                for row_offset, output_position in group_rows:
                    values = {
                        column: table[column][row_offset].as_py()
                        for column in IMAGE_COLUMNS
                    }
                    image = self._decode_image(values["image"], data_dir)
                    width, height = image.size
                    batch[output_position] = {
                        "img_id": values["img_id"],
                        "image": image,
                        # Use the decoded image as the source of truth. This also
                        # works if dimensions are absent or stale in the dataset.
                        "width": width,
                        "height": height,
                        "label": int(values["label"]),
                        # Team decision: tampered (class 2) counts as AI, so
                        # every non-real class maps to the binary AI label 1.
                        "binary_label": int(int(values["label"]) != 0),
                    }

        return [sample for sample in batch if sample is not None]

    @staticmethod
    def _decode_image(value: Any, data_dir: Path) -> Image.Image:
        """Decode a Hugging Face image struct (or raw encoded image bytes)."""
        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            image_path = value.get("path")
        else:
            image_bytes = value
            image_path = None

        if image_bytes is not None:
            if isinstance(image_bytes, memoryview):
                image_bytes = image_bytes.tobytes()
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                return image.copy()

        if image_path:
            path = Path(image_path)
            if not path.is_absolute():
                path = data_dir / path
            with Image.open(path) as image:
                image.load()
                return image.copy()

        raise ValueError("Image record contains neither encoded bytes nor a path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="train")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch = SIDDataset().get_random_batch(
        args.batch_size, args.split, args.data_dir, args.seed
    )

    for sample in batch:
        print(
            f"{sample['img_id']}: {sample['width']}x{sample['height']}, "
            f"label={sample['label']}, binary_label={sample['binary_label']}"
        )


if __name__ == "__main__":
    main()
