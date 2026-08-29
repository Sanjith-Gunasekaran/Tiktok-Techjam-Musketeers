"""Load batches of images from local, Hugging Face, or Kaggle datasets.

The loader deliberately uses Hugging Face ``datasets`` as the common backend:
local and downloaded Kaggle image folders are opened with ``ImageFolder``, while
Hub datasets are opened directly.  Source-specific code is therefore kept out
of the training loop.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Iterator, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import Dataset, IterableDataset, load_dataset
from PIL import Image


IMAGE_COLUMN_CANDIDATES = ("image", "img", "picture", "photo")
LABEL_COLUMN_CANDIDATES = ("label", "labels", "class", "target")

# Team decision: the model is binary (0 = real, 1 = AI) and tampered images
# count as AI. SID-Set has three classes (0 = real, 1 = fully synthetic,
# 2 = tampered), so classes 1 and 2 both map to the AI label. If the dataset's
# class order ever looks different, check original_label_name in a preview
# batch before training.
SID_SET_BINARY_LABEL_MAP = {0: 0, 1: 1, 2: 1}

# SID-Set stores label as a bare integer with no names attached (checked
# against the dataset's config.json), so readable names must be supplied by
# the caller via label_names=. These are SID-Set's classes in order.
SID_SET_LABEL_NAMES = ("real", "full_synthetic", "tampered")


class ImageDatasetLoader:
    """Present different image dataset sources through one batching API.

    Supported ``source`` values:

    * a local directory, such as ``./data/archive``;
    * a Hugging Face ID or URI, such as ``saberzl/SID_Set`` or
      ``hf://saberzl/SID_Set``;
    * a Kaggle URI, such as
      ``kaggle://birdy654/cifake-real-and-ai-generated-synthetic-images``;
    * a full Hugging Face or Kaggle dataset URL.

    A bare ``owner/name`` is interpreted as a Hugging Face ID because the same
    syntax is used by both services.  Prefix Kaggle handles with ``kaggle://``.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        split: str = "train",
        config: str | None = None,
        image_column: str | None = None,
        label_column: str | None = None,
        label_names: tuple[str, ...] | None = None,
        metadata_columns: tuple[str, ...] = (),
        revision: str | None = None,
        streaming: bool | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        shuffle_buffer_size: int = 100,
        convert_mode: str | None = "RGB",
        transform: Callable[[Image.Image], Any] | None = None,
        label_map: Mapping[Any, Any] | Callable[[Any], Any] | None = None,
    ) -> None:
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be greater than zero")

        self.source = source
        self.split = split
        self.config = config
        # Extra columns to carry along per sample. Opt-in on purpose: SID-Set's
        # unused columns include a full-size mask image per row (heavy), and
        # img_id text reveals the label -- metadata must never reach a model.
        self.metadata_columns = tuple(metadata_columns)
        self.revision = revision
        self._warned_unmapped = False
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.shuffle_buffer_size = shuffle_buffer_size
        self.convert_mode = convert_mode
        self.transform = transform
        self.label_map = label_map

        source_kind, source_value = self._parse_source(source)
        # Streaming is the safe default for potentially huge Hub datasets.
        # Local/Kaggle data is already on disk and supports exact random access.
        if streaming is None:
            streaming = source_kind == "huggingface"
        self.streaming = streaming
        self.local_root: Path | None = None

        self.dataset = self._open_dataset(
            source_kind, source_value, token=token
        )
        # Work out the column names once up front (see _detect_columns).
        self._columns = self._detect_columns()
        self.image_column = image_column or self._find_column(
            IMAGE_COLUMN_CANDIDATES, required=True
        )
        self.label_column = label_column or self._find_column(
            LABEL_COLUMN_CANDIDATES, required=False
        )
        self._validate_column(self.image_column)
        if self.label_column is not None:
            self._validate_column(self.label_column)
        elif self.label_map is not None:
            raise ValueError("label_map requires a label column")

        # Prefer names published by the dataset itself; otherwise use the
        # caller-supplied ones (e.g. SID_SET_LABEL_NAMES).
        self.label_names = self._get_label_names() or (
            tuple(label_names) if label_names else None
        )

    def get_batch(
        self, batch_size: int, *, seed: int | None = None
    ) -> list[dict[str, Any]]:
        """Return one batch, sampled without replacement when random-accessible."""
        self._validate_batch_size(batch_size)

        if isinstance(self.dataset, Dataset):
            if batch_size > len(self.dataset):
                raise ValueError(
                    f"batch_size ({batch_size}) exceeds dataset size "
                    f"({len(self.dataset)})"
                )
            rng = random.Random(seed)
            indices = rng.sample(range(len(self.dataset)), batch_size)
            batch = [
                self._normalise_sample(self.dataset[index]) for index in indices
            ]
        else:
            dataset = self.dataset
            shuffle_seed = (
                seed
                if seed is not None
                else random.SystemRandom().randrange(2**32)
            )
            dataset = dataset.shuffle(
                seed=shuffle_seed, buffer_size=self.shuffle_buffer_size
            )
            # Consume and normalise only the requested rows, then explicitly
            # close the generator so HTTP streams do not outlive a short CLI
            # command on Windows.
            batch = []
            row_iterator = iter(dataset)
            try:
                for _ in range(batch_size):
                    try:
                        row = next(row_iterator)
                    except StopIteration:
                        break
                    batch.append(self._normalise_sample(row))
            finally:
                close = getattr(row_iterator, "close", None)
                if close is not None:
                    close()

        if len(batch) != batch_size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds the number of available "
                f"samples ({len(batch)})"
            )
        return batch

    def get_images(
        self, batch_size: int, *, seed: int | None = None
    ) -> list[Any]:
        """Convenience wrapper that returns only images (or transformed images)."""
        return [sample["image"] for sample in self.get_batch(batch_size, seed=seed)]

    def iter_batches(
        self,
        batch_size: int,
        *,
        seed: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterator[list[dict[str, Any]]]:
        """Iterate through an epoch as normalised batches."""
        self._validate_batch_size(batch_size)
        dataset: Dataset | IterableDataset = self.dataset
        if shuffle:
            if isinstance(dataset, Dataset):
                dataset = dataset.shuffle(seed=seed)
            else:
                dataset = dataset.shuffle(
                    seed=seed, buffer_size=self.shuffle_buffer_size
                )

        batch: list[dict[str, Any]] = []
        for row in dataset:
            batch.append(self._normalise_sample(row))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch and not drop_last:
            yield batch

    def _open_dataset(
        self, source_kind: str, source_value: str, *, token: str | bool | None
    ) -> Dataset | IterableDataset:
        """Open the data from the right place: Hugging Face, Kaggle, or a local folder."""
        common_kwargs: dict[str, Any] = {
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.cache_dir is not None:
            common_kwargs["cache_dir"] = str(self.cache_dir)

        if source_kind == "huggingface":
            if token is not None:
                common_kwargs["token"] = token
            if self.revision is not None:
                common_kwargs["revision"] = self.revision
            return load_dataset(source_value, self.config, **common_kwargs)

        if source_kind == "kaggle":
            try:
                import kagglehub
            except ImportError as exc:
                raise ImportError(
                    "Kaggle sources require kagglehub; install requirements.txt"
                ) from exc

            download_kwargs: dict[str, Any] = {}
            if self.cache_dir is not None:
                kaggle_dir = self.cache_dir / "kaggle" / source_value.replace("/", "__")
                download_kwargs["output_dir"] = str(kaggle_dir)
            self.local_root = Path(
                kagglehub.dataset_download(source_value, **download_kwargs)
            )
        else:
            self.local_root = Path(source_value).expanduser().resolve()

        # A local folder can hold either Parquet shards (e.g. an `hf download`
        # of SID-Set) or plain image files in class subfolders. Shards named
        # like `validation-00000-of-00034.parquet` are matched to the split.
        shards = sorted(self.local_root.rglob(f"{self.split}-*.parquet"))
        if shards:
            self._check_shards(shards)
            return load_dataset(
                "parquet",
                data_files={self.split: [str(shard) for shard in shards]},
                **common_kwargs,
            )
        other_shards = sorted(self.local_root.rglob("*.parquet"))
        if other_shards:
            # Parquet data exists but not for this split; name the splits that
            # are actually present instead of failing confusingly later.
            available = sorted(
                {shard.name.split("-")[0] for shard in other_shards}
            )
            raise FileNotFoundError(
                f"No Parquet shards for split {self.split!r} under "
                f"{self.local_root}. Splits found: {', '.join(available)}"
            )
        return load_dataset(
            "imagefolder", data_dir=str(self.local_root), **common_kwargs
        )

    @staticmethod
    def _check_shards(shards: list[Path]) -> None:
        """Catch two silent download problems: the same shard file found
        twice under the folder, and fewer shards than the filenames promise
        (names look like ``validation-00007-of-00034.parquet``)."""
        names = [shard.name for shard in shards]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate shard filenames under the data folder: {duplicates}. "
                "Point the loader at a narrower folder."
            )
        totals = {
            int(name.rsplit("-of-", 1)[1].split(".")[0])
            for name in names
            if "-of-" in name
        }
        if len(totals) > 1:
            raise ValueError(
                f"Shards from different dataset versions are mixed together "
                f"(conflicting shard totals {sorted(totals)})."
            )
        if totals:
            expected = totals.pop()
            if len(names) != expected:
                import warnings

                warnings.warn(
                    f"Found {len(names)} shard(s) but filenames indicate "
                    f"{expected}; the download may be partial. Fine if "
                    "intentional (e.g. a quick subset).",
                    stacklevel=3,
                )

    def _normalise_sample(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Turn one raw dataset row into a clean {image, label, metadata} sample."""
        image = self._decode_image(row[self.image_column])
        if self.convert_mode is not None:
            image = image.convert(self.convert_mode)
        # Detach lazy PIL file handles before returning the sample.
        image.load()
        image = image.copy()
        output_image = self.transform(image) if self.transform else image

        metadata = {key: row[key] for key in self.metadata_columns if key in row}
        original_label = (
            row.get(self.label_column) if self.label_column is not None else None
        )
        label_name = self._get_label_name(original_label)
        return {
            "image": output_image,
            "label": self._map_label(original_label, label_name),
            "original_label": original_label,
            "original_label_name": label_name,
            "metadata": metadata,
        }

    def _map_label(self, original_label: Any, label_name: str | None) -> Any:
        """Map a source label to the task label while retaining its raw value."""
        if original_label is None:
            return original_label
        if self.label_map is None:
            # Our task is binary; a raw 3-class label reaching a trainer would
            # either crash the loss or train on an invalid target. Warn once.
            if original_label not in (0, 1) and not self._warned_unmapped:
                import warnings

                self._warned_unmapped = True
                warnings.warn(
                    f"Label {original_label!r} is not 0/1 and no label_map was "
                    "given. Pass label_map=SID_SET_BINARY_LABEL_MAP for binary "
                    "training.",
                    stacklevel=3,
                )
            return original_label
        if callable(self.label_map):
            return self.label_map(original_label)

        try:
            if original_label in self.label_map:
                return self.label_map[original_label]
        except TypeError:
            # Produce the useful error below if a provider returns an unhashable
            # label instead of leaking a dictionary implementation detail.
            pass
        if label_name is not None and label_name in self.label_map:
            return self.label_map[label_name]
        raise ValueError(
            f"No label_map entry for raw label {original_label!r}"
            + (f" ({label_name!r})" if label_name is not None else "")
        )

    def _get_label_names(self) -> tuple[str, ...] | None:
        if self.label_column is None:
            return None
        # Streamed datasets may not publish their schema; then no names exist.
        features = getattr(self.dataset, "features", None)
        if features is None:
            return None
        feature = features.get(self.label_column)
        names = getattr(feature, "names", None)
        return tuple(names) if names is not None else None

    def _get_label_name(self, label: Any) -> str | None:
        if self.label_names is None or not isinstance(label, int):
            return None
        if 0 <= label < len(self.label_names):
            return self.label_names[label]
        return None

    def _decode_image(self, value: Any) -> Image.Image:
        """Datasets store images in different forms (already-decoded image, raw
        bytes, or a file path); load whichever form this row uses."""
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            return Image.open(BytesIO(bytes(value)))
        if isinstance(value, Mapping):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(bytes(value["bytes"])))
            value = value.get("path")
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_absolute() and self.local_root is not None:
                path = self.local_root / path
            return Image.open(path)
        raise TypeError(
            f"Unsupported value in image column {self.image_column!r}: "
            f"{type(value).__name__}"
        )

    def _detect_columns(self) -> list[str]:
        """Return the dataset's column names.

        Streamed datasets sometimes do not know their columns until the first
        sample arrives, so peek at one sample in that case.  ``iter`` opens a
        fresh pass over the stream, so the peek does not skip any data later.
        """
        columns = self.dataset.column_names
        if columns is None:
            columns = list(next(iter(self.dataset)).keys())
        return list(columns)

    def _find_column(
        self, candidates: tuple[str, ...], *, required: bool
    ) -> str | None:
        # Guess which column holds the image (or label) from common names.
        columns = set(self._columns)
        for candidate in candidates:
            if candidate in columns:
                return candidate
        if required:
            raise ValueError(
                "Could not identify an image column. Available columns: "
                f"{', '.join(sorted(columns))}. Pass image_column explicitly."
            )
        return None

    def _validate_column(self, column: str) -> None:
        if column not in self._columns:
            raise ValueError(
                f"Column {column!r} does not exist. Available columns: "
                f"{', '.join(self._columns)}"
            )

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

    @staticmethod
    def _parse_source(source: str | Path) -> tuple[str, str]:
        """Decide whether ``source`` is a local folder, a Kaggle handle, or a
        Hugging Face ID/URL."""
        value = str(source)
        local_path = Path(value).expanduser()
        if local_path.exists():
            if not local_path.is_dir():
                raise ValueError("Local image source must be a directory")
            return "local", str(local_path)

        if value.startswith("kaggle://"):
            return "kaggle", value.removeprefix("kaggle://").strip("/")
        if value.startswith("hf://"):
            return "huggingface", value.removeprefix("hf://").strip("/")

        parsed = urlparse(value)
        if parsed.netloc == "www.kaggle.com" or parsed.netloc == "kaggle.com":
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 3 and parts[0] == "datasets":
                return "kaggle", "/".join(parts[1:3])
        if parsed.netloc == "huggingface.co":
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 3 and parts[0] == "datasets":
                return "huggingface", "/".join(parts[1:3])

        # Both hubs use owner/name; choose HF unless the caller disambiguates.
        return "huggingface", value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local path, Hub ID/URL, or kaggle:// URI")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config")
    parser.add_argument("--image-column")
    parser.add_argument("--label-column")
    parser.add_argument(
        "--label-map",
        type=_parse_label_map,
        help='comma-separated mapping, for example "0=0,1=1,2=1"',
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="save the selected images as JPEG previews in this directory",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=100,
        help="number of streamed images used for approximate shuffling",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download a Hugging Face dataset instead of streaming it",
    )
    args = parser.parse_args()

    loader = ImageDatasetLoader(
        args.source,
        split=args.split,
        config=args.config,
        image_column=args.image_column,
        label_column=args.label_column,
        label_map=args.label_map,
        shuffle_buffer_size=args.shuffle_buffer_size,
        streaming=False if args.download else None,
    )
    batch = loader.get_batch(args.batch_size, seed=args.seed)
    for index, sample in enumerate(batch, start=1):
        image = sample["image"]
        print(
            f"{index}: size={image.size}, mode={image.mode}, "
            f"label={sample['label']!r}"
        )
    if args.preview_dir is not None:
        saved_paths = save_batch_preview(batch, args.preview_dir)
        print(f"Saved {len(saved_paths)} previews to {args.preview_dir.resolve()}")


def save_batch_preview(
    batch: list[dict[str, Any]], output_dir: str | Path
) -> list[Path]:
    """Save a batch as numbered JPEGs for quick visual inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for index, sample in enumerate(batch, start=1):
        image = sample.get("image")
        if not isinstance(image, Image.Image):
            raise TypeError(
                "Preview images must be PIL images; save previews before applying "
                "a tensor-producing transform"
            )
        label = _filename_value(sample.get("label"))
        original_label = _filename_value(sample.get("original_label"))
        output_path = output_dir / (
            f"{index:03d}_label-{label}_original-{original_label}.jpg"
        )
        image.save(output_path, format="JPEG", quality=92)
        saved_paths.append(output_path)
    return saved_paths


def _filename_value(value: Any) -> str:
    """Convert a label to a short value that is safe in Windows filenames."""
    text = "none" if value is None else str(value)
    invalid = '<>:"/\\|?*'
    for character in invalid:
        text = text.replace(character, "-")
    return text[:50]


def _parse_label_map(value: str) -> dict[Any, Any]:
    """Parse CLI mappings while accepting numeric or string class names."""
    mapping: dict[Any, Any] = {}
    for entry in value.split(","):
        try:
            source_label, target_label = entry.split("=", maxsplit=1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "label map entries must use SOURCE=TARGET"
            ) from exc
        mapping[_parse_scalar(source_label)] = _parse_scalar(target_label)
    return mapping


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    main()
