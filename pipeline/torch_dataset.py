"""PyTorch datasets and loaders for the two-branch SID-Set pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info

from data_loader.image_dataset_loader import (
    SID_SET_BINARY_LABEL_MAP,
    SID_SET_LABEL_NAMES,
    ImageDatasetLoader,
)

from .augmentations import RandomAugment
from .preprocess import two_views
from .splits import TEST_FRACTION, split_dataset

BranchSample = tuple[torch.Tensor, torch.Tensor, int, int]


class BranchViewDataset(Dataset[BranchSample]):
    """Decode one row, augment once, then create both branch views."""

    def __init__(
        self,
        source: ImageDatasetLoader,
        rows: HFDataset | None = None,
        augmentation: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        if source.transform is not None:
            raise ValueError("ImageDatasetLoader.transform must be None")
        selected_rows = source.dataset if rows is None else rows
        if not isinstance(selected_rows, HFDataset):
            raise TypeError("BranchViewDataset requires streaming=False")
        self.source = source
        self.rows = selected_rows
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> BranchSample:
        sample = self.source.normalise_sample(self.rows[index])
        image = sample["image"]
        if self.augmentation is not None:
            image = self.augmentation(image)
        if not isinstance(image, Image.Image):
            raise TypeError("augmentation must return a PIL image")

        label = sample["label"]
        original_label = sample["original_label"]
        if label not in (0, 1) or original_label is None:
            raise ValueError("Expected binary labels after SID-Set filtering")

        dino_tensor, patch_tensor = two_views(image)
        return dino_tensor, patch_tensor, int(label), int(original_label)


class DataLoaderBundle(NamedTuple):
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def _seed_worker(worker_id: int) -> None:
    """Give each worker an independent augmentation stream."""
    worker = get_worker_info()
    if worker is None:
        return
    augmentation = getattr(worker.dataset, "augmentation", None)
    reseed = getattr(augmentation, "reseed", None)
    if callable(reseed):
        reseed(worker.seed % 2**32)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def create_dataloaders(
    source: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    train_split: str = "train",
    validation_split: str = "validation",
    test_fraction: float = TEST_FRACTION,
    id_column: str = "img_id",
    augmentation_probability: float = 0.5,
    second_augmentation_probability: float = 0.3,
    pin_memory: bool = False,
    drop_last: bool = False,
    config: str | None = None,
    image_column: str | None = None,
    label_column: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> DataLoaderBundle:
    """Create shuffled train and fixed validation/test DataLoaders.

    Train uses the published training split. The published validation split is
    deterministically divided into development and frozen internal-test sets.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool):
        raise TypeError("num_workers must be an integer")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    common = {
        "config": config,
        "image_column": image_column,
        "label_column": label_column,
        "label_names": SID_SET_LABEL_NAMES,
        "revision": revision,
        "streaming": False,
        "cache_dir": cache_dir,
        "token": token,
        "transform": None,
        "label_map": SID_SET_BINARY_LABEL_MAP,
    }
    train_source = ImageDatasetLoader(source, split=train_split, **common)
    validation_source = ImageDatasetLoader(
        source, split=validation_split, **common
    )
    if id_column not in validation_source.dataset.column_names:
        raise ValueError(f"Validation split has no {id_column!r} column")

    validation_rows, test_rows = split_dataset(
        validation_source.dataset,
        id_column=id_column,
        fraction=test_fraction,
    )
    if not len(train_source.dataset):
        raise ValueError("Training split is empty after label filtering")
    if not len(validation_rows) or not len(test_rows):
        raise ValueError("Validation/test split is empty; use more data")

    augmentation = RandomAugment(
        probability=augmentation_probability,
        second_probability=second_augmentation_probability,
        seed=seed,
    )
    train_dataset = BranchViewDataset(train_source, augmentation=augmentation)
    validation_dataset = BranchViewDataset(validation_source, validation_rows)
    test_dataset = BranchViewDataset(validation_source, test_rows)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
    }
    return DataLoaderBundle(
        train=DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=drop_last,
            generator=_generator(seed),
            **loader_options,
        ),
        validation=DataLoader(
            validation_dataset,
            shuffle=False,
            generator=_generator(seed + 1),
            **loader_options,
        ),
        test=DataLoader(
            test_dataset,
            shuffle=False,
            generator=_generator(seed + 2),
            **loader_options,
        ),
    )
