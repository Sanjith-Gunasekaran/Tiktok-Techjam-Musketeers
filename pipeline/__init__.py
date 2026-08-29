"""Feature pipeline: everything between raw images and model-ready inputs."""

from .augmentations import EVAL_GRID, RandomAugment

__all__ = ["EVAL_GRID", "RandomAugment"]
