"""Feature pipeline: everything between raw images and model-ready inputs."""

from .augmentations import EVAL_GRID, RandomAugment
from .canonicalize import CANONICAL_JPEG_QUALITY, canonicalize_encoding
from .evaluate import EvaluationReport, EvaluationRow, binary_auc, evaluate_model
from .preprocess import dino_view, simplest_patch, two_views
from .splits import exclude_heldout_families, family_id, is_internal_test, split_dataset
from .torch_dataset import BranchViewDataset, DataLoaderBundle, create_dataloaders

__all__ = [
    "EVAL_GRID",
    "RandomAugment",
    "CANONICAL_JPEG_QUALITY",
    "canonicalize_encoding",
    "EvaluationReport",
    "EvaluationRow",
    "binary_auc",
    "evaluate_model",
    "dino_view",
    "simplest_patch",
    "two_views",
    "exclude_heldout_families",
    "family_id",
    "is_internal_test",
    "split_dataset",
    "BranchViewDataset",
    "DataLoaderBundle",
    "create_dataloaders",
]
