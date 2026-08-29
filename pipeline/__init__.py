"""Feature pipeline: everything between raw images and model-ready inputs."""

from .augmentations import EVAL_GRID, RandomAugment
from .evaluate import EvaluationReport, EvaluationRow, binary_auc, evaluate_model
from .preprocess import dino_view, simplest_patch, two_views
from .splits import family_id, is_internal_test, split_dataset
from .torch_dataset import BranchViewDataset, DataLoaderBundle, create_dataloaders

__all__ = [
    "EVAL_GRID",
    "RandomAugment",
    "EvaluationReport",
    "EvaluationRow",
    "binary_auc",
    "evaluate_model",
    "dino_view",
    "simplest_patch",
    "two_views",
    "family_id",
    "is_internal_test",
    "split_dataset",
    "BranchViewDataset",
    "DataLoaderBundle",
    "create_dataloaders",
]
