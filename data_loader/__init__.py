"""Dataset loading utilities."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .image_dataset_loader import ImageDatasetLoader

__all__ = ["ImageDatasetLoader"]


def __getattr__(name: str) -> Any:
    """Import lazily so ``python -m data_loader.image_dataset_loader`` is safe."""
    if name == "ImageDatasetLoader":
        from .image_dataset_loader import ImageDatasetLoader

        return ImageDatasetLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
