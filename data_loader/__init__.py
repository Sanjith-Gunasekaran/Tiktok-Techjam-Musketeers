"""Dataset loading utilities."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .image_dataset_loader import SID_SET_BINARY_LABEL_MAP, ImageDatasetLoader

__all__ = ["ImageDatasetLoader", "SID_SET_BINARY_LABEL_MAP"]


def __getattr__(name: str) -> Any:
    """Import lazily so ``python -m data_loader.image_dataset_loader`` is safe."""
    if name in __all__:
        from . import image_dataset_loader

        return getattr(image_dataset_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
