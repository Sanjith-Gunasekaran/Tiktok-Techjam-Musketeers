"""DINOv2-based AI-generated image detector."""

from .dinov2 import DINO_MODEL_ID, DINOv2BinaryClassifier, load_detector_checkpoint

__all__ = [
    "DINO_MODEL_ID",
    "DINOv2BinaryClassifier",
    "load_detector_checkpoint",
]
