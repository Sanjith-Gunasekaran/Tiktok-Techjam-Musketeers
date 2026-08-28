from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL.Image import Image


class BatchLoader(ABC):
    @abstractmethod
    def get_random_batch(self, batch_size: int, split: str, data_dir: Path, seed: int):
        pass

    @staticmethod
    @abstractmethod
    def _decode_image(value: Any, data_dir: Path) -> Image:
        pass
