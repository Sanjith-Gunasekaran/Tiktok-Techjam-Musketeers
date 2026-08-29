"""The two model-ready views of one image.

Our detector has two branches with opposite needs, so one image becomes two
tensors:

* ``dino_view`` -- semantic branch. Resized to 224x224 and ImageNet-normalized,
  because that is the input language the pretrained DINOv2 backbone expects.
* ``srm_view``  -- forensics branch. The "simplest" patch of the image (least
  texture, after the ESSP paper), as RAW untouched pixels: no resize of the
  patch, no normalization, because interpolation and rescaling smear the
  faint noise fingerprint this branch detects.

Call these AFTER any augmentation, so both branches see the same degraded
image. PIL image in, torch tensor out.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# The average color statistics of ImageNet, which DINOv2 was trained with.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DINO_SIZE = 224     # DINOv2's standard input side length
PATCH_SIZE = 32     # simplest-patch side length (ESSP's ablation optimum)


def dino_view(image: Image.Image, size: int = DINO_SIZE) -> torch.Tensor:
    """Semantic-branch view: float32 tensor of shape (3, size, size),
    zero-centered the way DINOv2's pretraining expects."""
    resized = image.convert("RGB").resize((size, size), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    # Height x Width x Channels -> Channels x Height x Width (torch's layout).
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def srm_view(image: Image.Image, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Forensics-branch view: the lowest-texture patch as a float32 tensor of
    shape (3, patch_size, patch_size), raw pixel values 0-255.

    Why the *simplest* patch: generators concentrate effort on rich textures;
    flat regions keep the clearest gap between real camera noise and
    generated smoothness. Left unnormalized on purpose -- the SRM filters in
    the model operate on raw pixel values.
    """
    image = image.convert("RGB")
    width, height = image.size
    # Tiny images only: upscale just enough to fit one patch.
    if width < patch_size or height < patch_size:
        image = image.resize(
            (max(width, patch_size), max(height, patch_size)), Image.BILINEAR
        )
    array = np.asarray(image, dtype=np.float32)

    # Cut the image into a grid of non-overlapping patch_size x patch_size
    # tiles (dropping any leftover right/bottom edge).
    rows, cols = array.shape[0] // patch_size, array.shape[1] // patch_size
    array = array[: rows * patch_size, : cols * patch_size]
    tiles = array.reshape(rows, patch_size, cols, patch_size, 3).transpose(0, 2, 1, 3, 4)

    # ESSP's texture-diversity score: sum of absolute differences between
    # neighboring pixels in four directions. Lowest score = simplest tile.
    horiz = np.abs(tiles[:, :, :, :-1] - tiles[:, :, :, 1:]).sum(axis=(2, 3, 4))
    vert = np.abs(tiles[:, :, :-1, :] - tiles[:, :, 1:, :]).sum(axis=(2, 3, 4))
    diag = np.abs(tiles[:, :, :-1, :-1] - tiles[:, :, 1:, 1:]).sum(axis=(2, 3, 4))
    anti = np.abs(tiles[:, :, 1:, :-1] - tiles[:, :, :-1, 1:]).sum(axis=(2, 3, 4))
    scores = horiz + vert + diag + anti

    row, col = np.unravel_index(np.argmin(scores), scores.shape)
    return torch.from_numpy(tiles[row, col].transpose(2, 0, 1).copy())


def two_views(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: both branch views of the same (already augmented) image."""
    return dino_view(image), srm_view(image)
