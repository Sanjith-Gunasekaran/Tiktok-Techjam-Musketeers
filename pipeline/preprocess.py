"""The two model-ready views of one image.

Our detector has two branches with opposite needs, so one image becomes two
tensors:

* ``dino_view`` -- semantic branch. Aspect-preserving resize (shortest edge
  to 256) then center crop to 224x224, ImageNet-normalized: the exact
  geometry DINOv2's official processor uses. No stretching -- stretching
  distorts shapes and can leak class information when one class has more
  square images than the other.
* ``simplest_patch`` -- forensics branch. Following the official SSP
  pipeline, the whole image is first standardized to 256x256 so EVERY image
  offers the same 64 candidate tiles; then the lowest-texture 32x32 tile is
  returned as raw pixels. Standardizing first matters: at native resolution
  a 4K photo would offer thousands of candidates and therefore find
  systematically smoother minima than a small image -- the model could read
  image resolution instead of forensic evidence.

Note the patch is NOT yet SRM-filtered; the model applies SRM filters as its
first (frozen) layer. Call these AFTER any augmentation, so both branches
see the same degraded image. PIL image in, torch tensor out.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# The average color statistics of ImageNet, which DINOv2 was trained with.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DINO_RESIZE = 256    # shortest edge before the crop (official processor config)
DINO_CROP = 224      # DINOv2's input side length
STANDARD_SIZE = 256  # SSP: fixed candidate population of (256/32)^2 = 64 tiles
PATCH_SIZE = 32      # simplest-patch side length (SSP's choice)


def dino_view(image: Image.Image) -> torch.Tensor:
    """Semantic-branch view: float32 tensor (3, 224, 224), zero-centered the
    way DINOv2's pretraining expects, geometry per the official processor."""
    image = image.convert("RGB")
    width, height = image.size
    scale = DINO_RESIZE / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
    width, height = image.size
    left, top = (width - DINO_CROP) // 2, (height - DINO_CROP) // 2
    image = image.crop((left, top, left + DINO_CROP, top + DINO_CROP))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    # Height x Width x Channels -> Channels x Height x Width (torch's layout).
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def simplest_patch(image: Image.Image) -> torch.Tensor:
    """Forensics-branch view: the lowest-texture tile as a float32 tensor
    (3, 32, 32), raw pixel values 0-255.

    Why the *simplest* tile: generators concentrate effort on rich textures;
    flat regions keep the clearest gap between real camera noise and
    generated smoothness. Left unnormalized on purpose -- SRM filters in the
    model operate on raw pixel values.
    """
    # Standardize so every image offers the same 8x8 grid of candidates.
    image = image.convert("RGB").resize((STANDARD_SIZE, STANDARD_SIZE), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32)
    grid = STANDARD_SIZE // PATCH_SIZE
    tiles = array.reshape(grid, PATCH_SIZE, grid, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4)

    # SSP's texture-diversity score: sum of absolute differences between
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
    return dino_view(image), simplest_patch(image)
