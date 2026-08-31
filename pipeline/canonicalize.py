"""Label-independent input canonicalization.

SID-Set stores authentic images as JPEG and fully synthetic images as PNG.
Decoding both to RGB removes the container metadata, but not the JPEG
quantization artifacts in the pixels.  Give every sample one common,
moderately strong JPEG pass before augmentation or branch preprocessing so
the original container convention is a less direct pixel-artifact shortcut.

This mitigates source-encoding bias; it cannot restore information already
lost from an originally compressed image.  Generalization claims therefore
still require a separately sourced, encoding-matched external evaluation set.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image

CANONICAL_JPEG_QUALITY = 75


def canonicalize_encoding(
    image: Image.Image, *, quality: int = CANONICAL_JPEG_QUALITY
) -> Image.Image:
    """Return an RGB image after the same deterministic JPEG pass for all data."""
    if not isinstance(image, Image.Image):
        raise TypeError("canonicalization requires a PIL image")
    if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer between 1 and 100")

    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        # Copy detaches the result from the in-memory file before it is closed.
        return decoded.convert("RGB").copy()
