# pipeline — from raw images to model-ready inputs

Working docs for the feature-pipeline layer. See the repo root README for
overall project scope.

## `pipeline/augmentations.py` — image degradations

The brief's six real-world transformations, implemented once and exposed two
ways so training and evaluation can never drift apart:

```python
from pipeline import EVAL_GRID, RandomAugment

# Training: one random transform at a random strength on ~50% of images.
augment = RandomAugment(probability=0.5, seed=42)
degraded = augment(pil_image)

# Evaluation: the brief's exact parameter grid, deterministic.
for name, transform in EVAL_GRID.items():   # clean, jpeg_q90 ... crop_80 (16 entries)
    score_model_on(transform(pil_image))
```

Notes: JPEG is a real encode/decode, not an approximation; eval noise is
seeded so runs are repeatable; `crop_80` outputs a smaller image on purpose
(branch preprocessing resizes later where needed); with DataLoader workers,
give each worker its own `RandomAugment` seed.

## `pipeline/preprocess.py` — the two branch views

One (already augmented) image becomes the two tensors our branches need:

```python
from pipeline import dino_view, srm_view, two_views

dino_tensor = dino_view(pil_image)   # (3, 224, 224) resized + ImageNet-normalized
srm_tensor = srm_view(pil_image)     # (3, 32, 32) simplest patch, RAW pixels 0-255
dino_tensor, srm_tensor = two_views(pil_image)   # both at once
```

- `dino_view`: bicubic resize to 224x224, scaled to 0-1, then zero-centered
  with ImageNet statistics — the input format DINOv2's pretraining expects.
- `srm_view`: cuts the image into a grid of 32x32 tiles, scores each tile's
  texture (ESSP's four-direction pixel-difference measure), and returns the
  *lowest*-texture tile as raw, untouched pixels. No resize of the patch and
  no normalization on purpose: interpolation and rescaling smear the faint
  noise fingerprint this branch detects. Images smaller than one patch are
  upscaled just enough to fit.
- Order matters: augment first, then take the views, so both branches see the
  same degraded image.

## `pipeline/splits.py` — frozen internal test set

We need a test set no training decision ever touches, for the final
robustness table. Membership is decided by hashing each image's ID, so the
same image lands in the same split on every machine, every run — nothing to
store or sync:

```python
from pipeline import is_internal_test, split_dataset

is_internal_test("img_1234")          # True -> frozen test set (default 20%)
dev, test = split_dataset(hf_dataset) # filter a whole dataset in one call
```

Rules: `dev` is for day-to-day work (validation during training, tuning);
`test` is only for the final evaluation. md5 is used instead of Python's
`hash()` because the built-in is randomized per process.

