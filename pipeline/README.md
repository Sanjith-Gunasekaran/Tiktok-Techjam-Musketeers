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
