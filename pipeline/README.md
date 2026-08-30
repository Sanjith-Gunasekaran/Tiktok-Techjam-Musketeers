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
for name, transform in EVAL_GRID.items():   # clean through compound chain (17 cells)
    score_model_on(transform(pil_image))
```

Notes:

- JPEG is a real encode/decode, not an approximation.
- Eval noise is seeded **from each image's own pixels**: repeatable across
  runs, but no two images share a noise pattern (a shared pattern would be a
  giveaway to the noise-forensics branch).
- Real uploads stack degradations, so training chains a second, different
  transform 30% of the time (`second_probability=`) and the eval grid
  includes `chain_crop_resize_jpeg` (crop + thumbnail + JPEG together).
- `crop_80` and the chain output smaller images on purpose; branch
  preprocessing resizes where needed.
- **DataLoader workers** need separate random states. `create_dataloaders`
  handles this automatically.

## `pipeline/preprocess.py` — the two branch views

One (already augmented) image becomes the two tensors our branches need:

```python
from pipeline import dino_view, simplest_patch, two_views

dino_tensor = dino_view(pil_image)        # (3, 224, 224) resize-256 + centre-crop, normalized
patch_tensor = simplest_patch(pil_image)  # (3, 32, 32) lowest-texture tile, RAW pixels 0-255
dino_tensor, patch_tensor = two_views(pil_image)
```

- `dino_view`: shortest edge to 256 preserving aspect ratio, centre-crop to
  224, then ImageNet normalization — the geometry DINOv2's official
  processor uses. Deliberately *not* a straight stretch to 224x224:
  stretching distorts shapes and can leak class information if one class
  holds more square images than the other.
- `simplest_patch`: standardizes the image to 256x256 first (as the official
  SSP implementation does), so every image offers the same 64 candidate
  tiles, then returns the lowest-texture 32x32 tile as raw pixels.
  Standardizing matters: scanning at native resolution would give a 4K photo
  thousands of candidates and systematically smoother minima than a small
  image, letting the model read *resolution* instead of forensic evidence.
- The patch is **not** SRM-filtered here — the model applies SRM as its first
  frozen layer. (Hence the name `simplest_patch`, not `srm_view`.)
- Order matters: augment first, then take the views, so both branches see the
  same degraded image.

## `pipeline/splits.py` — frozen internal test set

SID-Set's own 60K test split is gated (request-only), so we carve a frozen
test set out of the validation split: images no training decision touches.
Membership comes from hashing each image's **family ID** — the raw ID with
its class prefix stripped (`real_12ab` / `tampered_12ab` -> `12ab`) — so
images derived from the same source photo stay on the same side and we never
test on an edit of a training image:

```python
from pipeline import family_id, is_internal_test, split_dataset

is_internal_test("tampered_12ab")     # True -> frozen test set (default 20%)
dev, test = split_dataset(hf_dataset) # filter a whole dataset in one call
```

Scope, stated plainly: this split supports **held-out images, same
generators** claims. It does *not* measure unseen-generator generalization —
that comes from the organizers' external benchmark (DALL·E Advanced + COCO),
which we never train on. md5 is used instead of Python's `hash()` because the
built-in is randomized per process.

The binary loader excludes tampered rows. Family handling remains here so a
raw SID dataset can still be partitioned safely before label filtering.

`create_dataloaders` applies this split before constructing the validation and
test loaders, so the frozen test set cannot enter training or validation.

## `pipeline/torch_dataset.py` — training input pipeline

The factory loads SID-Set without streaming, excludes tampered rows, augments
each training image once, and then creates both branch views. Validation and
test are not augmented or shuffled.

```python
from pipeline import create_dataloaders

loaders = create_dataloaders(
    "data", batch_size=64, num_workers=4, seed=42
)
for dino_tensor, patch_tensor, label, original_label in loaders.train:
    train_step(dino_tensor, patch_tensor, label)
```

Train uses SID-Set's published training split. The published validation split
is deterministically divided into validation and frozen internal test sets.
Their membership never changes between epochs or runs; only training order and
augmentation change. `original_label` is retained for provenance, but is only
`0` or `1` because tampered rows are excluded. The patch tensor is raw input
for the model's frozen SRM layer—it is not yet SRM-filtered. Freeze
`test_fraction` before experiments; changing it changes the holdout and
invalidates comparisons. This is one stable split, not cross-validation.

## `pipeline/evaluate.py` — clean-vs-transformed report

The evaluator accepts the full loader bundle so it can select the frozen test
partition itself. For every grid cell it applies the transform before both
views, calls `model(dino_batch, patch_batch)`, and writes accuracy/AUC plus
their change from clean to CSV and Markdown.

```python
from pipeline import evaluate_model

report = evaluate_model(
    model, loaders, output_dir="evaluation", save_predictions=True
)
print(report.csv_path, report.markdown_path, report.error_path)
```

The model must return one synthetic-class probability per image (`[B]` or
`[B, 1]`). Use `from_logits=True` for logits. The classification threshold is
fixed at `0.5` by default; select any other threshold on validation before
calling the evaluator. Never tune the model or threshold from this test table.

Each `report.rows` entry carries TP/TN/FP/FN, accuracy, precision, recall, F1,
AUC, and changes versus clean. `error_analysis.csv` always contains only false
positives/negatives with their image IDs and scores. Full predictions are large,
so `predictions.jsonl.gz` is written only with `save_predictions=True`.
Passing a custom `transforms=` mapping must still include a `"clean"` entry.
