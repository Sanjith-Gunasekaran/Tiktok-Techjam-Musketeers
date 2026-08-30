# Data Loaders

The data-loading layer for our AI-generated-image detector. It fetches images
from a local download, Hugging Face, or Kaggle, decodes them, attaches labels,
and hands out batches for the model code to consume.

## Label convention (team decision): tampered counts as AI

The model is a **binary** classifier: `0 = real`, `1 = AI-generated`.
SID-Set has three classes — real, fully synthetic, and tampered (a real photo
with an AI-edited region) — and we map **both** synthetic and tampered to `1`.
The mapping lives in the code as `SID_SET_BINARY_LABEL_MAP` so every script
uses the same rule:

```python
from data_loader import SID_SET_BINARY_LABEL_MAP   # {0: 0, 1: 1, 2: 1}
```

Every sample also keeps the original 3-class label (`original_label`,
`original_label_name`), so error analysis can still report tampered images
separately.

## What each file does

| File | Purpose |
| --- | --- |
| `data_loader/__init__.py` | Makes `data_loader` importable as a package; exposes `ImageDatasetLoader` and `SID_SET_BINARY_LABEL_MAP`. |
| `data_loader/image_dataset_loader.py` | **The main loader.** One API over local downloads (Parquet shards or image folders), Hugging Face, and Kaggle. Auto-detects image/label columns, remaps labels via `label_map`, iterates whole epochs (`iter_batches`), and has a CLI for previewing batches. |
| `data_loader/batch_loader.py` | Tiny abstract base class: the contract a batch loader follows (return a random batch, decode one image). |
| `data_loader/local_image_batch_loader.py` | Preview/debug tool for a locally downloaded SID-Set copy. Reads random samples straight out of the Parquet shards. Good for spot-checking a download; too slow to feed training. |

## Setup

```bash
python -m pip install -r requirements.txt
```

## 1. Already downloaded the data? Point the loader at it

If you have a local SID-Set copy (Parquet shards, e.g. from `hf download`),
just give the loader that folder — it finds the shards, matches them to the
requested split, and reads from disk with fast, truly shuffled random access:

```python
from data_loader import ImageDatasetLoader, SID_SET_BINARY_LABEL_MAP

loader = ImageDatasetLoader(
    "data",                               # folder containing the shards
    split="validation",
    label_map=SID_SET_BINARY_LABEL_MAP,   # tampered counts as AI
)

batch = loader.get_batch(8, seed=42)
```

Asking for a split that is not in the folder raises an error listing the
splits that are actually there.

## 2. No local copy? Two ways to get one (pick one, run once)

Sections 1 and 2 are alternatives, not steps: if section 1 applied to you,
skip this entirely.

**Option A — let the program download it.** Use the Hugging Face ID with
`streaming=False`. The `datasets` library downloads the split into its own
cache the first time this runs, and every later run reads from disk:

```python
loader = ImageDatasetLoader(
    "hf://saberzl/SID_Set",
    split="validation",
    streaming=False,                      # download once into the cache
    label_map=SID_SET_BINARY_LABEL_MAP,
)
```

**Option B — download manually with the Hugging Face CLI.** This gives you a
visible folder of Parquet files (which section 1 then loads). The full dataset
is ~116 GB, so consider starting with the validation split:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf download saberzl/SID_Set --type dataset \
  --include "data/validation-*.parquet" \
  --local-dir data
```

Both options end the same way — data on disk, fast loading. Never do both.

## 3. Check the data once it's loaded

Every sample is a dict with the decoded image, the binary training label, and
the original 3-class truth:

```python
for sample in loader.get_batch(8, seed=42):
    print(
        sample["label"],                  # 0 = real, 1 = AI (what the model trains on)
        sample["original_label_name"],    # "real" / "full_synthetic" / "tampered"
        sample["image"].size,             # decoded RGB PIL image
    )
```

**Do this once before training:** confirm that class 0 prints as "real". If it
ever does not, fix `SID_SET_BINARY_LABEL_MAP` first — a flipped mapping wastes
a whole training run.

Two more ways to eyeball the data:

```bash
# Random samples straight from downloaded Parquet shards (IDs, sizes, labels):
python data_loader/local_image_batch_loader.py \
  --batch-size 8 --split validation --data-dir data/data

# Save a previewed batch as JPEGs to look at:
python -m data_loader.image_dataset_loader hf://saberzl/SID_Set \
  --split validation --batch-size 8 --seed 42 \
  --label-map "0=0,1=1,2=1" --preview-dir batch_preview
```

## 4. Other ways to stream or extract data

**Streaming (quick exploration only).** The default for Hugging Face sources —
nothing is downloaded up front:

```python
loader = ImageDatasetLoader(
    "hf://saberzl/SID_Set",
    split="train",
    label_map=SID_SET_BINARY_LABEL_MAP,
)
batch = loader.get_batch(8, seed=0)
```

Caveat: streamed data is shuffled through a small buffer (100 images by
default), which over a 300K-image stream is close to no shuffling at all —
fine for previews, not good enough for training. `shuffle_buffer_size=` mixes
somewhat better at the cost of a slower start; a local download (option 1 or
2) is the real fix.

**Kaggle.** Prefix the handle with `kaggle://` (a bare `owner/name` is treated
as a Hugging Face ID). String class names work in `label_map` for folder-style
datasets:

```python
loader = ImageDatasetLoader(
    "kaggle://birdy654/cifake-real-and-ai-generated-synthetic-images",
    split="train",
    label_map={"REAL": 0, "FAKE": 1},
)
```

**Local image folders.** Standard ImageFolder layout
(`split/class_name/img.jpg`); folder names become labels:

```python
loader = ImageDatasetLoader("my_dataset", split="train")
```

**Unusual column names.** Pass them explicitly:

```python
loader = ImageDatasetLoader("owner/dataset", image_column="photo", label_column="is_generated")
```

An unknown label always raises an error instead of silently training on the
wrong target.

## Training loops

`iter_batches(batch_size)` walks a whole epoch, and the constructor accepts
`transform=` (a function applied to every image) for tensor conversion or
augmentation.

## Still to build on top of this layer

- Augmentation module (JPEG / blur / resize / noise / color jitter / crop)
- PyTorch `Dataset` wrapper producing the two-branch tensors, with a held-out
  test split carved out for the robustness table
- Evaluation harness (clean vs. transformed metrics)
- Balanced training-subset download script

## DINOv2 directory inference

`predict_aigc.py` is compatible with `best_dinov2_model.pt` produced by
`DINOv2_ML.ipynb`. It recursively scans an unlabeled image directory, processes
images in batches, and writes the probability that each image is AI-generated:

```bash
python predict_aigc.py path/to/images \
  --checkpoint path/to/best_dinov2_model.pt \
  --output predictions.json \
  --batch-size 16
```

The checkpoint must contain the notebook's full `model_state_dict`, `model_id`,
and `class_to_idx`. On its first run, Transformers downloads the DINOv2 model
configuration and image processor. Later runs use the local Hugging Face cache.

The output contains one entry per supported image, with paths relative to the
input directory and `pred` in the range `[0, 1]`:

```json
[
  {
    "image_path": "example.jpg",
    "pred": 0.9721
  },
  {
    "image_path": "nested/photo.png",
    "pred": 0.0314
  }
]
```

Higher `pred` values indicate a greater likelihood of AI-generated content.
Use `--device cpu` to force CPU inference or `--device cuda` to require a GPU.
The default `--num-workers 0` is the safest choice on Windows.

Before scoring a large directory, run a small deterministic smoke test:

```bash
python predict_aigc.py path/to/images \
  --checkpoint path/to/best_dinov2_model.pt \
  --output test_predictions.json \
  --max-images 32 \
  --seed 42
```

This samples 32 paths without changing the source directory. The command also
prints the selected device, processing rate, and estimated time remaining.

## DINOv2 training pipeline

`DINOv2_ML.ipynb` does train a model. `train_dinov2.py` ports that notebook's
augmentation, weighted loss, early stopping, checkpoint selection, and test
evaluation into a reusable command. For a local ImageFolder dataset:

```text
archive/
├── train/
│   ├── REAL/
│   └── FAKE/
├── validation/       # optional
│   ├── REAL/
│   └── FAKE/
└── test/             # optional
    ├── REAL/
    └── FAKE/
```

Run a one-epoch smoke test before full training:

```bash
python train_dinov2.py path/to/archive \
  --output checkpoints/smoke_dinov2.pt \
  --epochs 1 \
  --max-train-images 128 \
  --max-validation-images 64 \
  --max-test-images 64 \
  --device cpu
```

When `validation/` is absent, 10% of `train/` is held out deterministically.
The smoke-test limits are stratified so both classes are represented. After it
works, omit the three `--max-*-images` options for full training. A CUDA GPU is
strongly recommended:

```bash
python train_dinov2.py path/to/archive \
  --output checkpoints/best_dinov2_model.pt \
  --epochs 10 \
  --patience 3 \
  --batch-size 16 \
  --device cuda
```

Folder labels are normalized to `human=0` and `AI=1`; change `--human-class`
and `--ai-class` if the folders use different names. The best checkpoint is
directly compatible with `predict_aigc.py`. Training also creates a neighboring
`best_dinov2_model.metrics.json` containing validation and test metrics.
