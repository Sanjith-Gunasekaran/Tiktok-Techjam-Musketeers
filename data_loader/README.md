# data_loader — fetching and decoding images

Working docs for the data-loading layer. See the repo root README for
overall project scope.

## Label convention

The binary task is `0 = real`, `1 = fully synthetic`. SID-Set label `2`
contains partially tampered photos, so it is excluded rather than mixed into
the fully synthetic class. A mapping target of `None` filters that class before
sampling or batching:

```python
from data_loader import SID_SET_BINARY_LABEL_MAP   # {0: 0, 1: 1, 2: None}
```

Returned samples keep their original label for traceability. Pass
`label_names=SID_SET_LABEL_NAMES` because SID-Set stores labels as bare
integers without class names.

Other loader notes:

- **Metadata is opt-in** (`metadata_columns=(...)`, default: keep nothing).
  SID-Set's unused columns include a full-size mask image per row, and
  `img_id` text reveals the label — metadata must never be fed to a model.
- **Pin the dataset version** with `revision="<commit hash>"` for
  reproducible runs against Hugging Face sources.
- **Always pass `label_map`** for training so label `2` is excluded before it
  can reach the binary trainer.
- Streaming is for exploration only (weak shuffle buffer). Training uses
  `pipeline.create_dataloaders`, which forces random-access loading.
- Local shard folders are sanity-checked: duplicate shard filenames raise an
  error; fewer shards than the filenames promise triggers a warning.

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
from data_loader import (
    ImageDatasetLoader,
    SID_SET_BINARY_LABEL_MAP,
    SID_SET_LABEL_NAMES,
)

loader = ImageDatasetLoader(
    "data",                               # folder containing the shards
    split="validation",
    label_map=SID_SET_BINARY_LABEL_MAP,
    label_names=SID_SET_LABEL_NAMES,
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
    label_names=SID_SET_LABEL_NAMES,
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

Every returned sample contains a decoded image and a binary label; tampered
rows never reach the batch:

```python
for sample in loader.get_batch(8, seed=42):
    print(
        sample["label"],                  # 0 = real, 1 = fully synthetic
        sample["original_label_name"],    # "real" / "full_synthetic"
        sample["image"].size,             # decoded RGB PIL image
    )
```

Before training, confirm class `0` prints as `real` and no tampered row appears.

Two more ways to eyeball the data:

```bash
# Random samples straight from downloaded Parquet shards (IDs, sizes, labels):
python data_loader/local_image_batch_loader.py \
  --batch-size 8 --split validation --data-dir data/data

# Save a previewed batch as JPEGs to look at:
python -m data_loader.image_dataset_loader hf://saberzl/SID_Set \
  --split validation --batch-size 8 --seed 42 \
  --label-map "0=0,1=1,2=null" --preview-dir batch_preview
```

## 4. Other ways to stream or extract data

**Streaming (quick exploration only).** The default for Hugging Face sources —
nothing is downloaded up front:

```python
loader = ImageDatasetLoader(
    "hf://saberzl/SID_Set",
    split="train",
    label_map=SID_SET_BINARY_LABEL_MAP,
    label_names=SID_SET_LABEL_NAMES,
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
