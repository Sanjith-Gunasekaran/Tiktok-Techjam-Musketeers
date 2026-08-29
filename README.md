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
| `data_loader/image_dataset_loader.py` | **The main loader.** One API over local image folders, Hugging Face, and Kaggle. Downloads-and-caches or streams, auto-detects image/label columns, remaps labels via `label_map`, iterates whole epochs (`iter_batches`), and has a CLI for previewing batches. |
| `data_loader/batch_loader.py` | Tiny abstract base class: the contract a batch loader follows (return a random batch, decode one image). |
| `data_loader/local_image_batch_loader.py` | Preview/debug tool for a locally downloaded SID-Set copy. Reads random samples straight out of the Parquet shards. Good for spot-checking a download; too slow to feed training. |

## Setup

```bash
python -m pip install -r requirements.txt
```

## Recommended workflow: local downloaded data

Streaming straight from Hugging Face is handy for a quick look, but it
shuffles poorly and re-downloads every epoch (see below). For real work,
download once and load from disk.

### 1. Download

The full SID-Set is ~116 GB, so start with the validation split or a filtered
subset. Install the Hugging Face CLI, then from the repository root:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf download saberzl/SID_Set --type dataset \
  --include "data/validation-*.parquet" \
  --local-dir data
```

Drop the `--include` filter to fetch everything. Expected layout:

```text
data/
└── data/
    ├── train-00000-of-00249.parquet   (if downloaded)
    ├── ...
    └── validation-00000-of-00034.parquet
```

### 2. Spot-check the download

```bash
cd data_loader
python local_image_batch_loader.py --batch-size 8 --split validation --data-dir ../data/data
```

Prints each sampled image's ID, size, 3-class `label`, and the team-convention
`binary_label` (0 = real, 1 = AI).

### 3. Load in Python for real use

Open the dataset with `streaming=False`: the `datasets` library downloads it
into its cache on first use, and every later run gets fast, truly shuffled
random access from disk.

```python
from data_loader import ImageDatasetLoader, SID_SET_BINARY_LABEL_MAP

loader = ImageDatasetLoader(
    "hf://saberzl/SID_Set",
    split="validation",
    streaming=False,                      # download once, reuse from cache
    label_map=SID_SET_BINARY_LABEL_MAP,   # tampered counts as AI
)

batch = loader.get_batch(8, seed=42)
for sample in batch:
    print(sample["label"], sample["original_label_name"], sample["image"].size)
```

`label` is the binary label; `original_label` / `original_label_name` keep the
3-class truth. **Check the printed names once**: if class 0 turns out not to
be "real", fix `SID_SET_BINARY_LABEL_MAP` before training anything.

Note: the Parquet shards fetched with `hf download` are read only by the
spot-check tool for now; model code should load through `ImageDatasetLoader`
as above. A script that pulls a smaller balanced training subset is the next
planned addition.

## Streaming (quick exploration only)

The default for Hugging Face sources — nothing is downloaded up front:

```python
loader = ImageDatasetLoader(
    "hf://saberzl/SID_Set",
    split="train",
    label_map=SID_SET_BINARY_LABEL_MAP,
)
batch = loader.get_batch(8, seed=0)
```

**Caveat:** streamed data is shuffled through a small buffer (100 images by
default), which over a 300K-image stream is close to no shuffling at all.
Fine for previews; not good enough for training. `shuffle_buffer_size=` mixes
somewhat better at the cost of a slower start, but a local download is the
real fix.

## Other sources

Kaggle — prefix the handle with `kaggle://` (a bare `owner/name` is treated as
a Hugging Face ID). String class names work in `label_map` for folder-style
datasets:

```python
loader = ImageDatasetLoader(
    "kaggle://birdy654/cifake-real-and-ai-generated-synthetic-images",
    split="train",
    label_map={"REAL": 0, "FAKE": 1},
)
```

Local image folders — standard ImageFolder layout (`split/class_name/img.jpg`);
folder names become labels:

```python
loader = ImageDatasetLoader("my_dataset", split="train")
```

Unusual column names — pass them explicitly:

```python
loader = ImageDatasetLoader("owner/dataset", image_column="photo", label_column="is_generated")
```

An unknown label always raises an error instead of silently training on the
wrong target.

## Command line

The main loader doubles as a CLI. Preview 8 mapped samples and save them as
JPEGs for eyeballing:

```bash
python -m data_loader.image_dataset_loader hf://saberzl/SID_Set \
  --split validation --batch-size 8 --seed 42 \
  --label-map "0=0,1=1,2=1" --preview-dir batch_preview
```

Add `--download` to use the cached (non-streaming) mode.

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
