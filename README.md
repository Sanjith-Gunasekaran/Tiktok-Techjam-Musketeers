# SID-Set Local Batch Loader

This project loads random image batches from a local copy of the
[SID-Set dataset](https://huggingface.co/datasets/saberzl/SID_Set). Each sample
contains a decoded PIL image together with its image ID, width, height, and
classification label.

## Install the Python dependencies

```bash
python -m pip install -r requirements.txt
```

## Download the data

The complete dataset is approximately 116 GB, so make sure there is enough
free disk space before downloading it. Install the Hugging Face CLI:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

Then, from the repository root, download SID-Set into the directory expected
by the loader:

```bash
hf download saberzl/SID_Set --type dataset --local-dir data
```

The resulting layout should include:

```text
data/
└── data/
    ├── train-00000-of-00249.parquet
    ├── train-00001-of-00249.parquet
    ├── ...
    └── validation-00000-of-00034.parquet
```

The dataset is public, so authentication is normally unnecessary. If Hugging
Face asks for authentication, run `hf auth login` and provide a read token.

To save space, download only one split by adding an include filter. For
example, to download only the validation shards:

```bash
hf download saberzl/SID_Set --type dataset \
  --include "data/validation-*.parquet" \
  --local-dir data
```

## Load a batch

Run the command-line example:

```bash
python local_image_batch_loader.py --batch-size 8 --split train --seed 42
```

Or use the loader from Python:

```python
from pathlib import Path

from local_image_batch_loader import SIDDataset

batch = SIDDataset().get_random_batch(
    batch_size=8,
    split="train",
    data_dir=Path("data/data"),
    seed=42,
)

for sample in batch:
    print(sample["img_id"], sample["width"], sample["height"], sample["label"])
```

Valid split names are `train`, `validation`, and `all`. Sampling is performed
without replacement and is reproducible when the same seed is used.

## Universal image loader

`ImageDatasetLoader` gives local image folders, Hugging Face datasets, and
Kaggle datasets the same batch format:

```python
from data_loader import ImageDatasetLoader

loader = ImageDatasetLoader(
    "hf://saberzl/SID_Set",
    split="train",
)
batch = loader.get_batch(batch_size=8, seed=42)

for sample in batch:
    image = sample["image"]       # decoded RGB PIL.Image.Image
    label = sample["label"]       # normalized label, or the raw label by default
    raw_label = sample["original_label"]
    metadata = sample["metadata"] # all remaining columns
```

Hugging Face datasets stream by default, which avoids downloading all of a
large dataset. Streamed datasets use approximate shuffling with a 100-image
buffer and also shuffle the order of data shards. To download and cache a
dataset for exact random access instead, pass `streaming=False`.

For Kaggle, prefix the dataset handle with `kaggle://` (a plain
`owner/dataset` is treated as a Hugging Face ID):

```python
loader = ImageDatasetLoader(
    "kaggle://birdy654/cifake-real-and-ai-generated-synthetic-images",
    split="train",
)
images = loader.get_images(batch_size=8, seed=42)
```

KaggleHub prompts for authentication if the dataset requires it. Public
datasets normally work without a token.

Local folders should follow the standard ImageFolder layout. Folder names
become labels automatically:

```text
my_dataset/
├── train/
│   ├── real/
│   │   └── image1.jpg
│   └── fake/
│       └── image2.jpg
└── test/
    ├── real/
    └── fake/
```

```python
loader = ImageDatasetLoader("my_dataset", split="train")
batch = loader.get_batch(16, seed=0)
```

If a dataset uses unusual column names, specify them explicitly:

```python
loader = ImageDatasetLoader(
    "owner/dataset",
    image_column="photo",
    label_column="is_generated",
)
```

### Normalize labels for binary classification

Pass `label_map` when a dataset's labels do not already use `0=human` and
`1=AI`. SID-Set has three classes, so synthetic and tampered images should both
map to the AI class:

```python
loader = ImageDatasetLoader(
    "hf://saberzl/SID_Set",
    split="train",
    label_map={0: 0, 1: 1, 2: 1},
)
```

For ImageFolder datasets, string class names can be used even though the
underlying dataset represents them as integers. This avoids depending on the
alphabetical class order:

```python
loader = ImageDatasetLoader(
    "kaggle://birdy654/cifake-real-and-ai-generated-synthetic-images",
    split="train",
    label_map={"REAL": 0, "FAKE": 1},
)
```

Every sample contains the mapped `label`, the provider's `original_label`, and
`original_label_name` when the dataset publishes class names. An unknown label
raises an error instead of silently training with the wrong target.

The command line supports the same mapping:

```bash
python -m data_loader.image_dataset_loader hf://saberzl/SID_Set \
  --label-map "0=0,1=1,2=1" --batch-size 8
```

For training loops, `iter_batches()` walks an entire epoch and accepts a
callable `transform=` in the constructor for conversion to tensors or other
preprocessing.

The same interface is available from the command line:

```bash
python -m data_loader.image_dataset_loader hf://saberzl/SID_Set \
  --split train --batch-size 8 --seed 42 \
  --preview-dir batch_preview
```

`--preview-dir` creates the directory when necessary and saves numbered JPEG
files such as `001_label-0_original-0.jpg`. These files are only for visual
inspection; training continues to use the original decoded images.

Increase `--shuffle-buffer-size` for stronger approximate shuffling during a
long training run. A larger value must fetch and hold more images before the
first batch, so the default is intentionally small for interactive tests.
