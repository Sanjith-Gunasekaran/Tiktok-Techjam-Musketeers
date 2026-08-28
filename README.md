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
