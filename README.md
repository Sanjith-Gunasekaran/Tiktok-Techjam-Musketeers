# Robust AI-Generated Image Detection

A two-branch image detector for distinguishing authentic images from fully
AI-generated images, with explicit testing under JPEG compression, blur,
resizing, noise, colour changes, and cropping.

The final submission interface is intentionally simple: give `predict.py` a
directory of images and a learned-fusion checkpoint, and it writes one AIGC
confidence score per image to JSON.

## Results

The submitted model was trained on a 10 GB SID-Set subset and evaluated once
on a separately sourced benchmark containing 5,000 COCO val2017 photographs
and 8,843 WildFake DALL·E Advanced images.

| Metric | Result |
| --- | ---: |
| ROC AUC | **0.9200** |
| Accuracy at 0.5 | 0.7756 |
| Balanced accuracy | 0.8142 |
| AIGC precision | 0.9623 |
| AIGC recall | 0.6751 |
| F1 | 0.7935 |

The high precision and lower recall show that the 0.5 operating point is
conservative. The submitted `pred` value is a continuous confidence score, so
threshold-independent measures such as ROC AUC better represent its ranking
quality.

## Architecture

Each decoded image is canonicalised with the same quality-75 JPEG pass and
then converted into two complementary views:

```text
                                  224 x 224 ImageNet-normalised image
                                -> DINOv2-small -> MLP head -> semantic logit --\
raw image -> JPEG canonicalise                                                +-> learned linear fusion -> AIGC score
                                -> lowest-texture 32 x 32 patch                /
                                  -> fixed SRM filters -> CNN -> forensic logit
```

- **Semantic branch:** DINOv2 supplies pretrained visual representations. Its
  MLP classification head is trained first with the backbone frozen, then the
  final transformer blocks are fine-tuned at a lower learning rate.
- **Forensic branch:** the pipeline selects the least textured patch from a
  fixed 8x8 grid. Frozen Spatial Rich Model (SRM) high-pass filters expose
  residual patterns, and a small trainable CNN learns low-level forensic cues.
- **Fusion:** the selected DINO and forensic checkpoints are frozen. A linear
  layer learns how to combine their two logits on the calibration partition;
  checkpoint selection remains on a separate validation partition.

Binary labels are `0 = real` and `1 = fully synthetic`. SID-Set's partially
tampered class is excluded because localised manipulation is a different task.

## Data safeguards

SID-Set stores real images primarily as JPEG and synthetic images as PNG. To
reduce that shortcut, every class and partition receives the same JPEG
canonicalisation before augmentation or preprocessing. The loader also:

- removes training families that occur anywhere in published validation;
- keeps related image IDs together when making validation, calibration, and
  frozen internal-test partitions;
- applies augmentation only to training;
- keeps the internal test loader out of training and checkpoint selection.

Canonicalisation mitigates encoding bias but cannot reconstruct details lost
before the data was published. The external COCO/WildFake result is therefore
more representative of cross-source generalisation than internal validation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data_loader/` | SID-Set loading, decoding, label mapping, and filtering |
| `pipeline/` | Canonicalisation, augmentation, deterministic splits, branch preprocessing, and robustness evaluation |
| `models/` | DINO classifier, fixed SRM filters, forensic CNN, and two-branch fusion |
| `model_runs/train.py` | Stage-aware training and checkpointing |
| `model_runs/calibrate_fusion.py` | Optional interpretable fixed-fusion baseline |
| `predict.py` | Submission inference: image directory to JSON scores |
| `tests/` | Unit and integration tests |

## Installation

Clone the repository, create an environment, and install the dependencies:

```bash
git clone https://github.com/Sanjith-Gunasekaran/Tiktok-Techjam-Musketeers.git
cd Tiktok-Techjam-Musketeers

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA is strongly recommended for training. DINOv2 weights are downloaded from
Hugging Face the first time a DINO stage or final checkpoint is loaded.

Run the test suite before training:

```bash
python -m pytest tests
```

## Dataset preparation

Training requires SID-Set's `train` and `validation` splits. `--data-dir` may
be either the Hugging Face dataset ID or a local directory containing its
downloaded Parquet shards:

```bash
# Download/cache through the datasets library when training starts:
DATA_DIR="saberzl/SID_Set"

# Or use an existing local download:
DATA_DIR="/path/to/SID_Set"
```

No manual split step is required. `create_dataloaders` deterministically makes
the family-disjoint validation, fusion-calibration, and frozen-test partitions
from published validation and removes overlapping families from training.

## Training

Training is staged so that every checkpoint can be inspected independently.
All commands below are run from the repository root.

```bash
DATA_DIR="saberzl/SID_Set"       # replace with a local path if downloaded
RUN_DIR="model_runs/checkpoints"
```

### 1. Smoke-test both branches

These short runs validate data loading, tensor shapes, GPU memory, and
checkpoint writing before an expensive experiment:

```bash
python -m model_runs.train \
  --stage dino_head \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR" \
  --epochs 1 \
  --max-train-batches 10 \
  --max-validation-batches 10

python -m model_runs.train \
  --stage forensic \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR" \
  --epochs 1 \
  --max-train-batches 10 \
  --max-validation-batches 10
```

Use a separate `RUN_DIR` for the real experiment after smoke testing.

### 2. Train the frozen DINO head

```bash
python -m model_runs.train \
  --stage dino_head \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR" \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 4
```

Only the MLP head changes during this stage. The best validation-AUC checkpoint
is written to `$RUN_DIR/dino_head/best.pt`.

### 3. Fine-tune the final DINO blocks

```bash
python -m model_runs.train \
  --stage dino_finetune \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR" \
  --dino-checkpoint "$RUN_DIR/dino_head/best.pt" \
  --unfreeze-blocks 2 \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 4
```

The head uses the default learning rate `1e-4`; the unfrozen DINO blocks use
`1e-5`. The output is `$RUN_DIR/dino_finetune/best.pt`.

### 4. Train the forensic branch

```bash
python -m model_runs.train \
  --stage forensic \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR" \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 4
```

The SRM kernels remain fixed while the CNN learns. The output is
`$RUN_DIR/forensic/best.pt`.

### 5. Train learned fusion

```bash
python -m model_runs.train \
  --stage fusion \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR" \
  --dino-checkpoint "$RUN_DIR/dino_finetune/best.pt" \
  --forensic-checkpoint "$RUN_DIR/forensic/best.pt" \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 4
```

Both branches remain frozen. Their logits are cached, and only the two-input
fusion layer is trained. The final inference checkpoint is
`$RUN_DIR/fusion/best.pt`.

Every stage also writes `last.pt` for resuming and `history.jsonl` for epoch
metrics. Resume with the same stage and the final target epoch, for example:

```bash
python -m model_runs.train \
  --stage forensic \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_DIR" \
  --resume "$RUN_DIR/forensic/last.pt" \
  --epochs 10
```

## Internal robustness evaluation

`pipeline.evaluate_model` runs the learned detector on the frozen internal
test partition under the clean condition and every deterministic degradation
in `pipeline.augmentations.EVAL_GRID`.

Create `run_evaluation.py` at the repository root:

```python
from pathlib import Path

from pipeline import create_dataloaders, evaluate_model, write_overall_summary
from predict import choose_device, load_model

DATA_DIR = "saberzl/SID_Set"  # or a local SID-Set directory
CHECKPOINT = Path("model_runs/checkpoints/fusion/best.pt")
REPORT_DIR = Path("reports/internal_test")

device = choose_device("auto")
loaders = create_dataloaders(
    DATA_DIR,
    batch_size=32,
    num_workers=4,
    seed=67,
    view="both",
    pin_memory=device.type == "cuda",
)
model = load_model(CHECKPOINT, device)

report = evaluate_model(
    model,
    loaders,
    output_dir=REPORT_DIR,
    threshold=0.5,
    from_logits=True,
    device=device,
    save_predictions=True,
)
write_overall_summary(
    report.rows,
    REPORT_DIR / "overall_summary.json",
    exclude_transforms={"chain_crop_resize_jpeg"},
)
```

Run it once after model selection is complete:

```bash
python run_evaluation.py
```

The report directory contains the clean-versus-transformed CSV and Markdown
tables, error analysis, optional compressed predictions, and aggregate AUC
summary. Do not use this frozen test report to select checkpoints or tune the
threshold.

## Submission inference

Run the learned-fusion checkpoint on a directory. Images are discovered
recursively; JPEG, PNG, WebP, BMP, and TIFF are supported.

```bash
python predict.py /path/to/images \
  --checkpoint model_runs/checkpoints/fusion/best.pt \
  --output predictions.json \
  --batch-size 32 \
  --device auto
```

The output is a JSON array containing the required fields:

```json
[
  {
    "image_path": "/path/to/images/example.jpg",
    "pred": 0.8734
  }
]
```

`pred` is the model's confidence that the image is fully AI-generated, from
`0.0` to `1.0`. Inference automatically uses CUDA, then Apple MPS, then CPU
when `--device auto` is selected.

## Limitations

- The reported checkpoint used a 10 GB subset rather than the complete
  SID-Set training split.
- Internal validation is much easier than cross-source evaluation; the
  external AUC fell from near-perfect internal values to 0.9200.
- Source-level JPEG history can remain after canonicalisation.
- Recall at threshold 0.5 is lower than precision under external domain shift.
- The binary model does not claim to detect partially manipulated images or
  localise edited regions.

## Further documentation

- [`data_loader/README.md`](data_loader/README.md)
- [`pipeline/README.md`](pipeline/README.md)
- [`models/README.md`](models/README.md)
- [`model_runs/README.md`](model_runs/README.md)
