# model runs

`train.py` trains one stage at a time. It never uses the frozen internal test
loader; checkpoint and threshold decisions belong to validation.

```text
dino_head → dino_finetune ┐
                           ├→ fusion → final frozen-test evaluation
forensic ──────────────────┘
```

Run it from the repository root with either form:

```bash
python -m model_runs.train --stage dino_head --data-dir saberzl/SID_Set
# or
python model_runs/train.py --stage dino_head --data-dir saberzl/SID_Set
```

## Stages

| Stage | What trains | Required input checkpoint(s) |
| --- | --- | --- |
| `dino_head` | DINO classification head; backbone remains frozen. | None |
| `dino_finetune` | DINO head and final transformer blocks. | `--dino-checkpoint` from `dino_head` |
| `forensic` | SRM-forensic CNN; SRM filters remain fixed. | None |
| `fusion` | Learned linear fusion only; both branches are frozen. | `--dino-checkpoint` and `--forensic-checkpoint` |

Each run writes `best.pt` and `last.pt` under:

```text
model_runs/checkpoints/<stage>/
```

`best.pt` is selected by validation AUC. `last.pt` supports resuming and
retains the optimizer, scheduler, scaler, and RNG states. Each epoch also
appends metrics and learning rates to `history.jsonl`.

## Recommended commands

First verify each stage with a few batches. This exercises loading, shapes,
GPU memory, and checkpoint handling cheaply:

```bash
python -m model_runs.train --stage dino_head --data-dir saberzl/SID_Set \
  --epochs 1 --max-train-batches 10 --max-validation-batches 10

python -m model_runs.train --stage forensic --data-dir saberzl/SID_Set \
  --epochs 1 --max-train-batches 10 --max-validation-batches 10
```

Then run the branch stages:

```bash
python -m model_runs.train --stage dino_head --data-dir saberzl/SID_Set --epochs 5

python -m model_runs.train --stage dino_finetune --data-dir saberzl/SID_Set \
  --dino-checkpoint model_runs/checkpoints/dino_head/best.pt --epochs 10

python -m model_runs.train --stage forensic --data-dir saberzl/SID_Set --epochs 10
```

Finally fit learned fusion from the selected branch checkpoints:

```bash
python -m model_runs.train --stage fusion --data-dir saberzl/SID_Set \
  --dino-checkpoint model_runs/checkpoints/dino_finetune/best.pt \
  --forensic-checkpoint model_runs/checkpoints/forensic/best.pt --epochs 5
```

Fusion is not trained on `loaders.train`: it runs the frozen branches once on
the dedicated family-disjoint calibration loader and caches their logits. The
learned layer fits those scores; `best.pt` is selected on the separate
validation loader. Neither operation touches the frozen test loader. Each
branch is rebuilt from its checkpoint configuration, including the forensic
SRM clipping setting.

For interpretable fixed fusion, calibrate rather than train a layer:

```bash
python -m model_runs.calibrate_fusion --data-dir saberzl/SID_Set \
  --dino-checkpoint model_runs/checkpoints/dino_finetune/best.pt \
  --forensic-checkpoint model_runs/checkpoints/forensic/best.pt
```

It searches fixed DINO weights on validation AUC, calibrates the probability
threshold on the separate calibration partition, and writes `fusion/fixed.json`.

## External benchmark

After fusion is fixed, evaluate a separately sourced image benchmark without
retraining or recalibrating anything:

```bash
python -m model_runs.evaluate_external \
  --input-dir /path/to/benchmark \
  --fusion-config /path/to/run/fusion.json \
  --dino-checkpoint /path/to/run/dino_finetune/best.pt \
  --forensic-checkpoint /path/to/run/forensic/best.pt \
  --output-dir /path/to/run/external_report
```

The directory form scans recursively. Standard class directories including
`real`, `authentic`, `fake`, `synthetic`, and `ai_generated` are mapped to
binary labels; unrecognised directories remain unlabelled and still receive
predictions. Use `--no-infer-labels` for a deliberately unlabelled benchmark.

For benchmarks distributed with metadata, pass a CSV manifest instead:

```csv
image_id,path,label
sample_001,images/sample_001.png,real
sample_002,images/sample_002.jpg,synthetic
```

```bash
python -m model_runs.evaluate_external \
  --manifest /path/to/benchmark.csv \
  --image-root /path/to/benchmark_root \
  --fusion-config /path/to/run/fusion.json \
  --dino-checkpoint /path/to/run/dino_finetune/best.pt \
  --forensic-checkpoint /path/to/run/forensic/best.pt \
  --output-dir /path/to/run/external_report
```

The report directory contains `predictions.csv`, `errors.csv`, `summary.json`,
`summary.md`, and `run_metadata.json`. Metrics use only rows with supplied
labels. External images receive the same deterministic JPEG canonicalization
and branch preprocessing used during training, but no random or robustness
augmentation. A moved run often leaves old Colab paths inside `fusion.json`;
the two explicit checkpoint arguments above safely override those paths.

Resume an interrupted run with the same stage and settings:

```bash
python -m model_runs.train --stage forensic --data-dir saberzl/SID_Set \
  --resume model_runs/checkpoints/forensic/last.pt --epochs 10
```

`--epochs` is the final target epoch. If a checkpoint is already at that epoch,
there is nothing left to train.

## Tuning safely

- Default augmentation is robust training (`0.5` first transform, `0.3`
  second transform). Use `--no-augment` only for a clean baseline.
- Start with the default learning rates: head `1e-4`, DINO blocks `1e-5`,
  forensic `1e-3`, fusion `1e-3`. Change one setting at a time.
- CUDA runs use mixed precision, gradient clipping (`1.0`), one warmup epoch,
  then cosine decay to 10% of the starting rate. Use `--no-amp`,
  `--no-scheduler`, `--grad-clip-norm`, `--warmup-epochs`, or
  `--min-lr-scale` only for a deliberate comparison.
- Use `--unfreeze-blocks 1` or `2` for DINO fine-tuning. Do not fully unfreeze
  DINO as the first experiment.
- `--srm-clip-value 3.0` is the baseline. Measure clipping before changing it.
- Lower `--batch-size` if GPU memory is exhausted; increase `--num-workers`
  only after confirming CPU loading is the bottleneck.
- DINO stages build only DINO tensors; the forensic stage builds only simplest
  patches. Fusion and final evaluation still build both views.
- Keep checkpoint architecture settings consistent. The trainer rejects a
  wrong stage and rejects branch checkpoints with incompatible DINO/SRM setup.
- Do not choose a model, fusion rule, or threshold from `loaders.test`.

The trainer currently requires both classes in validation to calculate AUC. If
a bounded smoke run raises a one-class validation error, increase
`--max-validation-batches` or inspect the split; do not suppress the error.

## Before the expensive run

Use a GPU machine with enough disk/cache space for the dataset and DINO model.
Dataset construction removes every training family found in published
validation and applies the same quality-75 JPEG canonicalization to all
partitions before augmentation. These preprocessing changes invalidate older
checkpoints; start every branch in a new output directory after updating.
Confirm all four smoke commands complete and that each checkpoint can be used
by the following stage. After all choices are frozen, run
`pipeline.evaluate_model(..., from_logits=True)` once on the internal test set.
