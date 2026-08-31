# pipeline — the layer between data and the model

These files turn raw SID-Set images into consistent model inputs, then run the
final held-out evaluation. They are shared by training and evaluation so the
model sees the same preprocessing in both places.

```text
raw image
  -> augmentation (training only)
  -> DINO view + simplest-patch view
  -> model
  -> held-out evaluation report
```

## Files

| File | Role |
| --- | --- |
| `augmentations.py` | Random real-world degradation for training and the fixed clean/transform grid for evaluation. |
| `preprocess.py` | Makes the normalized 224×224 DINO tensor and raw 32×32 lowest-texture patch from the same image. |
| `splits.py` | Deterministically divides the published validation split into validation and one frozen internal test set. Related image families stay together. |
| `torch_dataset.py` | PyTorch `Dataset`/`DataLoader` factory: load, filter, augment, preprocess, batch, and shuffle training data. |
| `evaluate.py` | Runs a model on the frozen test set, clean and transformed, and writes the required report. |

## Training inputs

```python
from pipeline import create_dataloaders

loaders = create_dataloaders("data", batch_size=64, num_workers=4, seed=42)

for dino, patch, label, original_label in loaders.train:
    train_step(dino, patch, label)
```

Single-branch training can pass `view="dino"` or `view="forensic"` to avoid
building the unused input. The default `view="both"` is for fusion/evaluation.

- Training: published train split; shuffled and randomly augmented each epoch.
- Validation: fixed development subset; no augmentation or shuffle.
- Test: fixed holdout from the published validation split; no augmentation or
  shuffle. Do not use `loaders.test` while choosing models or thresholds.
- Tampered rows are discarded. Labels are `0` real and `1` fully synthetic.

`num_workers` is the number of PyTorch helper **processes** that load and
preprocess batches ahead of the training process. It changes speed, not data
membership or model results.

## Final evaluation

```python
from pipeline import evaluate_model

report = evaluate_model(model, loaders, output_dir="evaluation")
```

The model receives `(dino_batch, patch_batch)` and returns one synthetic-class
score per image. The evaluator uses only the frozen test loader, reruns the
model for clean and every fixed transform, and writes:

- `clean_vs_transformed.csv` and `.md`: accuracy, AUC, and other
  confusion-matrix metrics per condition.
- `error_analysis.csv`: false positives and false negatives only.
- `predictions.jsonl.gz`: every prediction, only when `save_predictions=True`.

Choose the checkpoint and threshold on validation first. The test report is a
single final measurement, not cross-validation.
