# models — two complementary classifiers

The pipeline produces two views of each image. These models turn them into
**synthetic-class logits**: positive means more synthetic; negative means more
real. Use `BCEWithLogitsLoss` for training and `sigmoid` only for probabilities.

```text
224×224 normalized image ──> DINOClassifier ──> semantic logit ─┐
                                                                  ├─> fused logit
32×32 raw simplest patch ──> ForensicCNN ──────> forensic logit ─┘
```

## Files

| File | Role |
| --- | --- |
| `dino_classifier.py` | DINOv2 semantic branch. |
| `srm_filters.py` | Fixed high-pass SRM residual filters. |
| `forensic_cnn.py` | Small CNN over SRM residuals. |
| `two_branch_detector.py` | Fixed or learned score-level fusion. |

## DINO semantic branch

`DINOClassifier` receives the pipeline's ImageNet-normalized `(B, 3, 224,
224)` tensor. DINOv2 produces a CLS embedding, and a trainable MLP head returns
one logit per image.

Training stages are explicit:

```python
model.set_backbone_trainable(False)  # linear/MLP probe: head only
model.unfreeze_last_blocks(2)        # final two DINO blocks + final LayerNorm
model.set_backbone_trainable(True)   # full fine-tuning, only if justified
```

Frozen blocks stay in evaluation mode even when the classifier is training.
`revision=` and `local_files_only=` make model loading reproducible; save
`model.checkpoint_config()` beside each checkpoint.

## Forensic branch

`ForensicCNN` receives the raw `(B, 3, 32, 32)` simplest patches in `[0,255]`.
It normalizes them, applies frozen SRM filters, then trains a small 32/64/128
channel CNN to return one logit per patch.

```text
raw patch → normalize → fixed SRM residuals → CNN → forensic logit
```

SRM filters are buffers, not learnable parameters. `srm_clip_value=3.0` is the
SSP-style baseline; set it to `None` only for a validation-controlled ablation.
`model.srm.clipping_fraction(patches)` reports how much a chosen clipping value
would affect normalized patches.

## Fusion

Train and validate the branches independently first. Then combine their logits:

```python
from models import TwoBranchDetector

# Interpretable baseline; choose the weight on validation only.
fixed = TwoBranchDetector(dino, forensic, fusion_mode="fixed", dino_weight=0.8)

# Calibrated alternative; freeze branches and train only fusion on validation-derived training data.
learned = TwoBranchDetector(dino, forensic, fusion_mode="learned")
learned.freeze_branches()
```

Fixed fusion computes `weight × dino + (1 - weight) × forensic`. Learned fusion
uses a linear layer over both logits, learning their relative scale, weight, and
bias. Save `detector.checkpoint_config()` with its state dictionary. After
loading, reapply the intended branch freeze policy for the training stage.

Both detector modes implement:

```python
logits = detector(dino_images, patches)      # (B,)
probabilities = detector.predict_proba(dino_images, patches)
labels = detector.predict(dino_images, patches, threshold=0.5)
```

`predict_proba()` temporarily switches to evaluation mode and restores the
previous mode. The final evaluator must receive logits with `from_logits=True`.

## Out of scope

These files do not load images, augment data, calculate loss, optimize weights,
choose thresholds, or select checkpoints. Those decisions belong in the
training/validation layer and must use validation—not the frozen test set.
