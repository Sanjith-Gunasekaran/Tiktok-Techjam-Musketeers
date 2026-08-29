# Robust AI-Generated Image Detection

Hackathon prototype that distinguishes AI-generated images from authentic
ones and stays accurate after real-world degradations (JPEG re-encoding,
blur, resizing, noise, color filters, cropping).

**Approach:** two complementary detectors fused at the score level — a
semantic branch (frozen DINOv2 features + small trained head) and a low-level
forensics branch (noise-fingerprint analysis of the simplest image patch,
after ESSP) — trained with the challenge's transformations as augmentation.

## Team conventions

- **Binary labels: `0 = real`, `1 = fully synthetic`. Tampered images are
  excluded** because partial editing is a separate detection task. The shared
  rule is `data_loader.SID_SET_BINARY_LABEL_MAP`.
- Primary dataset: [SID-Set](https://huggingface.co/datasets/saberzl/SID_Set).
  The WildFake DALL·E-Advanced subset and COCO val2017 are the organizers'
  demo benchmark — never used in training.

## Repository layout

| Folder | What lives there | Docs |
| --- | --- | --- |
| `data_loader/` | Fetching + decoding images (local Parquet, Hugging Face, Kaggle); label mapping | [data_loader/README.md](data_loader/README.md) |
| `pipeline/` | Augmentations, preprocessing, fixed splits, and torch DataLoaders | [pipeline/README.md](pipeline/README.md) |

## Setup

```bash
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/
```

Covers loading, label exclusion, augmentation, branch views, fixed splits,
and the torch DataLoader factory.

## Roadmap

- [x] Dataset loader (local Parquet / HF / Kaggle) with team label convention
- [x] Augmentation module (brief's six transforms; train + eval modes)
- [x] Committed pytest suite (`tests/`)
- [x] Two-branch preprocessing (DINOv2 view + simplest-patch view)
- [x] Internal test-set carve-out
- [x] PyTorch Dataset wrapper + DataLoader factory
- [ ] Evaluation harness (clean vs. transformed table)
- [ ] DINOv2 head, SRM forensics branch, score fusion
- [ ] Inference script (image dir → JSON of `image_path`, `pred`)

This root README will become the final consolidated project documentation
(overview, setup, reproduction steps, limitations) before submission.
