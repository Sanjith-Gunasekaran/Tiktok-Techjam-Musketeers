# Robust AI-Generated Image Detection

Hackathon prototype that distinguishes AI-generated images from authentic
ones and stays accurate after real-world degradations (JPEG re-encoding,
blur, resizing, noise, color filters, cropping).

**Approach:** two complementary detectors fused at the score level — a
semantic branch (frozen DINOv2 features + small trained head) and a low-level
forensics branch (noise-fingerprint analysis of the simplest image patch,
after ESSP) — trained with the challenge's transformations as augmentation.

## Team conventions

- **Binary labels: `0 = real`, `1 = AI`. Tampered counts as AI.** The shared
  mapping is `data_loader.SID_SET_BINARY_LABEL_MAP`; samples keep their
  original 3-class label for error analysis.
- Primary dataset: [SID-Set](https://huggingface.co/datasets/saberzl/SID_Set).
  The WildFake DALL·E-Advanced subset and COCO val2017 are the organizers'
  demo benchmark — never used in training.

## Repository layout

| Folder | What lives there | Docs |
| --- | --- | --- |
| `data_loader/` | Fetching + decoding images (local Parquet, Hugging Face, Kaggle); label mapping | [data_loader/README.md](data_loader/README.md) |
| `pipeline/` | Augmentations; (upcoming) two-branch preprocessing, splits, torch Dataset, eval harness | [pipeline/README.md](pipeline/README.md) |

## Setup

```bash
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/
```

Covers augmentations, both preprocessing views, the split carve-out, and the
loader against SID-Set's real schema (bare int64 labels, mask column,
duplicate/partial shard downloads).

## Roadmap

- [x] Dataset loader (local Parquet / HF / Kaggle) with team label convention
- [x] Augmentation module (brief's six transforms; train + eval modes)
- [x] Committed pytest suite (`tests/`)
- [ ] Two-branch preprocessing (DINOv2 view + simplest-patch view)
- [ ] Internal test-set carve-out
- [ ] PyTorch Dataset wrapper + DataLoader factory
- [ ] Evaluation harness (clean vs. transformed table)
- [ ] DINOv2 head, SRM forensics branch, score fusion
- [ ] Inference script (image dir → JSON of `image_path`, `pred`)

This root README will become the final consolidated project documentation
(overview, setup, reproduction steps, limitations) before submission.
