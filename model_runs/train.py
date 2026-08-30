"""Run one SID-Set image through the complete two-branch detector."""

from pathlib import Path

import torch

from data_loader.local_image_batch_loader import SIDDataset
from models.dino_classifier import DINOClassifier
from models.forensic_cnn import ForensicCNN
from models.two_branch_detector import TwoBranchDetector
from pipeline.preprocess import two_views


if __name__ == "__main__":
    sample = SIDDataset().get_random_batch(
        batch_size=1,
        split="train",
        data_dir=Path("data/data"),
        seed=0,
    )[0]

    dino_image, forensic_patch = two_views(sample["image"])
    dino_image = dino_image.unsqueeze(0)
    forensic_patch = forensic_patch.unsqueeze(0)

    torch.manual_seed(67)

    model = TwoBranchDetector(
        dino=DINOClassifier(freeze_backbone=True),
        forensic=ForensicCNN(),
    )
    model.eval()

    with torch.no_grad():
        probability = model.predict_proba(dino_image, forensic_patch)

    print("image:", sample["img_id"])
    print("true label:", sample["binary_label"], "(0=real, 1=synthetic)")
    print("synthetic probability:", probability.item())
