#!/usr/bin/env python3
"""
effnetb4_validate.py
--------------------
Loads the trained EfficientNet-B4 model and runs inference on the
NTIRE validation dataset, writing submission CSVs to effnetb4_outputs/.

Run
---
  python effnetb4_validate.py
"""

import os
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

BATCH_SIZE = 32

_ROOT = os.path.dirname(os.path.abspath(__file__))

# EfficientNet-B4: Resize(352) → CenterCrop(320) → ToTensor(), no normalization
_effnet_transform = transforms.Compose([
    transforms.Resize(352),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
])

# ══════════════════════════════════════════════════════════════════════════════
# Load Model
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading EfficientNet-B4 …")
effnet_model = timm.create_model("efficientnet_b4", num_classes=1)
effnet_model.load_state_dict(
    torch.load(
        os.path.join(_ROOT, "training", "baseline", "effnetb4_finetuned.pth"),
        map_location=DEVICE,
        weights_only=False,
    )
)
effnet_model = effnet_model.to(DEVICE).eval()
print("  ✓ EfficientNet-B4")

# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class InferenceDataset(Dataset):
    """Images-only dataset. Returns (tensor, filename) pairs."""

    def __init__(self, img_dir: str):
        self.img_dir   = img_dir
        self.filenames = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname   = self.filenames[idx]
        pil_img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        return _effnet_transform(pil_img), fname


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════
_AMP_ENABLED = torch.cuda.is_available()


def run_submission(img_dir: str, output_csv: str) -> None:
    """Run EfficientNet-B4 inference on every image in img_dir and write a CSV."""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found:\n  {img_dir}")

    dataset = InferenceDataset(img_dir)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    print(f"\nSubmission : {img_dir}")
    print(f"  Images   : {len(dataset)}")

    image_names: list[str]   = []
    scores:      list[float] = []

    for tensors, fnames in tqdm(
        loader, desc=f"Inference ({os.path.basename(img_dir)})"
    ):
        tensors = tensors.to(DEVICE, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=_AMP_ENABLED):
            logits = effnet_model(tensors)

        batch_scores = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        image_names.extend(fnames)
        scores.extend(batch_scores)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame({"image_name": image_names, "score": scores}).to_csv(
        output_csv, index=False
    )
    print(f"  Saved  → {output_csv}")


if __name__ == "__main__":
    _VAL_DIR  = os.path.join(_ROOT, "ntire_val_dataset", "val_images")
    _HARD_DIR = os.path.join(_ROOT, "ntire_val_dataset", "val_images_hard")
    _OUT_DIR  = os.path.join(_ROOT, "effnetb4_outputs")

    run_submission(_VAL_DIR,  os.path.join(_OUT_DIR, "submission.csv"))
    run_submission(_HARD_DIR, os.path.join(_OUT_DIR, "submission_hard.csv"))
