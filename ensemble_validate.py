#!/usr/bin/env python3
"""
ensemble_validate.py
--------------------
Loads three trained AI-generated image detection models and evaluates
a weighted ensemble on a random subset of the dataset.

Models
------
  • EfficientNet-B4  (spatial)
  • F3Net            (RGB + FFT frequency domain)
  • DINOv2 + MLP     (self-supervised ViT embedding)

Ensemble weights
----------------
  final_score = 0.5 * effnet + 0.3 * f3net + 0.2 * dino

Run
---
  python ensemble_validate.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. Imports + Setup
# ══════════════════════════════════════════════════════════════════════════════
import os
import random
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

SEED        = 42
SAMPLE_SIZE = 3000
BATCH_SIZE  = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Model Definitions
# ══════════════════════════════════════════════════════════════════════════════

# ── F3Net ─────────────────────────────────────────────────────────────────────
def _conv_block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
    """Conv2d(3×3) → BN → ReLU [ → MaxPool(2×2) ]."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class F3Net(nn.Module):
    """
    Frequency-Focused Forgery Detection Network.
    Dual-stream: RGB (spatial) + FFT magnitude (frequency domain).
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.rgb_stream = nn.Sequential(
            _conv_block(3,   32),
            _conv_block(32,  64),
            _conv_block(64, 128),
        )
        self.fft_stream = nn.Sequential(
            _conv_block(1,   32),
            _conv_block(32,  64),
            _conv_block(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, rgb: torch.Tensor, fft: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.gap(self.rgb_stream(rgb)).flatten(1)   # (B, 128)
        fft_feat = self.gap(self.fft_stream(fft)).flatten(1)   # (B, 128)
        return self.head(torch.cat([rgb_feat, fft_feat], dim=1))  # (B, 1)


# ── DINOv2 + MLP ──────────────────────────────────────────────────────────────
class DINOv2Embedder(nn.Module):
    """Frozen DINOv2 backbone — always runs without gradients."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MLPClassifier(nn.Module):
    """Two-layer MLP head: embed_dim → 256 → 1."""

    def __init__(self, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOv2Classifier(nn.Module):
    """End-to-end: image → DINOv2 backbone (frozen) → MLP head → logit."""

    def __init__(self, backbone: nn.Module, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedder   = DINOv2Embedder(backbone)
        self.classifier = MLPClassifier(embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedder(x))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Load Models
# ══════════════════════════════════════════════════════════════════════════════
_ROOT = os.path.dirname(os.path.abspath(__file__))

print("\nLoading models …")

# EfficientNet-B4
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

# F3Net
f3net_model = F3Net(dropout=0.3)
f3net_model.load_state_dict(
    torch.load(
        os.path.join(_ROOT, "training", "freq", "f3net_best.pth"),
        map_location=DEVICE,
        weights_only=False,
    )
)
f3net_model = f3net_model.to(DEVICE).eval()
print("  ✓ F3Net")

# DINOv2 + MLP  —  ViT-S/14, EMBED_DIM = 384
EMBED_DIM = 384
_dino_backbone = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=False,   # weights come from our checkpoint
)
dino_model = DINOv2Classifier(_dino_backbone, embed_dim=EMBED_DIM, dropout=0.0)
dino_model.load_state_dict(
    torch.load(
        os.path.join(_ROOT, "training", "embedding", "dinov2_mlp_best.pth"),
        map_location=DEVICE,
        weights_only=False,
    )
)
dino_model = dino_model.to(DEVICE).eval()
print("  ✓ DINOv2 + MLP")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Dataset
# ══════════════════════════════════════════════════════════════════════════════
DATASET_PATH = (
    "/home/daniel/datasets/ai-gen/mnt/c/Development/ai-gen-detection/shard_0/shard_0"
)

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

# EfficientNet-B4: effnetb4_finetuned.pth was saved at the end of Stage 3,
# which used Resize(352) → CenterCrop(320) → ToTensor() with NO normalization.
_effnet_transform = transforms.Compose([
    transforms.Resize(352),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
])

# F3Net: RGB resized directly to 256 × 256 (matches training val_transform)
_f3net_rgb_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

# DINOv2: Resize(256, BICUBIC) → CenterCrop(224)  (ImageNet eval protocol)
_dino_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def _fft_magnitude(pil_img: Image.Image, size: int = 256) -> torch.Tensor:
    """Log-scaled 2-D FFT magnitude spectrum → (1, size, size) float32 tensor."""
    gray = np.array(
        pil_img.convert("L").resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    fft       = np.fft.fft2(gray)
    magnitude = np.log1p(np.abs(np.fft.fftshift(fft)))

    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max - mag_min > 1e-8:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)

    return torch.from_numpy(magnitude.astype(np.float32)).unsqueeze(0)  # (1, H, W)


class EnsembleDataset(Dataset):
    """
    Produces pre-processed tensors for all three models in a single pass so the
    DataLoader can batch them efficiently.

    Returns
    -------
    rgb_effnet : (3, 380, 380)  — normalised for EfficientNet-B4
    rgb_f3net  : (3, 256, 256)  — normalised for F3Net RGB stream
    fft_f3net  : (1, 256, 256)  — log-FFT magnitude for F3Net frequency stream
    rgb_dino   : (3, 224, 224)  — normalised for DINOv2
    label      : LongTensor scalar
    """

    def __init__(self, shard_path: str, indices: list):
        self.shard_path = shard_path
        self.labels = (
            pd.read_csv(os.path.join(shard_path, "labels.csv"))
            .iloc[indices]
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        row      = self.labels.iloc[idx]
        img_path = os.path.join(self.shard_path, "images", row["image_name"])
        pil_img  = Image.open(img_path).convert("RGB")

        rgb_effnet = _effnet_transform(pil_img)
        rgb_f3net  = _f3net_rgb_transform(pil_img)
        fft_f3net  = _fft_magnitude(pil_img, size=256)
        rgb_dino   = _dino_transform(pil_img)

        label = torch.tensor(row["label"], dtype=torch.long)
        return rgb_effnet, rgb_f3net, fft_f3net, rgb_dino, label


# ── Randomly sample ~3000 images ──────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at:\n  {DATASET_PATH}")

all_labels  = pd.read_csv(os.path.join(DATASET_PATH, "labels.csv"))
all_indices = list(range(len(all_labels)))
random.shuffle(all_indices)
sample_indices = all_indices[:SAMPLE_SIZE]

dataset = EnsembleDataset(DATASET_PATH, indices=sample_indices)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

print(f"\nDataset : {len(dataset)} / {len(all_labels)} total images sampled")
print(f"Batches : {len(loader)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Prediction Functions  (single-image convenience API)
# ══════════════════════════════════════════════════════════════════════════════
_AMP_ENABLED = torch.cuda.is_available()


@torch.no_grad()
def predict_effnet(image: Image.Image) -> float:
    """Return P(AI-generated) for one PIL image using EfficientNet-B4."""
    tensor = _effnet_transform(image).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=_AMP_ENABLED):
        logit = effnet_model(tensor)
    return torch.sigmoid(logit).item()


@torch.no_grad()
def predict_f3net(image: Image.Image) -> float:
    """Return P(AI-generated) for one PIL image using F3Net (RGB + FFT)."""
    rgb = _f3net_rgb_transform(image).unsqueeze(0).to(DEVICE)
    fft = _fft_magnitude(image, size=256).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=_AMP_ENABLED):
        logit = f3net_model(rgb, fft)
    return torch.sigmoid(logit).item()


@torch.no_grad()
def predict_dino(image: Image.Image) -> float:
    """Return P(AI-generated) for one PIL image using DINOv2 + MLP."""
    tensor = _dino_transform(image).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=_AMP_ENABLED):
        logit = dino_model(tensor)
    return torch.sigmoid(logit).item()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Ensemble Validation
# ══════════════════════════════════════════════════════════════════════════════
def run_ensemble_validation() -> None:
    all_ensemble_scores: list[float] = []
    all_effnet_scores:   list[float] = []
    all_f3net_scores:    list[float] = []
    all_dino_scores:     list[float] = []
    all_true_labels:     list[int]   = []

    for rgb_effnet, rgb_f3net, fft_f3net, rgb_dino, labels in tqdm(
        loader, desc="Ensemble inference"
    ):
        rgb_effnet = rgb_effnet.to(DEVICE, non_blocking=True)
        rgb_f3net  = rgb_f3net.to(DEVICE,  non_blocking=True)
        fft_f3net  = fft_f3net.to(DEVICE,  non_blocking=True)
        rgb_dino   = rgb_dino.to(DEVICE,   non_blocking=True)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=_AMP_ENABLED):
            effnet_score = torch.sigmoid(effnet_model(rgb_effnet)).squeeze(1)
            f3net_score  = torch.sigmoid(f3net_model(rgb_f3net, fft_f3net)).squeeze(1)
            dino_score   = torch.sigmoid(dino_model(rgb_dino)).squeeze(1)

        final_score = 0.5 * effnet_score + 0.3 * f3net_score + 0.2 * dino_score

        all_ensemble_scores.extend(final_score.cpu().numpy())
        all_effnet_scores.extend(effnet_score.cpu().numpy())
        all_f3net_scores.extend(f3net_score.cpu().numpy())
        all_dino_scores.extend(dino_score.cpu().numpy())
        all_true_labels.extend(labels.numpy())

    ensemble_arr = np.array(all_ensemble_scores)
    labels_arr   = np.array(all_true_labels)

    roc_auc = roc_auc_score(labels_arr, ensemble_arr)

    print("\n" + "=" * 56)
    print("  Ensemble Validation Results")
    print("=" * 56)
    print(f"  Samples evaluated   : {len(ensemble_arr)}")
    print(f"  ROC-AUC (ensemble)  : {roc_auc:.5f}")
    print(f"  ── Average prediction scores ──")
    print(f"  Ensemble            : {ensemble_arr.mean():.4f}")
    print(f"  EfficientNet-B4     : {np.array(all_effnet_scores).mean():.4f}")
    print(f"  F3Net               : {np.array(all_f3net_scores).mean():.4f}")
    print(f"  DINOv2 + MLP        : {np.array(all_dino_scores).mean():.4f}")
    print("=" * 56)


if __name__ == "__main__":
    run_ensemble_validation()
