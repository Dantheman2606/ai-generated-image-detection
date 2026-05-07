#!/usr/bin/env python3
import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ==========================================
# 1. Config & Setup
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
SUBSET_SIZE = 500  # "Part of the dataset"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "/home/daniel/datasets/ai-gen/mnt/c/Development/ai-gen-detection/shard_0/shard_0"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ensemble_results.json")

# ==========================================
# 2. Models (from your architecture)
# ==========================================
def _conv_block(in_ch, out_ch, pool=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class F3Net(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.rgb_stream = nn.Sequential(_conv_block(3, 32), _conv_block(32, 64), _conv_block(64, 128))
        self.fft_stream = nn.Sequential(_conv_block(1, 32), _conv_block(32, 64), _conv_block(64, 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(p=dropout/2), nn.Linear(64, 1))
    def forward(self, rgb, fft):
        r_feat = self.gap(self.rgb_stream(rgb)).flatten(1)
        f_feat = self.gap(self.fft_stream(fft)).flatten(1)
        return self.head(torch.cat([r_feat, f_feat], dim=1))

class DINOv2Embedder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    @torch.no_grad()
    def forward(self, x):
        return self.backbone(x)

class MLPClassifier(nn.Module):
    def __init__(self, embed_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(embed_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
    def forward(self, x): return self.net(x)

class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, embed_dim, dropout=0.3):
        super().__init__()
        self.embedder = DINOv2Embedder(backbone)
        self.classifier = MLPClassifier(embed_dim, dropout=dropout)
    def forward(self, x):
        return self.classifier(self.embedder(x))

# ==========================================
# 3. Load Respective Weights
# ==========================================
print("Loading Models...")
effnet = timm.create_model("efficientnet_b4", num_classes=1)
effnet.load_state_dict(torch.load(os.path.join(BASE_DIR, "training", "baseline", "effnetb4_finetuned.pth"), map_location=DEVICE, weights_only=False))
effnet = effnet.to(DEVICE).eval()

f3net = F3Net(dropout=0.3)
f3net.load_state_dict(torch.load(os.path.join(BASE_DIR, "training", "freq", "f3net_best.pth"), map_location=DEVICE, weights_only=False))
f3net = f3net.to(DEVICE).eval()

embed_dim = 384
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
dino = DINOv2Classifier(backbone, embed_dim=embed_dim, dropout=0.0)
dino.load_state_dict(torch.load(os.path.join(BASE_DIR, "training", "embedding", "dinov2_mlp_best.pth"), map_location=DEVICE, weights_only=False))
dino = dino.to(DEVICE).eval()

# ==========================================
# 4. Dataset & Transforms
# ==========================================
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tf_effnet = transforms.Compose([transforms.Resize(352), transforms.CenterCrop(320), transforms.ToTensor()])
tf_f3net_rgb = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
tf_dino = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])

def get_fft(img):
    gray = np.array(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    fft = np.fft.fft2(gray)
    mag = np.log1p(np.abs(np.fft.fftshift(fft)))
    if mag.max() - mag.min() > 1e-8:
        mag = (mag - mag.min()) / (mag.max() - mag.min())
    return torch.from_numpy(mag.astype(np.float32)).unsqueeze(0)

class ShardSubsetDataset(Dataset):
    def __init__(self, root, size):
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.labels = df.sample(n=size, random_state=SEED).reset_index(drop=True)
        self.root = root

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        row = self.labels.iloc[i]
        img = Image.open(os.path.join(self.root, "images", row["image_name"])).convert("RGB")
        return tf_effnet(img), tf_f3net_rgb(img), get_fft(img), tf_dino(img), row["label"]

# ==========================================
# 5. Run Inference & Save
# ==========================================
dataset = ShardSubsetDataset(DATA_DIR, SUBSET_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds, all_labels = [], []

print(f"Testing ensemble on {SUBSET_SIZE} images...")
with torch.no_grad():
    for rgb_eff, rgb_f3, fft_f3, rgb_dn, labels in loader:
        rgb_eff, rgb_f3, fft_f3, rgb_dn = rgb_eff.to(DEVICE), rgb_f3.to(DEVICE), fft_f3.to(DEVICE), rgb_dn.to(DEVICE)
        
        prob_eff = torch.sigmoid(effnet(rgb_eff)).flatten()
        prob_f3 = torch.sigmoid(f3net(rgb_f3, fft_f3)).flatten()
        prob_dn = torch.sigmoid(dino(rgb_dn)).flatten()
        
        # Weighted Ensemble: 0.5 * EffNet + 0.3 * F3Net + 0.2 * DINO
        ensemble_prob = (0.5 * prob_eff) + (0.3 * prob_f3) + (0.2 * prob_dn)
        
        all_preds.extend(ensemble_prob.cpu().tolist())
        all_labels.extend(labels.tolist())

# Metrics
all_preds_np = np.array(all_preds)
all_labels_np = np.array(all_labels)

preds_bin = (all_preds_np > 0.5).astype(int)
acc = accuracy_score(all_labels_np, preds_bin)
auc = roc_auc_score(all_labels_np, all_preds_np)
f1 = f1_score(all_labels_np, preds_bin)

results = {
    "evaluation_size": SUBSET_SIZE,
    "accuracy": acc,
    "roc_auc": auc,
    "f1_score": f1,
    "predictions": [
        {"true_label": int(l), "ensemble_prob": float(p)}
        for l, p in zip(all_labels_np, all_preds_np)
    ]
}

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(f"Done! Results saved to {OUT_JSON}")
print(f"Metrics: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
