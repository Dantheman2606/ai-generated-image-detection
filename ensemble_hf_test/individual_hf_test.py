#!/usr/bin/env python3
import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from datasets import load_dataset

# ==========================================
# 1. Config & Setup
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
SUBSET_SIZE = 2000  # Evaluates on 1k images (or up to subset size)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_individual_results.json")

# ==========================================
# 2. Models (from your architecture)
# ==========================================
def _conv_block(in_ch, out_ch, pool=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), 
              nn.BatchNorm2d(out_ch), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class F3Net(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.rgb_stream = nn.Sequential(_conv_block(3, 32), _conv_block(32, 64), _conv_block(64, 128))
        self.fft_stream = nn.Sequential(_conv_block(1, 32), _conv_block(32, 64), _conv_block(64, 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Linear(256, 64), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=dropout/2), 
            nn.Linear(64, 1)
        )
        
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

class HFSubsetDataset(Dataset):
    def __init__(self, hf_ds, size):
        self.ds = hf_ds.shuffle(seed=SEED).select(range(min(size, len(hf_ds))))

    def __len__(self): 
        return len(self.ds)
        
    def __getitem__(self, i):
        item = self.ds[i]
        
        # Try to find the image key
        if "image" in item:
            img = item["image"]
        elif "img" in item:
            img = item["img"]
        else:
            # just take the first PIL Image we find
            for v in item.values():
                if isinstance(v, Image.Image):
                    img = v
                    break
                    
        img = img.convert("RGB")
        
        # Try to find the label key
        label_keys = ["label", "labels", "target", "class", "is_ai", "fake"]
        label = 0
        for k in label_keys:
            if k in item:
                label = item[k]
                break
        else:
            # If standard keys fail, try any int/bool column
            for k, v in item.items():
                if isinstance(v, (int, bool)):
                    label = int(v)
                    break
                elif isinstance(v, str) and v.lower() in ["fake", "real", "ai", "human"]:
                    label = 1 if v.lower() in ["fake", "ai"] else 0
                    break
        
        # Invert the label from the dataset
        label = 1 - int(label)
        
        return tf_effnet(img), tf_f3net_rgb(img), get_fft(img), tf_dino(img), label

# ==========================================
# 5. Run Inference & Save
# ==========================================
if __name__ == "__main__":
    # print("Downloading/Loading Hugging Face Dataset 'Parveshiiii/AI-vs-Real'...")
    # from huggingface_hub import HfFileSystem
    # fs = HfFileSystem()
    # parquet_files = fs.glob("datasets/Parveshiiii/AI-vs-Real/**/*.parquet")
    # selected_files = ["hf://" + f for f in parquet_files][:2]
    # print(f"Loading exactly {len(selected_files)} parquets: {selected_files}")
    # 
    # raw_dataset = load_dataset(
    #     "parquet", 
    #     data_files=selected_files, 
    #     split="train", 
    #     verification_mode="no_checks"
    # ) 

    print("Downloading/Loading Hugging Face Dataset 'Hemg/AI-Generated-vs-Real-Images-Datasets'...")
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    parquet_files = fs.glob("datasets/Hemg/AI-Generated-vs-Real-Images-Datasets/**/*.parquet")
    
    # Grab the first and last parquet files to ensure we get a mix of both classes 
    # (in case the parquets are sorted by class folder)
    if len(parquet_files) > 1:
        selected_files = ["hf://" + f for f in [parquet_files[0], parquet_files[-1]]]
    else:
        selected_files = ["hf://" + f for f in parquet_files]
        
    print(f"Loading exactly {len(selected_files)} parquets: {selected_files}")
    
    raw_dataset = load_dataset(
        "parquet", 
        data_files=selected_files, 
        split="train", 
        verification_mode="no_checks"
    ) 
    
    dataset = HFSubsetDataset(raw_dataset, SUBSET_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_preds_eff, all_preds_f3, all_preds_dn = [], [], []
    all_labels = []

    print(f"Testing individual models on {len(dataset)} images from HF...")
    with torch.no_grad():
        for batch_idx, (rgb_eff, rgb_f3, fft_f3, rgb_dn, labels) in enumerate(loader):
            rgb_eff = rgb_eff.to(DEVICE)
            rgb_f3 = rgb_f3.to(DEVICE)
            fft_f3 = fft_f3.to(DEVICE)
            rgb_dn = rgb_dn.to(DEVICE)
            
            prob_eff = torch.sigmoid(effnet(rgb_eff)).flatten()
            prob_f3 = torch.sigmoid(f3net(rgb_f3, fft_f3)).flatten()
            prob_dn = torch.sigmoid(dino(rgb_dn)).flatten()
            
            all_preds_eff.extend(prob_eff.cpu().tolist())
            all_preds_f3.extend(prob_f3.cpu().tolist())
            all_preds_dn.extend(prob_dn.cpu().tolist())
            all_labels.extend(labels.tolist())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(loader)}")

    # Metrics
    all_labels_np = np.array(all_labels)
    unique_labels, counts = np.unique(all_labels_np, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")

    def calculate_metrics(preds):
        preds_np = np.array(preds)
        preds_bin = (preds_np > 0.5).astype(int)
        acc = accuracy_score(all_labels_np, preds_bin)
        
        # roc_auc_score fails/returns nan if only one class is present in true labels
        try:
            auc = roc_auc_score(all_labels_np, preds_np)
        except ValueError:
            auc = float('nan')
            
        f1 = f1_score(all_labels_np, preds_bin)
        return {"accuracy": acc, "roc_auc": auc, "f1_score": f1}

    results = {
        "dataset": "Hemg/AI-Generated-vs-Real-Images-Datasets",
        "evaluation_size": len(dataset),
        "models": {
            "EfficientNetB4": calculate_metrics(all_preds_eff),
            "F3Net": calculate_metrics(all_preds_f3),
            "DINOv2": calculate_metrics(all_preds_dn)
        }
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Done! Results saved to {OUT_JSON}")
    for model_name, metrics in results["models"].items():
        print(f"{model_name} -> Acc={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}")
