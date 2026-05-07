"""
Ensemble Model Test Script
Tests the ensemble method on a subset of the validation dataset.
Saves results as JSON.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configuration
DATASET_PATH = "/workspace/datasets/ai-gen/mnt/c/Development/ai-gen-detection/shard_0/shard_0"
VAL_DATASET_PATH = "/home/daniel/ai-gen-detection/ntire_val_dataset"
SUBSET_SIZE = 500  # Test on 500 images
BATCH_SIZE = 16

# Model paths
EFFICIENTNET_PATH = "/home/daniel/ai-gen-detection/training/baseline/effnetb4_finetuned.pth"
DINOV2_PATH = "/home/daniel/ai-gen-detection/training/embedding/dinov2_mlp_final.pth"
F3NET_PATH = "/home/daniel/ai-gen-detection/training/freq/f3net_final.pth"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Load Models
# ============================================================================

def load_efficientnet():
    """Load EfficientNet B4 model (1-logit version)"""
    import timm
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1)
    if os.path.exists(EFFICIENTNET_PATH):
        state_dict = torch.load(EFFICIENTNET_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded EfficientNet from {EFFICIENTNET_PATH}")
    else:
        print(f"✗ EfficientNet model not found at {EFFICIENTNET_PATH}")
    return model.to(device).eval()

def load_dinov2():
    """Load DinoV2 + MLP model"""
    try:
        import timm
        # Load DinoV2 backbone
        backbone = timm.create_model("vit_base_patch14_dinov2", pretrained=False)
        
        # Build MLP head
        class DinoV2MLP(nn.Module):
            def __init__(self, backbone, num_classes=1):
                super().__init__()
                self.backbone = backbone
                self.mlp = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.backbone(x)
                return self.mlp(x)
        
        model = DinoV2MLP(backbone, num_classes=1)
        
        if os.path.exists(DINOV2_PATH):
            state_dict = torch.load(DINOV2_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded DinoV2 from {DINOV2_PATH}")
        else:
            print(f"✗ DinoV2 model not found at {DINOV2_PATH}")
        
        return model.to(device).eval()
    except Exception as e:
        print(f"Error loading DinoV2: {e}")
        return None

def load_f3net():
    """Load F3Net model"""
    try:
        # Assuming F3Net is a simple CNN model
        class F3Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(128, 1)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = F3Net()
        if os.path.exists(F3NET_PATH):
            state_dict = torch.load(F3NET_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded F3Net from {F3NET_PATH}")
        else:
            print(f"✗ F3Net model not found at {F3NET_PATH}")
        
        return model.to(device).eval()
    except Exception as e:
        print(f"Error loading F3Net: {e}")
        return None

# ============================================================================
# Dataset
# ============================================================================

class PartialValDataset:
    """Load subset of validation images"""
    def __init__(self, dataset_path, subset_size=500):
        self.dataset_path = dataset_path
        self.subset_size = subset_size
        self.image_dir = os.path.join(dataset_path, "val_images")
        
        # Get list of images
        self.images = []
        if os.path.exists(self.image_dir):
            self.images = sorted(os.listdir(self.image_dir))[:subset_size]
        
        print(f"Loaded {len(self.images)} images from validation set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        return image, img_name

# ============================================================================
# Inference
# ============================================================================

def preprocess_image(image, target_size=384):
    """Preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def get_ensemble_predictions(images_batch):
    """Get predictions from all models and ensemble them"""
    predictions = {}
    
    with torch.no_grad():
        # EfficientNet
        try:
            effnet_logits = effnet(images_batch)
            effnet_probs = torch.sigmoid(effnet_logits).squeeze(-1).cpu().numpy()
            predictions['efficientnet'] = effnet_probs
        except Exception as e:
            print(f"EfficientNet error: {e}")
        
        # DinoV2
        if dinov2 is not None:
            try:
                dinov2_logits = dinov2(images_batch)
                dinov2_probs = torch.sigmoid(dinov2_logits).squeeze(-1).cpu().numpy()
                predictions['dinov2'] = dinov2_probs
            except Exception as e:
                print(f"DinoV2 error: {e}")
        
        # F3Net
        if f3net is not None:
            try:
                f3net_logits = f3net(images_batch)
                f3net_probs = torch.sigmoid(f3net_logits).squeeze(-1).cpu().numpy()
                predictions['f3net'] = f3net_probs
            except Exception as e:
                print(f"F3Net error: {e}")
    
    # Ensemble: average probabilities
    ensemble_prob = np.mean(list(predictions.values()), axis=0)
    
    return ensemble_prob, predictions

# ============================================================================
# Main Testing Loop
# ============================================================================

print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)

effnet = load_efficientnet()
dinov2 = load_dinov2()
f3net = load_f3net()

print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)

dataset = PartialValDataset(VAL_DATASET_PATH, subset_size=SUBSET_SIZE)

if len(dataset) == 0:
    print("✗ No images found in validation dataset")
    print("Attempting to use training dataset...")
    # Fallback to training dataset
    class TrainDataset:
        def __init__(self, dataset_path, subset_size=500):
            self.dataset_path = dataset_path
            self.subset_size = subset_size
            self.image_dir = os.path.join(dataset_path, "images")
            
            import pandas as pd
            labels_file = os.path.join(dataset_path, "labels.csv")
            self.labels = pd.read_csv(labels_file)
            self.images = self.labels['image_name'].values[:subset_size]
            
            print(f"Loaded {len(self.images)} images from training set")
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            label = self.labels[self.labels['image_name'] == img_name]['label'].values[0]
            return image, img_name, label
    
    dataset = TrainDataset(DATASET_PATH, subset_size=SUBSET_SIZE)
    has_labels = True
else:
    has_labels = False

print("\n" + "="*60)
print("TESTING ENSEMBLE")
print("="*60)

results = {
    "metadata": {
        "total_images": len(dataset),
        "models": ["efficientnet", "dinov2", "f3net"],
        "ensemble_method": "average"
    },
    "predictions": [],
    "metrics": {}
}

all_ensemble_probs = []
all_labels = []

with torch.no_grad():
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Testing"):
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(dataset))))
        
        images_batch = []
        image_names = []
        batch_labels = []
        
        for idx in batch_indices:
            if has_labels:
                image, img_name, label = dataset[idx]
                batch_labels.append(label)
            else:
                image, img_name = dataset[idx]
            
            images_batch.append(preprocess_image(image))
            image_names.append(img_name)
        
        images_tensor = torch.stack(images_batch).to(device)
        
        ensemble_prob, model_preds = get_ensemble_predictions(images_tensor)
        
        for j, img_name in enumerate(image_names):
            pred_result = {
                "image": img_name,
                "ensemble_probability": float(ensemble_prob[j]),
                "ensemble_label": int(ensemble_prob[j] >= 0.5)
            }
            
            # Add individual model predictions
            for model_name, probs in model_preds.items():
                pred_result[f"{model_name}_probability"] = float(probs[j])
            
            # Add true label if available
            if has_labels:
                pred_result["true_label"] = int(batch_labels[j])
            
            results["predictions"].append(pred_result)
            all_ensemble_probs.append(ensemble_prob[j])
            
            if has_labels:
                all_labels.append(batch_labels[j])

# Calculate metrics if labels are available
if has_labels and len(all_labels) > 0:
    all_ensemble_preds = np.array([1 if p >= 0.5 else 0 for p in all_ensemble_probs])
    all_ensemble_probs = np.array(all_ensemble_probs)
    all_labels = np.array(all_labels)
    
    results["metrics"] = {
        "accuracy": float(accuracy_score(all_labels, all_ensemble_preds)),
        "precision": float(precision_score(all_labels, all_ensemble_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_ensemble_preds, zero_division=0)),
        "f1_score": float(f1_score(all_labels, all_ensemble_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_labels, all_ensemble_probs))
    }
    
    print("\n" + "="*60)
    print("METRICS")
    print("="*60)
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.4f}")

# Save results to JSON
output_file = "ensemble_test_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_file}")
print(f"✓ Tested {len(dataset)} images")
print(f"✓ Total predictions: {len(results['predictions'])}")
