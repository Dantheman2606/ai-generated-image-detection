# AI-Generated Image Detection

Entry for the [NTIRE 2026 Robust AI-Generated Image Detection in the Wild Challenge](https://www.codabench.org/competitions/12761/) (CVPR 2026 Workshop).

The goal is to classify images as **real** or **AI-generated** while remaining robust to common post-processing (cropping, resizing, compression, etc.) and unseen generators. The final submission uses a weighted ensemble of three complementary models.

---

## Results

| Model | Val Accuracy | Val AUC |
|---|---|---|
| EfficientNet-B4 (spatial) | **96.66 %** | **99.57 %** |
| F3Net (RGB + FFT) | 81.94 % | 88.63 % |
| DINOv2 + MLP (embeddings) | 83.56 % | 93.71 % |
| **Weighted Ensemble** | — | — |

Ensemble weights: `0.5 × EfficientNet-B4 + 0.3 × F3Net + 0.2 × DINOv2`.

---

## Models

### 1. EfficientNet-B4 (Baseline)
Fine-tuned EfficientNet-B4 from [`timm`](https://github.com/huggingface/pytorch-image-models). Trained in multiple stages on the NTIRE training set with a multi-step LR schedule. Input: `Resize(352) → CenterCrop(320)`.

Saved weights: `training/baseline/effnetb4_finetuned.pth`

### 2. F3Net — Frequency-Focused Forgery Detection
A custom dual-stream CNN that processes the **RGB image** and its **log-scaled 2-D FFT magnitude spectrum** (frequency domain) in parallel. Both streams share the same CNN architecture (Conv→BN→ReLU×3 + MaxPool), their global-average-pooled features are concatenated, and a small MLP head produces the final logit.

Saved weights: `training/freq/f3net_best.pth`

### 3. DINOv2 + MLP (Embedding-Based)
Frozen **DINOv2 ViT-S/14** backbone (`facebookresearch/dinov2`) with a trainable two-layer MLP head (embed_dim → 256 → 1). Only the MLP is updated during training; the backbone provides powerful self-supervised features at no gradient cost.

Saved weights: `training/embedding/dinov2_mlp_best.pth`

---

## Project Structure

```
ai-gen-detection/
├── Dockerfile                       # PyTorch 2.1 + CUDA 12.1 dev environment
├── download_val.py                  # Download NTIRE validation set from HuggingFace
├── ensemble_validate.py             # Evaluate / run inference with the 3-model ensemble
├── main.ipynb                       # High-level exploration notebook
├── training/
│   ├── baseline/
│   │   ├── baseline.ipynb           # EfficientNet-B4 training notebook
│   │   ├── effnetb4_finetuned.pth   # Final fine-tuned weights (Stage 3)
│   │   └── model_metrics.csv        # Validation metrics
│   ├── freq/
│   │   ├── f3net_training.ipynb     # F3Net training notebook
│   │   ├── f3net_best.pth           # Best F3Net checkpoint
│   │   └── f3net_training_history.csv
│   └── embedding/
│       ├── dinov2_mlp_training.ipynb # DINOv2 + MLP training notebook
│       ├── dinov2_mlp_best.pth       # Best DINOv2 MLP checkpoint
│       └── dinov2_mlp_training_history.csv
├── ntire_val_dataset/               # NTIRE 2026 validation images (downloaded)
│   ├── val_images/
│   └── val_images_hard/
├── outputs/
│   ├── submission.csv               # Final competition submission
│   ├── submission_hard.csv          # Submission for the hard split
│   └── check_submission.py          # Validates submission CSV format
└── experiments/
    └── comparison.py                # Model comparison experiments
```

---

## Setup

### Option A — Docker (recommended)

```bash
docker build -t ai-gen-detection .
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace ai-gen-detection
```

This launches JupyterLab at `http://localhost:8888`.

### Option B — Python virtual environment

```bash
python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install torch torchvision timm tqdm pandas scikit-learn huggingface_hub
```

---

## Data

Download the NTIRE 2026 validation set from HuggingFace:

```bash
python download_val.py
```

The dataset is saved to `./ntire_val_dataset/`. The full training set (~227 k images) is available at [`deepfakesMSU/NTIRE-RobustAIGenDetection-train`](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train).

---

## Running Inference

### Ensemble validation (random sample from training split)

```bash
python ensemble_validate.py
```

This loads all three models, samples 3 000 images from the configured dataset path, runs the weighted ensemble, and prints accuracy and AUC.

### Submission predictions

The `InferenceDataset` class in `ensemble_validate.py` handles images-only directories and produces a `submission.csv` with columns `image_name` and `score` (probability of being AI-generated).

Validate the output format before submitting:

```bash
python outputs/check_submission.py outputs/submission.csv
```

---

## Requirements

| Package | Version |
|---|---|
| PyTorch | 2.1.0 |
| CUDA | 12.1 |
| timm | latest |
| scikit-learn | latest |
| pandas | latest |
| huggingface_hub | latest |

---

## Challenge

- **Competition**: [NTIRE 2026 Robust AI-Generated Image Detection in the Wild](https://www.codabench.org/competitions/12761/)
- **Workshop**: New Trends in Image Restoration and Enhancement @ [CVPR 2026](https://cvpr.thecvf.com/)
- **Dataset**: [`deepfakesMSU/NTIRE-RobustAIGenDetection-val`](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-val)
