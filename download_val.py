# download_ntire_val.py

from huggingface_hub import snapshot_download

REPO_ID = "deepfakesMSU/NTIRE-RobustAIGenDetection-val"
LOCAL_DIR = "./ntire_val_dataset"

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Download complete. Dataset saved to:", LOCAL_DIR)