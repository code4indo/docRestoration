#!/usr/bin/env python3
"""
Download dataset dan model dari Hugging Face
Usage: python download_from_huggingface.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Konfigurasi
HF_USERNAME = "jatnikonm"  # Ganti dengan username Anda
REPO_NAME = "HTR_VOC"
REPO_TYPE = "dataset"

def download_file(repo_id, filename, local_path):
    """Download file dari Hugging Face"""
    print(f"📥 Downloading: {filename}")
    
    # Create directory
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Disable hf_transfer for small files (avoid dependency issue)
        env_backup = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        if filename.endswith(".txt"):
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=REPO_TYPE,
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False
        )
        
        # Restore environment
        if filename.endswith(".txt"):
            if env_backup:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = env_backup
            else:
                os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        
        # Rename to target path if needed
        if downloaded_path != str(local_path):
            os.rename(downloaded_path, str(local_path))
        
        file_size = os.path.getsize(local_path) / (1024**3)
        print(f"✅ Downloaded: {local_path} ({file_size:.2f} GB)")
        return True
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def main():
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    
    print(f"=== Downloading from Hugging Face ===")
    print(f"Repository: {repo_id}")
    print()
    
    # Files to download
    files = [
        ("dataset/dataset_gan.tfrecord", "dual_modal_gan/data/dataset_gan.tfrecord"),
        ("model/best_model.weights.h5", "models/best_htr_recognizer/best_model.weights.h5"),
        ("charlist/real_data_charlist.txt", "real_data_preparation/real_data_charlist.txt"),
    ]
    
    success_count = 0
    for hf_path, local_path in files:
        if download_file(repo_id, hf_path, local_path):
            success_count += 1
    
    print()
    print(f"=== Download Complete ===")
    print(f"✅ {success_count}/{len(files)} files downloaded successfully")
    
    if success_count < len(files):
        exit(1)

if __name__ == "__main__":
    main()
