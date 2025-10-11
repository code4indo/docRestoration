#!/usr/bin/env python3
"""
Download dataset dan model dari Hugging Face
Usage: python download_from_huggingface.py
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download

# Check hf_transfer availability
try:
    subprocess.run([sys.executable, "-c", "import hf_transfer"], 
                   check=True, capture_output=True)
    HF_TRANSFER_AVAILABLE = True
    print("✅ hf_transfer available")
except (subprocess.CalledProcessError, ImportError):
    HF_TRANSFER_AVAILABLE = False
    print("⚠️ hf_transfer not available, will use standard download")

# Konfigurasi
HF_USERNAME = "jatnikonm"  # Ganti dengan username Anda
REPO_NAME = "HTR_VOC"
REPO_TYPE = "dataset"

def download_file(repo_id, filename, local_path, disable_hf_transfer=False):
    """Download file dari Hugging Face"""
    print(f"📥 Downloading: {filename}")
    
    # Create directory
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Determine if we should use hf_transfer
        use_hf_transfer = HF_TRANSFER_AVAILABLE and not disable_hf_transfer
        if use_hf_transfer:
            print("  🚀 Using hf_transfer for faster download")
        else:
            if disable_hf_transfer:
                print("  📄 Standard download (small file)")
            else:
                print("  📄 Standard download (hf_transfer unavailable)")
        
        # Set environment variables
        env_backup = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        if use_hf_transfer and not disable_hf_transfer:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        else:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False
            )
            
            # Rename to target path if needed
            if downloaded_path != str(local_path):
                import shutil
                shutil.move(downloaded_path, str(local_path))
            
            file_size = os.path.getsize(local_path) / (1024**2)  # MB
            print(f"✅ Downloaded: {local_path.name} ({file_size:.2f} MB)")
            return True
            
        finally:
            # Always restore environment
            if env_backup:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = env_backup
            else:
                os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def main():
    # Get config from environment or use defaults
    hf_username = os.environ.get("HF_USERNAME", HF_USERNAME)
    hf_repo_name = os.environ.get("HF_REPO_NAME", REPO_NAME)
    repo_id = f"{hf_username}/{hf_repo_name}"
    
    print(f"=== Downloading from Hugging Face ===")
    print(f"Repository: {repo_id}")
    print()
    
    # Base path untuk workspace (support Docker dan local)
    workspace_root = os.environ.get("WORKSPACE_ROOT", "/workspace")
    
    # Use current workspace directory (docRestoration) sebagai base
    current_dir = Path.cwd()
    
    # Files to download (path, local_path, disable_hf_transfer)
    files = [
        ("dataset/dataset_gan.tfrecord", str(current_dir / "dual_modal_gan" / "data" / "dataset_gan.tfrecord"), False),
        ("model/best_model.weights.h5", str(current_dir / "models" / "best_htr_recognizer" / "best_model.weights.h5"), False),
        ("charlist/real_data_charlist.txt", str(current_dir / "real_data_preparation" / "real_data_charlist.txt"), True),  # Disable for small text file
    ]
    
    success_count = 0
    for hf_path, local_path, disable_transfer in files:
        if download_file(repo_id, hf_path, local_path, disable_hf_transfer=disable_transfer):
            success_count += 1
    
    print()
    print(f"=== Download Complete ===")
    print(f"✅ {success_count}/{len(files)} files downloaded successfully")
    
    if success_count < len(files):
        exit(1)

if __name__ == "__main__":
    main()
