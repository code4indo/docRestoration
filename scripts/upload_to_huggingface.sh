#!/bin/bash
# Upload dataset dan model ke Hugging Face
# Optimized: Check existing files before upload to avoid unnecessary uploads

set -e

# Konfigurasi
HF_USERNAME="jatnikonm"  # Ganti dengan username Hugging Face Anda
REPO_NAME="HTR_VOC"  # Nama repository
REPO_TYPE="dataset"  # "dataset" atau "model"
FORCE_UPLOAD=false  # Default: skip if files exist

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--force)
      FORCE_UPLOAD=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  -f, --force    Force upload even if files already exist"
      echo "  -h, --help     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h for help"
      exit 1
      ;;
  esac
done

echo "=== Upload to Hugging Face ==="
echo "Repository: ${HF_USERNAME}/${REPO_NAME}"
echo "Force Upload: $FORCE_UPLOAD"
echo ""

# File yang akan di-upload
DATASET_FILE="dual_modal_gan/data/dataset_gan.tfrecord"
MODEL_FILE="models/best_htr_recognizer/best_model.weights.h5"
CHARLIST_FILE="real_data_preparation/real_data_charlist.txt"

# 1. Create repository (jika belum ada)
echo "📦 Creating repository..."
python3 << EOF
from huggingface_hub import HfApi, create_repo, hf_hub_download
from pathlib import Path
import os
import hashlib

api = HfApi()
repo_id = "${HF_USERNAME}/${REPO_NAME}"

try:
    create_repo(
        repo_id=repo_id,
        repo_type="${REPO_TYPE}",
        exist_ok=True,
        private=False  # Ubah ke True jika ingin private
    )
    print(f"✅ Repository created/exists: {repo_id}")
except Exception as e:
    print(f"⚠️  {e}")
EOF

# 2. Check existing files before upload
echo ""
echo "🔍 Checking existing files on HuggingFace..."

CHECK_SCRIPT=$(cat << 'EOF'
import hashlib
import os
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path

def get_file_hash(filepath):
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def check_file_exists_on_hf(repo_id, file_path, local_hash=None):
    """Check if file exists on HuggingFace and optionally compare hashes"""
    try:
        # Try to download just metadata (not the full file)
        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id, repo_type="${REPO_TYPE}")
        
        # Check if file exists in repo
        file_info = None
        for file_in_repo in repo_info.siblings:
            if file_in_repo.rfilename == file_path:
                file_info = file_in_repo
                break
        
        if not file_info:
            return {"exists": False, "size": 0, "hash_diff": None}
        
        # If we want to compare hashes, download file and compare
        if local_hash:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="${REPO_TYPE}",
                    local_dir="./.tmp_check",
                    local_dir_use_symlinks=False
                )
                
                remote_hash = get_file_hash(downloaded_path)
                os.unlink(downloaded_path)
                
                return {
                    "exists": True, 
                    "size": file_info.size,
                    "hash_diff": remote_hash != local_hash,
                    "local_hash": local_hash,
                    "remote_hash": remote_hash
                }
            except Exception as e:
                print(f"Hash check failed for {file_path}: {e}")
                return {"exists": True, "size": file_info.size, "hash_diff": None}
        
        return {"exists": True, "size": file_info.size, "hash_diff": None}
        
    except Exception as e:
        # Repository doesn't exist or other error
        return {"exists": False, "size": 0, "hash_diff": None}
EOF
)

python3 << EOF
$CHECK_SCRIPT

repo_id = "${HF_USERNAME}/${REPO_NAME}"
files_to_upload = [
    "${DATASET_FILE}",
    "${MODEL_FILE}", 
    "${CHARLIST_FILE}"
]

force_upload = "${FORCE_UPLOAD}" == "true"

if not force_upload:
    print("Checking files before upload...")
    skip_count = 0
    upload_count = 0
    
    for local_file in files_to_upload:
        if not os.path.exists(local_file):
            print(f"⚠️  Local file not found: {local_file}")
            skip_count += 1
            continue
            
        local_hash = get_file_hash(local_file)
        hf_path = {
            "${DATASET_FILE}": "dataset/dataset_gan.tfrecord",
            "${MODEL_FILE}": "model/best_model.weights.h5", 
            "${CHARLIST_FILE}": "charlist/real_data_charlist.txt"
        }.get(local_file)
        
        result = check_file_exists_on_hf(repo_id, hf_path, local_hash)
        
        if result["exists"]:
            if result["hash_diff"] is None:
                # No hash comparison available, size exists
                print(f"⚠️  {hf_path} exists on HuggingFace (size: {result['size']/(1024**2):.1f} MB)")
                print(f"   Use --force to upload anyway")
                skip_count += 1
            elif result["hash_diff"]:
                print(f"🔄 {hf_path} exists but content differs")
                upload_count += 1
            else:
                print(f"✅ {hf_path} already exists with same content - skipping")
                skip_count += 1
        else:
            print(f"📤 {hf_path} not found on HuggingFace")
            upload_count += 1
    
    if upload_count == 0 and skip_count > 0:
        print(f"\n✅ All {skip_count} files already exist (use --force to override)")
        exit(0)
    elif upload_count > 0:
        print(f"\n📤 Need to upload {upload_count} new/updated files")
else:
    print("Force upload mode: uploading all files")
EOF

# 3. Upload files dengan progress bar
echo ""
echo "📤 Uploading files..."

python3 << EOF
from huggingface_hub import HfApi
from pathlib import Path
import os
import hashlib

def get_file_hash(filepath):
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

api = HfApi()
repo_id = "${HF_USERNAME}/${REPO_NAME}"

# File list dengan path target di HF
files_to_upload = [
    ("${DATASET_FILE}", "dataset/dataset_gan.tfrecord"),
    ("${MODEL_FILE}", "model/best_model.weights.h5"),
    ("${CHARLIST_FILE}", "charlist/real_data_charlist.txt"),
]

upload_count = 0
skip_count = 0

for local_path, hf_path in files_to_upload:
    if not os.path.exists(local_path):
        print(f"⚠️  File not found: {local_path}")
        skip_count += 1
        continue
    
    file_size = os.path.getsize(local_path) / (1024**3)  # GB
    print(f"\n🔍 Processing: {local_path} ({file_size:.2f} GB)")
    
    # Check if we need to upload (skip if force disabled and file exists)
    if "${FORCE_UPLOAD}" == "false":
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="${REPO_TYPE}")
            file_exists = any(f.rfilename == hf_path for f in repo_info.siblings)
            
            if file_exists:
                print(f"📤 {hf_path} exists on HuggingFace (use --force to override)")
                print(f"   Skipping upload...")
                skip_count += 1
                continue
        except:
            pass  # Repository doesn't exist or other error, proceed with upload
    
    print(f"   Target: {hf_path}")
    print(f"   Uploading...")
    
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="${REPO_TYPE}",
        )
        print(f"✅ Uploaded: {hf_path}")
        upload_count += 1
    except Exception as e:
        print(f"❌ Error uploading {hf_path}: {e}")
        skip_count += 1

print(f"\n🎉 Upload Summary:")
print(f"✅ Successfully uploaded: {upload_count} files")
if skip_count > 0:
    print(f"⏭️  Skipped: {skip_count} files")

if upload_count > 0:
    print(f"🔗 View at: https://huggingface.co/${REPO_TYPE}s/${repo_id}")
else:
    print(f"ℹ️  No new files uploaded")
EOF

echo ""
echo "=== Done ==="
echo ""
echo "Summary:"
echo " Repository: ${HF_USERNAME}/${REPO_NAME}"
echo " Force Upload: $FORCE_UPLOAD"
echo " Use -f or --force to override existing files"
