# GAN-HTR Document Restoration

Direktori structure lengkap untuk menjalankan training GAN-HTR menggunakan smoke test train32_smoke_test.sh

## 🎯 Purpose
- Menjalankan smoke test GAN-HTR dengan Pure FP32
- Validasi training pipeline sebelum full production
- Development environment yang terisolasi

## 📁 Struktur Direktori
```
docRestoration/
├── dual_modal_gan/                    # 🏗️  Training module
│   ├── scripts/                       # ⚙️  All training scripts  
│   ├── src/models/                    # 🧠  Network architectures
│   ├── data/                          # 🎯  Dataset files
│   └── outputs/                       # 💾  Training outputs
├── models/best_htr_recognizer/        # 🧠  HTR weights
├── real_data_preparation/             # 📝  Character sets
├── network/                           # 🧱  Custom layers
├── scripts/                           # 📜  Utility scripts
├── mlruns/                            # 📊  MLflow experiments
├── logbook/                           # 📔  Research logs
├── pyproject.toml                     # 📦  Python dependencies
├── poetry.lock                        # 🔒  Locked dependencies
└── requirements.txt                   # 📋  Pip requirements
```

## 🚀 Quick Start

### Method 1: Docker Compose (Recommended - Cloud Ready) 🐳

```bash
# 1. View available training options
./quick_start_training.sh

# 2. Run smoke test (quick validation)
docker-compose up -d gan-htr-smoke-test
docker logs -f gan-htr-smoke-test

# 3. Or run production training
docker-compose up -d gan-htr-prod
docker logs -f gan-htr-prod

# 4. Or run with custom script
TRAINING_SCRIPT=scripts/train32_smoke_test.sh docker-compose up -d gan-htr-prod
```

**✅ Benefits:**
- Auto-download dataset from HuggingFace
- Zero manual setup
- Cloud-ready (RunPod, AWS, GCP, Azure)
- Persistent volumes
- Easy resume training

📚 **Documentation:**
- `catatan/FLEXIBLE_TRAINING_CONFIG.md` - Flexible training guide
- `catatan/CLOUD_DEPLOYMENT.md` - Cloud deployment guide
- `catatan/README_CLOUD.md` - Quick cloud start
- `catatan/README.md` - Index semua dokumentasi

---

### Method 2: Local/Manual Training 💻

#### 1. Install Dependencies
```bash
cd docRestoration
poetry install
# atau
pip install -r requirements.txt
```

#### 2. Download Required Data
```bash
# Option 1: Simple wrapper (recommended)
./scripts/download_data.sh

# Option 2: Direct python script
python scripts/download_from_huggingface.py

# Option 3: Complete setup (install + download)
./scripts/setup.sh
```

#### 3. Build and Push Docker Image (Optional)
```bash
# Option 1: Build and push (recommended)
./scripts/build_and_push_to_dockerhub.sh

# Option 2: Push existing image
./scripts/push_to_dockerhub.sh
```

#### 4. Run Smoke Test
```bash
chmod +x scripts/train32_smoke_test.sh
./scripts/train32_smoke_test.sh
```

#### 5. Check Results
```bash
ls -la dual_modal_gan/outputs/checkpoints_fp32_smoke_test/
ls -la dual_modal_gan/outputs/samples_fp32_smoke_test/
```

## 📋 Critical Files Checklist

### ✅ Core Files (Dibutuhkan untuk training)
- [ ] `dual_modal_gan/data/dataset_gan.tfrecord` (4.6GB) - **Download from HF**
- [ ] `real_data_preparation/real_data_charlist.txt` (254B) - **Download from HF**
- [ ] `models/best_htr_recognizer/best_model.weights.h5` (0.5GB) - **Download from HF**
- [x] `dual_modal_gan/scripts/train32.py` (37KB) - ✅ Included

### ✅ Network Architecture  
- [ ] `network/layers.py` (Custom layers)
- [ ] `dual_modal_gan/src/models/generator.py`
- [ ] `dual_modal_gan/src/models/discriminator.py`
- [ ] `dual_modal_gan/src/models/recognizer.py`

### ✅ Utility Scripts
- [ ] `scripts/setup.sh` (Complete setup - dependencies + data)
- [ ] `scripts/build_and_push_to_dockerhub.sh` (Build + push Docker image)
- [ ] `scripts/push_to_dockerhub.sh` (Push existing Docker image)
- [ ] `scripts/download_data.sh` (Data download wrapper)
- [ ] `scripts/check_required_data.sh` (Check what data is missing)
- [ ] `scripts/train32_smoke_test.sh` (Main smoke test)
- [ ] `scripts/train32_production.sh` (Full training)
- [ ] `scripts/download_from_huggingface.py` (Raw data download)
- [ ] `scripts/upload_to_huggingface.sh` (Data upload)
- [ ] `scripts/verify_dataset_provenance.sh` (Data integrity)

## ⚙️ Smoke Test Configuration

**Training Parameters:**
- Epochs: 300 (smoke test akan berhenti lebih awal untuk validasi)
- Steps per epoch: 200
- Batch size: 4
- Precision: Pure FP32 (no mixed precision)
- GPU ID: 1
- Loss weights: Pixel=1000.0, CTC=15.0, Adversarial=1.0

## 🔧 Troubleshooting

### File Not Found Error
```bash
# Check all critical files exist
find . -name "*.tfrecord" -exec ls -lh {} \;
find . -name "*.weights.h5" -exec ls -lh {} \;
find . -name "*charlist*" -exec ls -lh {} \;
```

### Permission Issues  
```bash
# Fix permissions
chmod +x scripts/*.sh
chmod -R 755 dual_modal_gan/ outputs/ mlruns/
```

### Memory Issues
```bash
# Reduce batch size if GPU memory insufficient
# Edit scripts/train32_smoke_test.sh and change --batch_size 2
```

## 📊 Expected Output

After successful smoke test:
```
dual_modal_gan/outputs/checkpoints_fp32_smoke_test/
├── best_model-*.data-00000-of-00001
├── best_model-*.index
├── checkpoint
└── metrics/
    └── training_metrics_fp32.json

dual_modal_gan/outputs/samples_fp32_smoke_test/
├── sample_0001_before.png
├── sample_0001_after.png
└── ... (training visualizations)
```

## 🔄 Auto-download Data (Required!)

**Before training, download required data:**
```bash
python scripts/download_from_huggingface.py
```

This will automatically download from HuggingFace:
- `dual_modal_gan/data/dataset_gan.tfrecord` (~5GB) 
- `models/best_htr_recognizer/best_model.weights.h5` (~500MB)
- `real_data_preparation/real_data_charlist.txt` (~1KB)

**⚠️ IMPORTANT**: Training will fail without these files!

## ❗ Important Notes

1. **GPU Required**: Training membutuhkan GPU dengan CUDA support
2. **Disk Space**: Minimum 10GB free untuk training outputs
3. **Memory**: Minimum 8GB RAM, 12GB+ recommended
4. **Poetry**: Pastikan Poetry terinstall: `pip install poetry`

---
*Created: $(date)*
*Source: GAN-HTR-ORI project*
