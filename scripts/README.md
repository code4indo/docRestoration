# GAN-HTR Training Scripts

## 📋 Overview

Scripts dalam direktori ini digunakan untuk training GAN-HTR model dengan berbagai konfigurasi. Semua script telah dimodifikasi untuk mendukung execution di lingkungan lokal maupun cloud provider (RunPod, etc.).

**🎯 Problem Solved:**
- ✅ **Absolute paths** → **Relative paths** untuk fleksibilitas
- ✅ **Hard-coded workspace** → **Dynamic detection** lokal vs cloud
- ✅ **Manual environment setup** → **Auto-detection** Python & dependencies
- ✅ **No validation** → **Comprehensive setup validation**

## 🚀 Quick Start

### 🏠 Local Execution (Recommended)
```bash
# Dari root project directory
cd docRestoration
./scripts/train32_smoke_test.sh
```

### ☁️ Cloud Execution (RunPod)
```bash
# Set workspace environment variable
export WORKSPACE=/workspace/docRestoration
./scripts/train32_smoke_test.sh
```

### 🔍 Setup Validation (Recommended First)
```bash
# Validasi semua dependencies dan paths
./scripts/test_script_setup.sh
```

## 📝 Scripts Description

### `train32_smoke_test.sh`
**Purpose**: Quick validation test untuk smoke testing
- **Duration**: 5 epochs, 200 steps per epoch
- **Batch Size**: 4
- **Purpose**: Fast iteration untuk testing changes
- **Output**: Smoke test results di `logbook/` dan `dual_modal_gan/outputs/`

### `test_script_setup.sh`
**Purpose**: Comprehensive setup validation
- **Validates**: File paths, Python environment, dependencies
- **Checks**: GPU availability, script permissions
- **Output**: Detailed status report with troubleshooting tips

## 🔧 Environment Requirements

### Local Environment
- Python 3.11+ (via Poetry)
- TensorFlow 2.15+ with CUDA support
- Dependencies terinstall melalui Poetry

### Cloud Environment
- Docker image dengan Python dan TensorFlow pre-installed
- CUDA support enabled
- Dataset dan model weights sudah di-preload

## 📁 Directory Structure

```
docRestoration/
├── scripts/
│   ├── train32_smoke_test.sh    # Smoke test script
│   └── README.md                 # This file
├── dual_modal_gan/
│   ├── scripts/train32.py        # Main training script
│   ├── data/dataset_gan.tfrecord # Dataset file
│   └── outputs/                  # Training outputs
├── models/
│   └── best_htr_recognizer/      # Pre-trained HTR model
├── real_data_preparation/
│   └── real_data_charlist.txt    # Character set file
└── logbook/                      # Training logs
```

## ⚙️ Configuration Parameters

### Smoke Test Configuration
- **GPU ID**: 1 (ubah sesuai availability)
- **Epochs**: 5 (quick validation)
- **Batch Size**: 4 (memory efficient)
- **Learning Rates**:
  - Generator: 0.0006
  - Discriminator: 0.0002
- **Loss Weights**:
  - Pixel Loss: 1000.0
  - CTC Loss: 15.0
  - Adversarial Loss: 1.0
- **Early Stopping**: Enabled (patience=3)

## 🐍 Python Environment Detection

Script secara otomatis mendeteksi dan menggunakan environment yang tersedia:

1. **Poetry Environment** (Prioritas utama untuk local)
   ```bash
   poetry run python
   ```

2. **Custom Python** (Via PYTHON_BIN environment variable)
   ```bash
   export PYTHON_BIN=/path/to/python
   ./scripts/train32_smoke_test.sh
   ```

3. **System Python** (Fallback)
   ```bash
   python3
   ```

## 📊 Output Files

### Training Metrics
- **Location**: `dual_modal_gan/outputs/checkpoints_fp32_smoke_test/metrics/training_metrics_fp32.json`
- **Contains**: Loss values, PSNR, SSIM, CER metrics per epoch

### Sample Images
- **Location**: `dual_modal_gan/outputs/samples_fp32_smoke_test/`
- **Contains**: Generated sample images per evaluation interval

### Log Files
- **Location**: `logbook/smoke_test_YYYYMMDD_HHMMSS.log`
- **Contains**: Complete training output dengan timestamps

## 🔍 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check GPU availability
   nvidia-smi
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **Dependencies Missing**
   ```bash
   # Local: Install via Poetry
   poetry install

   # Cloud: Dependencies should be pre-installed
   ```

3. **File Not Found Errors**
   ```bash
   # Verify required files exist
   ls -la dual_modal_gan/data/dataset_gan.tfrecord
   ls -la models/best_htr_recognizer/best_model.weights.h5
   ls -la real_data_preparation/real_data_charlist.txt
   ```

4. **Permission Issues**
   ```bash
   # Make script executable
   chmod +x scripts/train32_smoke_test.sh
   ```

### Debug Mode

Untuk debugging, jalankan script dengan verbose output:
```bash
# Set debug environment variable
export DEBUG=1
./scripts/train32_smoke_test.sh
```

## 🌐 Cloud Provider Setup

### RunPod Configuration
1. **Template**: Gunakan template dengan CUDA support
2. **Volume**: Mount project directory ke `/workspace`
3. **Environment Variables**:
   ```bash
   WORKSPACE=/workspace/docRestoration
   CUDA_VISIBLE_DEVICES=0
   ALLOW_GPU=1
   ```

### Local Development
Untuk development lokal, pastikan:
```bash
# Install dependencies
poetry install

# Set GPU environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export ALLOW_GPU=1

# Run script
./scripts/train32_smoke_test.sh
```

## 📈 Next Steps

Setelah smoke test berhasil:
1. **Full Training**: Jalankan training dengan lebih banyak epochs
2. **Hyperparameter Tuning**: Adjust learning rates dan loss weights
3. **Multi-GPU Training**: Modifikasi untuk multi-GPU setup
4. **Production Deployment**: Export trained model untuk inference

## 📞 Support

Untuk masalah atau pertanyaan:
1. Check log files untuk error messages
2. Verify environment configuration
3. Pastikan semua required files tersedia
4. Check GPU availability dan CUDA installation