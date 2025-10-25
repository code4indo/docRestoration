# Full-Size Document Restoration Inference Guide
## Production V3 Model - Enhanced Generator

**Date:** 2025-10-22  
**Model:** production_v3_academic_split_70_15_15  
**Best Checkpoint:** ckpt-88 (Epoch 44, PSNR: 30.91 dB)

---

## 📋 Overview

Script inferensi production-ready untuk restorasi dokumen terdegradasi dalam ukuran penuh menggunakan Enhanced U-Net Generator dengan strategi overlapping tiles dan alpha blending.

### ✨ Key Features

- ✅ **Full-size processing** - Memproses gambar dokumen dalam resolusi asli tanpa resize
- ✅ **Overlapping tiles** - Tiles 128×1024 pixels dengan overlap 32px untuk menghilangkan seam artifacts
- ✅ **Alpha blending** - Transisi smooth antar tiles menggunakan weighted blending
- ✅ **Batch processing** - GPU-efficient dengan batch_size=4
- ✅ **Comprehensive metrics** - PSNR, SSIM, F-Measure, NRM, MPM (standar DIBCO)
- ✅ **High-quality output** - PNG lossless compression
- ✅ **Visualization** - Side-by-side comparison (Degraded | Restored | GT)

---

## 🏗️ Architecture Characteristics

### Model Specifications

```yaml
Generator: Enhanced U-Net
  - ResBlocks: Residual connections untuk gradient flow lebih baik
  - Attention Gates: Focus pada fitur relevan
  - Parameters: ~21.8M
  
Input Format: (1024, 128, 1)  # Width × Height × Channel
  - CRITICAL: Model expects (W, H, C), NOT (H, W, C)!
  - Requires transpose before inference

Output Activation: tanh (range [-1, 1])
  - Normalization: [0, 255] → [0, 1] → [-1, 1]
  - Denormalization: [-1, 1] → [0, 1] → [0, 255]
```

### Processing Strategy

```
1. Extract overlapping tiles (128×1024, stride 96×992)
   ├─ Padding dengan white background (255) untuk tiles di boundary
   └─ Total tiles: ⌈H/96⌉ × ⌈W/992⌉

2. Preprocess tiles
   ├─ Normalize: [0, 255] → [0, 1]
   ├─ Transpose: (H, W) → (W, H)
   └─ Scale: [0, 1] → [-1, 1] untuk tanh

3. Batch inference (batch_size=4)
   └─ GPU-efficient processing

4. Postprocess tiles
   ├─ Denormalize: [-1, 1] → [0, 1]
   ├─ Transpose: (W, H) → (H, W)
   └─ Scale: [0, 1] → [0, 255]

5. Merge dengan alpha blending
   ├─ Fade-in/fade-out 32px di edges
   └─ Weighted averaging untuk smooth transitions
```

---

## 📁 File Structure

```
dual_modal_gan/
├── scripts/
│   └── inference_production_v3.py    # Main inference script
└── checkpoints/
    └── production_v3_academic_split_70_15_15/
        └── best_model/
            ├── checkpoint
            ├── ckpt-88.index         # Best model weights
            └── ckpt-88.data-00000-of-00001

scripts/
└── run_inference_dibco2016.sh        # Launcher dengan validation

dibco_datasets/
├── DIPCO2016_dataset/                # Degraded images
│   ├── 1.bmp
│   ├── 2.bmp
│   └── ...
└── DIPCO2016_Dataset_GT/             # Ground truth (optional)
    ├── 1_gt.bmp
    ├── 2_gt.bmp
    └── ...

results/
└── inference_production_v3/
    └── dibco2016_YYYYMMDD_HHMMSS/
        ├── 1_restored.png            # Restored high-quality PNG
        ├── 1_comparison.png          # Side-by-side visualization
        ├── metrics.csv               # Per-image + average metrics
        ├── summary.json              # Complete processing summary
        └── inference_*.log           # Detailed execution log
```

---

## 🚀 Usage

### Method 1: Using Launcher Script (Recommended)

```bash
# Default: GPU 1
./scripts/run_inference_dibco2016.sh

# Specify GPU
./scripts/run_inference_dibco2016.sh 0    # Use GPU 0
./scripts/run_inference_dibco2016.sh 1    # Use GPU 1

# CPU only
./scripts/run_inference_dibco2016.sh -1
```

**Features:**
- ✅ Automatic validation (checkpoint, input directory, dependencies)
- ✅ GPU availability check dengan fallback
- ✅ Output directory dengan timestamp
- ✅ Comprehensive logging
- ✅ Results summary

### Method 2: Direct Python Script

```bash
source .venv/bin/activate

python dual_modal_gan/scripts/inference_production_v3.py \
    --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
    --checkpoint_name ckpt-88 \
    --input_dir dibco_datasets/DIPCO2016_dataset \
    --gt_dir dibco_datasets/DIPCO2016_Dataset_GT \
    --output_dir results/inference_production_v3/custom_run \
    --gpu_id 1 \
    --image_ext .bmp
```

**Arguments:**
- `--checkpoint_dir`: Directory containing checkpoint files (required)
- `--checkpoint_name`: Checkpoint file name without extension (default: ckpt-88)
- `--input_dir`: Directory with degraded images (required)
- `--gt_dir`: Directory with ground truth images (optional, for metrics)
- `--output_dir`: Output directory for results (required)
- `--gpu_id`: GPU device ID (default: 1, use -1 for CPU)
- `--image_ext`: Image file extension (default: .bmp)

---

## 📊 DIBCO2016 Test Results

**Test Date:** 2025-10-22  
**Checkpoint:** ckpt-88 (Best Model, Epoch 44)  
**Dataset:** DIBCO2016 (10 images)  
**GPU:** NVIDIA RTX A4000 (GPU 1)  
**Processing Time:** ~35 seconds (10 images)

### Average Metrics

| Metric     | Value  | Target | Status |
|------------|--------|--------|--------|
| PSNR       | 16.43  | ~20-25 | ⚠️ Below target |
| SSIM       | 0.8990 | ~0.95  | ⚠️ Below target |
| F-Measure  | 0.9807 | >0.95  | ✅ Excellent |
| NRM        | 0.0396 | <0.05  | ✅ Good |
| MPM        | 0.4424 | <0.50  | ✅ Good |

### Per-Image Results

| Image | Size (px)  | Tiles | PSNR  | SSIM   | F-Measure | NRM    | MPM    |
|-------|------------|-------|-------|--------|-----------|--------|--------|
| 1     | 1510×1067  | 24    | 19.33 | 0.9479 | 0.9926    | 0.0148 | 0.5164 |
| 2     | 2259×1023  | 33    | 19.73 | 0.9668 | 0.9938    | 0.0124 | 0.5364 |
| 3     | 2417×1064  | 36    | 20.29 | 0.9587 | 0.9938    | 0.0125 | 0.4843 |
| 4     | 2363×615   | 21    | 15.17 | 0.9085 | 0.9824    | 0.0357 | 0.4227 |
| 5     | 2375×1024  | 33    | 18.89 | 0.9435 | 0.9910    | 0.0180 | 0.4247 |
| 6     | 1364×788   | 18    | 16.86 | 0.9227 | 0.9872    | 0.0258 | 0.5571 |
| 7     | 963×656    | 7     | 15.23 | 0.8923 | 0.9790    | 0.0428 | 0.1594 |
| 8     | 1782×334   | 8     | 10.00 | 0.7629 | 0.9410    | 0.1245 | 0.5639 |
| 9     | 1339×302   | 8     | 15.46 | 0.8743 | 0.9804    | 0.0395 | 0.2686 |
| 10    | 378×315    | 4     | 13.35 | 0.8123 | 0.9654    | 0.0699 | 0.4903 |

### Analysis

**Strengths:**
- ✅ **Structural preservation** - SSIM rata-rata 0.90 menunjukkan struktur dokumen terjaga dengan baik
- ✅ **Binary quality** - F-Measure 0.98 menunjukkan kualitas binarisasi excellent
- ✅ **Low misclassification** - NRM 0.04 dan MPM 0.44 dalam range acceptable
- ✅ **Seamless reconstruction** - Alpha blending menghilangkan visible seams

**Weaknesses:**
- ⚠️ **PSNR below target** - Rata-rata 16.43 dB, target ~20-25 dB
- ⚠️ **Variable performance** - Image 8 PSNR 10.00 dB (outlier)
- ⚠️ **Small images struggle** - Image 10 (378×315) memiliki SSIM 0.81 (lowest)

**Possible Causes:**
1. Model dilatih dengan patch-based training (128×1024), bukan full document
2. DIBCO2016 adalah dataset **handwritten historical documents** (lebih challenging)
3. Training data mungkin dominan printed documents
4. Model belum melihat variasi degradasi ekstrem (image 8)

---

## 🔧 Troubleshooting

### GPU Out of Memory

```bash
# Reduce batch size di inference_production_v3.py
BATCH_SIZE = 2  # Default: 4

# Atau gunakan CPU
python ... --gpu_id -1
```

### Seam Artifacts Visible

```bash
# Increase overlap di inference_production_v3.py
OVERLAP = 64  # Default: 32

# Trade-off: More tiles = slower processing
```

### Metrics Not Calculated

```bash
# Pastikan GT directory structure benar
dibco_datasets/DIPCO2016_Dataset_GT/
├── 1_gt.bmp    # Naming: {image_name}_gt.bmp
├── 2_gt.bmp
└── ...
```

### Import Error: scikit-image

```bash
# SSIM akan menggunakan approximation
# Untuk accuracy maksimal, install:
poetry add scikit-image

# Atau
pip install scikit-image
```

---

## 📈 Performance Optimization

### Processing Speed

| Configuration   | Speed (tiles/sec) | Memory (GB) | Notes |
|-----------------|-------------------|-------------|-------|
| GPU 1, batch=4  | ~8-10             | ~2-3        | Recommended |
| GPU 1, batch=8  | ~12-15            | ~4-5        | Faster, more VRAM |
| GPU 1, batch=1  | ~3-4              | ~1          | Slower, safe |
| CPU, batch=4    | ~0.5-1            | ~2-3        | Very slow |

**Tips:**
- Gunakan GPU jika tersedia (15-20x faster than CPU)
- Batch size 4 optimal untuk RTX A4000 (16GB VRAM)
- Untuk GPU <8GB VRAM, gunakan batch_size=2
- Parallel processing NOT recommended (GPU memory contention)

---

## 🎯 Next Steps for Improvement

### 1. Model Retraining Strategy

```yaml
Problem: PSNR below target (16.43 vs 20-25 dB)

Solutions:
  A. Full-document training:
    - Train dengan full-size images (bukan patches)
    - Requires: Lebih banyak VRAM (multi-GPU training)
    
  B. Augment training data:
    - Add more DIBCO handwritten documents
    - Include extreme degradation samples
    
  C. Loss function tuning:
    - Increase weight pada perceptual loss
    - Add SSIM loss component
    
  D. Progressive training:
    - Stage 1: Patch-based (current)
    - Stage 2: Full-document fine-tuning
```

### 2. Post-processing Enhancement

```python
# Add optional post-processing steps:
- Adaptive histogram equalization
- Morphological operations
- Edge enhancement
- Noise reduction
```

### 3. Multi-scale Processing

```python
# Process image at multiple scales:
1. Original resolution
2. 2x downsampled
3. 4x downsampled

# Combine outputs dengan weighted fusion
```

---

## 📝 Citation & References

### This Implementation

```bibtex
@software{production_v3_inference,
  title={Full-Size Document Restoration Inference},
  author={AI Assistant},
  year={2025},
  note={Production V3 - Enhanced Generator with Overlapping Tiles}
}
```

### Base Research

```bibtex
@inproceedings{souibgui2022enhance,
  title={Enhance to Read Better: A Multi-Task Adversarial Network for Handwritten Document Image Enhancement},
  author={Souibgui, Mohamed Ali and Kessentini, Yousri},
  booktitle={Pattern Recognition},
  year={2022}
}
```

---

## 📞 Support

**Issues & Questions:**
- Check log files: `results/inference_production_v3/*/inference_*.log`
- Validate checkpoint: `dual_modal_gan/checkpoints/.../best_model/checkpoint`
- Verify GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

**Common Errors:**
1. **Checkpoint not found** → Verify training completed, best_model exists
2. **GPU not available** → Check CUDA installation, use CPU fallback
3. **Memory error** → Reduce batch_size or use smaller images
4. **JSON serialization** → Update to latest inference script (fixed 2025-10-22)

---

**End of Guide**
