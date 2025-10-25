# 🚀 Quick Start: Document Restoration Inference

**Date**: 2025-10-24  
**Status**: ✅ PRODUCTION READY

---

## 🎯 RECOMMENDED: Line-Aware Processing (BEST QUALITY)

**Script**: `dual_modal_gan/scripts/inference_line_aware_highres.py`  
**Quality Proven**: Contrast 58.7, Components: 19 ✅

### Single Document
```bash
poetry run python dual_modal_gan/scripts/inference_line_aware_highres.py \
    --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
    --checkpoint_name ckpt-88 \
    --input path/to/document.png \
    --output_dir results/restored \
    --gpu_id 1
```

### Batch Processing (Directory)
```bash
poetry run python dual_modal_gan/scripts/inference_line_aware_highres.py \
    --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
    --checkpoint_name ckpt-88 \
    --input path/to/documents/ \
    --output_dir results/batch_restored \
    --gpu_id 1
```

### DIBCO2016 Test
```bash
poetry run python dual_modal_gan/scripts/inference_line_aware_highres.py \
    --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
    --checkpoint_name ckpt-88 \
    --input dibco_datasets/DIPCO2016_dataset/ \
    --output_dir results/dibco2016_restored \
    --gpu_id 1
```

---

## 📊 Quality Comparison (DIBCO2016)

| Approach | Contrast | Components | Status |
|----------|----------|------------|--------|
| **Line-Aware** ✅ | **58.7** | 19 | **BEST** |
| Grid-Based (v3) | 52.9 | 20 | Good |

---

## 🔧 Alternative: Grid-Based (Fallback)

**Use when**: Line detection fails or for very dense documents

## Inference Cepat (1 Command)

```bash
# Default: GPU 1, DIBCO2016 dataset
./scripts/run_inference_dibco2016.sh
```

## Custom Inference

### Pilih GPU berbeda
```bash
./scripts/run_inference_dibco2016.sh 0    # GPU 0
./scripts/run_inference_dibco2016.sh 1    # GPU 1
./scripts/run_inference_dibco2016.sh -1   # CPU only
```

### Custom dataset
```bash
python dual_modal_gan/scripts/inference_production_v3.py \
    --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
    --checkpoint_name ckpt-88 \
    --input_dir path/to/your/images \
    --output_dir results/custom_run \
    --gpu_id 1 \
    --image_ext .bmp
```

### Dengan Ground Truth (untuk metrics)
```bash
python dual_modal_gan/scripts/inference_production_v3.py \
    --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
    --checkpoint_name ckpt-88 \
    --input_dir path/to/degraded/images \
    --gt_dir path/to/ground/truth \
    --output_dir results/with_metrics \
    --gpu_id 1 \
    --image_ext .png
```

## Output Files

```
results/inference_production_v3/dibco2016_YYYYMMDD_HHMMSS/
├── 1_restored.png              # Restored image (high-quality PNG)
├── 1_comparison.png            # Side-by-side: Degraded | Restored | GT
├── metrics.csv                 # All metrics + average
├── summary.json                # Complete run summary
└── inference_*.log             # Detailed execution log
```

## View Results

### Check metrics
```bash
cat results/inference_production_v3/dibco2016_*/metrics.csv
```

### View comparison images (Linux)
```bash
xdg-open results/inference_production_v3/dibco2016_*/*_comparison.png
```

### View comparison images (macOS)
```bash
open results/inference_production_v3/dibco2016_*/*_comparison.png
```

## Expected Performance

| Metric     | DIBCO2016 Result | Status |
|------------|------------------|--------|
| PSNR       | 16.43 dB         | ⚠️ Below target (20-25) |
| SSIM       | 0.8990           | ⚠️ Below target (0.95) |
| F-Measure  | 0.9807           | ✅ Excellent (>0.95) |
| NRM        | 0.0396           | ✅ Good (<0.05) |
| MPM        | 0.4424           | ✅ Good (<0.50) |

**Processing Speed:**
- ~21 seconds for 10 images (GPU 1, RTX A4000)
- ~2.1 seconds per image average
- Variable based on image size

## Troubleshooting

### GPU Out of Memory
```bash
# Edit inference_production_v3.py
BATCH_SIZE = 2  # Reduce from 4

# Or use CPU
./scripts/run_inference_dibco2016.sh -1
```

### Metrics not calculated
```bash
# Check GT naming: {image_name}_gt.bmp
# Example:
# Input:  1.bmp
# GT:     1_gt.bmp
```

### Seam artifacts visible
```bash
# Edit inference_production_v3.py
OVERLAP = 64  # Increase from 32
```

## Need More Help?

📖 **Full documentation:** `catatan/INFERENCE_PRODUCTION_V3_GUIDE.md`
