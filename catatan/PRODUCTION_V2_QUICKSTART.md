# 🚀 PRODUCTION TRAINING V2 - QUICK START GUIDE

**Config**: `configs/production_v2_range_fixed_20251021.json`  
**Date**: 2025-10-21  
**Status**: ✅ Ready to Launch (Bug Fixed & Verified)

---

## 🎯 Quick Launch

```bash
# Metode 1: Interactive (dengan konfirmasi)
./scripts/launch_production_v2_20251021.sh

# Metode 2: Direct (tanpa konfirmasi)
nohup ./scripts/universal_train_from_json.sh \
  configs/production_v2_range_fixed_20251021.json &
```

---

## 📋 Configuration Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Experiment** | production_v2_range_fixed_20251021 | Clean slate |
| **Generator** | enhanced | U-Net + ResBlocks + Attention |
| **Discriminator** | enhanced_v2_fixed | Dual-Modal |
| **Epochs** | 100 | Early stopping enabled |
| **Batch Size** | 2 | |
| **Steps/Epoch** | 2133 | Full dataset (4266 samples) |
| **GPU** | 0 | Single GPU |
| **Early Stopping** | Patience 25 | Combined metric |
| **Learning Rate** | 0.0002 | Both G & D |
| **LR Schedule** | ✅ Enabled | Warmup 10, Anneal 20 |

### Loss Weights:
- Pixel Loss: 50.0
- Adversarial: 3.0
- Recognition Feature: 8.0
- CTC Loss: 0.15
- Perceptual: 1.0

---

## 📊 Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| **PSNR** | >30 dB | Realistic (no range bug) |
| **SSIM** | >0.95 | Structure similarity |
| **CER** | <5% | Character error rate |
| **Text Color** | BLACK | Pixel 0-50 (not gray 117-150) |
| **Convergence** | 40-60 epochs | With early stopping |
| **Duration** | 6-8 hours | GPU dependent |

---

## 🔍 Monitoring Commands

```bash
# Watch training log
tail -f logbook/production_v2_range_fixed_20251021_*.log

# Check samples directory
ls -lh dual_modal_gan/outputs/samples_production_v2_range_fixed_20251021/

# Verify black pixels in output
python3 -c "
import cv2
img = cv2.imread('dual_modal_gan/outputs/samples_production_v2_range_fixed_20251021/comparison_epoch_0001_sample_0.png', 0)
print(f'Min: {img.min()}, Max: {img.max()}, Black pixels (<50): {(img<50).sum()}')
"

# MLflow UI
poetry run mlflow ui
# Open: http://localhost:5000

# Check GPU usage
nvidia-smi -l 5
```

---

## 🎯 Validation Checkpoints

### After Epoch 1:
- [ ] Sample images generated
- [ ] Black pixels present (not all gray)
- [ ] Min pixel < 50 (should be 0-20)
- [ ] No training errors

### After Epoch 10 (Warmup complete):
- [ ] PSNR > 20 dB
- [ ] CTC loss decreasing
- [ ] Sample quality improving

### After Epoch 30:
- [ ] PSNR > 28 dB
- [ ] SSIM > 0.90
- [ ] Text readable in samples

### Convergence (40-60 epochs):
- [ ] PSNR > 30 dB
- [ ] SSIM > 0.95
- [ ] CER < 5%
- [ ] Early stopping triggered (no improvement)

---

## ⚠️ CRITICAL REMINDERS

1. **Previous runs INVALID**: Checkpoint ckpt-91 dari run 704979cb9bbe40c3aeb4201f65141990 JANGAN digunakan
2. **Clean slate**: Training harus dari epoch 1, tidak ada resume
3. **Verify samples**: Cek epoch 1 samples untuk black text
4. **Monitor closely**: First 10 epochs kritis untuk validasi fix

---

## 🛑 Stop Training

```bash
# Find PID
ps aux | grep production_v2_range_fixed_20251021 | grep -v grep

# Kill process
kill <PID>
```

---

## 📁 Output Locations

- **Checkpoints**: `dual_modal_gan/checkpoints/production_v2_range_fixed_20251021/`
- **Best Model**: `dual_modal_gan/checkpoints/production_v2_range_fixed_20251021/best_model/`
- **Samples**: `dual_modal_gan/outputs/samples_production_v2_range_fixed_20251021/`
- **MLflow**: `mlruns/`
- **Log**: `logbook/production_v2_range_fixed_20251021_YYYYMMDD_HHMMSS.log`

---

## ✅ Pre-Launch Checklist

- [x] Range fix verified (129,684 black pixels in test)
- [x] Config file created and validated
- [x] Launch script prepared
- [x] GPU 0 available
- [x] TFRecord dataset exists
- [x] Recognizer weights loaded
- [x] Sufficient disk space
- [ ] Ready to launch!

---

**🚀 Ready to launch production training with validated fix!**
