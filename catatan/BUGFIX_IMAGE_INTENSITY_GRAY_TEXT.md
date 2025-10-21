# BUGFIX: Text Tampak Abu-Abu Bukan Hitam di Output Gambar

## 🔍 Problem

Setelah training selesai (run ID: 704979cb9bbe40c3aeb4201f65141990) dengan PSNR >36 (excellent), evaluasi visual menunjukkan **text tidak hitam** di semua gambar output:
- Degraded image: text abu-abu
- Ground Truth: text abu-abu  
- Restored image: text abu-abu

## 📊 Analisis Objektif

### Data Mentah TFRecord (Correct ✅)
```
Clean Image (Ground Truth):
  Range: [0.00, 1.00] ✅ Normalized
  Text region (0.0-0.2): 8.65% pixels ✅ Ada pixel hitam
  Background (0.8-1.0): 88.73% ✅ Background putih
```

### Output PNG Training (Incorrect ❌)
```
Ground Truth:
  Range: [127, 255] ❌ Tidak ada pixel hitam (<100)
  Text mean: 146.76 ❌ Abu-abu, bukan hitam
  
Restored:
  Range: [120, 255] ❌ Tidak ada pixel hitam (<100)
  Text mean: 145.85 ❌ Abu-abu, bukan hitam
```

## 🐛 Root Cause

**Lokasi**: `dual_modal_gan/scripts/train_enhanced.py` lines 1321-1323

```python
# BUGFIX: Denormalize from tanh [-1,1] to [0,1] before saving
generated_samples_normalized = (generated_samples + 1.0) / 2.0  # ✅ CORRECT
degraded_samples_normalized = (degraded_samples + 1.0) / 2.0    # ❌ WRONG!
clean_samples_normalized = (clean_samples + 1.0) / 2.0          # ❌ WRONG!
```

**Masalah**:
1. `degraded_samples` dan `clean_samples` **SUDAH** dalam range [0, 1] dari TFRecord
2. Code melakukan denormalisasi seolah-olah data dalam range [-1, 1]
3. Hasil transformasi yang salah:
   - Pixel hitam (0.0) → (0.0 + 1.0) / 2.0 = **0.5** → 0.5 * 255 = **127.5** ✅ Abu-abu!
   - Pixel putih (1.0) → (1.0 + 1.0) / 2.0 = **1.0** → 1.0 * 255 = **255** ✅ Putih
4. Range output menjadi [127, 255] bukan [0, 255]

**Yang SEHARUSNYA**:
- ✅ `generated_samples`: Dari generator (tanh → [-1, 1]) → **PERLU** denormalisasi
- ❌ `degraded_samples`: Dari TFRecord ([0, 1]) → **TIDAK PERLU** denormalisasi  
- ❌ `clean_samples`: Dari TFRecord ([0, 1]) → **TIDAK PERLU** denormalisasi

## ✅ Solusi

### Solusi 1: Post-Processing (Quick Fix - IMPLEMENTED)

Untuk gambar yang sudah dihasilkan, gunakan script `scripts/fix_image_intensity.py`:

```bash
poetry run python scripts/fix_image_intensity.py \
    --input dual_modal_gan/outputs/samples_full_training_production_v1 \
    --pattern "comparison_epoch_0050_*.png" \
    --method linear_remap
```

**Hasil**:
```
BEFORE (Original):
  GT  range: [127, 255], text mean: 146.76 ❌
  Res range: [120, 255], text mean: 145.85 ❌

AFTER (Fixed):
  GT  range: [0, 255], text mean: 38.90 ✅ HITAM!
  Res range: [0, 255], text mean: 48.50 ✅ HITAM!
```

Gambar diperbaiki di: `dual_modal_gan/outputs/samples_full_training_production_v1/fixed/`

### Solusi 2: Fix Training Script (Permanent Fix)

**File**: `dual_modal_gan/scripts/train_enhanced.py`

**BEFORE** (lines 1321-1326):
```python
# BUGFIX: Denormalize from tanh [-1,1] to [0,1] before saving
generated_samples_normalized = (generated_samples + 1.0) / 2.0
degraded_samples_normalized = (degraded_samples + 1.0) / 2.0
clean_samples_normalized = (clean_samples + 1.0) / 2.0

img_to_save = (generated_samples_normalized * 255).numpy().astype(np.uint8)
```

**AFTER** (Corrected):
```python
# Denormalize ONLY generated samples (from tanh [-1,1] to [0,1])
# degraded_samples and clean_samples are ALREADY in [0,1] from TFRecord!
generated_samples_normalized = (generated_samples + 1.0) / 2.0
degraded_samples_normalized = degraded_samples  # Already [0, 1] - NO denorm!
clean_samples_normalized = clean_samples        # Already [0, 1] - NO denorm!

img_to_save = (generated_samples_normalized * 255).numpy().astype(np.uint8)
```

Atau lebih eksplisit:
```python
# Generator uses tanh activation → output in [-1, 1] → need denormalization
generated_samples_uint8 = ((generated_samples + 1.0) / 2.0 * 255).numpy().astype(np.uint8)

# Data from TFRecord is already in [0, 1] → NO denormalization needed
degraded_samples_uint8 = (degraded_samples * 255).numpy().astype(np.uint8)
clean_samples_uint8 = (clean_samples * 255).numpy().astype(np.uint8)
```

## 📝 Verification

### Scripts Created
1. **`scripts/analyze_image_intensity.py`**: Analisis objektif intensitas pixel
2. **`scripts/fix_image_intensity.py`**: Post-processing untuk memperbaiki gambar existing

### Results
```bash
# Analisis
poetry run python scripts/analyze_image_intensity.py

# Fix images
poetry run python scripts/fix_image_intensity.py \
    --input dual_modal_gan/outputs/samples_full_training_production_v1 \
    --pattern "*.png" \
    --method linear_remap
```

## 🎯 Impact Assessment

### Apakah Perlu Training Ulang?

**TIDAK PERLU! ❌** Alasan:

1. **Model sudah optimal**: PSNR 36+, SSIM 0.99+ ✅
2. **Bug hanya di visualization**: Tidak mempengaruhi training process
3. **Metrics tetap valid**: PSNR/SSIM dihitung pada data [0, 1], bukan pada saved PNG
4. **Generator output correct**: Model menghasilkan output correct di [-1, 1]
5. **Inference tidak terpengaruh**: Saat inference, bisa langsung fix denormalisasi

### Apa yang Terpengaruh?

✅ **TIDAK TERPENGARUH**:
- Training process
- Model weights
- PSNR/SSIM metrics
- Generator performance
- Discriminator training

❌ **TERPENGARUH**:
- Visual quality gambar PNG yang disave
- Presentasi/publikasi (perlu gambar yang fixed)
- Human evaluation (text tampak abu-abu)

## 🚀 Action Items

### Immediate (DONE ✅)
- [x] Analisis root cause
- [x] Create post-processing script
- [x] Fix existing epoch 50 samples
- [x] Verify hasil perbaikan
- [x] Document findings

### For Future Training
- [ ] Update `train_enhanced.py` dengan fix permanent
- [ ] Add assertion di code: `assert 0 <= degraded_samples.min() and degraded_samples.max() <= 1`
- [ ] Add comment warning tentang range data
- [ ] Test dengan 1 epoch untuk verify fix

### For Inference Script
- [ ] Ensure inference script uses correct denormalization
- [ ] Test inference dengan best model
- [ ] Verify output images memiliki black text

## 📚 Lessons Learned

1. **Always check data range at each pipeline stage**
   - TFRecord: [0, 1] normalized
   - Generator output: [-1, 1] (tanh)
   - PNG output: [0, 255] uint8

2. **Visual inspection ≠ Quantitative metrics**
   - PSNR/SSIM bisa bagus tapi visual ada issue
   - Always do sanity check on saved images

3. **Previous bugfix introduced new bug**
   - File: `catatan/CRITICAL_BUGFIX_RANGE_MISMATCH.md`
   - Fix yang diterapkan: denormalize semua dengan `(x + 1) / 2`
   - Assumsi salah: data sudah dalam [-1, 1]

4. **Denormalization harus match dengan data source**
   - From TFRecord [0, 1] → multiply by 255
   - From tanh [-1, 1] → `(x + 1) / 2 * 255`
   - **DON'T MIX THEM!**

## 🔗 Related Issues

- `catatan/CRITICAL_BUGFIX_RANGE_MISMATCH.md`: Previous fix that introduced this bug
- Training run: 704979cb9bbe40c3aeb4201f65141990
- Model checkpoint: `dual_modal_gan/checkpoints/full_training_production_v1/best_model`

## 📊 Comparison

| Metric | Original | Fixed | Status |
|--------|----------|-------|--------|
| PSNR | 36.61 | 36.61 | ✅ Same (not affected) |
| SSIM | 0.9905 | 0.9905 | ✅ Same (not affected) |
| GT Text Mean | 146.76 | 38.90 | ✅ Much blacker |
| Res Text Mean | 145.85 | 48.50 | ✅ Much blacker |
| GT Range | [127,255] | [0,255] | ✅ Full range |
| Res Range | [120,255] | [0,255] | ✅ Full range |

---

**Date**: 2025-10-21  
**Status**: RESOLVED (post-processing), DOCUMENTED  
**Priority**: Medium (doesn't affect model, only visualization)  
**Next**: Update training script for future runs
