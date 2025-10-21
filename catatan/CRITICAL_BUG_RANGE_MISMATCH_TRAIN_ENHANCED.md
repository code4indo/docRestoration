# 🚨 CRITICAL BUG: Range Mismatch in Training - Generator Cannot Produce Black Text

**Status**: DISCOVERED - Explains why model PSNR 17.88 dB (vs expected >30 dB)  
**Date**: 2025-10-21  
**Run ID**: 704979cb9bbe40c3aeb4201f65141990 (INVALID TRAINING)

## 🔴 THE BUG

In `dual_modal_gan/scripts/train_enhanced.py`, there is a **CRITICAL RANGE MISMATCH** between clean images and generated images when calculating losses:

### Current (WRONG) Code:

```python
# Line 594
generated_images = generator(degraded_images, training=True)

# Line 599-600: ONLY recognizer gets denormalized
clean_images_normalized = (clean_images + 1.0) / 2.0  # ❌ BUG! Already [0,1]
generated_images_normalized = (generated_images + 1.0) / 2.0  # ✅ Correct

# Line 625-628: Discriminator receives MISMATCHED ranges
real_output = discriminator([clean_images, ...], training=True)     # [0, 1]
fake_output = discriminator([generated_images, ...], training=True) # [-1, 1]

# Line 635: Pixel Loss compares DIFFERENT ranges!
pixel_loss = mae_loss_fn(clean_images, generated_images)  # [0,1] vs [-1,1] ❌

# Line 643: Perceptual Loss also mismatched
perceptual_loss = perceptual_loss_layer(clean_images, generated_images)  # ❌
```

## 🎯 ROOT CAUSE

1. **TFRecord data** stores images as `float32` in range **[0, 1]**
2. **Generator** with `tanh` activation outputs range **[-1, 1]**
3. **Training losses** (pixel, perceptual, discriminator) compare:
   - `clean_images`: [0, 1] 
   - `generated_images`: [-1, 1] 
   - ❌ **DIFFERENT RANGES!**

## 💥 IMPACT

### Model Behavior:
- Generator learns to output **compromise range**: [-0.08, 1.0] instead of [-1, 1]
- **Cannot produce black text**: minimum output ~ 117/255 = 0.46 (gray) instead of 0/255 (black)
- **All restored images have gray text** instead of black

### Metrics Impact:
- **PSNR calculation WRONG**: comparing mismatched ranges gives artificially low values
- Validation PSNR: **17.88 dB** (should be >30 dB)
- SSIM: **0.9187** (should be >0.95)
- **Model appears to underperform** when it's actually a calculation error

### Visual Impact:
```
Ground Truth:    pixel range [0, 255]   - ✅ Black text (0-50)
Restored Output: pixel range [117, 255] - ❌ Gray text (117-150)
                 Missing range: 0-116    - 🚫 No black pixels!
```

## 🔍 EVIDENCE

### From Evaluation:
```python
# Validation sample analysis:
GT       - Range: [0, 255], Black pixels (<50): 9898-13271  ✅
Restored - Range: [117, 255], Black pixels (<50): 0         ❌
```

### Generator Output Analysis:
```python
Generator raw output (tanh):  [-0.08, 1.00]  # Should be [-1.0, 1.0]
After (x+1)/2 denormalization: [0.46, 1.00]  # Should be [0.0, 1.0]
In uint8:                      [117, 255]    # Should be [0, 255]
```

## ✅ THE FIX

Three options to align ranges:

### Option 1: Normalize Clean Images to [-1, 1] (Recommended)
```python
# In train_step() function:
clean_images_tanh = clean_images * 2.0 - 1.0  # [0,1] → [-1,1]
degraded_images_tanh = degraded_images * 2.0 - 1.0

generated_images = generator(degraded_images_tanh, training=True)  # [-1,1]

# Now all losses compare same range
pixel_loss = mae_loss_fn(clean_images_tanh, generated_images)  # [-1,1] vs [-1,1] ✅
```

### Option 2: Denormalize Generated to [0, 1] Everywhere
```python
generated_images_raw = generator(degraded_images, training=True)  # [-1,1]
generated_images = (generated_images_raw + 1.0) / 2.0  # [-1,1] → [0,1]

# All losses in [0,1] space
pixel_loss = mae_loss_fn(clean_images, generated_images)  # [0,1] vs [0,1] ✅
```

### Option 3: Change Generator Activation
```python
# Replace tanh with sigmoid in generator's final layer
# Output directly to [0,1] - no denormalization needed
```

## ✅ FIX IMPLEMENTED (2025-10-21)

**Status**: FIXED - All range mismatches corrected in train_enhanced.py

### Changes Made:

1. **Training Step (Line ~593-603)**:
   - Added normalization: `clean_images_tanh = clean_images * 2.0 - 1.0`
   - Added normalization: `degraded_images_tanh = degraded_images * 2.0 - 1.0`
   - Generator now receives [-1,1] input
   - All losses now compare same range [-1,1]

2. **Validation Step (Line ~282-291)**:
   - Added normalization before generator call
   - PSNR/SSIM calculation now uses properly denormalized [0,1] values

3. **Sample Generation (Line ~1325-1335)**:
   - Normalized samples to [-1,1] before generator
   - Proper denormalization for saving images

### Verification:
```bash
poetry run python scripts/verify_range_fix.py
# ✅ ALL CHECKS PASSED (4/4)
# ✅ Generator can produce black pixels (range [0, 255])
```

## 📋 NEXT STEPS

1. ✅ **Fix implemented**: All range mismatches corrected
2. ✅ **Verified**: Test script confirms generator can produce black text
3. ⬜ **Retrain from scratch**: Previous checkpoint (ckpt-91) is INVALID
4. ⬜ **Expected results after retrain**:
   - Restored pixel range: [0, 255] with proper black text
   - Validation PSNR: >30 dB
   - Validation SSIM: >0.95

## 🔬 WHY THIS WASN'T CAUGHT EARLIER

1. **Recognizer was correctly denormalized** → CTC loss worked
2. **Training seemed stable** → No NaN or divergence
3. **PSNR improved over epochs** → But from wrong baseline
4. **Visual inspection needed** → Gray text not obvious in small samples

## ⚠️ INVALIDATED RESULTS

- **Run ID**: 704979cb9bbe40c3aeb4201f65141990
- **Checkpoint**: ckpt-91 (epoch 50)
- **Reported PSNR**: 36.61 dB (training set with data leakage + wrong calculation)
- **Actual PSNR**: 17.88 dB (validation set with wrong range comparison)
- **Status**: ❌ INVALID - Do NOT use for publication

## 📊 NEXT STEPS

1. ✅ Document bug in this file
2. ⬜ Fix train_enhanced.py (normalize clean/degraded to [-1,1])
3. ⬜ Verify fix with dry run (1 epoch)
4. ⬜ Launch full training with corrected code
5. ⬜ Re-evaluate with proper metrics
6. ⬜ Compare visual output quality (black text)

---
**Discoverer**: Analysis of validation results showing zero black pixels in restored images  
**Impact**: HIGH - Invalidates all training runs using train_enhanced.py before this fix  
**Priority**: CRITICAL - Must fix before any production training
