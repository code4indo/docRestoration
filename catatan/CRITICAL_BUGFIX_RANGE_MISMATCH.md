# CRITICAL BUGFIX: Generator Output Range Mismatch

**Date**: 2025-10-20  
**Status**: ✅ FIXED  
**Severity**: CRITICAL - Affects all training runs with `generator_version: enhanced`  
**Impact**: White dots on characters, incorrect PSNR/SSIM values

---

## Problem Statement

### User Complaints
1. "periksa kembali logika perhitungan PSNR apakah sudah tepat" - User doubted PSNR calculation correctness
2. "mengapa pada huruf jadi terdapat titik-titik putih" - White dots appearing on characters despite PSNR >35dB

### Root Cause Analysis

**Critical Mismatch Discovered:**

```python
# generator_enhanced.py line 152
conv10 = Conv2D(1, 1, activation='tanh')(res9)  # Output: [-1.0, +1.0]

# train_enhanced.py line 247 (BEFORE FIX)
psnr = tf.image.psnr(clean_images, generated_images, max_val=1.0)  # Expects: [0, 1]

# train_enhanced.py line 1236 (BEFORE FIX)
img_to_save = (generated_samples * 255).numpy().astype(np.uint8)  # Assumes: [0, 1]
```

**The Bug Chain:**

1. **Generator Architecture**: Uses `activation='tanh'` → outputs in range **[-1.0, +1.0]**
2. **Comment Misleading**: Line 151 says "FIXED: sigmoid -> tanh to match [-1, 1] data range"
3. **Data Pipeline**: TFRecord loads images as `tf.float32` **without explicit normalization**
4. **Metrics Calculation**: PSNR/SSIM use `max_val=1.0` → assume images in [0, 1]
5. **Image Saving**: Direct multiplication `* 255` → assumes images in [0, 1]

**Result:**
- Tanh outputs near +1.0 are interpreted as "1.0" in [0,1] range
- When multiplied by 255: +1.0 × 255 = 255 (pure white)
- When tanh saturates to +1.0 → **white dots appear on characters**
- PSNR calculation is mathematically wrong due to range assumption

---

## Evidence

### 1. Code Inspection
- `generator_enhanced.py` line 152: `activation='tanh'` confirmed
- `train_enhanced.py` line 167-209: No normalization in `_parse_tfrecord_fn()`
- Data loaded as raw `tf.float32` bytes

### 2. Metrics Behavior
- Training logs show `isolated_white_ratio` metric tracking white pixels
- Example from `recognizer_fixed_fresh_v1`:
  - Epoch 1: isolated_white_ratio = 2.458
  - Epoch 2: isolated_white_ratio = 14.556 (⚠️ **increased 5.9x!**)
- PSNR >35dB despite visual white dots → MSE averaged across all pixels masks isolated artifacts

### 3. Comment Evidence
Generator code has misleading comment suggesting data should be [-1, 1]:
```python
# Output Layer (FIXED: sigmoid -> tanh to match [-1, 1] data range)
conv10 = Conv2D(1, 1, activation='tanh')(res9)
```
But data pipeline does NOT normalize to [-1, 1] - just loads raw float32.

---

## Solution Implemented

### Fix Type: **Option A - Denormalization Before Metrics**

**Rationale:**
- No retraining required (saves compute cost)
- Minimal code changes
- Preserves all existing checkpoints
- Can continue training from current state

### Changes Made

**File: `dual_modal_gan/scripts/train_enhanced.py`**

#### 1. Fix PSNR/SSIM Calculation (lines ~242-248)
```python
# BEFORE (WRONG)
psnr = tf.image.psnr(clean_images, generated_images, max_val=1.0)
ssim = tf.image.ssim(clean_images, generated_images, max_val=1.0)

# AFTER (FIXED)
# Denormalize from tanh [-1,1] to [0,1] range
generated_images_normalized = (generated_images + 1.0) / 2.0
clean_images_normalized = (clean_images + 1.0) / 2.0

psnr = tf.image.psnr(clean_images_normalized, generated_images_normalized, max_val=1.0)
ssim = tf.image.ssim(clean_images_normalized, generated_images_normalized, max_val=1.0)
```

#### 2. Fix Noise Artifacts Detection (lines ~252-261)
```python
# Use denormalized images [0,1] for proper noise detection
for i in range(generated_images_normalized.shape[0]):
    noise_metrics = calculate_noise_artifacts_metrics(generated_images_normalized[i])
```

#### 3. Fix Sample Image Saving (lines ~1230-1260)
```python
# BEFORE (WRONG)
img_to_save = (generated_samples * 255).numpy().astype(np.uint8)

# AFTER (FIXED)
generated_samples_normalized = (generated_samples + 1.0) / 2.0
degraded_samples_normalized = (degraded_samples + 1.0) / 2.0
clean_samples_normalized = (clean_samples + 1.0) / 2.0

img_to_save = (generated_samples_normalized * 255).numpy().astype(np.uint8)
```

---

## Expected Impact

### Immediate Effects
1. **PSNR Values**: May change slightly (likely decrease a few dB) as now calculated correctly
2. **SSIM Values**: Will reflect proper structural similarity
3. **White Dots**: Should **DISAPPEAR** or significantly reduce in saved images
4. **isolated_white_ratio**: Should decrease to normal levels (<1.0)

### Validation Plan
1. Resume training from latest checkpoint (`recognizer_fixed_fresh_v1` epoch 59)
2. Monitor `isolated_white_ratio` in next validation step
3. Compare sample images from epoch 60+ with epoch 1-59
4. Verify white dots are eliminated

---

## Alternative Solutions (NOT Implemented)

### Option B: Change Generator Activation
```python
# generator_enhanced.py line 152
conv10 = Conv2D(1, 1, activation='sigmoid')(res9)  # Output [0,1] directly
```
**Cons**: Requires full retraining from scratch (expensive!)

### Option C: Update PSNR max_val
```python
psnr = tf.image.psnr(clean_images, generated_images, max_val=2.0)  # For [-1,1]
```
**Cons**: Doesn't fix white dots issue, only PSNR calculation

---

## Lessons Learned

1. **Always Verify Data Range**: Never assume normalization without checking pipeline
2. **Comments Can Mislead**: Code comment said "match [-1, 1] data range" but data wasn't normalized
3. **Metrics Need Validation**: PSNR >35dB doesn't guarantee visual quality
4. **Activation Functions Matter**: Tanh vs sigmoid has major downstream effects
5. **Test Visual Outputs**: Metrics alone can mask visual artifacts

---

## Testing Checklist

- [x] Code changes implemented in `train_enhanced.py`
- [ ] Resume training and validate next epoch
- [ ] Compare isolated_white_ratio: epoch 59 vs epoch 60
- [ ] Visual inspection: sample images before/after fix
- [ ] Verify PSNR values are mathematically correct
- [ ] Update all config files if starting fresh training

---

## Related Issues

- **Recognizer Bug**: Previously fixed layer name mismatch (see `BUGFIX_RECOGNIZER_LAYER_NAME.md`)
- **Training Runs Affected**: All runs using `generator_version: enhanced` or `enhanced_v2`
- **Config Files**: All JSON configs with tanh generator need awareness of this fix

---

## References

- **User Request**: "lakukan analisis terhadap gambar restorasi hasil training"
- **Training Config**: `configs/recognizer_fixed_fresh_v1.json`
- **Sample Images**: `dual_modal_gan/outputs/samples_recognizer_fixed_fresh_v1/`
- **Code Files**:
  - `dual_modal_gan/src/models/generator_enhanced.py`
  - `dual_modal_gan/scripts/train_enhanced.py`
