# CRITICAL BUG: Best Model Checkpoint Lost (dibco_ft_v2_loss_rebalance)

## Executive Summary
Training completed but **best model checkpoint (epoch 3, PSNR 20.31 dB) was DELETED** during training. Final `best_model/` contains wrong checkpoint (`ckpt-125` from epoch 16, PSNR 20.00 dB).

## User Report
- **Symptom**: Gambar restorasi menghasilkan stroke dan edge yang sangat tebal
- **Investigation**: Inference script `inference_portrait_overlap_experiment.py` menggunakan checkpoint dari `best_model/ckpt-125`
- **Finding**: ckpt-125 adalah BUKAN best checkpoint! Best adalah epoch 3 yang sudah HILANG

## Root Cause Analysis

### Bug #1: Checkpoint Saved Every Epoch (Ignores Config)
**Location**: `train_enhanced.py:1663`
```python
# Line 1663: Save checkpoint EVERY EPOCH (for resume capability)
ckpt_save_path = ckpt_manager.save()  # ❌ IGNORES save_interval config!
```

**Config**: `save_interval: 3` (expects save every 3 epochs)  
**Actual**: Saves **EVERY epoch**

**Impact**: 
- Creates 16 checkpoints instead of 5
- Fills max_checkpoints limit quickly
- Triggers automatic deletion of older checkpoints

---

### Bug #2: max_checkpoints=3 Causes Premature Deletion
**Config**: `max_checkpoints: 3`  
**Behavior**: TensorFlow CheckpointManager keeps only last 3 checkpoints

**Timeline of Checkpoint Deletion**:
```
Epoch 1: ckpt-121 saved
Epoch 2: ckpt-122 saved
Epoch 3: ckpt-123 saved ✅ BEST (PSNR 20.31 dB)
Epoch 4: ckpt-124 saved, DELETES ckpt-121
Epoch 5: ckpt-125 saved, DELETES ckpt-122
Epoch 6: ckpt-126 saved, DELETES ckpt-123 ❌ BEST MODEL LOST!
...
Epoch 16: ckpt-136 saved (final)
```

**Result**: Best checkpoint deleted at epoch 6, only 3 epochs after creation!

---

### Bug #3: best_model_ckpt_manager Has Hardcoded max_to_keep=1
**Location**: `train_enhanced.py:645`
```python
best_model_ckpt_manager = tf.train.CheckpointManager(
    checkpoint, best_model_dir, 
    max_to_keep=1  # ❌ HARDCODED! Should be max_to_keep=None for permanent storage
)
```

**Expected**: Best model directory should preserve ALL best models  
**Actual**: Only keeps 1 checkpoint (can be overwritten)

**Impact**: Even if best model is saved separately, it can be overwritten by later "improvements"

---

### Bug #4: Early Stopping Fallback Saves Wrong Checkpoint
**What Happened**:
1. Training triggers early stopping at epoch 16 (patience=15 exceeded)
2. Config has `restore_best_weights: true`
3. Code attempts to restore from `best_weights_path` (ckpt-123)
4. **ckpt-123 does NOT exist** (deleted by Bug #2)
5. Fallback mechanism restores `ckpt_manager.latest_checkpoint` = `ckpt-125`
6. Saves ckpt-125 to `best_model/` directory

**Code Location**: `train_enhanced.py:1475-1515`
```python
if args.restore_best_weights and best_weights_path:
    if os.path.exists(best_weights_path + '.index'):
        checkpoint.restore(best_weights_path).expect_partial()
    else:
        # ❌ BUG: Fallback to wrong checkpoint!
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
```

**Result**: `best_model/ckpt-125` is NOT the actual best model!

---

## Evidence

### Epoch Performance Comparison
| Epoch | PSNR (dB) | SSIM | Local Var | Isolated White % | Best Saved? | Status |
|-------|-----------|------|-----------|------------------|-------------|--------|
| **3** | **20.31** | 0.9059 | 861.2 | 37.8 | ✅ Yes | ❌ **DELETED** |
| 16 | 20.00 | 0.9059 | 896.8 | 23.2 | ❌ No | ✅ In best_model/ |

### Checkpoint Files
```bash
# Main checkpoint directory (only keeps last 3)
ckpt-139.{data,index}  # Epoch 14 (approx)
ckpt-140.{data,index}  # Epoch 15 (approx)
ckpt-141.{data,index}  # Epoch 16 (final)

# best_model directory (wrong checkpoint!)
ckpt-125.{data,index}  # Epoch 5 fallback, NOT epoch 3!
```

### Training Metrics
```json
{
  "best_epoch": 3,              // ✅ Correct
  "best_psnr": 20.31,          // ✅ Correct
  "last_completed_epoch": 16,  // Training stopped early
  "patience_counter": 14       // Almost hit patience=15
}
```

---

## Impact on Inference

### Why Restored Images Have Thick Strokes?
**Not directly caused by wrong checkpoint!**

Analysis shows epoch 16 actually has **BETTER stroke control** than epoch 3:
- Epoch 3: Isolated White Ratio = 37.8% (thicker strokes)
- Epoch 16: Isolated White Ratio = 23.2% (thinner strokes)

**Real issue**: 
1. Model was trained on DIBCO dataset (binarized documents)
2. Real paleography documents have different characteristics
3. Both checkpoints (3 and 16) have poor performance (PSNR ~20 dB vs target 24+ dB)

---

## Immediate Fixes

### Fix #1: Respect save_interval Config
**File**: `train_enhanced.py:1663`

**Current**:
```python
# Save checkpoint EVERY EPOCH (for resume capability)
ckpt_save_path = ckpt_manager.save()
```

**Fixed**:
```python
# Save checkpoint at save_interval for disk efficiency
if (epoch + 1) % args.save_interval == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f"  💾 Regular checkpoint saved: {ckpt_save_path}")
```

---

### Fix #2: Increase max_checkpoints to Prevent Best Model Deletion
**File**: Config JSON

**Current**:
```json
"max_checkpoints": 3
```

**Fixed**:
```json
"max_checkpoints": 10  // Or patience + 5 to ensure best is never deleted
```

**Rationale**: 
- With patience=15, training can run up to 15 epochs without improvement
- Need at least patience + buffer checkpoints to preserve best model
- Alternative: Set max_checkpoints = None (unlimited)

---

### Fix #3: Remove max_to_keep Limit for Best Model
**File**: `train_enhanced.py:645`

**Current**:
```python
best_model_ckpt_manager = tf.train.CheckpointManager(
    checkpoint, best_model_dir, 
    max_to_keep=1
)
```

**Fixed**:
```python
best_model_ckpt_manager = tf.train.CheckpointManager(
    checkpoint, best_model_dir, 
    max_to_keep=None  # ✅ Keep ALL best models permanently
)
```

---

### Fix #4: Abort Training if Best Checkpoint Not Found
**File**: `train_enhanced.py:1475-1515`

**Current**: Silently falls back to latest checkpoint

**Fixed**:
```python
if args.restore_best_weights and best_weights_path:
    if os.path.exists(best_weights_path + '.index'):
        checkpoint.restore(best_weights_path).expect_partial()
        print(f"   ✅ Best weights restored successfully")
    else:
        print(f"   ❌ CRITICAL: Best checkpoint not found: {best_weights_path}")
        print(f"   🚨 This indicates a BUG in checkpoint management!")
        print(f"   🛑 ABORTING to prevent saving wrong checkpoint as 'best'")
        raise FileNotFoundError(f"Best checkpoint lost: {best_weights_path}")
```

---

## Long-term Improvements

### 1. Separate Best Model from CheckpointManager
Don't rely on TensorFlow's CheckpointManager for best model. Manually save:
```python
if is_best_model:
    # Save manually to prevent automatic deletion
    best_model_path = os.path.join(best_model_dir, f"ckpt-epoch{epoch+1}")
    checkpoint.write(best_model_path)
```

### 2. Add Checkpoint Integrity Check
Before early stopping restoration:
```python
def verify_checkpoint_exists(path):
    if not (os.path.exists(path + '.index') and 
            os.path.exists(path + '.data-00000-of-00001')):
        raise FileNotFoundError(f"Checkpoint incomplete: {path}")
```

### 3. Log Checkpoint Operations
Add detailed logging:
```python
logger.info(f"Saving best model: epoch={epoch}, path={path}")
logger.info(f"Current checkpoints: {ckpt_manager.checkpoints}")
logger.info(f"Best model path tracked: {best_weights_path}")
```

---

## Recommendations for Current Situation

### Option 1: Retrain from ANRI V1 Checkpoint
**Pros**: 
- Start from proven checkpoint (PSNR 23.75 dB)
- Apply all bug fixes before training

**Cons**: 
- Requires retraining time (~3-4 hours)

**Command**:
```bash
nohup ./scripts/universal_train_from_json.sh configs/dibco_ft_v2_loss_rebalance_FIXED.json &
```

### Option 2: Use ckpt-125 with Understanding
**Pros**:
- Already trained
- Actually has better stroke control than epoch 3

**Cons**:
- Not the "true" best based on PSNR
- PSNR 20.00 dB is below target (24+ dB)

**Recommendation**: Use for testing, but plan retrain with fixes

### Option 3: Check if ckpt-123 Exists Anywhere
**Search for backups**:
```bash
find dual_modal_gan/checkpoints -name "*ckpt-123*" -o -name "*ckpt-3*"
```

If found in backups/snapshots, restore it!

---

## Prevention Checklist

Before starting ANY future training:

- [ ] Verify `save_interval` is respected in code
- [ ] Set `max_checkpoints >= patience + 5`
- [ ] Set `best_model_ckpt_manager max_to_keep=None`
- [ ] Add checkpoint integrity checks
- [ ] Enable detailed checkpoint logging
- [ ] Test early stopping restoration with mock training
- [ ] Document checkpoint paths in MLflow

---

## References
- Config: `configs/dibco_ft_v2_loss_rebalance.json`
- Training Script: `dual_modal_gan/scripts/train_enhanced.py`
- Metrics: `dual_modal_gan/checkpoints/dibco_ft_v2_loss_rebalance/metrics/training_metrics_fp32.json`
- Inference Script: `dual_modal_gan/scripts/inference_portrait_overlap_experiment.py`

---

**Date**: 2025-10-30  
**Severity**: CRITICAL  
**Status**: IDENTIFIED - FIXES PENDING  
**Impact**: Best model lost, incorrect checkpoint used for inference
