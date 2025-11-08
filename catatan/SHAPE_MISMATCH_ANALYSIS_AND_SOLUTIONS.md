# 🔧 SHAPE MISMATCH ANALYSIS & SOLUTIONS

**Date**: 2025-10-31
**Problem**: ckpt-115 checkpoint fails to load (BatchNorm shape mismatch)
**Impact**: Transfer learning blocked, must train from scratch

---

## 🚨 ERROR DETAILS

```
❌ Error loading pretrained checkpoint: Received incompatible tensor with shape (64,)
when attempting to restore variable with shape (512,) and name
batch_normalization_24/moving_mean:0
```

**Analysis**:
- **Layer**: BatchNorm layer `batch_normalization_24`
- **Parameter**: `moving_mean`
- **Expected**: 512 channels
- **Received**: 64 channels
- **Difference**: 8x mismatch (64 → 512)

---

## 🔍 ROOT CAUSE ANALYSIS

### Architecture Mismatch Between Source and Target

**Source Checkpoint (ckpt-115)**:
- Model: `anri_finetuning_stage1_full_model_v2`
- Generator: Enhanced V2 (CBAM + RDB + Multi-Scale)
- Architecture: More complex, different channel dimensions
- BatchNorm Layers: Different channel sizes

**Target Configuration (dibco_transfer_learning_v2_memory_efficient)**:
- Generator: BASE (Standard U-Net)
- Architecture: Simpler, fewer channels
- BatchNorm Layers: Smaller channel dimensions

**Mismatch Point**:
```
Source (Enhanced V2): 512 channels in some layers
Target (BASE):        64 channels in corresponding layers

→ BatchNorm moving_mean/moving_variance shapes incompatible
```

---

## 💡 SOLUTIONS (Priority Order)

### Solution 1: Use Compatible Checkpoint (RECOMMENDED) ⭐
**Strategy**: Use ckpt-99 instead of ckpt-115

**Rationale**:
- ckpt-99 from `thin_stroke_preservation_v1_academic`
- BASE model architecture (matches target)
- PSNR: 30.56 dB (proven performance)
- No shape mismatch expected

**Implementation**:
```bash
# Edit config: dibco_transfer_learning_v2_memory_efficient.json
"pretrained_checkpoint": "dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/ckpt-99"
```

**Expected Result**:
- ✅ Transfer learning will work
- ✅ Faster convergence (starting from 30.56 dB level)
- ✅ Better final performance (potentially 18-20 dB)

### Solution 2: Create Architecture-Compatible Config ⭐
**Strategy**: Use Enhanced generator to match ckpt-115 architecture

**Implementation**:
```json
{
  "generator_version": "enhanced",
  "discriminator_version": "enhanced_v2_fixed",
  "pretrained_checkpoint": "dual_modal_gan/checkpoints/anri_finetuning_stage1_full_model_v2/best_model/ckpt-115",
  "batch_size": 1,  // Reduce for memory
  "perceptual_loss_weight": 15.0
}
```

**Expected Result**:
- ✅ Transfer learning will work
- ⚠️ May hit memory limits (Enhanced + Perceptual)
- ✅ Maximum performance potential

### Solution 3: Manual Weight Transfer (ADVANCED)
**Strategy**: Extract compatible weights manually

**Steps**:
1. Load ckpt-115 in separate script
2. Extract weights layer by layer
3. Map to BASE architecture
4. Save as new checkpoint

**Code Sketch**:
```python
# Pseudocode - manual weight mapping
source_model = load_model('ckpt-115')  # Enhanced V2
target_model = build_base_generator()  # BASE

# Map compatible layers
for i, layer in enumerate(target_model.layers):
    source_layer = source_model.layers[i]
    if layer.shape == source_layer.shape:
        layer.set_weights(source_layer.get_weights())
```

**Expected Result**:
- ✅ Partial transfer learning
- ⚠️ Time consuming
- ⚠️ May lose some learned features

### Solution 4: Progressive Transfer Learning (LONG-TERM)
**Strategy**: Chain through compatible checkpoints

**Chain**:
```
ckpt-99 (BASE) → [Train Enhanced V2] → ckpt-X → DIBCO
```

**Steps**:
1. Train Enhanced V2 from ckpt-99
2. Save checkpoint (compatible)
3. Use for DIBCO transfer

**Expected Result**:
- ✅ Full pipeline compatibility
- ⚠️ Requires intermediate training
- ✅ Optimal long-term solution

---

## 🎯 RECOMMENDED APPROACH (IMMEDIATE)

### Option A: Quick Fix with ckpt-99 ⭐
**Timeline**: 5 minutes
**Effort**: Minimal
**Expected Result**: 18-20 dB PSNR

```bash
# Edit config
sed -i 's|ckpt-115|ckpt-99|g' configs/dibco_transfer_learning_v2_memory_efficient.json

# Launch training
nohup ./scripts/universal_train_from_json.sh configs/dibco_transfer_learning_v2_memory_efficient.json &
```

**Why This Works**:
- ckpt-99 is BASE architecture (matches target)
- Proven performance (30.56 dB)
- No shape mismatch
- Faster convergence

### Option B: Architecture Match with Enhanced
**Timeline**: 30 minutes setup
**Effort**: Moderate
**Expected Result**: 20-25 dB PSNR

```bash
# Create new config
cp configs/dibco_transfer_learning_v2_memory_efficient.json configs/dibco_transfer_learning_v2_enhanced.json

# Edit:
# - generator_version: "enhanced"
# - discriminator_version: "enhanced_v2_fixed"
# - pretrained_checkpoint: "ckpt-115"
# - batch_size: 1
# - perceptual_loss_weight: 15.0

# Launch
nohup ./scripts/universal_train_from_json.sh configs/dibco_transfer_learning_v2_enhanced.json &
```

**Why This Works**:
- Matches ckpt-115 architecture exactly
- Full transfer learning benefits
- Maximum performance potential

---

## 📊 COMPARISON OF SOLUTIONS

| Solution | Time | Effort | Compatibility | Expected PSNR | Memory |
|----------|------|--------|---------------|---------------|--------|
| **ckpt-99** | 5 min | Low | ✅ Perfect | 18-20 dB | ✅ Safe |
| **Enhanced Match** | 30 min | Medium | ✅ Perfect | 20-25 dB | ⚠️ Risky |
| **Manual Transfer** | 2 hours | High | ⚠️ Partial | 16-18 dB | ✅ Safe |
| **Progressive** | 6 hours | High | ✅ Perfect | 22-28 dB | ✅ Safe |

---

## 🚀 IMPLEMENTATION PLAN

### Phase 1: Quick Fix (TODAY) ⚡
1. **Use ckpt-99 for immediate transfer learning**
   - Edit config: change ckpt-115 → ckpt-99
   - Launch training
   - Expected: 18-20 dB in 5 epochs

### Phase 2: Architecture Match (THIS WEEK) 📅
2. **Create Enhanced config for maximum performance**
   - New config with Enhanced V2 + ckpt-115
   - Memory optimization (batch_size=1)
   - Expected: 20-25 dB in 10 epochs

### Phase 3: Long-term Solution (NEXT MONTH) 🔮
3. **Progressive transfer learning pipeline**
   - Document architecture compatibility matrix
   - Create automated checkpoint selection
   - Establish best practices

---

## ✅ VALIDATION CHECKLIST

- [x] **Problem Identified**: BatchNorm shape mismatch (64 vs 512)
- [x] **Root Cause**: Architecture incompatibility (Enhanced V2 vs BASE)
- [x] **Solution 1**: Use ckpt-99 (BASE compatible)
- [x] **Solution 2**: Match architecture (Enhanced V2 + ckpt-115)
- [x] **Solution 3**: Manual weight transfer (advanced)
- [x] **Solution 4**: Progressive pipeline (long-term)
- [x] **Recommendation**: Start with ckpt-99 for immediate results

---

## 📝 NEXT STEPS

### Immediate Action (Next 30 minutes)
1. **Choose solution**: ckpt-99 (quick) or Enhanced (optimal)
2. **Edit configuration**: Update pretrained_checkpoint path
3. **Launch training**: Monitor for successful checkpoint loading
4. **Validate transfer**: Check logs for "✅ Pretrained weights loaded successfully"

### If Transfer Learning Works
- Expected convergence: 5-8 epochs to optimal
- Target PSNR: 18-25 dB (depending on solution)
- Document: Architecture compatibility matrix

### If Still Fails
- Check: Generator/discriminator version matches checkpoint
- Verify: Model building order matches training script
- Consider: Manual architecture inspection required

---

**Conclusion**: Shape mismatch is SOLVABLE with correct checkpoint selection! 🚀

**Recommended**: Start with ckpt-99 for immediate 18-20 dB results, then optimize with Enhanced V2 for 20-25 dB potential.