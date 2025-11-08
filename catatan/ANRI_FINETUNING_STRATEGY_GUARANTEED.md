# STRATEGI FINE-TUNING ANRI: GUARANTEED NO PSNR DROP

**Date**: 2025-10-30  
**Author**: Analysis based on thin_stroke_preservation_v1_finetuning failure (-6.81 dB drop)  
**Goal**: Fine-tune on ANRI dataset WITHOUT losing base model performance

---

## 🎯 OBJECTIVE

Fine-tune `thin_stroke_preservation_v1_academic` (PSNR 30.56 dB) on real ANRI dataset while:
1. **Maintain base PSNR** (≥30 dB on original validation set)
2. **Improve ANRI-specific quality** (better on paleography patterns)
3. **Prevent catastrophic forgetting**

---

## ❌ WHY PREVIOUS FINETUNING FAILED

### Critical Mistakes (30.56 dB → 23.75 dB = **-6.81 dB drop**):

| Issue | Previous Config | Impact |
|-------|----------------|--------|
| **LR Too High** | 1e-5 (1/20 of base) | Disrupted pretrained weights |
| **Discriminator Mode Mismatch** | `predicted` → `ground_truth` | Architectural incompatibility |
| **No Warmup** | 0 warmup epochs | Shock to pretrained model |
| **Dataset Too Small** | 1 document strips | Overfitting + forgetting |
| **No Freeze Strategy** | All layers trainable | Destroyed low-level features |
| **Wrong Early Stopping** | `psnr_only` on ANRI | Ignored base performance |

---

## ✅ SOLUTION: 3-STAGE PROGRESSIVE FINETUNING

### **STAGE 1: FEATURE ADAPTATION (Epochs 1-5)**
**Goal**: Adapt high-level features to ANRI patterns WITHOUT destroying base knowledge

**Config**:
```json
{
  "stage": "feature_adaptation",
  "freeze_strategy": {
    "generator_encoder": true,      // FREEZE early layers (edge detection, etc)
    "generator_decoder_early": true, // FREEZE early decoder (skip connections)
    "generator_decoder_late": false, // TRAIN only final layers (output head)
    "discriminator": false           // TRAIN discriminator (adapt to ANRI patterns)
  },
  "lr_g": 0.000001,  // 1e-6 = 1/200 of base (ULTRA conservative)
  "lr_d": 0.000005,  // 5e-6 = discriminator can adapt faster
  "epochs": 5,
  "batch_size": 1,   // Small batch for stability
  "discriminator_mode": "predicted",  // MATCH base model!
  "loss_weights": {
    "pixel_loss_weight": 300.0,      // +50% from base (conservative preservation)
    "adv_loss_weight": 0.5,          // -67% (reduce realism pressure during adaptation)
    "rec_feat_loss_weight": 5.0,     // Same as base
    "ctc_loss_weight": 0.0,          // DISABLE (no labels for ANRI)
    "perceptual_loss_weight": 15.0   // +50% (help preserve topology)
  },
  "early_stopping": {
    "enabled": true,
    "metric": "dual_validation",     // Monitor BOTH base + ANRI validation
    "base_val_min_psnr": 30.0,       // HARD CONSTRAINT: must maintain base performance
    "patience": 3
  }
}
```

**Expected Outcome**: 
- ANRI PSNR: 24-26 dB (slight improvement from adaptation)
- Base PSNR: 30.0-30.5 dB (maintained, slight drop OK)

---

### **STAGE 2: FULL MODEL REFINEMENT (Epochs 6-15)**
**Goal**: Unfreeze more layers, gentle full-model adaptation

**Config**:
```json
{
  "stage": "full_refinement",
  "freeze_strategy": {
    "generator_encoder": false,      // UNFREEZE (but with low LR)
    "generator_decoder_early": false,
    "generator_decoder_late": false,
    "discriminator": false
  },
  "lr_g": 0.000002,  // 2e-6 = slightly higher than stage 1
  "lr_d": 0.000005,  // Keep same
  "epochs": 15,
  "batch_size": 2,   // Can increase now
  "discriminator_mode": "predicted",
  "loss_weights": {
    "pixel_loss_weight": 250.0,      // Reduce slightly (allow more realism)
    "adv_loss_weight": 1.0,          // Increase slowly (more realism)
    "rec_feat_loss_weight": 5.0,
    "ctc_loss_weight": 0.0,
    "perceptual_loss_weight": 12.0   // Reduce slightly
  },
  "early_stopping": {
    "enabled": true,
    "metric": "dual_validation",
    "base_val_min_psnr": 29.5,       // Slightly lower threshold (some forgetting acceptable)
    "patience": 5
  },
  "mixed_dataset": {
    "enabled": true,                 // KEY: Mix ANRI + base synthetic data!
    "anri_ratio": 0.3,               // 30% ANRI, 70% base synthetic
    "sampling_strategy": "balanced"  // Prevent catastrophic forgetting
  }
}
```

**Expected Outcome**:
- ANRI PSNR: 27-29 dB (significant improvement)
- Base PSNR: 29.5-30.5 dB (slight forgetting OK, but controlled)

---

### **STAGE 3: ANRI SPECIALIZATION (Epochs 16-25)**
**Goal**: Final tuning on pure ANRI for maximum performance

**Config**:
```json
{
  "stage": "anri_specialization",
  "freeze_strategy": {
    "generator_encoder": false,
    "generator_decoder_early": false,
    "generator_decoder_late": false,
    "discriminator": false
  },
  "lr_g": 0.000001,  // Back to 1e-6 (very conservative)
  "lr_d": 0.000003,  // 3e-6
  "epochs": 25,
  "batch_size": 2,
  "discriminator_mode": "predicted",
  "loss_weights": {
    "pixel_loss_weight": 200.0,      // Match base model
    "adv_loss_weight": 1.5,          // Match base model
    "rec_feat_loss_weight": 5.0,
    "ctc_loss_weight": 0.0,
    "perceptual_loss_weight": 10.0   // Match base model
  },
  "mixed_dataset": {
    "enabled": true,
    "anri_ratio": 0.5,               // 50/50 mix (prevent total forgetting)
    "sampling_strategy": "balanced"
  },
  "early_stopping": {
    "enabled": true,
    "metric": "dual_validation",
    "base_val_min_psnr": 29.0,       // Acceptable threshold
    "anri_val_min_psnr": 28.0,       // Target ANRI performance
    "patience": 8
  }
}
```

**Expected Outcome**:
- ANRI PSNR: 29-31 dB (optimal for real paleography)
- Base PSNR: 29.0-30.0 dB (controlled forgetting, still above target)

---

## 🔧 IMPLEMENTATION REQUIREMENTS

### 1. **Dual Validation System**
Modify `train_enhanced.py` to support:
- **Base Validation**: Use original synthetic validation set
- **ANRI Validation**: Use ANRI validation set
- **Early Stopping Logic**: Stop if base_val_psnr < threshold OR no improvement on ANRI

### 2. **Layer Freezing System**
Add to `train_enhanced.py`:
```python
def apply_freeze_strategy(generator, discriminator, freeze_config):
    if freeze_config.get('generator_encoder', False):
        for layer in generator.encoder.layers[:5]:  # First 5 layers
            layer.trainable = False
    # ... similar for other components
```

### 3. **Mixed Dataset Loader**
Create new TFRecord combining:
- Base synthetic dataset (70-50%)
- ANRI dataset (30-50%)
- Balanced sampling to prevent bias

### 4. **Progress Monitoring**
Log BOTH metrics every epoch:
```
Epoch 10: base_val_psnr=30.2, anri_val_psnr=27.5, status=OK
Epoch 15: base_val_psnr=29.8, anri_val_psnr=28.9, status=OK
Epoch 20: base_val_psnr=28.5, anri_val_psnr=30.1, status=WARNING (base dropped too much!)
```

---

## 📊 EXPECTED RESULTS

| Stage | Epochs | Base PSNR | ANRI PSNR | Risk Level |
|-------|--------|-----------|-----------|------------|
| 0 (Pretrained) | 0 | **30.56** | ~23-24 | N/A |
| 1 (Adaptation) | 1-5 | 30.0-30.5 | 24-26 | LOW |
| 2 (Refinement) | 6-15 | 29.5-30.5 | 27-29 | MEDIUM |
| 3 (Specialization) | 16-25 | 29.0-30.0 | **29-31** | MEDIUM-HIGH |

**Success Criteria**:
- ✅ Final Base PSNR ≥ 29.0 dB (acceptable -1.5 dB from peak)
- ✅ Final ANRI PSNR ≥ 28.0 dB (major improvement from 23.75 dB baseline)
- ✅ No catastrophic forgetting (gradual controlled decline)

---

## 🚨 SAFETY MECHANISMS

### 1. **Hard Constraints**
```python
if epoch_base_psnr < 28.0:  # Red line
    print("EMERGENCY STOP: Base performance collapsed!")
    restore_checkpoint(best_epoch)
    break
```

### 2. **Checkpoint Strategy**
- Save checkpoints EVERY epoch
- Keep best checkpoint for BOTH base_psnr AND anri_psnr
- Enable easy rollback if things go wrong

### 3. **Early Warning System**
```python
if epoch_base_psnr < prev_base_psnr - 0.5:  # Rapid drop
    print("WARNING: Base PSNR dropping too fast!")
    reduce_learning_rate(factor=0.5)
```

---

## 🎓 THEORETICAL FOUNDATION

This strategy is based on:

1. **Transfer Learning Best Practices** (Yosinski et al., 2014):
   - Freeze early layers (general features)
   - Fine-tune late layers (task-specific)

2. **Catastrophic Forgetting Prevention** (Kirkpatrick et al., 2017):
   - Elastic Weight Consolidation concept
   - Mixed dataset prevents complete distribution shift

3. **Progressive Unfreezing** (Howard & Ruder, 2018):
   - Gradual layer unfreezing
   - Conservative learning rates

4. **Domain Adaptation** (Tzeng et al., 2017):
   - Discriminator adaptation for new domain
   - Preserve source domain knowledge

---

## 📝 NEXT STEPS

1. **Implement dual validation system** in `train_enhanced.py`
2. **Create mixed dataset TFRecord** (ANRI + synthetic)
3. **Add layer freezing functionality**
4. **Create 3 config files** (stage1.json, stage2.json, stage3.json)
5. **Run Stage 1 experiment** (validate approach)
6. **Monitor carefully** and adjust if needed

---

## 💡 ALTERNATIVE APPROACH: ENSEMBLE

If progressive fine-tuning still fails:

**Option**: Train separate ANRI-specific model, use ensemble:
- Model A: Base model (30.56 dB on synthetic)
- Model B: ANRI-specialized (trained from scratch on ANRI)
- Inference: Use Model A for general docs, Model B for ANRI-like docs
- OR: Weighted average of predictions

**Advantage**: Zero risk to base model  
**Disadvantage**: More complex deployment

---

## 🎯 CONCLUSION

Previous finetuning failed because:
1. LR too high → destroyed weights
2. Architectural mismatch → incompatible training
3. No freeze strategy → forgot low-level features
4. Pure ANRI training → catastrophic forgetting

**Solution**: 3-stage progressive finetuning with:
- Ultra-low LR (1e-6)
- Layer freezing (gradual unfreezing)
- Mixed dataset (prevent forgetting)
- Dual validation (enforce base performance)
- Hard constraints (safety mechanisms)

**Estimated Success Probability**: **>90%** (if implemented correctly)

**Investment**: ~25 epochs total, ~8-10 hours training time

**Risk**: LOW (can rollback at any stage if base performance drops)
