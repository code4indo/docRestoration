# ANRI FINETUNING IMPLEMENTATION CHECKLIST

**Date**: 2025-10-30  
**Goal**: Implement 3-stage progressive finetuning WITHOUT PSNR drop

---

## ✅ PRE-IMPLEMENTATION (DO THIS FIRST)

### 1. Code Modifications Required
- [ ] **Add Layer Freezing Support** to `train_enhanced.py`
  - Function: `apply_freeze_strategy(generator, discriminator, freeze_config)`
  - Read `freeze_strategy` from config JSON
  - Set `layer.trainable = False` for specified layers

- [ ] **Add Dual Validation System** to `train_enhanced.py`
  - Load TWO validation sets: base synthetic + ANRI
  - Compute metrics on BOTH datasets each epoch
  - Log: `base_val_psnr`, `anri_val_psnr`
  - Early stopping: check BOTH thresholds

- [ ] **Add Mixed Dataset Support** (for Stage 2+3)
  - Create TFRecord combining ANRI + base synthetic
  - Implement balanced sampling (configurable ratio)
  - File: `dual_modal_gan/data/finetuning/mixed_dataset.tfrecord`

- [ ] **Add Safety Mechanisms**
  - Hard stop if `base_val_psnr < base_psnr_red_line` (default: 28.0)
  - Warning if `base_val_psnr < warning_threshold` (default: 29.0)
  - Auto LR reduction if rapid PSNR drop

### 2. Data Preparation
- [ ] Verify ANRI TFRecord exists: `dual_modal_gan/data/finetuning/finetuning_train.tfrecord`
- [ ] Check ANRI dataset size (should have enough samples for train/val/test)
- [ ] Create mixed dataset TFRecord (70% base + 30% ANRI for Stage 2)
- [ ] Verify base validation set is intact: `dual_modal_gan/data/dataset_gan.tfrecord`

### 3. Checkpoint Verification
- [ ] Base checkpoint exists: `dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/ckpt-99`
- [ ] Base checkpoint intact (not corrupted)
- [ ] Base model PSNR confirmed: 30.56 dB at epoch 49

---

## 🚀 STAGE 1: FEATURE ADAPTATION (5 epochs)

### Config: `configs/anri_finetuning_stage1_adaptation.json`

**Key Parameters**:
- LR_G: **1e-6** (ultra-conservative)
- LR_D: **5e-6** (discriminator adapts faster)
- Freeze: Encoder + early decoder
- Pixel loss: **300.0** (high preservation)
- Adv loss: **0.5** (low realism pressure)
- Perceptual loss: **15.0** (topology preservation)
- Batch size: **1** (stability)
- Discriminator mode: **predicted** (MATCH base!)

**Launch Command**:
```bash
nohup ./scripts/universal_train_from_json.sh configs/anri_finetuning_stage1_adaptation.json > logs/stage1_adaptation.log 2>&1 &
```

**Monitor**:
```bash
tail -f logs/stage1_adaptation.log
# Watch for: base_val_psnr, anri_val_psnr
```

**Expected Results**:
- Base PSNR: 30.0-30.5 dB ✅
- ANRI PSNR: 24-26 dB ✅
- Duration: ~1-2 hours

**Success Criteria**:
- ✅ Base PSNR stays ≥ 30.0 dB
- ✅ ANRI PSNR improves from ~23.75 dB
- ✅ No warning triggers

**If Failed**:
- Check freeze strategy is working (encoder layers should have `trainable=False`)
- Verify discriminator_mode = "predicted"
- Check dual validation is monitoring base dataset

---

## 🚀 STAGE 2: FULL MODEL REFINEMENT (10 epochs)

### Config: `configs/anri_finetuning_stage2_refinement.json` (CREATE THIS)

**Key Changes from Stage 1**:
- LR_G: **2e-6** (slightly higher)
- Freeze: **NONE** (unfreeze all layers)
- Pixel loss: **250.0** (reduce slightly)
- Adv loss: **1.0** (increase realism)
- Perceptual loss: **12.0** (reduce slightly)
- Batch size: **2** (can increase now)
- **Mixed dataset**: 30% ANRI + 70% base synthetic

**Launch Command**:
```bash
nohup ./scripts/universal_train_from_json.sh configs/anri_finetuning_stage2_refinement.json > logs/stage2_refinement.log 2>&1 &
```

**Expected Results**:
- Base PSNR: 29.5-30.5 dB ✅ (slight drop OK)
- ANRI PSNR: 27-29 dB ✅
- Duration: ~3-4 hours

**Success Criteria**:
- ✅ Base PSNR stays ≥ 29.5 dB
- ✅ ANRI PSNR significantly improves
- ✅ No red line trigger (28.0 dB)

---

## 🚀 STAGE 3: ANRI SPECIALIZATION (10 epochs)

### Config: `configs/anri_finetuning_stage3_specialization.json` (CREATE THIS)

**Key Changes from Stage 2**:
- LR_G: **1e-6** (back to conservative)
- LR_D: **3e-6**
- Loss weights: **MATCH base model** (200, 1.5, 10.0)
- **Mixed dataset**: 50% ANRI + 50% base synthetic
- Early stopping patience: **8** epochs

**Launch Command**:
```bash
nohup ./scripts/universal_train_from_json.sh configs/anri_finetuning_stage3_specialization.json > logs/stage3_specialization.log 2>&1 &
```

**Expected Results**:
- Base PSNR: 29.0-30.0 dB ✅
- ANRI PSNR: **29-31 dB** ✅ (target!)
- Duration: ~3-4 hours

**Success Criteria**:
- ✅ Base PSNR stays ≥ 29.0 dB
- ✅ ANRI PSNR ≥ 28.0 dB
- ✅ Overall quality improved on ANRI

---

## 📊 MONITORING DASHBOARD

Create simple monitoring script:

```bash
# Check all stages progress
watch -n 10 '
echo "=== STAGE 1 (Latest) ==="
tail -n 5 logs/stage1_adaptation.log | grep -E "Epoch|PSNR|base_val"

echo ""
echo "=== STAGE 2 (Latest) ==="
tail -n 5 logs/stage2_refinement.log | grep -E "Epoch|PSNR|base_val"

echo ""
echo "=== STAGE 3 (Latest) ==="
tail -n 5 logs/stage3_specialization.log | grep -E "Epoch|PSNR|base_val"
'
```

---

## 🚨 EMERGENCY PROCEDURES

### If Base PSNR Drops Below 28.0 dB (Red Line):
1. **STOP TRAINING IMMEDIATELY**
2. Restore previous stage checkpoint
3. Analyze what went wrong:
   - LR too high?
   - Freeze strategy not working?
   - Mixed dataset ratio wrong?
4. Adjust config and retry

### If ANRI PSNR Not Improving:
1. Check if ANRI dataset has enough samples
2. Verify data augmentation is working
3. Consider increasing ANRI ratio in mixed dataset
4. Check if discriminator is adapting (monitor discriminator loss)

### If Training Crashes:
1. Check GPU memory (reduce batch size if needed)
2. Verify all paths are correct
3. Check if freeze strategy conflicts with checkpoint loading

---

## 📈 SUCCESS METRICS

### Overall Goal:
| Metric | Previous Finetuning | Target | Stretch Goal |
|--------|-------------------|--------|--------------|
| Base PSNR | 23.75 dB (FAIL) | ≥29.0 dB | ≥30.0 dB |
| ANRI PSNR | 23.75 dB | ≥28.0 dB | ≥30.0 dB |
| PSNR Drop | -6.81 dB | ≤-1.5 dB | ≤-0.5 dB |

### Stage-by-Stage:
- **Stage 1**: Base ≥30.0, ANRI 24-26
- **Stage 2**: Base ≥29.5, ANRI 27-29
- **Stage 3**: Base ≥29.0, ANRI **29-31**

---

## 🎯 IMPLEMENTATION ORDER

1. **TODAY**: Implement code modifications (freeze, dual validation)
2. **TEST**: Run Stage 1 for 1-2 epochs (verify it works)
3. **LAUNCH**: Full Stage 1 (5 epochs)
4. **ANALYZE**: Check if base PSNR maintained
5. **CREATE**: Stage 2 config based on Stage 1 results
6. **LAUNCH**: Stage 2 (10 epochs)
7. **CREATE**: Stage 3 config based on Stage 2 results
8. **LAUNCH**: Stage 3 (10 epochs)
9. **EVALUATE**: Final model on both base + ANRI test sets

**Total Time Estimate**: 2-3 days (including implementation + training)

---

## 💡 TIPS

1. **Save checkpoints EVERY epoch** in Stage 1 (max_checkpoints: 10)
2. **Monitor logs actively** during Stage 1 (critical stage)
3. **Don't rush**: If Stage 1 fails, fix it before Stage 2
4. **Document everything**: Log all results for paper
5. **Be ready to adjust**: This is experimental, may need tweaking

---

## ✅ DONE WHEN:

- [ ] All 3 stages completed
- [ ] Final base PSNR ≥ 29.0 dB
- [ ] Final ANRI PSNR ≥ 28.0 dB
- [ ] Inference tested on ANRI test set
- [ ] Results documented in logbook
- [ ] Best checkpoint identified and backed up

**Expected Success Rate**: >90% (based on conservative strategy)
