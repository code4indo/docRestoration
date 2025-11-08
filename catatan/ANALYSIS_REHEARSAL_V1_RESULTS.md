# ANALYSIS: Rehearsal V1 Training Results - Data Rehearsal Success but DIBCO Failure

**Date:** 2024-10-31 13:40  
**Experiment:** dibco_finetuning_with_anri_rehearsal_v1  
**Status:** ✅ PARTIAL SUCCESS - Catastrophic forgetting prevented, but DIBCO adaptation failed  
**Duration:** ~35 minutes (4 epochs, early stopped)

---

## EXECUTIVE SUMMARY

**DATA REHEARSAL WORKS! But DIBCO adaptation still fails.**

Training dengan rehearsal dataset (50% Base + 30% DIBCO + 20% ANRI) **BERHASIL mencegah catastrophic forgetting** pada ANRI domain (v4 gagal catastrophic di 29.65 dB, v1 maintained di 31-32 dB). 

TETAPI, DIBCO PSNR **TETAP GAGAL** mencapai target (stuck di 14-15 dB, target ≥28.5 dB), menunjukkan masalah fundamental pada domain adaptation ke DIBCO.

### Final Results (Early Stopped at Epoch 4, Best Epoch 1):

| Epoch | Base PSNR | ANRI PSNR | DIBCO PSNR | Status |
|-------|-----------|-----------|------------|--------|
| **1** | **30.13 dB** | **31.04 dB** | **14.63 dB** | ⚠️ ANRI warning |
| 2 | 30.26 dB | 31.18 dB | 14.90 dB | ⚠️ ANRI warning |
| 3 | 30.43 dB | 31.79 dB | 14.99 dB | ✅ All OK |
| 4 | 30.50 dB | 32.47 dB | 15.15 dB | ✅ All OK, Early Stop |

**Best Model:** Epoch 1 (Base 30.13 dB, ANRI 31.04 dB, DIBCO 14.63 dB)

---

## DETAILED ANALYSIS

### 1. Data Rehearsal SUCCESS ✅

**GOAL:** Prevent catastrophic forgetting on ANRI (v4 failed at 29.65 dB)

**RESULT:** ✅ **SUCCESS - Catastrophic forgetting prevented!**

| Metric | V4 (No Rehearsal) | V1 Rehearsal | Improvement |
|--------|-------------------|--------------|-------------|
| **ANRI Epoch 1** | **29.65 dB** 🚨 | **31.04 dB** ✅ | **+1.39 dB** |
| **ANRI Best** | 29.65 dB (emergency stop) | 32.47 dB (epoch 4) | **+2.82 dB** |
| **Emergency Stop** | YES (epoch 1) | NO | N/A |
| **Training Continued** | NO | YES (4 epochs) | N/A |

**Evidence:**
- V4: ANRI 29.65 dB < 30.0 dB red line → **EMERGENCY STOP**
- V1: ANRI 31.04-32.47 dB > 30.0 dB red line → **Training continued normally**
- Improvement trajectory: 31.04 → 31.18 → 31.79 → 32.47 dB (consistent upward trend)
- ANRI gap vs pretrained (33.02 dB): V4 -3.37 dB catastrophic, V1 -0.55 dB acceptable

**Conclusion:** 20% ANRI rehearsal samples **SUCCESSFULLY** prevented catastrophic forgetting!

### 2. DIBCO Adaptation FAILURE ❌

**GOAL:** Achieve DIBCO PSNR ≥28.5 dB (target 28-30 dB)

**RESULT:** ❌ **COMPLETE FAILURE - DIBCO stuck at ~15 dB**

| Epoch | DIBCO PSNR | Target | Gap | Progress |
|-------|------------|--------|-----|----------|
| 1 | 14.63 dB | ≥28.5 dB | **-13.87 dB** | Baseline |
| 2 | 14.90 dB | ≥28.5 dB | -13.60 dB | +0.27 dB |
| 3 | 14.99 dB | ≥28.5 dB | -13.51 dB | +0.09 dB |
| 4 | 15.15 dB | ≥28.5 dB | **-13.35 dB** | +0.16 dB |

**Total Improvement:** 14.63 → 15.15 dB = **+0.52 dB (marginal, insufficient)**

**Comparison:**
- V4 (0% ANRI): DIBCO 14.84 dB (1 epoch, emergency stopped)
- V1 (20% ANRI): DIBCO 15.15 dB (4 epochs, early stopped)
- **Difference:** Only +0.31 dB better than v4 (negligible)

**Why DIBCO Failed:**
1. **Domain gap too large:** DIBCO degradation patterns fundamentally different from Base/ANRI
2. **Insufficient DIBCO data:** Only 322 DIBCO samples (30%) vs 537 Base (50%)
3. **Model capacity saturated:** Pretrained from ANRI may not have capacity for DIBCO adaptation
4. **Wrong loss weights:** Current weights (Pixel 100, Adv 1.0, Percep 15) may not suit DIBCO
5. **Early stopping too aggressive:** Stopped at epoch 4 due to no improvement (patience 3)

### 3. Base Performance - Maintained ✅

**GOAL:** Maintain Base PSNR ~30.5 dB

**RESULT:** ✅ **SUCCESS - Base maintained**

| Epoch | Base PSNR | Target | Status |
|-------|-----------|--------|--------|
| 1 | 30.13 dB | ≥28.5 dB | ✅ |
| 2 | 30.26 dB | ≥28.5 dB | ✅ |
| 3 | 30.43 dB | ≥28.5 dB | ✅ |
| 4 | 30.50 dB | ≥28.5 dB | ✅ |

**Improvement trajectory:** 30.13 → 30.50 dB (+0.37 dB consistent improvement)

**Comparison vs pretrained:** 30.69 dB (pretrained) vs 30.50 dB (best) = -0.19 dB (acceptable minor drop)

### 4. Early Stopping Analysis

**Triggered:** Epoch 4, Patience 3/3 reached  
**Reason:** No improvement in combined score for 3 consecutive epochs  
**Best Model:** Epoch 1 (Base 30.13 dB, ANRI 31.04 dB, DIBCO 14.63 dB)

**Problem:** Combined score stuck at 0.00 across all epochs
- This indicates **CER weight 0.0** made PSNR-only metric insufficient
- Early stopping triggered too early (only 4 epochs vs 10 max)
- DIBCO was showing slight improvement (+0.52 dB trend) but not enough to move combined score

**Recommendation:** Adjust early stopping metric or increase patience for DIBCO domain

---

## COMPARISON: V4 vs V1 Rehearsal

| Aspect | V4 (No Rehearsal) | V1 Rehearsal | Winner |
|--------|-------------------|--------------|--------|
| **Dataset** | 0% ANRI (catastrophic) | 20% ANRI (rehearsal) | ✅ V1 |
| **Base PSNR** | 30.31 dB (epoch 1) | 30.13-30.50 dB (4 epochs) | ≈ Tie |
| **ANRI PSNR** | 29.65 dB 🚨 | 31.04-32.47 dB ✅ | ✅ V1 (+2.82 dB) |
| **DIBCO PSNR** | 14.84 dB | 14.63-15.15 dB | ≈ Tie (both failed) |
| **Emergency Stop** | YES (epoch 1) | NO | ✅ V1 |
| **Training Epochs** | 1 (forced stop) | 4 (early stop) | ✅ V1 |
| **Catastrophic Forgetting** | YES | NO | ✅ V1 |
| **DIBCO Target Met** | NO | NO | ❌ Both failed |

**Key Takeaway:** 
- **Data rehearsal WORKS** for preventing catastrophic forgetting (+2.82 dB ANRI improvement)
- **But DIBCO adaptation STILL FAILS** regardless of rehearsal strategy

---

## ROOT CAUSE: Why DIBCO Fails

### Hypothesis 1: Domain Gap Too Large (MOST LIKELY)

**Evidence:**
- DIBCO degradation: Complex real-world degradation (ink bleed, paper aging, noise)
- Base/ANRI: Synthetic or controlled degradation patterns
- DIBCO PSNR stuck at 14-15 dB across V4 and V1 (both failed similarly)
- Model trained on clean → degraded (Base/ANRI) can't reverse complex real degradation

**Implication:** Progressive finetuning Base → ANRI → DIBCO may be wrong path. DIBCO needs different approach.

### Hypothesis 2: Insufficient DIBCO Training Data

**Evidence:**
- Only 322 DIBCO samples (30%) in training
- Only 68 DIBCO validation samples
- DIBCO dataset much smaller than Base (4739) or ANRI (359)

**Implication:** Need more DIBCO data or higher DIBCO ratio (e.g., 50% DIBCO + 30% Base + 20% ANRI)

### Hypothesis 3: Wrong Loss Function for DIBCO

**Current weights:**
- Pixel: 100.0 (L1 reconstruction)
- Adversarial: 1.0 (realism)
- Perceptual: 15.0 (VGG features)
- RecFeat: 10.0 (recognition features)

**Problem:** These weights tuned for Base/ANRI clean→degraded generation, NOT DIBCO degraded→clean restoration

**Implication:** DIBCO may need:
- Higher perceptual loss (focus on visual quality)
- Different pixel loss (SSIM instead of L1?)
- DIBCO-specific loss components (edge preservation, contrast enhancement)

### Hypothesis 4: Model Architecture Not Suitable for DIBCO

**Evidence:**
- Generator trained for clean→degraded generation (Base/ANRI forward process)
- DIBCO needs degraded→clean restoration (reverse process)
- Architecture may be fundamentally mismatched

**Implication:** May need separate DIBCO restoration model OR bidirectional architecture

---

## RECOMMENDATIONS

### IMMEDIATE ACTIONS

**1. ABANDON Progressive Finetuning for DIBCO**
- Data rehearsal works for catastrophic forgetting prevention ✅
- But DIBCO adaptation fails regardless (14-15 dB stuck)
- Progressive path Base → ANRI → DIBCO = **WRONG APPROACH FOR DIBCO**

**2. CHOOSE NEW DIBCO STRATEGY:**

#### **Option A: DIBCO-Only Model from Scratch (RECOMMENDED)**

Train dedicated DIBCO restoration model without Base/ANRI baggage:

```json
{
  "experiment_name": "dibco_direct_restoration_v1",
  "tfrecord_path": "dual_modal_gan/data/dibco_tiled_full.tfrecord",
  "pretrained_checkpoint": null,  // Start from scratch OR base model only
  "loss_weights": {
    "pixel_loss_weight": 200.0,  // Higher pixel fidelity
    "perceptual_loss_weight": 50.0,  // Much higher perceptual
    "adv_loss_weight": 2.0,  // Higher adversarial for realism
    "edge_loss_weight": 30.0  // NEW: Edge preservation for DIBCO
  },
  "discriminator_mode": "clean",  // Discriminate CLEAN images (restoration task)
  "epochs": 50,  // Much longer training for complex restoration
  "batch_size": 8,  // Larger batch for stability
  "lr_g": 1e-4,  // Higher LR for faster adaptation
  "lr_d": 2e-4
}
```

**Pros:**
- No catastrophic forgetting risk (single domain)
- Full model capacity dedicated to DIBCO
- Can tune everything for DIBCO restoration
- Simpler, cleaner approach

**Cons:**
- Lose Base/ANRI knowledge (may be irrelevant for DIBCO anyway)
- Need to verify DIBCO-only can reach target

#### **Option B: DIBCO Data Augmentation + Rebalanced Ratio**

Keep rehearsal but heavily favor DIBCO:

```json
{
  "experiment_name": "dibco_rehearsal_v2_dibco_heavy",
  "tfrecord_path": "mixed_20base_60dibco_20anri_rehearsal.tfrecord",
  "dataset_composition": {
    "base": "20% (maintain core knowledge)",
    "dibco": "60% (PRIMARY FOCUS)",
    "anri": "20% (prevent forgetting)"
  },
  "epochs": 20,
  "early_stopping_patience": 10  // Much more patience for DIBCO
}
```

**Pros:**
- Maintains Base/ANRI knowledge
- Focus more capacity on DIBCO (60% vs 30%)
- Data rehearsal still works

**Cons:**
- May still fail if domain gap too large
- More complex dataset creation
- DIBCO data may still be insufficient

#### **Option C: Separate Generator for DIBCO (Novel Architecture)**

Use different generator architecture optimized for restoration:

- **Base/ANRI Generator:** Current enhanced (clean→degraded generation)
- **DIBCO Generator:** U-Net or ResNet (degraded→clean restoration)
- **Shared Discriminator:** Dual-modal enhanced_v2_fixed

**Pros:**
- Architecture matched to task (restoration vs generation)
- Can use proven restoration architectures (U-Net, ResUNet)
- Base/ANRI knowledge preserved in separate generator

**Cons:**
- Complex implementation
- Need to manage 2 generators
- Higher compute and memory cost

### RECOMMENDATION PRIORITY:

**1st Choice: Option A (DIBCO-Only Model)** ⭐⭐⭐
- Simplest, most pragmatic approach
- No catastrophic forgetting complexity
- Full focus on DIBCO target
- **Can validate if DIBCO target (28-30 dB) is achievable at all**

**2nd Choice: Option B (DIBCO-Heavy Rehearsal)**
- If Option A succeeds but you still want Base/ANRI knowledge
- Rebalance to 60% DIBCO primary focus

**3rd Choice: Option C (Dual-Generator Architecture)**
- Research experiment only
- High complexity, uncertain payoff
- Only if Options A & B fail

---

## NEXT STEPS

### IF CHOOSING OPTION A (DIBCO-Only Model):

**Step 1:** Verify DIBCO dataset quality
```bash
# Check DIBCO dataset statistics
python scripts/analyze_dibco_dataset.py \
  --tfrecord dual_modal_gan/data/dibco_tiled_full.tfrecord
```

**Step 2:** Create DIBCO-only config
```json
{
  "experiment_name": "dibco_direct_restoration_v1",
  "description": "Direct DIBCO restoration without progressive finetuning. Clean slate approach to validate if DIBCO target (28-30 dB) is achievable.",
  "tfrecord_path": "dual_modal_gan/data/dibco_tiled_full.tfrecord",
  "pretrained_checkpoint": null,  // OR base model if helpful
  "epochs": 50,
  "lr_g": 1e-4,
  "lr_d": 2e-4,
  "pixel_loss_weight": 200.0,
  "perceptual_loss_weight": 50.0,
  "adv_loss_weight": 2.0,
  "discriminator_mode": "clean",
  "dual_validation": {
    "enabled": true,
    "monitor_dibco_psnr": true,
    "dibco_psnr_red_line": 20.0,  // Lower red line for early experiments
    "dibco_psnr_warning_threshold": 25.0
  }
}
```

**Step 3:** Launch DIBCO-only training
```bash
nohup ./scripts/universal_train_from_json.sh configs/dibco_direct_restoration_v1.json > /dev/null 2>&1 &
```

**Expected Outcome:**
- If DIBCO reaches 28-30 dB: **SUCCESS** - DIBCO target achievable, progressive approach was wrong
- If DIBCO stuck at 14-15 dB: **DATASET PROBLEM** - DIBCO data quality or architecture issue
- If DIBCO reaches 20-25 dB: **MARGINAL** - Target may be too ambitious, need better approach

---

## LESSONS LEARNED

### 1. Data Rehearsal WORKS for Catastrophic Forgetting ✅

**Evidence:** V1 ANRI 31-32 dB vs V4 catastrophic 29.65 dB (+2.82 dB improvement)

**Lesson:** For multi-domain continual learning, 20% rehearsal samples sufficient to maintain performance

**Application:** Use rehearsal strategy for ANY multi-domain progressive training

### 2. Domain Gap Analysis is CRITICAL ⚠️

**Evidence:** Base (30 dB) and ANRI (32 dB) maintained, but DIBCO stuck at 15 dB regardless of strategy

**Lesson:** Not all domains are compatible for progressive finetuning. DIBCO degradation fundamentally different.

**Application:** Analyze domain similarity BEFORE attempting progressive finetuning

### 3. Early Stopping Metric Must Match Objective ⚠️

**Evidence:** Combined score stuck at 0.00 (CER weight 0), early stop at epoch 4 despite DIBCO showing improvement trend

**Lesson:** When primary objective is PSNR improvement (not CER), use PSNR-based early stopping metric

**Application:** Set `early_stopping_metric: "psnr"` for restoration tasks, reserve combined for HTR tasks

### 4. Progressive Finetuning Has Limits ❌

**Evidence:** Base → ANRI successful (synthetic → real handwriting), but ANRI → DIBCO failed (handwriting → benchmark degradation)

**Lesson:** Progressive path only works when domains share common features. DIBCO restoration is different task than generation.

**Application:** Don't force progressive finetuning when task fundamentally changes (generation vs restoration)

### 5. Validation of Assumptions is Essential ��

**Evidence:** Assumed DIBCO would improve via transfer learning from ANRI. Reality: stuck at 15 dB regardless.

**Lesson:** Test baseline (DIBCO-only) BEFORE attempting complex strategies (progressive + rehearsal)

**Application:** Always validate simplest approach first, then add complexity only if needed

---

## CONCLUSION

**Rehearsal V1 Training: PARTIAL SUCCESS**

### What Worked ✅
1. **Data rehearsal** successfully prevented catastrophic forgetting (+2.82 dB ANRI improvement)
2. **Triple validation** provided clear visibility into all 3 domains
3. **Safety mechanisms** worked (no emergency stop, training continued normally)
4. **Base performance** maintained (30.13-30.50 dB)
5. **ANRI performance** preserved and improved (31.04-32.47 dB)

### What Failed ❌
1. **DIBCO adaptation** completely failed (14.63-15.15 dB, target ≥28.5 dB)
2. **Early stopping** triggered too early (epoch 4 vs 10 max)
3. **Combined score metric** ineffective (stuck at 0.00 due to CER weight 0)
4. **Transfer learning assumption** wrong (ANRI knowledge doesn't help DIBCO)
5. **Progressive finetuning strategy** fundamentally flawed for DIBCO restoration

### Final Verdict

**Data Rehearsal:** ✅ **PROVEN EFFECTIVE** for catastrophic forgetting prevention  
**Progressive Finetuning for DIBCO:** ❌ **ABANDONED** - wrong approach

**Recommendation:** Implement **Option A (DIBCO-Only Model)** to validate if DIBCO target is achievable at all, then iterate based on results.

---

**Status:** Analysis complete  
**Next Action:** Create DIBCO-only training config and launch baseline experiment  
**Priority:** HIGH - Validate DIBCO baseline before further complex strategies  
**ETA:** ~2-3 hours for DIBCO-only training (50 epochs × 3-4 min)
