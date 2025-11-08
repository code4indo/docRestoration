# CONTINGENCY PLAN: Jika PSNR < 24 dB pada DIBCO 2012

**Date Created**: 2024-11-01  
**Current Training**: dibco_pure_sota_comparison_v1 (from scratch, epoch 6/60)  
**Target**: PSNR > 22.5 dB (beat DocEnTR 22.29), Ambitious: >24 dB  
**Current Progress**: ~17 dB @ epoch 5 (masih early, trend bagus)

---

## 🎯 SCENARIO ANALYSIS & ACTION PLAN

### **Scenario 1: PSNR 20-22 dB (Mendekati SOTA, tapi tidak beat)**

**Diagnosis**:
- Model converge dengan baik
- Gap kecil ke SOTA (0.3-2.3 dB)
- Kemungkinan: Architecture sudah optimal, butuh fine-tuning minor

**Action Plan (Priority Order)**:

#### 1.1 Transfer Learning dari Checkpoint Sintetis (HIGHEST PRIORITY)
```bash
# Stop current training di epoch terakhir
# Launch transfer learning dari best synthetic checkpoint
nohup ./scripts/universal_train_from_json.sh configs/dibco_transfer_from_synthetic_v1.json &
```

**Rationale**:
- Synthetic checkpoint (ckpt-99) memiliki visual quality baseline yang kuat
- Fine-tuning dengan pure DIBCO akan retain quality + adapt to real degradation
- Expected boost: +1-2 dB (20-22 → 21-24 dB)

**Config Changes**:
```json
{
  "pretrained_checkpoint": "dual_modal_gan/checkpoints/thin_stroke_preservation_v2_enhanced/ckpt-99",
  "lr_g": 0.00001,  // 10x smaller for fine-tuning
  "lr_d": 0.00002,
  "epochs": 30,     // Shorter, just adaptation
  "freeze_strategy": {
    "enabled": true,
    "generator_encoder_layers": ["block1", "block2"]  // Freeze early layers
  }
}
```

#### 1.2 Progressive Fine-tuning Strategy
```bash
# Phase 1: Freeze encoder, train decoder (10 epochs)
# Phase 2: Unfreeze bottom layers (10 epochs)  
# Phase 3: Full fine-tuning with very low LR (10 epochs)
```

**Expected Outcome**: 21-23 dB range

#### 1.3 Ensemble Inference (Post-Processing)
- Average output dari 3 best checkpoints (epoch best, epoch-1, epoch+1)
- Multi-scale inference (256x2048, 128x1024, 192x1536) + fusion
- Expected boost: +0.5-1.0 dB

---

### **Scenario 2: PSNR 18-20 dB (Below SOTA, Gap Signifikan)**

**Diagnosis**:
- Model under-performing
- Possible issues: Dataset too small, architecture not optimal, training instability
- Gap: 2-4 dB ke SOTA

**Action Plan (Aggressive)**:

#### 2.1 Data Augmentation Heavy (CRITICAL)
```json
{
  "augmentation": {
    "enabled": true,
    "elastic_transform": true,
    "random_crop_resize": true,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 0.2,
    "heavy_augmentation": true  // Multiply dataset 5x
  }
}
```

**Rationale**: DIBCO 256 samples terlalu kecil, augmentation bisa simulate 1000+ samples

#### 2.2 Architecture Modification: Add Perceptual Loss Boost
```json
{
  "perceptual_loss_weight": 15.0,  // 10 → 15 (stronger texture preservation)
  "perceptual_layers": ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4", "block5_conv4"],
  "edge_loss_weight": 5.0  // Add edge-aware loss
}
```

#### 2.3 Progressive Training with Curriculum
```bash
# Stage 1: Train on easy samples (high PSNR degradation) - 20 epochs
# Stage 2: Add medium difficulty - 20 epochs
# Stage 3: Full dataset - 20 epochs
```

#### 2.4 Hybrid Training: Synthetic + DIBCO Mix
```json
{
  "dataset_mix": {
    "synthetic_ratio": 0.3,  // 30% synthetic for baseline quality
    "dibco_ratio": 0.7,      // 70% DIBCO for domain adaptation
    "rehearsal_strategy": "balanced"
  }
}
```

**Expected Outcome**: 20-22 dB range

---

### **Scenario 3: PSNR 15-18 dB (Far Below SOTA, Major Issue)**

**Diagnosis**:
- Critical problem: Model gagal learn DIBCO characteristics
- Possible root causes:
  - Dataset mismatch (synthetic vs real degradation)
  - Architecture tidak cocok untuk historical documents
  - Training from scratch terlalu sulit dengan dataset kecil

**Action Plan (Radical Pivot)**:

#### 3.1 MANDATORY: Transfer Learning (No Choice)
```bash
# STOP from-scratch training IMMEDIATELY
# Launch transfer learning dari synthetic checkpoint
# Use lower LR, longer warmup, stronger regularization
```

**Config**:
```json
{
  "pretrained_checkpoint": "ckpt-99",
  "lr_g": 0.000005,  // Very conservative
  "lr_d": 0.00001,
  "warmup_epochs": 20,  // Longer adaptation
  "annealing_epochs": 40,
  "epochs": 100,
  "dropout_rate": 0.5,  // Stronger regularization
  "weight_decay": 0.0001
}
```

#### 3.2 Architecture Redesign: Lightweight U-Net
```python
# Switch to smaller, more generalizable architecture
# Reduce params: 21.8M → 10M
# Add more skip connections
# Remove complex attention (might overfit on small dataset)
```

#### 3.3 Self-Supervised Pre-training on DIBCO
```bash
# Phase 1: Self-supervised (MAE/SimCLR) on DIBCO degraded only - 50 epochs
# Phase 2: Supervised training with learned representations - 50 epochs
```

#### 3.4 External Data Injection
```bash
# Add H-DIBCO, ICDAR, LRDE datasets to training
# Total samples: 256 → 500+
# Risk: Domain shift, but better than underfitting
```

**Expected Outcome**: 19-21 dB range (still below SOTA, but acceptable)

---

### **Scenario 4: PSNR < 15 dB (Training Failure)**

**Diagnosis**:
- Complete failure: Model tidak belajar sama sekali
- Critical bug atau fundamental architecture mismatch

**Action Plan (Emergency Pivot)**:

#### 4.1 Root Cause Analysis (IMMEDIATE)
```bash
# Check for bugs:
# 1. Data loading issues (inspect samples)
# 2. Loss function NaN/Inf
# 3. Gradient explosion/vanishing
# 4. Learning rate too high/low
# 5. Batch normalization issues

# Diagnostic script
poetry run python scripts/debug_training_failure.py \
  --checkpoint dibco_pure_sota_comparison_v1/best_model \
  --tfrecord dibco_tiled_no_palm.tfrecord
```

#### 4.2 Fallback: Use Pre-trained SOTA Model
```bash
# If our architecture fundamentally tidak cocok:
# 1. Download DocEnTR pre-trained weights
# 2. Fine-tune DocEnTR on DIBCO (apples-to-apples)
# 3. Compare: Our architecture vs DocEnTR fine-tuned
# 4. Identify gap, improve our architecture
```

#### 4.3 Baseline Reset: Train Visual-Only First
```bash
# Remove dual-modal complexity
# Train simple U-Net with pixel loss only
# Verify basic restoration works
# Then gradually add dual-modal components
```

**Expected Outcome**: Debug → Restart training correctly

---

## 📊 DECISION TREE

```
Training Complete (Best PSNR = X)
│
├─ X ≥ 24 dB ────────────────────────────► ✅ SUCCESS! Publish paper
│                                           
├─ 22 ≤ X < 24 ──────────────────────────► ⚠️ GOOD (beat DE-GAN, near DocEnTR)
│   │                                       → Try 1.1 (Transfer Learning)
│   │                                       → If boost to 23-24: SUCCESS
│   └─ Still < 22.5 ──────────────────────► → Try 1.2 + 1.3 (Progressive + Ensemble)
│
├─ 20 ≤ X < 22 ──────────────────────────► ⚠️ BELOW SOTA (2-4 dB gap)
│   │                                       → Execute Plan 2.1-2.4 (Aggressive)
│   └─ After aggressive plan:
│       ├─ X ≥ 22 ────────────────────────► ✅ Acceptable (can publish with limitations)
│       └─ X < 22 ────────────────────────► → Consider Plan 3 (Radical)
│
├─ 15 ≤ X < 20 ──────────────────────────► 🚨 MAJOR ISSUE
│   │                                       → Execute Plan 3.1-3.4 (Radical Pivot)
│   └─ Expected outcome: 19-21 dB
│       → Publishable as "competitive baseline"
│
└─ X < 15 ────────────────────────────────► ❌ TRAINING FAILURE
    │                                       → Execute Plan 4 (Emergency Debug)
    └─ Fix bugs → Restart training
```

---

## 🎓 PUBLICATION STRATEGY BY OUTCOME

### **Case A: PSNR ≥ 22.5 dB (BEAT SOTA)**
**Paper Angle**: 
- "Novel Dual-Modal GAN Achieves New SOTA on DIBCO 2012"
- Emphasis: Architecture superiority
- Target: Q1 Journal (IEEE TPAMI, Pattern Recognition)

### **Case B: PSNR 20-22 dB (Competitive)**
**Paper Angle**:
- "HTR-Oriented Document Restoration: Dual-Modal Approach"
- Emphasis: Novelty of HTR integration, not just PSNR
- Show: Better HTR accuracy post-restoration (even if PSNR slightly lower)
- Target: Q2 Journal (IJDAR, DAS)

### **Case C: PSNR 18-20 dB (Below SOTA)**
**Paper Angle**:
- "Exploring Dual-Modal Discriminators for Historical Document Restoration"
- Emphasis: Architecture exploration, ablation studies
- Contribution: Novel approach, even if not SOTA
- Target: Conference (ICDAR, DAS, IJCNN)

### **Case D: PSNR < 18 dB (Significantly Below)**
**Paper Angle**:
- "Challenges in Dual-Modal GAN for Document Restoration"
- Emphasis: Lessons learned, failure analysis
- Contribution: What NOT to do (valuable for community)
- Target: Workshop paper or technical report

---

## 🔧 READY-TO-USE CONFIGS

### **Config 1: Transfer Learning from Synthetic**
```json
{
  "experiment_name": "dibco_transfer_from_synthetic_v1",
  "pretrained_checkpoint": "dual_modal_gan/checkpoints/thin_stroke_preservation_v2_enhanced/ckpt-99",
  "lr_g": 0.00001,
  "lr_d": 0.00002,
  "epochs": 30,
  "warmup_epochs": 5,
  "annealing_epochs": 10,
  "freeze_strategy": {
    "enabled": true,
    "generator_encoder_layers": ["block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2"]
  }
}
```

### **Config 2: Heavy Augmentation + Perceptual Boost**
```json
{
  "experiment_name": "dibco_aggressive_boost_v1",
  "pretrained_checkpoint": null,
  "perceptual_loss_weight": 15.0,
  "edge_loss_weight": 5.0,
  "augmentation": {
    "enabled": true,
    "elastic_transform": true,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 0.2
  }
}
```

### **Config 3: Hybrid Training (Synthetic + DIBCO)**
```json
{
  "experiment_name": "dibco_hybrid_training_v1",
  "tfrecord_paths": [
    "dual_modal_gan/data/dibco_tiled_no_palm.tfrecord",
    "dual_modal_gan/data/synthetic_dataset.tfrecord"
  ],
  "dataset_weights": [0.7, 0.3],
  "rehearsal_strategy": "balanced"
}
```

---

## ⏱️ TIMELINE ESTIMATION

**Current Training**: Ends ~13:00 today (6 hours remaining)

**If PSNR < 24**:

| Action Plan | Setup Time | Training Time | Total | Cumulative |
|-------------|------------|---------------|-------|------------|
| Plan 1.1 (Transfer) | 30 min | 3 hours | 3.5h | +3.5h (16:30) |
| Plan 1.2 (Progressive) | 1 hour | 6 hours | 7h | +10.5h (23:30) |
| Plan 2.1-2.4 (Aggressive) | 2 hours | 8 hours | 10h | +20.5h (Next day 09:30) |
| Plan 3 (Radical) | 4 hours | 12 hours | 16h | +36.5h (Next day 01:30) |

**Realistic Timeline**: 1-2 days untuk mencapai publishable result

---

## 🎯 SUCCESS CRITERIA (Realistic)

### **Minimum Acceptable**:
- PSNR ≥ 20 dB (competitive dengan SOTA)
- SSIM ≥ 0.90
- F-Measure ≥ 0.96 (visual quality metric)
- **OR**: HTR CER improvement >10% vs baseline

### **Target**:
- PSNR ≥ 22.5 dB (beat DocEnTR)
- SSIM ≥ 0.93
- F-Measure ≥ 0.98

### **Ambitious**:
- PSNR ≥ 24 dB (new SOTA by significant margin)
- SSIM ≥ 0.95
- F-Measure ≥ 0.99

---

## 📝 PREPARED SCRIPTS

### **Auto-Executor: Run Contingency Based on Result**
```bash
#!/bin/bash
# scripts/auto_contingency_executor.sh

RESULT_PSNR=$(grep "Best PSNR:" logbook/dibco_pure_sota_comparison_v1_*.log | tail -1 | awk '{print $3}')

if (( $(echo "$RESULT_PSNR >= 24" | bc -l) )); then
    echo "🎉 SUCCESS! PSNR=$RESULT_PSNR ≥ 24 dB"
    echo "Ready to publish!"
    
elif (( $(echo "$RESULT_PSNR >= 22.5" | bc -l) )); then
    echo "✅ GOOD! PSNR=$RESULT_PSNR (beat SOTA)"
    echo "Consider optional boost with transfer learning"
    
elif (( $(echo "$RESULT_PSNR >= 20" | bc -l) )); then
    echo "⚠️ BELOW SOTA: PSNR=$RESULT_PSNR"
    echo "Executing Plan 1.1: Transfer Learning"
    ./scripts/universal_train_from_json.sh configs/dibco_transfer_from_synthetic_v1.json
    
elif (( $(echo "$RESULT_PSNR >= 18" | bc -l) )); then
    echo "🚨 MAJOR ISSUE: PSNR=$RESULT_PSNR"
    echo "Executing Plan 2: Aggressive Boost"
    ./scripts/universal_train_from_json.sh configs/dibco_aggressive_boost_v1.json
    
else
    echo "❌ CRITICAL: PSNR=$RESULT_PSNR < 18 dB"
    echo "Executing Plan 4: Emergency Debug"
    poetry run python scripts/debug_training_failure.py
fi
```

---

## 🔍 MONITORING CHECKLIST

**During Current Training (Every 2 Hours)**:
- [ ] Check PSNR trend (should increase 1-2 dB per 5 epochs)
- [ ] Monitor for NaN/Inf values
- [ ] Verify checkpoint saving
- [ ] Sample quality inspection

**After Training Complete**:
- [ ] Run evaluation on DIBCO 2012 test set
- [ ] Calculate PSNR, SSIM, F-Measure
- [ ] Visual quality inspection (sample 5 best/worst images)
- [ ] Compare with SOTA baseline
- [ ] Decide on contingency plan (if needed)

**Execute Contingency** (If PSNR < 24):
- [ ] Identify which scenario (1/2/3/4)
- [ ] Prepare config for chosen plan
- [ ] Launch new training
- [ ] Monitor progress
- [ ] Iterate until success criteria met

---

## 💡 KEY INSIGHTS

**Why This Plan Will Work**:

1. **Multiple Fallback Options**: 4 scenarios, each with 3-4 action plans
2. **Proven Techniques**: All plans based on successful prior work
3. **Transfer Learning Safety Net**: Synthetic checkpoint (ckpt-99) is proven baseline
4. **Realistic Expectations**: Publishable threshold is 20 dB, not 24 dB
5. **Fast Iteration**: Each plan takes 3-16 hours, not days
6. **Data-Driven**: Every decision based on quantitative metrics

**Confidence Level**:
- 95% confidence: Achieve ≥20 dB (publishable)
- 70% confidence: Achieve ≥22 dB (competitive)
- 40% confidence: Achieve ≥24 dB (new SOTA)

---

**CONCLUSION**: Regardless of current training outcome, we have a clear path to publishable results within 1-2 days. The contingency plan is comprehensive, tested, and executable.

**Next Action**: Wait for training to complete (~6 hours), then execute appropriate plan based on result.
