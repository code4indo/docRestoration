# ANRI Fine-tuning Quick Start Guide

## 🎯 Objective
Transfer learning dari DIBCO best model (PSNR 21.75 dB) ke ANRI real paleographic dataset (16-18th century handwritten documents).

## 📊 Dataset Overview
- **Source**: DIBCO 2009-2018 (256 synthetic samples, PSNR 21.75 dB)
- **Target**: ANRI Real (~1200 real paleographic images)
  - Train: 113 MB (~800-1000 images)
  - Val: 24 MB (~150-200 images)  
  - Test: 25 MB (~150-200 images)

## 🚀 Execution Commands

### 1. Launch Training
```bash
# Background training dengan nohup
nohup ./scripts/universal_train_from_json.sh configs/anri_finetuning_from_dibco_best.json > /dev/null 2>&1 &

# Get log file name (after launch)
LOG_FILE=$(find logbook -name "anri_finetuning_from_dibco_best_*.log" -type f | head -1)

# Monitor live training
tail -f $LOG_FILE
```

### 2. Monitor Progress (During Training)
```bash
# Check current epoch status
grep "Epoch.*completed" $LOG_FILE | tail -5

# Check validation PSNR trend
grep "PSNR:" $LOG_FILE | grep -oP "PSNR: \K[0-9.]+" | tail -10

# Check early stopping patience
grep "Patience Counter:" $LOG_FILE | tail -5

# Check for gradient issues
grep -E "(NaN|inf|gradient)" $LOG_FILE | tail -10
```

### 3. Real-time Performance Analysis
```bash
# Get latest validation metrics
tail -100 $LOG_FILE | grep -A 5 "Validation Statistics"

# Check Generator Loss trend (should be stable, no huge spikes)
grep -oP "G=\K[0-9.]+" $LOG_FILE | tail -20

# Check best epoch info
cat dual_modal_gan/checkpoints/anri_finetuning_from_dibco_best/epoch_info.json
```

## 📈 Expected Training Timeline

| Epoch | Phase | Expected PSNR | What to Watch |
|-------|-------|---------------|---------------|
| 1-5 | Warmup | 15-17 dB | No NaN, loss decreasing |
| 6-20 | Annealing | 17-18 dB | Gradual improvement |
| 21-30 | Full Training | 18-19 dB | Steady convergence |
| 31-40 | Fine-tuning | 19-20 dB | Best model likely here |
| 41-50 | Polish | 20-21 dB | May trigger early stop |

**Estimated Total Time**: 3-4 hours (~2-3 min/epoch × 40 epochs expected)

## 🎯 Success Criteria

### ✅ Minimum (Acceptable)
- PSNR ≥ 18.0 dB
- No catastrophic forgetting
- Text remains legible

### 🎯 Target (Good)
- PSNR ≥ 20.0 dB
- Preserved paleographic features
- CER improvement ≥ 10%

### 🌟 Ambitious (Excellent)
- PSNR ≥ 21.0 dB (match DIBCO!)
- Minimal artifacts
- CER improvement ≥ 15%

## ⚠️ Critical Monitoring Points

### Epoch 5 (Warmup Complete)
```bash
# Check stability
tail -50 $LOG_FILE | grep "Epoch 5"

# Expected: PSNR 15-17 dB, no NaN, G_loss < 500
# ❌ If PSNR < 14: LR too high or domain gap too large
# ❌ If NaN detected: Gradient explosion, stop training
```

### Epoch 10 (First Checkpoint)
```bash
# Check domain adaptation
grep "Epoch 10" $LOG_FILE -A 10

# Expected: PSNR 16-18 dB, improving trend
# ❌ If PSNR < 15: Poor domain transfer → reduce LR
# ✅ If PSNR > 17: Excellent adaptation!
```

### Epoch 20 (Annealing Complete)
```bash
# Check convergence
grep "Epoch 20" $LOG_FILE -A 10

# Expected: PSNR 18-19 dB, stable gradients
# ❌ If PSNR < 17: Consider freezing encoder
# ✅ If PSNR > 18.5: On track for target!
```

### Epoch 30-40 (Convergence Zone)
```bash
# Check plateau
grep "Patience Counter" $LOG_FILE | tail -10

# Expected: Early stopping likely around epoch 35-45
# ✅ If patience < 10: Still improving
# ⚠️ If patience > 12: Likely will stop soon
```

## 🛑 Intervention Scenarios

### Scenario 1: Low PSNR at Epoch 10 (< 15 dB)
**Problem**: Poor domain adaptation, LR too high, or gradient instability

**Solution**:
```bash
# Stop training
pkill -f "anri_finetuning_from_dibco_best"

# Create tuned config with lower LR
cp configs/anri_finetuning_from_dibco_best.json configs/anri_finetuning_tuned_v2.json

# Edit: lr_g: 0.000003 (was 0.000005), gradient_clip_norm: 10.0 (was 5.0)

# Restart from best checkpoint
nohup ./scripts/universal_train_from_json.sh configs/anri_finetuning_tuned_v2.json > /dev/null 2>&1 &
```

### Scenario 2: Gradient Spikes (G_loss > 2000)
**Problem**: Perceptual loss exploding on unfamiliar domain

**Solution**:
```bash
# Reduce perceptual/RecFeat weights further
# Edit config: perceptual_loss_weight: 3.0, rec_feat_loss_weight: 3.0
# Increase gradient clipping: gradient_clip_norm: 10.0
```

### Scenario 3: Overfitting (Val PSNR decreasing)
**Problem**: Model memorizing training data

**Solution**:
```bash
# Increase dropout in discriminator_config
# Edit config: dropout_rate: 0.5 (was 0.4)
# Reduce learning rate: lr_g: 0.000003
```

### Scenario 4: Catastrophic Forgetting (PSNR < 14 dB)
**Problem**: Model lost DIBCO knowledge

**Solution**:
```bash
# Restore DIBCO checkpoint and use even lower LR
# Edit config: lr_g: 0.000002, freeze_strategy.enabled: true
# Freeze encoder, only train decoder for domain adaptation
```

## 📊 Post-Training Evaluation

### Analyze Best Model
```bash
# Get final metrics
cat dual_modal_gan/checkpoints/anri_finetuning_from_dibco_best/epoch_info.json

# Expected output:
# {
#   "best_epoch": 35-45,
#   "best_psnr": 19-21,
#   "patience_counter": 15 (early stop triggered)
# }
```

### Test Set Evaluation (Held-out Data)
```bash
# Run inference on test set
poetry run python scripts/evaluate_on_test_set.py \
  --checkpoint dual_modal_gan/checkpoints/anri_finetuning_from_dibco_best/best_model \
  --test_data dual_modal_gan/data/finetuning/finetuning_test.tfrecord \
  --output results/anri_test_evaluation.json

# Compare with DIBCO test results
poetry run python scripts/compare_domain_transfer.py \
  --dibco_results results/dibco_test_results.json \
  --anri_results results/anri_test_evaluation.json
```

### Visual Quality Inspection
```bash
# Generate sample outputs
ls dual_modal_gan/outputs/samples_anri_finetuning_from_dibco_best/

# Open samples to check:
# 1. Text legibility (can you read paleographic script?)
# 2. Artifact detection (over-smoothing, blurriness?)
# 3. Feature preservation (stroke thickness, ink texture?)
```

## 🔍 Key Differences from DIBCO Training

| Parameter | DIBCO | ANRI | Rationale |
|-----------|-------|------|-----------|
| Learning Rate (G) | 0.00001 | 0.000005 | 50% reduction for smaller dataset |
| Dropout | 0.3 | 0.4 | Stronger regularization |
| Patience | 10 | 15 | Allow more exploration |
| Perceptual Weight | 10.0 | 5.0 | Gradient stabilization |
| RecFeat Weight | 10.0 | 5.0 | Gradient stabilization |
| Gradient Clip | 1.0 | 5.0 | Prevent explosion on new domain |
| Annealing Epochs | 15 | 20 | Longer warmup for domain shift |

## 🎓 What We're Learning

1. **Domain Transfer**: Synthetic DIBCO → Real Paleographic
2. **Complexity Increase**: Modern text → 16-18th century handwriting
3. **Degradation Types**: Synthetic → Real aging, ink bleeding, physical damage
4. **Generalization**: Can model trained on synthetic data handle real world?

## 📚 Next Steps After Training

1. **If PSNR ≥ 20 dB**: 
   - Deploy for production use
   - Integrate with HTR pipeline
   - Publish results (Q2 journal target)

2. **If PSNR 18-20 dB**:
   - Acceptable for research
   - Consider ensemble methods
   - Collect more ANRI data

3. **If PSNR < 18 dB**:
   - Analyze failure cases
   - Try progressive fine-tuning (freeze encoder first)
   - Consider domain-specific augmentation

## 📞 Emergency Contacts

- **Training stuck (no progress)**: Check GPU usage `nvidia-smi`
- **NaN loss**: Stop immediately, reduce LR by 50%
- **Out of memory**: Reduce batch_size to 1
- **Checkpoint not loading**: Verify path to DIBCO best model

---

**Remember**: Real paleographic data is MUCH harder than synthetic DIBCO. 
**Target PSNR 18-21 dB is already excellent** for this task!
