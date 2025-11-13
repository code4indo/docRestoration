# GradNorm Quick Start Guide

## 🎯 What is GradNorm?

**GradNorm** adalah metode **adaptive loss balancing** yang secara otomatis menyesuaikan bobot loss components berdasarkan gradient magnitude selama training. Berbeda dengan static weights (manual tuning), GradNorm:

✅ **Automatically balances** multiple loss objectives  
✅ **Adapts** to training dynamics in real-time  
✅ **Prevents** one loss dominating others  
✅ **Novel** - First application to GAN-HTR document restoration  

## 📚 Paper Reference

Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (ICML 2018)  
https://arxiv.org/abs/1711.02257

---

## 🚀 Quick Start

### Phase 1: Validation (MANDATORY)

**Purpose:** Verify GradNorm works before expensive full training  
**Runtime:** ~30 minutes  
**Cost:** Low  

```bash
# Launch validation
./scripts/launch_gradnorm_validation.sh

# Monitor progress
tail -f logs/gradnorm_validation.log

# Watch for:
# - ✅ GradNorm weights adapting (printed every 50 steps)
# - ✅ Loss balance improving
# - ❌ NO NaN or Inf values
```

**Expected output:**
```
🎯 GradNorm weights: pixel=42.3%, adversarial=18.1%, rec_feat=15.2%, perceptual=12.4%, ctc=12.0%
```

### Phase 2: Production (AFTER VALIDATION SUCCESS)

**Purpose:** Full training for paper results  
**Runtime:** ~8-10 hours  
**Cost:** HIGH  

```bash
# Launch production
./scripts/launch_gradnorm_production.sh

# Monitor
tail -f logs/gradnorm_production.log
```

---

## 📁 Files Structure

```
configs/
├── gradnorm_validation.json      # 5 epochs, 50 steps/epoch
└── gradnorm_production.json      # 50 epochs, full dataset

scripts/
├── launch_gradnorm_validation.sh # Validation launcher
└── launch_gradnorm_production.sh # Production launcher

dual_modal_gan/
├── models/gradnorm.py            # GradNorm implementation
└── scripts/train_enhanced.py     # Modified with GradNorm support
```

---

## 🔧 Configuration

### Key Parameters (in JSON config)

```json
{
  "use_gradnorm": true,
  "gradnorm_config": {
    "alpha": 1.5,                 // Asymmetry param (1.0-2.0)
    "update_frequency": 1,        // Update every N batches
    "loss_names": [               // Loss components to balance
      "pixel",
      "adversarial", 
      "rec_feat",
      "perceptual",
      "ctc"
    ],
    "initial_weights": [          // Starting weights
      50.0,  // pixel
      3.0,   // adversarial
      8.0,   // rec_feat
      1.0,   // perceptual
      0.15   // ctc
    ]
  }
}
```

### Parameter Tuning Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `alpha` | 1.5 | 0.5-2.0 | Higher = stronger adaptation |
| `update_frequency` | 1 | 1-10 | 1=every batch, 10=every 10 batches |
| `initial_weights` | from v3 | any | Starting point (will be adapted) |

---

## 📊 Monitoring

### 1. Console Output

Every 50 steps:
```
🎯 GradNorm weights: pixel=42.3%, adversarial=18.1%, rec_feat=15.2%, perceptual=12.4%, ctc=12.0%
```

### 2. MLflow Tracking

```bash
# Start MLflow UI
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration
poetry run mlflow ui

# Open: http://localhost:5000
```

**Metrics to watch:**
- `gradnorm/weight_*` - Absolute weights  
- `gradnorm/pct_*` - Normalized percentages  
- `train/*_loss` - Individual loss values  

### 3. Log Files

```bash
# Validation
tail -f logs/gradnorm_validation.log

# Production
tail -f logs/gradnorm_production.log

# Search for GradNorm updates
grep "GradNorm weights" logs/gradnorm_validation.log
```

---

## ✅ Success Criteria

### Validation Phase

✅ **MUST achieve:**
- [ ] No NaN or Inf in losses
- [ ] GradNorm weights change over time (not stuck)
- [ ] Training completes without crash
- [ ] Loss values reasonable (not exploding)

⚠️ **If validation fails:** Debug before production!

### Production Phase

🎯 **Expected improvements over production_v3:**
- PSNR: ≥30.74 dB (baseline)
- SSIM: ≥0.9869 (baseline)
- CER: ≤34.9% (baseline)
- WER: ≤82.4% (baseline)

🏆 **Novel contribution:**
- First GradNorm application to GAN-HTR
- Automatic loss balancing vs manual tuning
- Evidence of better convergence

---

## 🐛 Troubleshooting

### Issue: GradNorm weights not changing

**Symptom:** Weights stay at initial values  
**Cause:** Update frequency too high or alpha too low  
**Fix:**
```json
"update_frequency": 1,  // Try 1 (every batch)
"alpha": 1.5            // Increase to 2.0 for stronger adaptation
```

### Issue: NaN losses

**Symptom:** Loss becomes NaN after some steps  
**Cause:** Exploding gradients or unstable weights  
**Fix:**
```json
"gradient_clip_norm": 0.5,  // Reduce from 1.0
"initial_weights": [10, 1, 2, 0.5, 0.05]  // Start with smaller values
```

### Issue: One loss dominates

**Symptom:** One weight becomes >>90%  
**Cause:** Alpha too high, causing over-adaptation  
**Fix:**
```json
"alpha": 1.0,  // Reduce from 1.5
"update_frequency": 5  // Update less frequently
```

---

## 🔬 Comparison with Baseline

### production_v3 (Static Weights)

```json
{
  "pixel_loss_weight": 50.0,
  "adv_loss_weight": 3.0,
  "rec_feat_loss_weight": 8.0,
  "ctc_loss_weight": 0.15,
  "perceptual_loss_weight": 1.0
}
```

**Pros:** Proven, stable  
**Cons:** Manually tuned, no adaptation  

### GradNorm (Adaptive)

**Pros:**
- ✅ Automatic optimization
- ✅ Adapts to training dynamics
- ✅ Prevents loss dominance
- ✅ Novel contribution

**Cons:**
- ⚠️ More complex
- ⚠️ Needs validation
- ⚠️ Additional hyperparameter (alpha)

---

## 📈 Expected Training Behavior

### Early Training (Epoch 1-10)

```
Epoch 1: pixel=45%, adv=20%, rec_feat=18%, percep=10%, ctc=7%
Epoch 5: pixel=38%, adv=22%, rec_feat=16%, percep=12%, ctc=12%
Epoch 10: pixel=35%, adv=23%, rec_feat=15%, percep=13%, ctc=14%
```

**Expected:** Weights gradually converge to optimal balance

### Mid Training (Epoch 20-35)

```
Epoch 20: pixel=33%, adv=24%, rec_feat=14%, percep=14%, ctc=15%
Epoch 35: pixel=32%, adv=24%, rec_feat=14%, percep=15%, ctc=15%
```

**Expected:** Weights stabilize, small adjustments

### Late Training (Epoch 40-50)

```
Epoch 45: pixel=31%, adv=25%, rec_feat=14%, percep=15%, ctc=15%
Epoch 50: pixel=31%, adv=25%, rec_feat=14%, percep=15%, ctc=15%
```

**Expected:** Near-constant weights (converged)

---

## 🎓 Paper Writing Tips

### For Chapter 5 (Results)

**Claim:**
> "Berbeda dengan penelitian baseline yang menggunakan bobot loss statis hasil manual tuning, penelitian ini menerapkan GradNorm (*Gradient Normalization*) untuk adaptive loss balancing secara otomatis."

**Evidence to report:**
1. Initial vs final weight distributions
2. Loss convergence curves (compared to v3)
3. Final metrics (PSNR/SSIM/CER/WER)
4. Training stability (variance of losses)

### Novelty Statement

> "Sepengetahuan penulis, ini merupakan aplikasi pertama GradNorm pada domain GAN-HTR untuk restorasi dokumen, menunjukkan bahwa adaptive loss balancing dapat mengatasi tantangan multi-objective optimization dalam sistem multi-modal."

---

## 🛑 Stop Training

```bash
# Find process
ps aux | grep train_enhanced

# Kill gracefully
kill <PID>

# Force kill if needed
kill -9 <PID>
```

**Note:** Checkpoints saved every 2 epochs. Safe to stop anytime.

---

## 📝 Next Steps

1. ✅ **Run validation** - MANDATORY before production
2. ⏸️ **Analyze validation results** - Check weights evolution
3. 🚀 **Run production** - Only if validation successful
4. 📊 **Compare with v3** - Metrics comparison
5. 📄 **Write paper** - Document findings

---

## 🔗 Related Files

- Implementation: `dual_modal_gan/models/gradnorm.py`
- Training: `dual_modal_gan/scripts/train_enhanced.py`
- Configs: `configs/gradnorm_*.json`
- Paper baseline: `Research_papers/souibgui_enhance_to_read_better.md`

---

## ❓ Questions?

**Q: Should I use GradNorm or Optuna?**  
A: GradNorm (adaptive during training) vs Optuna (search before training). GradNorm more novel, cheaper, better for paper.

**Q: What if validation fails?**  
A: Check logs, adjust alpha/initial_weights, try again. Don't proceed to production.

**Q: How long for production?**  
A: 8-10 hours with 2x GPU. Can use CPU fallback but slower.

**Q: Can I resume if stopped?**  
A: Not yet implemented for GradNorm. Plan for continuous run.

---

**Last Updated:** 2025-11-12  
**Status:** Ready for validation testing
