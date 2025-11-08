# QUICK REFERENCE: DIBCO FT V2 Loss Rebalance Experiment

**Created:** 30 Oktober 2025  
**Config File:** `configs/dibco_ft_v2_loss_rebalance.json`  
**Status:** ⏸️ **READY TO START - AWAITING CONFIRMATION**

---

## 🎯 OBJECTIVE

Fix **thick edges** and **rough strokes** dari DIBCO FT V1 experiment dengan **rebalancing loss weights** berdasarkan analisis kontribusi aktual.

---

## 🔄 CHECKPOINT STRATEGY

```
Resume from: dibco_finetuning_from_anri_v1/best_model/ckpt-121
Epoch: 1 (PSNR 21.30 dB - sebelum degradasi dimulai)
Mode: resume=true, no_restore=true
✅ VERIFIED: Checkpoint exists (383MB data + 49KB index)
```

**Rationale:** Epoch 1 memiliki PSNR terbaik (21.30 dB) sebelum loss imbalance menyebabkan degradasi progresif ke 20.52 dB di epoch 10.

---

## ⚖️ LOSS WEIGHTS CHANGES

| Loss | OLD | NEW | Change | Expected Contribution |
|------|-----|-----|--------|----------------------|
| **Pixel** | 200.0 | **50.0** | **-75%** | 3.3% (target 15-20%) |
| **Perceptual** | 10.0 | **1.0** | **-90%** 🔴 | 69.8% (target 50-60%) |
| **Adversarial** | 1.5 | **5.0** | **+233%** 🟢 | 26.8% (target 10-15%) |
| **Rec Feat** | 5.0 | **5.0** | **SAME** | 0.1% |

**Total Loss:** 523.5 → **72.75** (86% reduction)

---

## ⚠️ CRITICAL FIXES

### 1. Loss Contribution Imbalance (FIXED)
```
OLD: Perceptual 97% | Pixel 1.8% | Adv 1.1%  ❌ SEVERELY IMBALANCED
NEW: Perceptual 70% | Adv 27% | Pixel 3%   ⚠️ BETTER (adv masih tinggi)
```

**Note:** Adversarial contribution 26.8% lebih tinggi dari target 10-15%, tapi ini acceptable untuk strengthen discriminator feedback.

### 2. Gradient Stability
```
Gradient Clip Norm: 1.0 → 5.0 (+400%)
Generator LR: 1e-5 → 5e-6 (-50%)
```

---

## 📊 EXPECTED IMPROVEMENTS

| Metric | Current (Epoch 10) | Target | Improvement |
|--------|-------------------|--------|-------------|
| **PSNR** | 20.52 dB | **>23.5 dB** | +3+ dB |
| **Local Variance** | 909 | **<400** | -56% |
| **Noise Variance** | 8662 | **<2000** | -77% |
| **Isolated White** | 13.89% | **<10%** | -28% |
| **Avg Stroke Width** | 187px | **<10px** | -95% |

---

## 🚨 ABORT CRITERIA

1. **PSNR < 20.0 dB** after 5 epochs → Abort, investigate
2. **Total Loss > 1000** → Abort, check implementation  
3. **Generator gradient > 1000** → Abort, check LR/clip

---

## 🚀 TRAINING COMMAND

```bash
# 1. Verify checkpoint
ls -lh dual_modal_gan/checkpoints/dibco_finetuning_from_anri_v1/best_model/ckpt-121*

# 2. Start training (background)
nohup ./scripts/universal_train_from_json.sh configs/dibco_ft_v2_loss_rebalance.json > logs/dibco_ft_v2_loss_rebalance.log 2>&1 &

# 3. Monitor log
tail -f logs/dibco_ft_v2_loss_rebalance.log

# 4. Check process
ps aux | grep train_enhanced
```

---

## 📈 MONITORING CHECKLIST

**Every Epoch, check:**
- [ ] PSNR trend (must increase from 21.30)
- [ ] Loss contribution breakdown (target: ~70% perc, ~27% adv, ~3% pixel)
- [ ] Local variance (must decrease from 909)
- [ ] Generator gradient norm (must decrease from 556)

**Every 5 epochs, check samples:**
- [ ] Edge thickness (visual inspection)
- [ ] Stroke smoothness (visual inspection)
- [ ] Noise artifacts (isolated white pixels)

---

## 🎯 SUCCESS CRITERIA

### PRIMARY (Must Achieve)
✅ **PSNR ≥ 23.5 dB** (recover at least 3 dB)  
✅ **Local Variance < 400** (improve roughness by 50%)

### SECONDARY (Should Achieve)
⭕ **Noise Variance < 2000** (reduce noise artifacts)  
⭕ **Isolated White < 10%** (cleaner edges)

### STRETCH GOAL
🌟 **PSNR ≥ 24.5 dB + Local Variance < 300** (match/exceed ANRI V1)

---

## 🔄 NEXT STEPS (If Success)

1. **Phase 2A:** Add Total Variation loss (weight: 2.0)
2. **Phase 2B:** Add Edge-aware loss (weight: 10.0)  
3. **Phase 3:** Enhance discriminator (spectral norm, dropout 0.3)
4. **Phase 4:** Mixed dataset training (30% ANRI + 70% DIBCO)

---

## 🔙 ROLLBACK PLAN (If Failure)

Jika PSNR < 21 dB atau tidak ada improvement setelah 10 epochs:
1. Revert ke ANRI V1 checkpoint: `thin_stroke_preservation_v1_finetuning/best_model/ckpt-120`
2. Try **progressive fine-tuning** strategy (freeze encoder first)
3. Consider **mixed dataset** approach dari awal

---

## 📝 NOTES

- Training akan berjalan **di background** dengan `nohup`
- Log file: `logs/dibco_ft_v2_loss_rebalance.log`
- Checkpoints tersimpan di: `dual_modal_gan/checkpoints/dibco_ft_v2_loss_rebalance/`
- Sample outputs di: `dual_modal_gan/outputs/samples_dibco_ft_v2_loss_rebalance/`
- Early stopping patience: **15 epochs** (increased dari 10)

---

**⚠️ FINAL CHECKLIST BEFORE TRAINING:**

- [x] Config file created: `configs/dibco_ft_v2_loss_rebalance.json`
- [x] Checkpoint verified: `ckpt-121` exists (383MB)
- [x] Loss weights rebalanced (Pixel -75%, Perc -90%, Adv +233%)
- [x] Gradient stability improved (clip 5.0, gen_lr 5e-6)
- [x] Output directories configured (new paths)
- [x] Abort criteria defined
- [ ] **USER CONFIRMATION PENDING** ⏸️

---

**Status:** 🟡 Ready to launch, awaiting final confirmation from user.
