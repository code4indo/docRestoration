# 🧪 EXPERIMENT SERIES - CTC DOMINANCE HYPOTHESIS

**Tanggal**: 16 Oktober 2025
**Tujuan**: Membuktikan bahwa CTC loss dominance (97%) menyebabkan PSNR rendah

---

## 📋 HYPOTHESIS TO PROVE

### **Baseline Finding**
- Training 1000 steps dengan CTC weight 1.0
- **Result**: PSNR 18.91 dB, CER 0.3445
- **Problem**: CTC loss = 242.83 (97.6% dari total G_loss 248.86)
- **Root Cause**: Pixel/RecFeat/Adv gradients terlalu lemah (hanya 2.4%)

### **Hipotesis yang Akan Dibuktikan**

1. **H1: CTC Weight 0.1 → PSNR naik 5-7 dB**
   - CTC dominance turun → Pixel/RecFeat/Adv loss lebih berpengaruh
   - Expected: PSNR 24-26 dB, CER 0.6-0.7

2. **H2: CTC Weight 0.3 → Balanced image+text quality**
   - Balance antara image quality dan text recognition
   - Expected: PSNR 22-24 dB, CER 0.4-0.5

3. **H3: Strong Adversarial (Adv 5.0) → Better image quality**
   - Discriminator lebih kuat → Generator terpaksa produce better images
   - Expected: PSNR 23-25 dB, D_loss lebih rendah

4. **H4: Optimal Combination → Best overall PSNR**
   - CTC 0.2, Pixel 300, RecFeat 10, Adv 3.0
   - Expected: PSNR 25-27 dB (BEST)

---

## 🔬 EXPERIMENT CONFIGURATIONS

### **Experiment 1: CTC Annealing Low (CTC=0.1)**
```json
{
  "ctc_loss_weight": 0.1,
  "pixel_loss_weight": 200.0,
  "rec_feat_loss_weight": 5.0,
  "adv_loss_weight": 1.5
}
```
**Expected CTC contribution**: 24/30 = 80% → Still high, but pixel/recfeat/adv get 20%

---

### **Experiment 2: CTC Medium (CTC=0.3)**
```json
{
  "ctc_loss_weight": 0.3,
  "pixel_loss_weight": 200.0,
  "rec_feat_loss_weight": 5.0,
  "adv_loss_weight": 1.5
}
```
**Expected CTC contribution**: 72/85 = 85% → More balanced

---

### **Experiment 3: Strong Adversarial (Adv=5.0, CTC=0.5)**
```json
{
  "ctc_loss_weight": 0.5,
  "pixel_loss_weight": 200.0,
  "rec_feat_loss_weight": 5.0,
  "adv_loss_weight": 5.0
}
```
**Expected**: Discriminator lebih kuat, generator improve quality

---

### **Experiment 4: Optimal Balanced (CTC=0.2)**
```json
{
  "ctc_loss_weight": 0.2,
  "pixel_loss_weight": 300.0,
  "rec_feat_loss_weight": 10.0,
  "adv_loss_weight": 3.0
}
```
**Expected**: Best overall combination

---

## 🚀 HOW TO RUN

### **Run All Experiments (Sequential)**
```bash
# Total waktu: ~80 menit (4 × 20 menit)
poetry run bash scripts/run_experiment_series.sh
```

### **Run Individual Experiment**
```bash
# Experiment 1
poetry run bash scripts/run_training_from_json.sh configs/experiment1_ctc_annealing_low.json

# Experiment 2
poetry run bash scripts/run_training_from_json.sh configs/experiment2_ctc_medium.json

# Experiment 3
poetry run bash scripts/run_training_from_json.sh configs/experiment3_strong_adversarial.json

# Experiment 4
poetry run bash scripts/run_training_from_json.sh configs/experiment4_balanced_optimal.json
```

---

## 📊 COMPARE RESULTS

### **Extract Metrics**
```bash
# Compare all experiments
poetry run python scripts/compare_experiment_results.py
```

### **Manual Check**
```bash
# Check final PSNR dari setiap experiment
tail -50 logbook/experiment1_*.log | grep "PSNR:"
tail -50 logbook/experiment2_*.log | grep "PSNR:"
tail -50 logbook/experiment3_*.log | grep "PSNR:"
tail -50 logbook/experiment4_*.log | grep "PSNR:"
```

---

## 📈 EXPECTED RESULTS TABLE

| Experiment | CTC Weight | Expected PSNR | Expected CER | G_Loss Range | Verification |
|------------|------------|---------------|--------------|--------------|--------------|
| Baseline   | 1.0        | 18.91 dB      | 0.34         | ~248         | ✅ Done      |
| Exp 1      | 0.1        | 24-26 dB      | 0.6-0.7      | ~30-40       | ⏳ Pending   |
| Exp 2      | 0.3        | 22-24 dB      | 0.4-0.5      | ~80-100      | ⏳ Pending   |
| Exp 3      | 0.5, Adv 5 | 23-25 dB      | 0.4-0.5      | ~120-150     | ⏳ Pending   |
| Exp 4      | 0.2, Opt   | 25-27 dB ✨   | 0.5-0.6      | ~60-80       | ⏳ Pending   |

---

## 🎯 SUCCESS CRITERIA

### **Hypothesis PROVEN if:**
1. ✅ Exp 1 PSNR > 24 dB (improvement >5 dB dari baseline)
2. ✅ Exp 2 CER < Exp 1 CER (balance works)
3. ✅ Exp 3 D_loss < 1.2 (discriminator stronger)
4. ✅ Exp 4 PSNR adalah yang tertinggi (>25 dB)

### **If Proven:**
- **Full training strategy**: Gunakan Exp 4 config untuk 50 epochs
- **Expected final PSNR**: 32-35 dB (25 dB + 7-10 dB dari full training)
- **Expected final CER**: 0.2-0.3

### **If NOT Proven:**
- **Plan B**: Multi-stage training
  - Stage 1: CTC=0.0, 5 epochs (pure image quality)
  - Stage 2: CTC=0.1, 10 epochs
  - Stage 3: CTC=0.5, 15 epochs
  - Stage 4: CTC=1.0, 20 epochs

---

## 📝 LOGBOOK STRUCTURE

```
logbook/
├── experiment1_ctc_annealing_low_YYYYMMDD_HHMMSS.log
├── experiment2_ctc_medium_YYYYMMDD_HHMMSS.log
├── experiment3_strong_adversarial_YYYYMMDD_HHMMSS.log
└── experiment4_balanced_optimal_YYYYMMDD_HHMMSS.log
```

---

## 🔍 ANALYSIS WORKFLOW

1. **Run all experiments**: `bash scripts/run_experiment_series.sh`
2. **Compare results**: `python scripts/compare_experiment_results.py`
3. **Analyze findings**: Check which config gives best PSNR
4. **Decision**:
   - If PSNR >25 dB → Proceed to full training
   - If PSNR 22-25 dB → Refine config, test again
   - If PSNR <22 dB → Consider architectural changes

---

## 💡 NOTES

- Each experiment = 1 epoch × 1000 steps ≈ 20 minutes
- Total waktu series = ~80 minutes
- Bisa dijalankan sequential atau parallel (tapi hati-hati GPU memory)
- Log metrics ke MLflow untuk tracking

---

**Status**: Ready to run
**Next Action**: Execute `bash scripts/run_experiment_series.sh`
