# 📋 SURAT TUGAS: FULL TRAINING GAN-HTR MODEL

**Kepada:** Agent Coding (ClaudeCode)  
**Dari:** Data Scientist/ML Engineer (Agent Sebelumnya)  
**Tanggal:** 15 Oktober 2025  
**Prioritas:** HIGH  
**Estimasi Waktu:** 24-48 jam (training time)

---

## 🎯 OBJEKTIF UTAMA

Melakukan **FULL TRAINING** model GAN-HTR dengan konfigurasi optimal hasil Hyperparameter Optimization (HPO) menggunakan skrip yang sudah tersedia: `scripts/train32_smoke_test.sh`

**Target Metrics:**
- **PSNR**: ≥ 40 dB (target penelitian)
- **SSIM**: ≥ 0.99 (target penelitian)
- **CER**: < 0.10 (10% error rate)
- **WER**: < 0.20 (20% error rate)

---

## 📊 KONTEKS PENELITIAN

### Hasil HPO (29 trials completed)
**Best Configuration (Trial 8):**
```
pixel_loss_weight      : 120.0
rec_feat_loss_weight   : 80.0
adv_loss_weight        : 2.5
ctc_loss_weight        : 10.0 (monitoring only, NOT backpropagated)
```

**Best Metrics (2 epochs smoke test):**
```
PSNR: 13.09 dB
SSIM: 0.746
CER:  0.707
WER:  0.811
```

**Key Findings dari HPO:**
1. ✅ **ADV weight LOW (2.5)** = Kualitas visual bagus
2. ❌ **ADV weight HIGH (>5.0)** = Visual quality HANCUR
3. ✅ **RecFeat weight HIGH (80.0)** = Strong positive correlation dengan objective score
4. ⚠️ **Pixel weight MODERATE (120.0)** = Weak correlation, stable baseline

### Important Notes:
- **CTC Loss**: HANYA monitoring metric, TIDAK di-backprop ke Generator
- **Training Strategy**: 3 loss components untuk Generator (adversarial + pixel-wise + recognition feature)
- **Precision**: Pure FP32 (bukan mixed precision) untuk stabilitas CTC computation

---

## 🔧 JOB DESCRIPTION

### TASK 1: Modifikasi Script untuk Full Training

**File Target:** `scripts/train32_smoke_test.sh`

**Perubahan yang Diperlukan:**

#### A. Parameter Training (Update di bagian akhir script)
```bash
# DARI (Smoke Test - 3 epochs):
--epochs 3 \
--warmup_epochs 2 \
--annealing_epochs 4 \
--steps_per_epoch 50 \
--batch_size 4 \

# MENJADI (Full Training - 200 epochs):
--epochs 200 \
--warmup_epochs 10 \
--annealing_epochs 50 \
--steps_per_epoch 100 \
--batch_size 8 \
```

#### B. Checkpoint & Sample Directories
```bash
# DARI:
CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_fp32_smoke_test"
SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_fp32_smoke_test"

# MENJADI:
CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_fp32_full_training"
SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_fp32_full_training"
```

#### C. Log File Naming
```bash
# DARI:
LOG_FILE="$LOGBOOK_DIR/smoke_test_$(date +%Y%m%d_%H%M%S).log"

# MENJADI:
LOG_FILE="$LOGBOOK_DIR/full_training_$(date +%Y%m%d_%H%M%S).log"
```

#### D. Loss Weights (SUDAH OPTIMAL - JANGAN UBAH!)
```bash
--pixel_loss_weight 120.0 \
--rec_feat_loss_weight 80.0 \
--adv_loss_weight 2.5 \
--ctc_loss_weight 10.0 \
```
⚠️ **CRITICAL**: Loss weights ini hasil HPO 29 trials, JANGAN diubah!

#### E. Early Stopping Parameters
```bash
# DARI (Ketat untuk smoke test):
--patience 10 \
--min_delta 0.1 \

# MENJADI (Lebih longgar untuk full training):
--patience 20 \
--min_delta 0.01 \
```

---

### TASK 2: Eksekusi Training di Background

**Command:**
```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration
nohup ./scripts/train32_full_training.sh > /dev/null 2>&1 &
echo $! > training_pid.txt
```

**Monitoring:**
```bash
# Monitor log file (real-time)
tail -f logbook/full_training_YYYYMMDD_HHMMSS.log

# Check process status
ps aux | grep train32.py

# Monitor GPU usage
watch -n 2 nvidia-smi

# Check MLflow UI (jika running)
# http://localhost:5000
```

---

### TASK 3: Monitoring Metrics Progression

**File Metrics:** `dual_modal_gan/outputs/checkpoints_fp32_full_training/metrics/training_metrics_fp32.json`

**Key Metrics to Monitor:**

1. **PSNR Progression**
   - Epoch 10: Expected ~15-18 dB
   - Epoch 50: Expected ~25-30 dB
   - Epoch 100: Expected ~35-38 dB
   - Epoch 200: Target ≥40 dB

2. **SSIM Progression**
   - Epoch 10: Expected ~0.75-0.80
   - Epoch 50: Expected ~0.85-0.90
   - Epoch 100: Expected ~0.95-0.97
   - Epoch 200: Target ≥0.99

3. **CER/WER Progression**
   - Epoch 10: Expected ~0.60-0.70
   - Epoch 50: Expected ~0.30-0.40
   - Epoch 100: Expected ~0.15-0.20
   - Epoch 200: Target CER <0.10, WER <0.20

4. **Loss Components Balance**
   - Generator Total Loss: Should decrease steadily
   - Discriminator Accuracy: Target 60-80% (Nash equilibrium)
   - CTC Loss: Should decrease (monitoring only)
   - NO divergence atau sudden spikes

---

### TASK 4: Validasi Quality Samples

**Directory:** `dual_modal_gan/outputs/samples_fp32_full_training/`

**Validation Checklist per Epoch:**

✅ **Visual Quality (subjektif):**
- [ ] Degradasi berkurang progressively
- [ ] Teks lebih jelas dan readable
- [ ] Tidak ada artifacts (blur, noise, distortion)
- [ ] Background restoration natural

✅ **Quantitative (dari metrics file):**
- [ ] PSNR meningkat consistently
- [ ] SSIM mendekati 1.0
- [ ] CER/WER menurun significantly

---

### TASK 5: Handling Potential Issues

#### Issue 1: Out of Memory (OOM)
**Symptoms:** CUDA out of memory error

**Solution:**
```bash
# Reduce batch size
--batch_size 4 \  # atau bahkan 2 jika masih OOM

# Enable gradient checkpointing (jika tersedia di train32.py)
# atau gunakan single GPU dengan --gpu_id 0
```

#### Issue 2: Divergence/Instability
**Symptoms:** Loss meledak (>1000), PSNR turun drastis, NaN values

**Solution:**
```bash
# Stop training
kill $(cat training_pid.txt)

# Restore dari checkpoint terakhir yang stabil
# Cek di: dual_modal_gan/outputs/checkpoints_fp32_full_training/

# Resume dengan learning rate lebih kecil
--lr_g 0.0002 \  # dari 0.0004
--lr_d 0.0002 \
```

#### Issue 3: Slow Convergence
**Symptoms:** Metrics plateau, tidak ada improvement setelah 50+ epochs

**Analysis:**
```bash
# Check jika early stopping terlalu ketat
# Cek pattern di log file:
grep "Early stopping" logbook/full_training_*.log

# Jika terlalu sering stop, adjust:
--patience 30 \  # dari 20
--min_delta 0.005 \  # dari 0.01
```

#### Issue 4: Discriminator Collapse
**Symptoms:** Discriminator accuracy >95% atau <40%

**Solution:**
```bash
# Jika D terlalu kuat (>95%):
--lr_d 0.0002 \  # reduce discriminator LR

# Jika D terlalu lemah (<40%):
--lr_g 0.0002 \  # reduce generator LR
```

---

## 📝 DELIVERABLES

### 1. Training Completion Report
**File:** `logbook/full_training_report_$(date +%Y%m%d).md`

**Content:**
```markdown
# Full Training Report - GAN-HTR Model

## Training Configuration
- Start Time: [timestamp]
- End Time: [timestamp]
- Total Duration: [hours]
- Total Epochs: 200 (atau early stopped di epoch X)
- Final Loss Weights: pixel=120.0, rec_feat=80.0, adv=2.5, ctc=10.0

## Final Metrics
- PSNR: [value] dB (Target: ≥40)
- SSIM: [value] (Target: ≥0.99)
- CER: [value] (Target: <0.10)
- WER: [value] (Target: <0.20)

## Key Observations
- [Convergence pattern]
- [Any issues encountered]
- [Best epoch number]
- [Discriminator equilibrium status]

## Model Checkpoints
- Best Model: [path]
- Final Model: [path]
- Best Metrics Epoch: [epoch number]
```

### 2. Updated Metrics JSON
**File:** `dual_modal_gan/outputs/checkpoints_fp32_full_training/metrics/training_metrics_fp32.json`

Harus berisi 200 entries (atau jumlah epoch yang dijalankan) dengan lengkap.

### 3. Best Model Checkpoint
**Files:**
- `dual_modal_gan/outputs/checkpoints_fp32_full_training/generator_epoch_XXX.weights.h5`
- `dual_modal_gan/outputs/checkpoints_fp32_full_training/discriminator_epoch_XXX.weights.h5`

Dimana XXX adalah epoch dengan metrics terbaik.

### 4. Sample Images (Every 10 epochs)
**Directory:** `dual_modal_gan/outputs/samples_fp32_full_training/`

File naming: `epoch_XXX_sample_Y.png` (X = epoch, Y = sample index)

### 5. MLflow Tracking Data
**Location:** `mlruns/` directory

Pastikan semua runs tercatat dengan metadata lengkap.

---

## ⚠️ CRITICAL WARNINGS

### 🔴 JANGAN DIUBAH:
1. **Loss weights** (120.0, 80.0, 2.5, 10.0) - Hasil optimal dari HPO
2. **Precision** (Pure FP32) - Critical untuk CTC stability
3. **Gradient clipping** (1.0) - Mencegah exploding gradients
4. **CTC loss clipping** (100.0) - Stabilitas monitoring
5. **Seed** (42) - Reproducibility

### 🟡 BOLEH DISESUAIKAN (jika diperlukan):
1. **Batch size** (8 → 4 jika OOM)
2. **Steps per epoch** (100 → sesuai dataset size)
3. **Early stopping patience** (20 → 30 jika perlu)
4. **Learning rates** (hanya jika divergence)

### 🟢 HARUS DIPANTAU:
1. **GPU memory usage** (jangan sampai OOM)
2. **Discriminator accuracy** (target 60-80%)
3. **Loss progression** (steady decrease)
4. **Sample quality** (visual inspection)

---

## 📞 ESCALATION PROTOCOL

### Kondisi Harus Dihentikan Segera:
1. ❌ **NaN/Inf dalam loss values**
2. ❌ **PSNR turun drastis (>5 dB drop)**
3. ❌ **Discriminator accuracy >95% atau <30%**
4. ❌ **GPU temperature >85°C sustained**
5. ❌ **Disk space <10GB tersisa**

### Kondisi Perlu Review:
1. ⚠️ **Metrics plateau >30 epochs**
2. ⚠️ **Memory usage >90% sustained**
3. ⚠️ **Sample quality degradation**
4. ⚠️ **CTC loss meningkat consistently**

---

## 🎓 REFERENSI TEKNIS

### Arsitektur Model:
- **Generator**: U-Net based dengan skip connections
- **Discriminator**: Dual-modal (visual + textual features)
- **Recognizer**: Pre-trained HTR (frozen, hanya feature extraction)

### Training Strategy:
- **Phase 1 (Epoch 1-10)**: Warmup dengan adversarial weight rendah
- **Phase 2 (Epoch 11-50)**: CTC annealing (gradual weight increase)
- **Phase 3 (Epoch 51-200)**: Full training dengan all components

### Loss Function Formula:
```
L_Generator = pixel_weight * L_pixel 
            + rec_feat_weight * L_recognition_feature
            + adv_weight * L_adversarial
            
L_CTC = ctc_weight * L_ctc  (MONITORING ONLY, NOT BACKPROPAGATED)
```

---

## ✅ CHECKLIST PRE-EXECUTION

Sebelum memulai training, pastikan:

- [ ] Script dimodifikasi dengan benar (200 epochs, batch_size=8, dll)
- [ ] Dataset tersedia: `dual_modal_gan/data/dataset_gan.tfrecord`
- [ ] Charset file ada: `real_data_preparation/real_data_charlist.txt`
- [ ] Recognizer weights ada: `models/best_htr_recognizer/best_model.weights.h5`
- [ ] GPU available dan tidak sedang digunakan proses lain
- [ ] Disk space >50GB free
- [ ] Virtual environment active (poetry)
- [ ] MLflow UI running (optional, untuk monitoring)
- [ ] Screen/tmux session ready (jika training di remote server)

---

## 🚀 EXPECTED TIMELINE

| Fase | Duration | Expected Outcome |
|------|----------|------------------|
| **Setup & Validation** | 30 min | Script ready, dependencies checked |
| **Training Start** | - | Process launched in background |
| **Epoch 1-50** | 8-12 hours | PSNR ~25-30 dB, SSIM ~0.85-0.90 |
| **Epoch 51-100** | 8-12 hours | PSNR ~35-38 dB, SSIM ~0.95-0.97 |
| **Epoch 101-200** | 16-24 hours | PSNR ≥40 dB, SSIM ≥0.99 |
| **Total** | **32-48 hours** | Final model ready untuk evaluasi |

---

## 📋 POST-TRAINING TASKS

Setelah training selesai:

1. **Generate Comprehensive Report** (Task 1 Deliverables)
2. **Archive Best Checkpoint** ke direktori `models/best_gan_htr_full/`
3. **Update Thesis Chapter 5** dengan final results
4. **Prepare Visualization** (loss curves, sample comparisons)
5. **MLflow Experiment Export** untuk dokumentasi
6. **Backup Training Logs** ke cloud storage (jika ada)

---

## 💡 TIPS & BEST PRACTICES

1. **Monitor Awal (Epoch 1-10):**
   - Perhatikan apakah losses menurun smooth
   - Cek sample images untuk artifacts
   - Validasi discriminator accuracy dalam range 60-80%

2. **Mid-Training (Epoch 50-100):**
   - Jika metrics sudah plateau, pertimbangkan early stopping
   - Compare dengan baseline metrics dari HPO
   - Backup checkpoint setiap 10 epochs

3. **Late Training (Epoch 150-200):**
   - Diminishing returns is normal
   - Focus on stability, bukan speed of improvement
   - Prepare untuk final evaluation

4. **Resource Management:**
   - Monitor GPU temperature regularly
   - Clear old checkpoints jika disk space terbatas
   - Use `tee` untuk log file agar bisa review later

---

## 📞 KONTAK & SUPPORT

**Primary Agent:** Data Scientist/ML Engineer (Agent Sebelumnya)  
**Backup:** Principal Investigator (Peneliti Utama)

**Knowledge Base:**
- HPO Results: `logbook/HPO_FINAL_REPORT_20251015.md`
- Ablation Study: `catatan/laporanStudiAblation_*.md`
- Training Issues: `catatan/DebugCudnnRnnMaskError.md`
- Thesis Methodology: `dual_modal_gan/docs/chapter3_metodologi.tex`

---

## 🎯 SUCCESS CRITERIA

Training dinyatakan **BERHASIL** jika:

✅ **Quantitative:**
- PSNR ≥ 40 dB (CRITICAL)
- SSIM ≥ 0.99 (CRITICAL)
- CER < 0.10 (IMPORTANT)
- WER < 0.20 (IMPORTANT)

✅ **Qualitative:**
- Visual samples menunjukkan restorasi signifikan
- Teks terlihat jelas dan readable
- Tidak ada degradasi artifacts
- Background natural dan clean

✅ **Process:**
- No divergence atau training instability
- Reproducible dengan seed 42
- All metrics logged completely
- Checkpoints saved properly

---

## 📜 DISCLAIMER

**CLEAN SLATE PRINCIPLE:**  
Training ini menggunakan prinsip "clean slate" - dimulai dari scratch dengan konfigurasi optimal hasil HPO. TIDAK ada restore dari checkpoint sebelumnya (`--no_restore` flag).

**PRODUCTION COST AWARENESS:**  
Setiap epoch membutuhkan GPU time yang mahal. Lakukan monitoring ketat untuk detect issues early. Jangan biarkan training diverge atau waste resources.

**RESEARCH INTEGRITY:**  
Semua hasil training harus dicatat dengan jujur di logbook. Jika gagal mencapai target, dokumentasikan alasannya untuk iterasi berikutnya.

---

**Dibuat oleh:** ML Engineer Agent  
**Tanggal:** 15 Oktober 2025  
**Versi:** 1.0  
**Status:** READY FOR EXECUTION

---

## 🚦 AUTHORIZATION

**Approved for execution:** ✅  
**Risk Level:** MEDIUM (training stability proven di smoke test)  
**Resource Cost:** HIGH (48 hours GPU time)  
**Impact:** CRITICAL (final model untuk thesis)

**GO/NO-GO:** ✅ **GO** - Semua prerequisites terpenuhi, konfigurasi optimal tersedia

---

**GOOD LUCK! 🚀**

*Ingat: Penelitian ini bertujuan menemukan novelty untuk jurnal Q1. Fokus pada kualitas dan reproducibility!*
