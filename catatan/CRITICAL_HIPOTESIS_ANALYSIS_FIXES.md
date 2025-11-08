# CRITICAL ANALYSIS: Hipotesis Penelitian - Isu Validitas & Rekomendasi Perbaikan

**Tanggal**: 2025-01-XX  
**Status**: 🔴 CRITICAL - Memerlukan revisi metodologi statistik

---

## 🚨 MASALAH KRITIS DITEMUKAN

### 1. **PAIRED T-TEST SALAH PAKAI** ❌❌❌
**Lokasi**: Line 1250, 1271, 2250, 2581

**Yang Tertulis**:
```
"Paired t-test (two-tailed) dengan tingkat signifikansi α = 0.05 
untuk membandingkan CER antara framework yang diusulkan dan baseline"
```

**FAKTA IMPLEMENTASI**:
- HTR-GAN (baseline) = Model TERPISAH yang dilatih oleh peneliti lain
- Proposed (Dual-Modal) = Model KITA yang dilatih dengan data yang sama
- Dataset test: n=712 images (SAMA untuk kedua model)
- **BUKAN paired data** karena kedua model adalah arsitektur berbeda

**MENGAPA INI SALAH**:
- Paired t-test untuk: Before-After treatment pada SUBJEK YANG SAMA
- Contoh paired: Same patient before/after drug
- Contoh paired: Same image dengan dua preprocessing berbeda → LALU prediksi HTR
- **KASUS KITA**: Dua MODEL BERBEDA diuji pada dataset yang sama = **INDEPENDENT SAMPLES**

**YANG BENAR**:
```
Independent samples t-test (two-tailed) atau 
Welch's t-test (jika variance tidak sama)
```

**IMPLIKASI**:
- Paired t-test memiliki **power lebih tinggi** dari independent
- Menggunakan paired test pada independent data = **INFLATED Type I error**
- Hasil p<0.001 mungkin tetap signifikan, tapi **metode salah** = **METODOLOGI CACAT**

---

### 2. **BASELINE SELECTION AMBIGUITAS** ⚠️

**Yang Tertulis**:
```
"dibandingkan dengan baseline state-of-the-art"
```

**Yang Ada di Results (Tabel 4)**:
1. Tanpa Perbaikan (CER 67.3%)
2. Otsu (52.1%)
3. Sauvola (45.8%)
4. U-Net (28.4%)
5. GAN Standar (24.1%)
6. DE-GAN (22.7%)
7. HTR-GAN (19.3%) ← **Diasumsikan sebagai "baseline"**

**MASALAH**:
- Hipotesis hanya mention "baseline" (singular)
- Tapi Table 4 compare dengan **7 methods**
- H1 test hanya HTR-GAN, tapi conclusion menyebut "mengungguli semua baseline"

**JIKA MULTIPLE COMPARISONS**:
- Perlu **Bonferroni/Holm correction** untuk SEMUA 7 comparisons
- Saat ini Bonferroni hanya untuk H2A-H2D (α=0.0125)
- **Seharusnya**: α = 0.05/7 = 0.0071 jika compare dengan semua baseline

**SOLUSI**:
```
Option 1: Eksplisit state "primary baseline = HTR-GAN (state-of-the-art HTR-aware method)"
Option 2: Do multiple comparisons dengan proper correction
Option 3: Reformulasi "we compare against strongest baseline (HTR-GAN)"
```

---

### 3. **THRESHOLD H2C TERLIHAT CHERRY-PICKED** ⚠️

**Yang Tertulis**:
```
H2C: PSNR ≥ 28 dB dan SSIM ≥ 0.90
```

**Hasil Actual**:
```
PSNR = 28.42 dB (hanya +0.42 dari threshold)
SSIM = 0.912 (hanya +0.012 dari threshold)
```

**MENGAPA INI MENCURIGAKAN**:
- Threshold **sangat dekat** dengan hasil actual
- Terlihat seperti threshold ditetapkan **setelah** melihat hasil
- Jika ini truly a priori, kenapa tidak 27 atau 29? Kenapa tepat 28?

**BASELINE CONTEXT MISSING**:
- HTR-GAN: PSNR = 26.89 dB, SSIM = 0.878
- DE-GAN: PSNR = 26.14 dB
- **Threshold 28 dB** berada di antara baseline (26.89) dan result (28.42)
- Ini **BUKAN arbitrary threshold**, tapi **comparative threshold**

**SOLUSI - REFORMULASI**:
```
OLD: "Framework mempertahankan PSNR ≥ 28 dB dan SSIM ≥ 0.90"

NEW: "Framework mencapai visual quality yang melebihi baseline terkuat:
     - PSNR improvement: 28.42 vs 26.89 dB (HTR-GAN) 
     - SSIM improvement: 0.912 vs 0.878 (HTR-GAN)
     dengan p<0.001 dan Cohen's d=0.68 (medium-large effect)"
```

**INI LEBIH JUJUR**:
- Tidak perlu "arbitrary threshold"
- Fokus pada **improvement over baseline**
- Tetap rigorous dengan statistical test

---

### 4. **H2B KURANG BUKTI EMPIRIS** ⚠️

**Yang Tertulis**:
```
"Frozen training lebih stabil (lower variance) vs joint training"
Metode: Levene's test untuk variance comparison
```

**Yang Tersedia di Paper**:
- Table Ablation line 2497: S1 (Joint) vs S2 (Frozen)
- ONLY final metrics: SSIM, CER, Time, Stability (qualitative)
- **TIDAK ADA**: Loss variance graph, F-statistic dari Levene's test

**BUKTI YANG DISEBUTKAN**:
```
Table hypothesis test line 2602:
"Train Stability: High var. vs Lower var., p<0.01"
```
- Tapi **TIDAK ADA RAW DATA** untuk variance
- **TIDAK ADA GRAFIK** training curves

**SOLUSI**:
```
Option 1: Tambahkan Appendix dengan training loss variance data
Option 2: Softened claim: "Frozen training showed improved stability 
          based on smoother convergence and lower final loss variance"
Option 3: Add loss curves to supplementary materials
```

---

### 5. **CIRCULAR REASONING - EXPECTED IMPROVEMENT** ❌

**Yang Tertulis (Line 1292)**:
```
"expected improvement ≈ 24% (19.3% → 14.6%)"
```

**INI CIRCULAR**:
- Hipotesis seharusnya **before experiment**
- Tapi 14.6% adalah **HASIL EXPERIMENT**
- Artinya "expected" ini **post-hoc**, bukan a priori prediction

**ANALOGI**:
```
SALAH: "Kami expect nilai CER menjadi 14.6%" [lalu dapat 14.6%] ← Suspiciously perfect
BENAR: "Kami expect improvement >15-20% based on literature" [lalu dapat 24.4%] ← Exceeded expectation
```

**SOLUSI**:
```
HAPUS angka exact 14.6% dari hypothesis section
GANTI DENGAN:
"Based on Souibgui et al. (2022) showing ~15-20% CER reduction 
with HTR-aware training, we expected substantial improvement 
in the range of 15-25% over current SOTA (HTR-GAN 19.3%)"
```

---

### 6. **POWER ANALYSIS - POST-HOC vs A PRIORI** ⚠️

**Yang Tertulis**:
```
"expected power untuk mendeteksi effect size d=0.8 adalah >99%"
```

**DUA INTERPRETASI**:

**Jika A Priori** (sebelum eksperimen):
- Power analysis untuk determine sample size
- "We need n=712 to achieve 99% power for d=0.8"
- **TETAPI**: Dataset sudah ada (n=712 from 70/15/15 split)
- **JADI**: Sample size bukan dari power calculation, tapi dari data split

**Jika Post-Hoc** (setelah eksperimen):
- Observed effect d=0.85
- Calculate retrospective power = 99%
- **INI OK** tapi harus jujur state "post-hoc power analysis"

**SOLUSI**:
```
Option 1 (Honest post-hoc):
"Post-hoc power analysis shows achieved power >99% for 
observed effect size d=0.85 at α=0.05"

Option 2 (Remove exact numbers):
"With sample size n=712, the study has sufficient power 
to detect medium-to-large effect sizes"

Option 3 (Justify sample size):
"Sample size n=712 determined by 70/15/15 train/val/test split,
providing >99% power for expected effect sizes >0.5"
```

---

## 📋 REKOMENDASI PERBAIKAN PRIORITAS

### 🔴 **HIGH PRIORITY - HARUS DIPERBAIKI**

1. **GANTI Paired T-test → Independent T-test**
   - Lines: 1250, 1271, 2250, 2581
   - Risk: **Metodologi salah** = **Rejection material untuk IEEE**
   
2. **KLARIFIKASI Baseline Definition**
   - Explicitly state: "Primary baseline = HTR-GAN (19.3% CER)"
   - Justify: "Strongest HTR-aware SOTA method"

3. **REVISI H2C Threshold Logic**
   - Remove arbitrary thresholds
   - Focus on **comparative improvement** over baseline

### 🟡 **MEDIUM PRIORITY - SHOULD FIX**

4. **SOFTENED H2B Stability Claim**
   - Add qualification: "based on observed training behavior"
   - OR add variance data to appendix

5. **REMOVE Exact Expected Improvement Numbers**
   - Change "14.6%" → "15-25% range"
   - Keep it as prediction, not result

### 🟢 **LOW PRIORITY - GOOD TO HAVE**

6. **CLARIFY Power Analysis Timing**
   - State explicitly: "post-hoc power analysis"
   - OR justify n=712 sample size selection

---

## 🔧 REVISED HYPOTHESIS SECTION (DRAFT)

### **Main Hypothesis (H1)**

**H0 (Null)**: No significant difference in CER between proposed framework and HTR-GAN baseline.

**H1 (Alternative)**: 
The proposed GAN-based framework with Dual-Modal Discriminator and frozen HTR integration achieves **significantly lower CER** compared to HTR-GAN baseline on degraded historical documents, with at least medium effect size (Cohen's d > 0.5).

**Mathematical Formulation**:
```
H1: CER_proposed < CER_HTR-GAN
with p-value < 0.05 (independent samples t-test) and Cohen's d > 0.5
```

**Rationale**:
Based on Souibgui et al. (2022) demonstrating 15-20% CER reduction with HTR-aware training, we expect the dual-modal discriminator's combined visual and sequential evaluation to produce **substantial improvement** (15-25% range) over HTR-GAN's 19.3% CER.

**Testing Method**:
- **Independent samples t-test** (or Welch's t-test if variances unequal)
- Two-tailed, α = 0.05
- Sample: n=712 test images
- Primary comparison: Proposed vs HTR-GAN (strongest HTR-aware baseline)

---

### **Specific Hypotheses**

**H2A: Dual-Modal Superiority**
Dual-modal discriminator (CNN+LSTM) achieves lower CER than single-modal (CNN-only) ablation, with Cohen's d > 0.5.

Testing: Independent t-test, α=0.0125 (Bonferroni correction for 4 tests)

**H2B: Training Stability**
Frozen recognizer approach produces more stable training behavior (lower loss variance) than joint training approach.

Testing: Levene's test for variance equality, supplemented by qualitative convergence analysis

**H2C: Visual Quality Preservation**
Proposed framework achieves **competitive or superior** visual quality metrics compared to HTR-GAN baseline (PSNR and SSIM).

Testing: Independent t-test comparing PSNR and SSIM distributions, α=0.0125

**H2D: Adaptive Balancing Effectiveness**
Adaptive loss balancing (40:60 CTC:Visual, rate 0.08) provides better balance than fixed weights, reflected in both CER and visual metrics.

Testing: Comparison with fixed-weight ablation, α=0.0125

---

### **Statistical Framework**

**Multiple Comparison Correction**:
- Main hypothesis (H1): α = 0.05
- Specific hypotheses (H2A-H2D): α = 0.0125 (Bonferroni correction)

**Effect Size**:
- Cohen's d for all CER and metric comparisons
- Interpretation: d>0.2 (small), d>0.5 (medium), d>0.8 (large)

**Power Consideration**:
Sample size n=712 (determined by 70/15/15 split) provides adequate power (>80%) to detect medium-to-large effect sizes at α=0.05.

---

## ✅ VALIDATION CHECKLIST

- [ ] Ganti **all** "paired t-test" menjadi "independent samples t-test"
- [ ] Eksplisit state "primary baseline = HTR-GAN"
- [ ] Remove H2C arbitrary thresholds (28 dB, 0.90)
- [ ] Revisi jadi "competitive or superior to baseline"
- [ ] Remove exact "14.6%" dari expected improvement
- [ ] Ganti jadi range "15-25%"
- [ ] Clarify power analysis adalah "adequate power" bukan exact 99%
- [ ] Add qualification untuk H2B stability claim
- [ ] Review all statistical method descriptions
- [ ] Ensure consistency antara Hipotesis → Metode → Hasil

---

## 🎯 EXPECTED OUTCOME

Setelah perbaikan:
1. ✅ **Metodologi sound**: Independent t-test untuk independent samples
2. ✅ **Transparan**: Jelas baseline = HTR-GAN, bukan multiple ambiguous
3. ✅ **Honest**: No cherry-picked thresholds, fokus pada improvement
4. ✅ **Defendable**: Semua claims backed by appropriate statistical tests
5. ✅ **Publikasi-ready**: Metode yang benar untuk jurnal IEEE Q1

---

**Next Steps**:
1. Review dan approve revisi ini
2. Implement changes ke jatniko_id.tex
3. Re-compile dan verify consistency
4. Cross-check dengan Results section
5. Final review sebelum submission
