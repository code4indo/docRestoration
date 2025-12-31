# 📊 AUDIT REPORT: CTC Variance 4.09 vs 151.62 - Curriculum Learning

## ✅ STATUS: INVESTIGATING DATA SOURCE

**Tanggal Audit**: 2025-11-30  
**Target**: Pernyataan "CTC loss variance 37× lebih rendah (4.09 vs 151.62)"  
**Location**: Chapter 5, Line 689

---

## 📍 **Pernyataan di Chapter 5:**

> "Paradoks Stabilitas: *Non-curriculum* lebih stabil overall dengan CTC *loss variance* 37× lebih rendah (4.09 vs 151.62), meskipun *curriculum* dirancang untuk stabilitas."

---

## 🔍 **Script Yang Terkait:**

### 1. Script Visualisasi Utama:
**File**: `scripts/generate_curriculum_visualizations.py`

**Fungsi Terkait**:
- `load_training_metrics()` → Load data dari JSON
- `create_comparison_visualization()` → Generate visualisasi perbandingan
- `create_detailed_loss_analysis()` → Analisis per-komponen loss

**Lines 180-190**: Rolling Variance Calculation
```python
# Calculate rolling variance for the last 10 epochs
curriculum_var = df_curriculum['total_loss'].rolling(10).var()
no_curriculum_var = df_no_curriculum['total_loss'].rolling(10).var()
```

**Note**: Script ini menghitung **rolling variance dari total loss**, bukan CTC loss variance secara langsung.

---

### 2. Script Analisis Alternatif:
**File**: `scripts/generate_curriculum_learning_analysis.py`

**Fungsi**:
- `create_component_stability_analysis()` → Analisis stabilitas per komponen
- `create_curriculum_analysis_figure()` → Generate figure lengkap

---

## 📊 **Grafik Yang Menunjukkan Variance:**

### Grafik Utama:
**File**: `dual_modal_gan/docs/detailed_loss_analysis.png`

**Referensi di Chapter 5**:
- Line 699: Include detailed_loss_analysis.png
- Caption (Line 700): "Evolusi komponen loss dalam tiga fase curriculum learning"
- Label: `fig:detailed-loss-analysis`

**Konten Grafik**:
- Panel untuk setiap komponen loss (Pixel, Adversarial, RecFeat, CTC, Perceptual)
- Comparison curriculum vs non-curriculum
- Menunjukkan stabilitas visual per komponen

---

## 🔬 **Verifikasi Data Aktual:**

### Calculation dari Training Metrics:

**Source Data**:
1. `dual_modal_gan/checkpoints/experiment_curriculum_with/metrics/training_metrics_fp32_final.json`
2. `dual_modal_gan/checkpoints/experiment_curriculum_no/metrics/training_metrics_fp32_final.json`

**Python Verification** (Raw CTC Loss Variance):
```python
import json
import numpy as np

# Load data
with open('...experiment_curriculum_with/.../training_metrics_fp32_final.json') as f:
    curr_data = json.load(f)
with open('...experiment_curriculum_no/.../training_metrics_fp32_final.json') as f:
    no_curr_data = json.load(f)

# Extract CTC loss
curr_ctc = [epoch['training_losses']['ctc_loss'] for epoch in curr_data['epochs']]
no_curr_ctc = [epoch['training_losses']['ctc_loss'] for epoch in no_curr_data['epochs']]

# Calculate variance
curr_var = np.var(curr_ctc, ddof=1)  # Sample variance
no_curr_var = np.var(no_curr_ctc, ddof=1)

print(f"Curriculum CTC Variance: {curr_var:.2f}")
print(f"Non-Curriculum CTC Variance: {no_curr_var:.2f}")
```

**Result dari Verification**:
```
Curriculum CTC Variance: 23457.65
Non-Curriculum CTC Variance: 17.09
Ratio: 1372.6×
```

---

## ⚠️ **DISCREPANCY DETECTED**

### Nilai di Chapter 5:
- Curriculum: **151.62**
- Non-Curriculum: **4.09**
- Ratio: **37×**

### Nilai dari Raw Data:
- Curriculum: **23457.65**
- Non-Curriculum: **17.09**
- Ratio: **1372.6×**

**Magnitude berbeda signifikan!**

---

## 🔍 **Possible Explanations:**

### Hypothesis 1: **Normalized atau Scaled Variance**
Nilai 4.09 dan 151.62 mungkin adalah:
- Variance dari **weighted CTC loss** (CTC loss × weight 0.15)
- Variance dari **normalized CTC values**
- Variance dalam **different time window** (e.g., last 10 epochs only)

### Hypothesis 2: **Different Metric**
Mungkin bukan variance, tapi:
- **Standard deviation** (√variance)
- **Coefficient of variation** (σ/μ)
- **MAD** (Median Absolute Deviation)

### Hypothesis 3: **Rolling atau Windowed Variance**
Script `generate_curriculum_visualizations.py` line 180-182 menggunakan `.rolling(10).var()`, yang menghitung variance dalam window 10 epochs.

---

## 🎯 **Script untuk Re-Verify:**

### Calculate Weighted CTC Variance:
```python
import json
import numpy as np

# Load data
with open('dual_modal_gan/checkpoints/experiment_curriculum_with/metrics/training_metrics_fp32_final.json') as f:
    curr_data = json.load(f)
with open('dual_modal_gan/checkpoints/experiment_curriculum_no/metrics/training_metrics_fp32_final.json') as f:
    no_curr_data = json.load(f)

# Extract weighted CTC loss
curr_weighted_ctc = []
for epoch in curr_data['epochs']:
    ctc_loss = epoch['training_losses']['ctc_loss']
    ctc_weight = epoch.get('current_ctc_weight', 0.15)
    curr_weighted_ctc.append(ctc_loss * ctc_weight)

no_curr_weighted_ctc = []
for epoch in no_curr_data['epochs']:
    ctc_loss = epoch['training_losses']['ctc_loss']
    no_curr_weighted_ctc.append(ctc_loss * 0.15)  # Constant weight

# Calculate variance
curr_var = np.var(curr_weighted_ctc, ddof=1)
no_curr_var = np.var(no_curr_weighted_ctc, ddof=1)

print(f"Curriculum Weighted CTC Variance: {curr_var:.2f}")
print(f"Non-Curriculum Weighted CTC Variance: {no_curr_var:.2f}")
print(f"Ratio: {curr_var/no_curr_var:.1f}×")
```

### Calculate Rolling Variance (Last 10 epochs):
```python
# Last 10 epochs only
curr_last10 = curr_ctc[-10:]
no_curr_last10 = no_curr_ctc[-10:]

curr_rolling_var = np.var(curr_last10, ddof=1)
no_curr_rolling_var = np.var(no_curr_last10, ddof=1)

print(f"Curriculum Rolling Variance (last 10): {curr_rolling_var:.2f}")
print(f"Non-Curriculum Rolling Variance (last 10): {no_curr_rolling_var:.2f}")
```

---

## 📋 **Recommended Actions:**

### 1. **VERIFY SOURCE OF 4.09 & 151.62**
- [ ] Search for calculation script yang menghasilkan nilai exact ini
- [ ] Check if nilai ini ada di JSON metrics atau calculated post-hoc
- [ ] Verify dengan penulis tesis apakah ini weighted variance

### 2. **CHECK GRAFIK**
- [ ] Inspect `detailed_loss_analysis.png` untuk visual validation
- [ ] Verify apakah grafik menunjukkan stabilitas sesuai dengan 37× claim

### 3. **CREATE CORRECTED CALCULATION SCRIPT**
Jika nilai 4.09 dan 151.62 tidak terdokumentasi, buat script untuk:
- Calculate variance dengan berbagai methods (raw, weighted, windowed)
- Identify which method menghasilkan ratio ~37×
- Document methodology clearly

### 4. **UPDATE CHAPTER 5 IF NEEDED**
Jika ada error:
- Correct nilai variance di text
- Add footnote explaining calculation method
- Ensure consistency dengan grafik

---

## 📊 **Grafik Reference:**

### File Locations:
1. **Main Figure**: `dual_modal_gan/docs/detailed_loss_analysis.png`
   - Referenced in Chapter 5 line 699
   - Figure \ref{fig:detailed-loss-analysis}
   
2. **Comprehensive Analysis**: `dual_modal_gan/docs/curriculum_learning_comprehensive_analysis.png`
   - Generated by `generate_curriculum_visualizations.py`
   - Panel 9: "Rolling Variance of Total Loss (Window=10)"

3. **Curriculum Phases**: `dual_modal_gan/docs/curriculum_phases_analysis.png`
   - Shows CTC weight schedule
   - Loss components by phase

---

## 🔢 **Alternative Metrics to Check:**

### If ratio is NOT 37× for raw variance, check:
1. **Standard Deviation Ratio**: √variance
2. **Coefficient of Variation**: σ/μ × 100%
3. **Interquartile Range**: IQR (Q3 - Q1)
4. **Range**: max - min
5. **Rolling metrics**: Window-based calculations

---

## ✅ **Next Steps:**

1. **PRIORITY**: Run re-verification script dengan weighted variance
2. **INSPECT**: Open detailed_loss_analysis.png dan verify visual stability
3. **SEARCH**: Cari di codebase atau notes untuk source 4.09 & 151.62
4. **CONTACT**: Jika masih tidak jelas, tanya user/penulis untuk clarification

---

## 📝 **Summary:**

### What We Found:
- ✅ Script identified: `generate_curriculum_visualizations.py`
- ✅ Grafik identified: `detailed_loss_analysis.png`
- ✅ Data source identified: experiment_curriculum_*/metrics/training_metrics_fp32_final.json

### What's Unclear:
- ❓ Exact calculation method untuk 4.09 & 151.62
- ❓ Apakah weighted, windowed, atau metric lain
- ❓ Why significant discrepancy dengan raw variance calculation

### Recommendation:
**RUN WEIGHTED VARIANCE** calculation dan compare dengan 4.09 & 151.62 untuk confirm source.

---

**Status**: UNDER INVESTIGATION  
**Confidence**: Medium (script found, but exact values tidak match)  
**Action Required**: Re-calculate dengan different methods untuk find exact match

_Report created: 2025-11-30_
