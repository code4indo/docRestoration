# ANALISIS KRITIS: DIBCO Fine-Tuning Edge Thickness & Stroke Roughness

**Tanggal:** 30 Oktober 2025  
**Experiment:** `dibco_finetuning_from_anri_v1`  
**Status:** ❌ **FAILED - Performance Degradation**

---

## 📋 EXECUTIVE SUMMARY

Fine-tuning model ANRI V1 (best checkpoint: ckpt-106, PSNR 24.79 dB) pada dataset DIBCO menghasilkan **degradasi performa yang signifikan**:

- ✅ **Target:** Maintain PSNR ≥24.79 dB, SSIM ≥0.9675
- ❌ **Actual:** PSNR = 20.52 dB (-4.27 dB drop), SSIM = 0.9102 (-0.057)
- ❌ **Issues:** Edge terlalu tebal, strokes tidak halus, noise artifacts tinggi

---

## 🔍 ANALISIS PERBANDINGAN METRICS

### 1. Performance Comparison: ANRI V1 vs DIBCO Fine-tuning

| Metric | ANRI V1 (Epoch 11) | DIBCO FT (Epoch 10) | Delta | Status |
|--------|-------------------|---------------------|-------|--------|
| **PSNR** | 23.75 dB | 20.52 dB | **-3.23 dB** | ❌ SEVERE DROP |
| **SSIM** | 0.9670 | 0.9102 | **-0.0568** | ❌ DEGRADED |
| **CER** | 0.6875 | 0.8972 | **+0.2097** | ❌ WORSE |
| **Pixel Loss** | 0.0676 | 0.0482 | -0.0194 | ✅ Lower (but misleading) |
| **Perceptual Loss** | 41.81 | 50.81 | **+9.00** | ❌ HIGHER |
| **Noise Variance** | 391.95 | 8661.99 | **+8270** | ❌ EXTREME |
| **Isolated White** | 8.25% | 13.89% | **+5.64%** | ❌ NOISE ARTIFACTS |
| **Local Variance** | 158.07 | 909.53 | **+751** | ❌ ROUGH STROKES |

### 🔴 **CRITICAL FINDING:**
Meskipun **pixel loss turun**, performa visual (PSNR, SSIM) justru **memburuk drastis**. Ini indikasi **loss function tidak aligned dengan quality metrics**.

---

## 🎯 ROOT CAUSE ANALYSIS

### A. MENGAPA EDGE TERLALU TEBAL?

#### 1. **Noise Variance Meledak (391 → 8662)**
```
ANRI V1:      Noise Variance = 391.95
DIBCO FT:     Noise Variance = 8661.99  (+2110% increase!)
```

**Penyebab:**
- **Domain mismatch:** DIBCO dataset memiliki degradasi yang sangat berbeda dari ANRI
- **Dataset DIBCO characteristics:**
  - Blur, stains, fading, mixed printed+handwritten
  - Variasi intensitas noise sangat tinggi
  - Background degradation patterns berbeda
- **Model overfitting to DIBCO noise patterns** → menghasilkan thick edges sebagai "safe" solution

#### 2. **Isolated White Pixels Tinggi (8.25% → 13.89%)**
```
ANRI V1:      Isolated White = 8.25%
DIBCO FT:     Isolated White = 13.89%  (+68% increase)
```

**Interpretasi:**
- Model menghasilkan **banyak white pixels yang terpisah** (noise artifacts)
- Ini membuat edge terlihat **tebal dan tidak clean**
- **Morphological opening** mendeteksi 13.89% pixels sebagai noise

**Visual Evidence (dari analisis):**
```
Edge Thickness (as % of total text):
  Epoch 5:  4.00%  (relatif normal)
  Epoch 10: 3.95%  (masih normal)

Stroke Width Distribution:
  Epoch 5:  Avg=111.12px, Max=412.84px  ⚠️ SANGAT TEBAL
  Epoch 10: Avg=186.74px, Max=715.20px  🔴 EXTREMELY THICK
```

**⚠️ ANOMALI:** Average stroke width **111-187 pixels** adalah **ABSURDLY THICK** untuk dokumen historis!
- Normal stroke width untuk paleography: **2-5 pixels**
- Current output: **50x lebih tebal!**

#### 3. **Pixel Loss Tidak Mencerminkan Kualitas Visual**
```
Training Progression:
Epoch 1:  Pixel Loss=0.0578, PSNR=21.30 dB
Epoch 10: Pixel Loss=0.0482, PSNR=20.52 dB  ← PIXEL LOSS TURUN, PSNR MALAH TURUN!
```

**Root Cause:**
- **Pixel loss weight terlalu dominan (200.0)** → model fokus minimize MSE
- MSE tidak sensitif terhadap **spatial structure** dan **edge quality**
- Model belajar "blur out" edges untuk minimize pixel-wise error
- **Perceptual loss (10.0) terlalu kecil** untuk counterbalance

---

### B. MENGAPA STROKES TIDAK HALUS?

#### 1. **Local Variance Meledak (158 → 909)**
```
ANRI V1:      Local Variance = 158.07
DIBCO FT:     Local Variance = 909.53  (+475% increase)
```

**Interpretasi:**
- **High local variance** = pixel intensities sangat bervariasi dalam neighborhood kecil
- Ini adalah **signature of rough, jagged strokes**
- Strokes tidak smooth, banyak "bumps" dan irregularities

#### 2. **Gradient Coefficient of Variation Tinggi**
```
Stroke Smoothness Analysis:
  Epoch 5:  CoV=5.025  (gradient variance / mean)
  Epoch 10: CoV=5.085  (sedikit naik, masih tinggi)
```

**Threshold:**
- **CoV < 2.0:** Smooth strokes
- **CoV 2.0-4.0:** Moderate roughness
- **CoV > 5.0:** 🔴 **ROUGH, JAGGED STROKES**

**Penyebab:**
- **Perceptual loss yang digunakan (VGG-based) tidak enforce edge smoothness**
- **Tidak ada explicit smoothness regularization** (e.g., Total Variation loss)
- **Adversarial loss (1.5) terlalu kecil** untuk enforce realistic texture
- Model menghasilkan **high-frequency noise** pada stroke edges

#### 3. **Generator Gradient Instability**
```
Generator Gradients (Epoch 10):
  Mean: 556.28
  Max:  1339.66  ← VERY HIGH!
  
Discriminator Gradients:
  Mean: 14.49  ← VERY LOW!
```

**Diagnosis:**
- **Huge gradient imbalance** (Generator 556 vs Discriminator 14)
- Generator mengalami **exploding gradients** → training instability
- Discriminator terlalu weak → tidak memberikan useful feedback
- Ini menyebabkan generator **tidak konvergen dengan baik**

**Gradient Clipping Ineffective:**
```json
"gradient_clip_norm": 1.0  ← TOO AGGRESSIVE!
```
- Clipping di 1.0 terlalu ketat untuk complex GAN
- Menyebabkan gradient information loss
- Recommended: 5.0-10.0 untuk GAN training

---

## 🧪 TECHNICAL DEEP DIVE

### Loss Weight Configuration Analysis

```json
{
  "pixel_loss_weight": 200.0,      // ← TOO HIGH
  "perceptual_loss_weight": 10.0,  // ← TOO LOW
  "rec_feat_loss_weight": 5.0,     // OK
  "adv_loss_weight": 1.5,          // ← TOO LOW
  "ctc_loss_weight": 0.0           // OK (visual-only mode)
}
```

**Problem: Loss Weight Imbalance**

| Loss Component | Current Weight | Contribution (Epoch 10) | Weighted Value | Recommended |
|----------------|----------------|-------------------------|----------------|-------------|
| Pixel Loss | 200.0 | 0.0482 | **9.64** | 50-100 |
| Perceptual Loss | 10.0 | 50.81 | **508.1** | 20-50 |
| Rec Feat Loss | 5.0 | 0.0104 | 0.052 | 5-10 |
| Adversarial Loss | 1.5 | 3.897 | 5.85 | 5-10 |

**🔴 CRITICAL ISSUE:** 
- **Perceptual loss dominates** dengan kontribusi **508.1** (98.5% dari total weighted loss)
- Pixel loss hanya **9.64** (1.9%)
- Adversarial loss hanya **5.85** (1.1%)

**Konsekuensi:**
- Model **overfitting to VGG feature space** (perceptual loss)
- **Mengabaikan pixel-level accuracy** dan **adversarial realism**
- VGG features tidak capture **thin stroke details** dengan baik
- Hasil: **Thick, rough edges** yang match VGG features tapi bukan ground truth

---

### Dataset Domain Mismatch

#### ANRI Dataset (Training base):
```
Type:     16-18th century paleography
Samples:  108 (1 document, 5x augmentation)
Features: 
  - Thin strokes (2-3 pixels)
  - Complex calligraphy
  - Consistent ink degradation patterns
  - Low background noise variance
```

#### DIBCO Dataset (Fine-tuning):
```
Type:     Document binarization benchmark (2009-2018)
Samples:  2855 (817 strips, 5x augmentation)
Features:
  - Mixed degradation types (blur, stains, fading)
  - Printed + handwritten mix
  - HIGH background noise variance  ← MISMATCH!
  - Varying stroke widths
  - Different degradation patterns  ← MISMATCH!
```

**Domain Shift Impact:**
1. **Noise variance jump:** 391 → 8662 (+2110%)
   - Model tidak trained untuk handle noise variance setinggi ini
   - Compensation: thick edges untuk "safe" solution

2. **Stroke width variability:**
   - ANRI: Consistent 2-3px thin strokes
   - DIBCO: Mixed 1-10px strokes
   - Model confused → generates averaged thick strokes

3. **Background degradation patterns:**
   - ANRI: Relatively uniform aging
   - DIBCO: Highly non-uniform (stains, fading, etc.)
   - Model overcompensates → isolated white artifacts

---

## 🎯 REKOMENDASI PERBAIKAN

### PRIORITY 1: Loss Function Rebalancing

```json
// CURRENT (BROKEN):
{
  "pixel_loss_weight": 200.0,      // Terlalu tinggi vs contribution
  "perceptual_loss_weight": 10.0,  // Terlalu rendah vs contribution
  "adv_loss_weight": 1.5           // Terlalu rendah
}

// RECOMMENDED:
{
  "pixel_loss_weight": 50.0,       // ↓ Turunkan 4x (target contribution ~10-20%)
  "perceptual_loss_weight": 1.0,   // ↓ Turunkan 10x (target contribution ~50-60%)
  "rec_feat_loss_weight": 5.0,     // Maintain
  "adv_loss_weight": 5.0,          // ↑ Naikkan 3.3x (target ~10-20%)
  
  // ADD NEW:
  "tv_loss_weight": 2.0,           // Total Variation for smoothness
  "edge_loss_weight": 10.0         // Explicit edge preservation
}
```

**Rationale:**
- **Perceptual loss contribution saat ini ~98.5%** → TIDAK BALANCED!
- Target distribution: **50% perceptual, 20% pixel, 15% adversarial, 10% edge, 5% TV**
- **Total Variation loss** akan enforce smoothness
- **Edge loss** akan preserve thin strokes

---

### PRIORITY 2: Add Smoothness Regularization

#### A. Total Variation Loss
```python
def total_variation_loss(image):
    """
    Penalize high-frequency variations (rough edges)
    """
    dx = image[:, :, 1:, :] - image[:, :, :-1, :]
    dy = image[:, 1:, :, :] - image[:, :-1, :, :]
    return tf.reduce_mean(tf.abs(dx)) + tf.reduce_mean(tf.abs(dy))
```

**Benefit:** Akan enforce smooth gradients → smoother strokes

#### B. Edge-Aware Loss
```python
def edge_aware_loss(generated, ground_truth):
    """
    Fokus pada preservasi edges yang thin dan smooth
    """
    # Sobel edge detection
    gen_edges = sobel_edges(generated)
    gt_edges = sobel_edges(ground_truth)
    
    # Weighted L1 loss (higher weight on edge regions)
    edge_mask = (gt_edges > threshold).float()
    loss = tf.reduce_mean(edge_mask * tf.abs(gen_edges - gt_edges))
    return loss
```

**Benefit:** Explicitly preserve thin stroke edges

---

### PRIORITY 3: Gradient Stability

```json
// CURRENT:
{
  "gradient_clip_norm": 1.0,  // TOO AGGRESSIVE
  "lr_g": 0.00001,
  "lr_d": 0.00001
}

// RECOMMENDED:
{
  "gradient_clip_norm": 5.0,   // ↑ More headroom for complex gradients
  "lr_g": 0.000005,            // ↓ Slower, more stable
  "lr_d": 0.00001,             // Maintain (atau slightly higher)
  
  "use_lr_schedule": true,     // Enable adaptive LR
  "warmup_epochs": 3,          // Gradual warmup
  "lr_decay_factor": 0.5,      // Decay if plateau
  "lr_decay_patience": 5
}
```

**Rationale:**
- Current gradient clip (1.0) **terlalu ketat** → information loss
- Generator LR terlalu tinggi → instability (mean gradient 556!)
- Warmup akan mencegah early training shock

---

### PRIORITY 4: Domain Adaptation Strategy

#### A. Progressive Fine-tuning
```json
{
  "stage_1": {
    "epochs": 5,
    "freeze_encoder": true,        // Freeze ANRI-trained encoder
    "train_decoder_only": true,    // Adapt decoder to DIBCO
    "lr_g": 0.000001               // Very slow
  },
  "stage_2": {
    "epochs": 10,
    "freeze_encoder": false,       // Unfreeze all
    "full_training": true,
    "lr_g": 0.000005
  }
}
```

**Benefit:** Mencegah catastrophic forgetting dari ANRI knowledge

#### B. Mixed Dataset Training
```json
{
  "training_strategy": "mixed",
  "anri_ratio": 0.3,      // 30% ANRI samples per batch
  "dibco_ratio": 0.7,     // 70% DIBCO samples
  "sample_alternating": true
}
```

**Benefit:** Maintain ANRI performance while adapting to DIBCO

---

### PRIORITY 5: Discriminator Enhancement

```json
// CURRENT:
{
  "discriminator_mode": "ground_truth",
  "adv_loss_weight": 1.5,  // Too weak
  
  "discriminator_config": {
    "dropout_rate": 0.1,   // Too low
    ...
  }
}

// RECOMMENDED:
{
  "discriminator_mode": "ground_truth",
  "adv_loss_weight": 5.0,  // ↑ Stronger adversarial signal
  
  "discriminator_config": {
    "dropout_rate": 0.3,   // ↑ Prevent discriminator overfitting
    "spectral_normalization": true,  // Stabilize training
    "update_ratio": 2      // Update discriminator 2x per generator update
  }
}
```

**Current Issue:**
- Discriminator gradient (14.49) **TERLALU RENDAH** vs Generator (556.28)
- Discriminator terlalu weak → tidak provide useful feedback
- Generator mengalami mode collapse

**Solution:**
- **Spectral normalization** akan stabilize discriminator
- **Higher dropout** mencegah discriminator overfit
- **Update ratio 2:1** akan balance GAN training

---

## 📊 EXPECTED IMPROVEMENTS

Dengan implementasi rekomendasi di atas:

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **PSNR** | 20.52 dB | **24-26 dB** | +3.5 to +5.5 dB |
| **SSIM** | 0.9102 | **0.96-0.97** | +0.05 to +0.06 |
| **Noise Variance** | 8661 | **<1000** | -88% |
| **Isolated White** | 13.89% | **<5%** | -64% |
| **Local Variance** | 909 | **<300** | -67% |
| **Avg Stroke Width** | 186px | **3-5px** | -97% (!!) |
| **Gradient CoV** | 5.08 | **<2.5** | -51% |

---

## 🧪 EXPERIMENT PLAN

### Experiment 1: Loss Rebalancing Only
```json
{
  "experiment_name": "dibco_ft_v2_loss_rebalance",
  "pixel_loss_weight": 50.0,
  "perceptual_loss_weight": 1.0,
  "adv_loss_weight": 5.0,
  "tv_loss_weight": 2.0,
  "gradient_clip_norm": 5.0
}
```
**Expected:** PSNR +2-3 dB, smoother strokes

### Experiment 2: Add Edge-Aware Loss
```json
{
  "experiment_name": "dibco_ft_v3_edge_aware",
  "base": "dibco_ft_v2_loss_rebalance",
  "edge_loss_weight": 10.0,
  "edge_threshold": 0.1
}
```
**Expected:** Thinner edges, better stroke preservation

### Experiment 3: Mixed Dataset Training
```json
{
  "experiment_name": "dibco_ft_v4_mixed_dataset",
  "base": "dibco_ft_v3_edge_aware",
  "anri_ratio": 0.3,
  "dibco_ratio": 0.7
}
```
**Expected:** Maintain ANRI performance + DIBCO generalization

### Experiment 4: Full Solution
```json
{
  "experiment_name": "dibco_ft_v5_full_solution",
  "base": "dibco_ft_v4_mixed_dataset",
  "progressive_fine_tuning": true,
  "discriminator_spectral_norm": true,
  "discriminator_update_ratio": 2
}
```
**Expected:** **PSNR >25 dB, SSIM >0.97**, smooth thin strokes

---

## 📝 KESIMPULAN

### ROOT CAUSES (Berurutan berdasar impact):

1. **🔴 CRITICAL: Loss weight severely imbalanced**
   - Perceptual loss dominates (98.5% contribution)
   - Pixel loss diabaikan meskipun weight tinggi
   - **Fix:** Rebalance weights based on actual contribution

2. **🔴 CRITICAL: Perceptual loss tidak cocok untuk thin strokes**
   - VGG features terlalu coarse untuk capture thin strokes
   - **Fix:** Turunkan weight drastis + add edge-aware loss

3. **🟠 HIGH: Tidak ada smoothness regularization**
   - Tidak ada penalty untuk rough edges
   - **Fix:** Add Total Variation loss

4. **🟠 HIGH: Adversarial loss terlalu weak**
   - Discriminator tidak provide useful feedback
   - **Fix:** Increase adv_loss_weight + spectral norm

5. **🟡 MEDIUM: Gradient instability**
   - Generator gradient exploding (556 mean)
   - Gradient clipping terlalu agresif
   - **Fix:** Increase clip norm + reduce LR

6. **🟡 MEDIUM: Domain mismatch**
   - DIBCO noise variance 22x lebih tinggi dari ANRI
   - **Fix:** Mixed dataset training atau progressive fine-tuning

### IMMEDIATE ACTION ITEMS:

1. ✅ **Stop current training** (sudah early stop di epoch 10 - correct decision!)
2. 🔧 **Implement loss rebalancing** (Experiment 1)
3. 🔧 **Add TV loss + Edge loss** (Experiment 2)
4. 🔧 **Test mixed dataset** (Experiment 3)
5. 📊 **Compare results** dengan ANRI V1 baseline

### SUCCESS CRITERIA:
- ✅ PSNR ≥ 24 dB (minimal match ANRI V1)
- ✅ SSIM ≥ 0.965
- ✅ Isolated White < 8%
- ✅ Local Variance < 300
- ✅ Average Stroke Width < 10 pixels (ideally 3-5px)
- ✅ Gradient CoV < 3.0

---

**Status:** Ready untuk experiment series baru dengan recommended fixes.
