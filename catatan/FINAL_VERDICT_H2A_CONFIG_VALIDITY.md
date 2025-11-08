# FINAL VERDICT: Validitas Config h2a_single_modal

**Tanggal:** 6 November 2025  
**Pertanyaan:** Apakah config h2a_single_modal_experiment.json salah untuk dikatakan sebagai single-modal?

---

## 🎯 JAWABAN SINGKAT: **TIDAK SALAH** untuk Arsitektur, **SALAH** untuk Training Parameters

Config h2a_single_modal menggunakan **arsitektur single-modal yang BENAR** (CNN-only, verified), TETAPI memiliki **training parameters yang TIDAK OPTIMAL** untuk fair comparison.

---

## ✅ YANG SUDAH BENAR (Architecture)

### 1. Discriminator Architecture: ✅ **VALID**

```json
"discriminator_version": "single_modal"
```

**Mapping di train_enhanced.py:**
```python
if args.discriminator_version == 'single_modal':
    discriminator = build_single_modal_discriminator_enhanced(
        img_shape=(1024, 128, 1),
        config=disc_config
    )
```

**Arsitektur yang di-load:**
- Source: `dual_modal_gan/src/models/discriminator_single_modal.py`
- Function: `build_single_modal_discriminator_enhanced()`
- Input: **Image only (1024, 128, 1)** ✅
- Layers: **CNN-only (no LSTM, no Embedding, no Text)** ✅
- Output: Single validity score ✅
- Parameters: 15.6M ✅

**Verification Passed:**
- ✅ 1 input (image only)
- ✅ 0 LSTM layers
- ✅ 0 Embedding layers
- ✅ 0 Text-related layers
- ✅ Forward pass successful dengan image-only input
- ✅ Correctly rejects dual input [image, text]

**KESIMPULAN ARSITEKTUR: BENAR SINGLE-MODAL** ✅

---

## ❌ YANG SALAH (Training Parameters)

### 1. Loss Weights: ❌ **TIDAK DIREBALANCE**

```json
// h2a_single_modal_experiment.json (CURRENT - WRONG)
{
  "pixel_loss_weight": 50.0,
  "adv_loss_weight": 3.0,
  "rec_feat_loss_weight": 8.0,   // ❌ Set tapi tidak digunakan
  "ctc_loss_weight": 0.15,        // ❌ Set tapi tidak digunakan
  "perceptual_loss_weight": 1.0
}
```

**Problem:**
- Single-modal discriminator **TIDAK menerima HTR input**
- CTC loss (0.15) dan rec_feat loss (8.0) **TIDAK digunakan** di discriminator
- Total gradient magnitude **LEBIH RENDAH** dari dual-modal
- Under-optimization akibat loss magnitude tidak comparable

**Should be:**
```json
{
  "pixel_loss_weight": 60.0,      // Increased
  "adv_loss_weight": 4.0,         // Increased
  "rec_feat_loss_weight": 0.0,    // Disabled
  "ctc_loss_weight": 0.0,         // Disabled
  "perceptual_loss_weight": 2.0   // Increased
}
```

---

### 2. Warmup/Annealing: ❌ **TIDAK PERLU**

```json
// h2a_single_modal_experiment.json (CURRENT - WRONG)
{
  "warmup_epochs": 10,      // ❌ Single-modal tidak butuh warmup
  "annealing_epochs": 20,   // ❌ Tidak ada CTC untuk di-anneal
  "curriculum_aware": true  // ❌ Tidak ada curriculum
}
```

**Problem:**
- Epoch 1-10: Warmup (wasted karena single-modal tidak butuh HTR integration)
- Epoch 11-30: Annealing CTC weight 0→0.15 (meaningless karena CTC tidak digunakan)
- Epoch 31-50: Full training (hanya 20 epoch efektif)
- **Training budget**: 20 efektif vs dual-modal 52 efektif (2.6x lebih sedikit)

**Should be:**
```json
{
  "warmup_epochs": 0,       // No warmup
  "annealing_epochs": 0,    // No annealing
  "curriculum_aware": false // Disable
}
```

---

### 3. Early Stopping Metric: ❌ **SALAH TARGET**

```json
// h2a_single_modal_experiment.json (CURRENT - WRONG)
{
  "early_stopping_metric": "combined",  // ❌ Optimize CER+PSNR
  "cer_weight": 0.2
}
```

**Problem:**
- Single-modal optimize "combined" metric (PSNR - 0.2*CER)
- Tapi CER **TIDAK di-optimize** di generator loss (no CTC loss)
- Inconsistent: monitor CER tapi tidak optimize CER

**Should be:**
```json
{
  "early_stopping_metric": "psnr",  // Visual-only optimization
  "cer_weight": 0.0                 // Not used
}
```

---

### 4. Adaptive Loss Balancing: ❌ **TIDAK PERLU**

```json
// h2a_single_modal_experiment.json (CURRENT - WRONG)
{
  "adaptive_loss_balancing": true,
  "target_ctc_ratio": 0.40,
  "target_visual_ratio": 0.60
}
```

**Problem:**
- Adaptive balancing untuk menyeimbangkan CTC vs Visual loss
- Single-modal **TIDAK ada CTC loss** untuk dibalance
- Overhead komputasi tanpa manfaat

**Should be:**
```json
{
  "adaptive_loss_balancing": false
}
```

---

### 5. Training Budget: ⚠️ **BERBEDA**

```json
// h2a_single_modal_experiment.json
{
  "epochs": 50  // ⚠️ Hanya 20 epoch efektif (setelah warmup+annealing)
}

// production_v4_optimal (dual-modal)
{
  "epochs": 100  // 52 epoch efektif (early stopped)
}
```

**For fair comparison:**
```json
{
  "epochs": 100  // Same budget
}
```

---

## 📊 Impact Analysis

### Current h2a_single_modal Results:

```
Validation PSNR: 30.63 dB
Validation CER: 27.12%
Best epoch: 44/50
```

### Comparison dengan Dual-Modal:

| Metric | Single-Modal (h2a) | Dual-Modal (v4_optimal) | Δ | Status |
|--------|-------------------|------------------------|---|--------|
| **PSNR** | 30.63 dB | 30.86 dB | +0.23 dB | ❌ Not significant |
| **CER** | 27.12% | 27.07% | -0.05% | ❌ Negligible |
| **Effective epochs** | 20 | 52 | 2.6x less | ❌ Unfair |

### ⚠️ CRITICAL IMPLICATION:

Dengan config yang **UNDER-OPTIMIZED**, single-modal masih menghasilkan:
- PSNR hanya **0.23 dB lebih rendah** (margin sangat kecil)
- CER hampir **IDENTIK**

**Artinya:**
Jika single-modal di-optimize dengan benar (loss rebalanced, no warmup/annealing, full training budget), kemungkinan besar:
- PSNR single-modal ≥ dual-modal
- CER single-modal ≈ dual-modal
- **Dual-modal contribution TIDAK TERBUKTI** ❌

---

## 🎯 FINAL VERDICT

### Pertanyaan: "Apakah config salah untuk dikatakan sebagai single-modal?"

**Jawaban:**

1. **Arsitektur:** ✅ **BENAR SINGLE-MODAL**
   - Discriminator benar-benar CNN-only
   - Tidak ada LSTM/Embedding/Text processing
   - Verified dengan audit script

2. **Training Config:** ❌ **SALAH/TIDAK OPTIMAL**
   - Loss weights tidak direbalance
   - Warmup/annealing tidak perlu
   - Training budget berbeda
   - Early stopping metric salah

3. **Istilah "Single-Modal":** ✅ **TEPAT untuk arsitektur**
   - Model **IS** single-modal (hanya visual input)
   - **BUKAN** dual-modal

4. **Validitas untuk Ablation Study:** ❌ **TIDAK VALID**
   - Comparison tidak fair (under-optimized baseline)
   - Risk: Reviewer akan reject dengan alasan "unfair comparison"

---

## 📋 Summary Table

| Aspek | Status | Keterangan |
|-------|--------|-----------|
| **Arsitektur discriminator** | ✅ BENAR | CNN-only, truly single-modal |
| **Discriminator input** | ✅ BENAR | Image only (1024, 128, 1) |
| **Discriminator layers** | ✅ BENAR | No LSTM, no Embedding, no Text |
| **Loss weights** | ❌ SALAH | Tidak direbalance, masih include CTC+rec_feat |
| **Warmup/annealing** | ❌ SALAH | Tidak perlu untuk single-modal |
| **Training budget** | ❌ SALAH | 50 epoch vs 100 epoch |
| **Early stopping** | ❌ SALAH | Combined metric vs PSNR-only |
| **Adaptive balancing** | ❌ SALAH | Tidak perlu tanpa multi-modal |
| **Fair comparison** | ❌ TIDAK | Under-optimized baseline |
| **Q1 journal validity** | ❌ TIDAK | Reviewer akan flag unfair comparison |

---

## ✅ Kesimpulan

**Config h2a_single_modal TIDAK SALAH untuk istilah "single-modal"** karena arsitekturnya memang benar-benar single-modal (CNN-only).

**TETAPI config h2a_single_modal SALAH untuk ablation study** karena training parameters tidak di-optimize untuk fair comparison dengan dual-modal.

**Rekomendasi:**
1. Gunakan config baru: `configs/ablation_single_modal_fair.json` (sudah dibuat)
2. Re-train dengan parameters yang benar
3. Atau pivot paper fokus ke frozen recognizer (avoid ablation issue)

---

**STATUS:** Config h2a_single_modal membuktikan arsitektur single-modal sudah benar, tapi training setup tidak fair untuk Q1 journal publication.
