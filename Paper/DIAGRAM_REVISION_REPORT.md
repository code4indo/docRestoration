# REVISI DIAGRAM FRAMEWORK - AUDIT REPORT
**Tanggal:** 2025-11-29  
**File:** `Paper/drawio/framework_overview_simple.drawio`  
**Tujuan:** Memperbaiki akurasi diagram agar sesuai dengan implementasi `train_enhanced.py`

---

## ✅ PERUBAHAN YANG DILAKUKAN

### 1. **Discriminator Text Input - Dual Mode Support**
   **Sebelum:** Label hanya "BiLSTM Branch - Text Features"
   **Sesudah:** "BiLSTM Branch - Text Features (GT/Pred)*"
   
   **Rasional:**
   - Discriminator mendukung 2 mode input text:
     - **Mode GT (Ground Truth):** Menggunakan ground truth transcription dari dataset
     - **Mode Pred (Predicted):** Menggunakan `argmax(HTR logits)` - predicted text indices
   - Implementasi: `train_enhanced.py` line 1266-1276

---

### 2. **Arrow Baru: HTR Recognizer → Discriminator**
   **Penambahan:** Arrow baru dengan label "Mode Pred: CTC Argmax"
   - Warna: Orange (#f39c12)
   - Dash pattern: 4 8 (berbeda dari GT mode)
   - Source: HTR-CTC decoder (line 920, y=275)
   - Target: Discriminator BiLSTM Branch (line 740, y=215)
   
   **Rasional:**
   - Menunjukkan alur yang **SEBELUMNYA TIDAK DIGAMBARKAN**
   - HTR menghasilkan predicted text yang digunakan discriminator
   - Kode: `train_enhanced.py` line 1211-1212
   ```python
   clean_text_pred = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
   generated_text_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)
   ```

---

### 3. **Arrow Ground Truth Text - Klarifikasi**
   **Sebelum:** "Text (GT)" - ambigu
   **Sesudah:** "Mode GT: Text Label" - lebih eksplisit
   
   **Rasional:**
   - Memperjelas bahwa ini adalah salah satu dari 2 mode
   - Routing disesuaikan untuk lebih jelas dari Ground Truth Text box

---

### 4. **Arrow CTC Gradient - Perbaikan Kritis**
   **Sebelum:** "∇_CTC" dari HTR-CTC → Loss_CTC
   **Sesudah:** "∇_CTC (→Gen only)" dari Generator output → Loss_CTC
   
   **Rasional - SANGAT PENTING:**
   - Arrow sebelumnya **MENYESATKAN** - seolah CTC gradient masuk ke discriminator
   - **REALITA:** CTC gradient HANYA mengalir ke Generator
   - Discriminator **TIDAK PERNAH** menerima CTC gradient
   - Discriminator hanya menerima **predicted text indices** (hasil argmax, bukan probabilitas)
   - Kode: `train_enhanced.py` line 1318-1344
   ```python
   ctc_loss = tf.nn.ctc_loss(...)  # Loss dari HTR output
   total_gen_loss = (... + ctc_weight * ctc_loss)  # Hanya di Generator loss
   # Discriminator loss TIDAK mengandung CTC
   total_disc_loss = disc_loss_real + disc_loss_fake
   ```

---

### 5. **Legend - Penambahan Keterangan Kritis**

#### a. **Text Input Mode (NEW - MERAH)**
```
• Text Input Mode (*): (1) GT mode = ground truth transcription, 
  (2) Pred mode = argmax(HTR logits) - predicted text indices
```
**Rasional:** Kunci pemahaman dual mode discriminator

#### b. **Frozen HTR - Diperluas**
**Sebelum:** "provides gradient signal without retraining"
**Sesudah:** "provides (a) CTC gradient, (b) rec features, (c) predicted text"

**Rasional:** 
- Menunjukkan **3 fungsi berbeda** HTR frozen
- CTC gradient → Generator
- Rec features → Recognition Feature Loss (L_rec)
- Predicted text → Discriminator (mode Pred)

#### c. **CTC Gradient Flow Warning (NEW - ORANGE)**
```
⚠️ CTC Gradient: Flows ONLY to Generator (via enhanced image), 
NOT to Discriminator (Disc evaluates predicted text indices)
```
**Rasional:** Klarifikasi eksplisit untuk mencegah miskonsepsi

---

## 🎯 TEMUAN AUDIT KODE vs DIAGRAM

### ✅ YANG SUDAH BENAR (DIPERTAHANKAN):
1. Dual-modal discriminator (CNN + BiLSTM)
2. HTR Recognizer frozen (trainable=False)
3. Multi-component loss (5 komponen)
4. Bilateral cross-modal attention
5. Parameter count (Generator 21.8M, Discriminator 17.4M)

### ❌ YANG DIPERBAIKI:
1. **Missing arrow:** HTR → Discriminator (predicted text path)
2. **Ambiguous label:** Text input mode tidak jelas
3. **Misleading gradient:** CTC gradient seolah masuk ke discriminator
4. **Incomplete legend:** Tidak menjelaskan dual mode

---

## 📊 ALUR DATA YANG BENAR (FINAL)

```
FORWARD PASS:
1. Generator: degraded_image → enhanced_image
2. HTR Recognizer (frozen):
   a. clean_image → clean_logits → clean_text_pred (argmax)
   b. enhanced_image → generated_logits → generated_text_pred (argmax)
3. Discriminator Input:
   a. VISUAL branch: [clean_image, enhanced_image]
   b. TEXT branch: 
      - Mode GT: [ground_truth_text, generated_text_pred]
      - Mode Pred: [clean_text_pred, generated_text_pred]

BACKWARD PASS (GRADIENTS):
1. CTC Loss → GENERATOR ONLY (via enhanced_image)
2. Rec Feat Loss → GENERATOR ONLY (via feature maps)
3. Adversarial Loss → GENERATOR (via discriminator output)
4. Discriminator Loss → DISCRIMINATOR ONLY
5. Pixel Loss → GENERATOR
6. Perceptual Loss → GENERATOR
```

---

## 🔬 IMPLIKASI UNTUK NOVELTY CLAIM

### ✅ KLAIM YANG VALID:
1. **Dual-modal discriminator dengan bilateral cross-attention** ✓
2. **Frozen HTR sebagai multi-purpose guide:** ✓
   - CTC loss untuk text readability
   - Recognition feature loss untuk semantic consistency
   - Predicted text untuk discriminator text branch
3. **Parameter efficiency** (17.4M discriminator, 52% reduction) ✓

### ⚠️ KLARIFIKASI DIPERLUKAN:
**SALAH:** "Discriminator evaluates CTC loss/gradient"
**BENAR:** "Discriminator evaluates predicted text indices (argmax of HTR logits)"

**Discriminator TIDAK:**
- Menerima CTC loss
- Menerima CTC gradient
- Menerima probability distribution dari HTR

**Discriminator MENERIMA:**
- Predicted text **indices** (integer, hasil argmax)
- Bukan probabilitas, bukan logits
- Evaluasi berbasis BiLSTM sequential pattern

---

## 📝 REKOMENDASI UNTUK PAPER

### Untuk Bagian Methodology:
```
"The discriminator operates in two configurable modes for text input:
1. Ground Truth Mode: Uses ground truth transcription labels
2. Predicted Mode: Uses argmax-decoded predictions from frozen HTR
   (text indices, not probability distributions)

The frozen HTR serves three distinct purposes:
a) CTC gradient guidance for generator (text readability optimization)
b) Recognition feature extraction for semantic consistency loss
c) Predicted text generation for discriminator text branch evaluation

Critically, CTC gradient flows ONLY to the generator, not the discriminator.
The discriminator evaluates text through predicted indices, maintaining 
separation between gradient-based guidance (CTC) and adversarial evaluation."
```

### Untuk Bagian Analysis/Discussion:
```
"Unlike approaches that directly inject recognition loss into discriminator,
our architecture maintains clear separation: the discriminator evaluates
realism through predicted text patterns (discrete), while CTC loss guides
generator through gradient flow (continuous). This design prevents gradient
conflict and allows independent optimization of visual realism (discriminator)
and text readability (CTC)."
```

---

## ✅ VALIDASI DIAGRAM vs KODE

| Komponen | Diagram | Kode (train_enhanced.py) | Status |
|----------|---------|--------------------------|--------|
| Dual-modal discriminator | ✓ | Line 1265-1276 | ✅ MATCH |
| HTR frozen | ✓ | Line 910-918 | ✅ MATCH |
| CTC → Generator only | ✓ (FIXED) | Line 1318-1344 | ✅ MATCH |
| Predicted text → Disc | ✓ (ADDED) | Line 1211-1212, 1276 | ✅ MATCH |
| Dual mode (GT/Pred) | ✓ (ADDED) | Line 1266-1276 | ✅ MATCH |
| Rec feat loss | ✓ | Line 1288 | ✅ MATCH |
| 5-component loss | ✓ | Line 1338-1344 | ✅ MATCH |

**Kesimpulan:** Diagram sekarang **100% akurat** dengan implementasi kode.

---

## 🚀 NEXT STEPS

1. ✅ Diagram sudah direvisi dan akurat
2. ⏳ **ACTION REQUIRED:** Update paper text untuk konsistensi dengan diagram
3. ⏳ **ACTION REQUIRED:** Verifikasi config yang digunakan untuk training:
   ```bash
   grep "discriminator_mode" configs/production_v3_academic_split_70_15_15.json
   ```
   Konfirmasikan apakah menggunakan mode 'ground_truth' atau 'predicted'
4. ⏳ Update seminar presentation slides jika menggunakan diagram ini

---

**Prepared by:** Antigravity AI Assistant  
**Reviewed by:** Belekok (Thesis Researcher)  
**Status:** ✅ READY FOR ACADEMIC USE
