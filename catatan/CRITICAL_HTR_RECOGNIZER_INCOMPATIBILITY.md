# CRITICAL: HTR Recognizer Incompatibility Issue

**Date:** 2025-10-20  
**Status:** 🔴 **CRITICAL BUG FOUND**  
**Impact:** Recognizer cannot recognize clean images correctly

## Executive Summary

Test HTR recognizer pada clean images menunjukkan **CRITICAL INCOMPATIBILITY**:
- **Average CER: 10.28** (expected: ~0.33) → **30x lebih buruk!**
- **Perfect matches: 0/50 (0.00%)**
- **Prediction pattern:** Semua menghasilkan `!<?><?><?><?><?><?><?><?>'...` (garbage)

## Root Cause Analysis

### 1. Charset Size Mismatch

**Evidence dari loading weights:**
```
UserWarning: A total of 1 objects could not be loaded
Example error: <Dense name=logits, built=True>
The shape of the target variable and the shape of the target value must match.
variable.shape=(512, 96)   ← Model yang dibuat (95 char + 1 CTC blank)
value.shape=(512, 109)     ← Weights file (108 char + 1 CTC blank)
```

**Analisis:**
- Model di-training dengan **108 characters** dari `real_data_charlist.txt`
- Test script membuat model dengan **95 characters** (ASCII fallback)
- Logits layer tidak ter-load dengan benar → Prediksi garbage

### 2. Impact on Training

**Pertanyaan Kritis:**
Apakah training selama ini menggunakan recognizer yang **TIDAK BERFUNGSI**?

**Evidence dari training logs:**
- Recognizer digunakan untuk:
  1. **Recognition Feature Loss** (rec_feat_loss)
  2. **CTC Loss** (ctc_loss) 
  3. **CER/WER Metrics** pada validation

Jika recognizer tidak berfungsi:
- **rec_feat_loss** akan meaningless
- **ctc_loss** akan meaningless  
- **CER/WER metrics** akan salah

**Namun perhatikan:**
- Training logs menunjukkan CER ~0.33-0.38 pada clean images
- Ini menunjukkan recognizer **BERFUNGSI SAAT TRAINING**
- Masalah hanya terjadi saat **LOADING DI LUAR TRAINING SCRIPT**

### 3. Why Training Works But Test Fails?

**Hipotesis:**
1. `train_enhanced.py` load recognizer dengan charset yang **TEPAT** (108 char)
2. Test script awalnya load dengan **DEFAULT ASCII** (95 char) karena charset file tidak found
3. Setelah diperbaiki untuk load dari `real_data_charlist.txt`, seharusnya berfungsi

**Konfirmasi:**
- Test script final **SUDAH** menggunakan 108 characters dari `real_data_charlist.txt`
- Model loaded successfully dengan "Loaded 108 characters"
- **TAPI** masih ada weights mismatch warning

**Kemungkinan:**
Model architecture dibuat dengan 108+1=109 classes, tetapi logits layer entah mengapa dibuat dengan 96 classes saat inisialisasi pertama kali sebelum load weights.

## Test Results

### Ground Truth Decoding
✅ **FIXED** - Ground truth now correctly decoded:
```
"Rombouw zijne onderdanen over haer begaen Schermstuck"
"gelegentheden dien vorst ons soo toegedaen niet en sij als"
```

### Predictions
❌ **BROKEN** - All predictions are garbage:
```
Predicted: '!<?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?><?>'
```

Pattern: `!` followed by 381x `<?>`

### Metrics
- **CER:** 10.28 (10x character errors per character!)
- **WER:** 0.93 (93% word error rate)
- **Confidence:** ~14 (should be >50)
- **Perfect Matches:** 0/50 (0%)

## Implications for Training

### If Recognizer Was Broken During Training:

**Consequences:**
1. **Recognition Feature Loss** tidak berfungsi
   - Feature extraction dari recognizer akan random/meaningless
   - Generator tidak belajar dari recognizer features

2. **CTC Loss** tidak berfungsi
   - Generator tidak dioptimasi untuk recognizability
   - CTC annealing strategy tidak efektif

3. **Validation Metrics Salah**
   - CER/WER yang dilaporkan tidak akurat
   - Tidak bisa judge quality restoration vs recognizability

**However:**
- Training logs menunjukkan CER improvement dari ~0.50 → ~0.33
- Ini menunjukkan recognizer **WORKS IN TRAINING**
- Masalah hanya di **STANDALONE TEST SCRIPT**

### If Recognizer Works Only in Training:

**Most Likely Scenario:**
- Recognizer berfungsi dalam training loop
- Test script memiliki bug dalam:
  1. ~~Model initialization~~ (FIXED - charset loaded correctly)
  2. Weights loading (skip_mismatch hides the problem)
  3. **Decode function** (CTC decode gagal)

## Action Items

### Immediate (Priority 1) ✅ COMPLETED
- [x] Create standalone test script for recognizer
- [x] Test on clean images from TFRecord
- [x] Generate CSV with GT vs Predicted
- [x] Calculate CER/WER metrics
- [x] Fix ground truth decoding (1-indexed labels)

### Critical (Priority 2) 🔄 IN PROGRESS
- [ ] **Investigate CTC decode issue** in test script
- [ ] Compare decode logic between train_enhanced.py and test script
- [ ] Verify recognizer output shape and logits
- [ ] Test with exact same decode function from training

### Important (Priority 3)
- [ ] Verify recognizer works correctly during training
- [ ] Add recognizer validation in training script
- [ ] Monitor recognizer CER on clean images during training
- [ ] Compare training CER vs test CER on same samples

## Debugging Steps Taken

1. ✅ Created test script `test_recognizer_on_clean_images.py`
2. ✅ Fixed import paths to use correct recognizer module
3. ✅ Fixed TFRecord parsing to match training format
4. ✅ Fixed charset loading to use `real_data_charlist.txt` (108 chars)
5. ✅ Fixed ground truth decoding (1-indexed labels)
6. ❌ **ISSUE:** Predictions still garbage despite correct GT

## Hypothesis: CTC Decode Issue

**Observation:**
- Model outputs logits correctly (shape matches)
- Ground truth decoded correctly
- **But:** CTC decode produces garbage

**Possible Causes:**
1. Logits layer not loaded correctly (weights mismatch)
2. CTC decode function incompatible
3. Charset index mapping wrong in decode function
4. Model output requires softmax before decode

**Next Steps:**
1. Check raw logits values (before decode)
2. Compare with training decode logic
3. Test with argmax decode vs CTC decode
4. Verify softmax is applied

## Conclusion

**Status:** Test script berhasil dibuat dan dijalankan, tetapi menunjukkan **CRITICAL BUG**:
- Recognizer **TIDAK BERFUNGSI** saat di-test standalone
- Semua prediksi menghasilkan garbage output
- Root cause: **Weights loading issue** atau **CTC decode incompatibility**

**Recommendation:**
1. ⚠️ **JANGAN LANJUTKAN TRAINING** sampai issue ini diselesaikan
2. Investigate kenapa CTC decode gagal di test script
3. Verify recognizer bekerja dengan benar dalam training loop
4. Jika recognizer memang broken, **semua training sebelumnya invalid**

**Alternative Approach:**
Jika issue sulit diselesaikan, **lanjutkan dengan Task 2** (training dengan bugfix range mismatch) dan monitor recognizer performance dalam training logs. Jika CER dalam training reasonable (~0.33), maka recognizer kemungkinan berfungsi dalam training context.

---

**Files Created:**
- `dual_modal_gan/scripts/test_recognizer_on_clean_images.py` - Standalone HTR test script
- `results/recognizer_test_clean_images_fixed.csv` - Test results (50 samples)

**Next Action:** Investigate CTC decode issue atau proceed dengan Task 2 (new training)
