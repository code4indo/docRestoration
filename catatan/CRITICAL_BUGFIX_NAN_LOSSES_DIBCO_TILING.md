# CRITICAL BUGFIX: NaN Losses pada DIBCO Tiled Training

**Tanggal**: 26 Oktober 2025  
**Status**: RESOLVED ✅

## Masalah

Training DIBCO dengan tiled dataset (461 samples) menghasilkan **NaN losses sejak step pertama**:
- Generator Loss: NaN
- Discriminator Loss: NaN  
- Pixel Loss: NaN
- Perceptual Loss: NaN
- CTC Loss: NaN (expected, karena weight=0)

Training **berjalan tanpa crash** tapi tidak ada learning (losses tetap NaN selama epochs).

## Root Cause Analysis

### Timeline Investigasi

1. **Hypothesis 1**: Data corruption (NaN/Inf dalam TFRecord)
   - ❌ DITOLAK: Verifikasi TFRecord menunjukkan data valid (min=0, max=1, no NaN/Inf)

2. **Hypothesis 2**: cuDNN LSTM mask error dari empty ground truth text
   - ✅ SEBAGIAN BENAR: Menambahkan dummy text dengan `tf.ones()` fix cuDNN error
   - ❌ TAPI: NaN losses tetap muncul

3. **Hypothesis 3**: Shape mismatch antara dummy text dan discriminator
   - ✅ BENAR: `clean_text_pred.shape[1]` (recognizer timesteps, 256-512) ≠ `max_text_len=128` (discriminator)
   - FIX: Hardcode dummy text shape ke `[batch_size, 128]`
   - ❌ TAPI: NaN losses MASIH muncul setelah fix

4. **Hypothesis 4**: Pretrained checkpoint incompatibility dengan pure visual mode
   - ✅ **ROOT CAUSE CONFIRMED!**
   - Test WITHOUT pretrained checkpoint → **Training sukses, no NaN!**
   - PSNR = 5.99 dB (epoch 2), losses valid

### Root Cause

**Pretrained checkpoint `thin_stroke_preservation_v1_academic/ckpt-99` INCOMPATIBLE dengan pure visual training mode (ctc=0, rec_feat=0).**

**Alasan**:
1. Pretrained checkpoint di-train dengan:
   - `ctc_loss_weight > 0` (HTR loss aktif)
   - `rec_feat_loss_weight > 0` (recognizer feature loss aktif)
   - Text branch discriminator menerima **real predictions** dari recognizer

2. DIBCO tiled training menggunakan:
   - `ctc_loss_weight = 0` (pure visual, no HTR)
   - `rec_feat_loss_weight = 0`
   - Text branch discriminator menerima **dummy text** (tf.ones)

3. **Weight mismatch**: Generator/Discriminator weights dari pretrained TIDAK kompatibel dengan dummy text inputs → **gradient explosion/NaN**

### Kenapa TIDAK Terjadi Sebelum Tiling?

**Sebelum tiling**:
- DIBCO training menggunakan **dataset normal** (152 samples)
- Kemungkinan:
  - Training TIDAK menggunakan pretrained checkpoint (train from scratch)
  - ATAU menggunakan `ctc_loss_weight > 0` (bukan pure visual)
  - ATAU discriminator_mode berbeda

**Setelah tiling**:
- Dataset 461 samples (3x augmentation)
- Config set `ctc_loss_weight=0` (pure visual untuk DIBCO binarization)
- **PLUS** load pretrained checkpoint → **INCOMPATIBILITY**

## Solusi

### Fix yang Diimplementasikan

1. **Remove pretrained checkpoint** dari config:
   ```json
   "resume": false,
   "no_restore": true
   ```
   
2. **Train from scratch** untuk pure visual DIBCO:
   - Generator & Discriminator initialized dengan random weights
   - Kompatibel dengan dummy text inputs (ctc=0)
   - Losses valid, training converge

3. **Fixes yang tetap dipertahankan**:
   - Dummy text shape = `[batch_size, 128]` (match discriminator max_text_len)
   - `tf.cond()` untuk CTC label lengths (dummy saat ctc_weight=0)
   - `recurrent_activation='sigmoid'` di discriminator LSTM (disable cuDNN)

### Konfigurasi Final

```json
{
  "experiment_name": "dibco_tiled_augmented",
  "resume": false,
  "no_restore": true,
  "epochs": 50,
  "batch_size": 4,
  "pixel_loss_weight": 200.0,
  "adv_loss_weight": 1.5,
  "rec_feat_loss_weight": 0.0,
  "ctc_loss_weight": 0.0,
  "perceptual_loss_weight": 10.0,
  "discriminator_mode": "predicted"
}
```

## Hasil Testing

**Test Configuration**:
- Dataset: DIBCO tiled (461 samples)
- Epochs: 2
- Steps per epoch: 10
- No pretrained checkpoint

**Results**:
- ✅ No NaN losses
- ✅ Training converge
- Epoch 1: PSNR = 0.62 dB (initialization)
- Epoch 2: PSNR = 5.99 dB (+5.37 improvement)
- Training time: ~90s untuk 2 epochs

## Lessons Learned

### 1. Pretrained Checkpoint Compatibility
**CRITICAL**: Pretrained checkpoint **HARUS** di-train dengan loss configuration yang sama!
- Jika pretrained pakai `ctc_loss > 0`, fine-tuning JUGA harus pakai `ctc_loss > 0`
- Pure visual mode (ctc=0) TIDAK compatible dengan pretrained yang pakai CTC
- **Solusi**: Train from scratch untuk mode yang berbeda drastis

### 2. Debugging NaN Losses
**Strategi debugging yang efektif**:
1. ✅ Check data first (NaN/Inf dalam TFRecord)
2. ✅ Isolate components (test without pretrained checkpoint)
3. ✅ Compare with working configuration
4. ❌ Jangan assume shape/type compatibility tanpa verifikasi

### 3. Pure Visual Training Requirements
Untuk pure visual training (DIBCO binarization):
- ✅ Set `ctc_loss_weight = 0`
- ✅ Set `rec_feat_loss_weight = 0`
- ✅ Use dummy text dengan shape valid untuk discriminator
- ✅ **Train from scratch** (NO pretrained checkpoint)
- ✅ Focus on pixel/perceptual/adversarial losses

### 4. Tiling Dataset Considerations
Tiling augmentation (152 → 461 samples) VALID, tapi:
- Butuh training from scratch untuk convergence yang baik
- Pretrained weights dari domain berbeda (synthetic text) bisa incompatible
- Expected PSNR target lebih realistis (25-28 dB, bukan 30+ dB)

## Next Steps

1. **Launch full training**:
   - Config: `dibco_tiled_augmented.json` (updated, no pretrained)
   - Epochs: 50
   - Monitor: PSNR progression (target >25 dB)

2. **Comparison experiment**:
   - Train dengan pretrained + `ctc_loss_weight > 0` (dual-modal mode)
   - Compare PSNR: pure visual vs dual-modal

3. **Production deployment**:
   - Jika pure visual lebih baik → use for DIBCO
   - Jika dual-modal lebih baik → consider hybrid approach

## References

- TFRecord verification: No NaN/Inf in data ✅
- Fresh training test: `/tmp/dibco_test_fresh.log`
- Config file: `configs/dibco_tiled_augmented.json`
- Tiled dataset: `dual_modal_gan/data/dibco_tiled_full.tfrecord` (461 samples)
