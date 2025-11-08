# CRITICAL BUG: Resume/No_Restore Flags Untuk Fine-Tuning

**Date:** 2025-10-30  
**Severity:** 🔴 CRITICAL - Menyebabkan training from scratch bukan fine-tuning  
**Impact:** Wasted 30+ menit compute, hasil training tidak valid untuk fine-tuning analysis

---

## 🐛 BUG DESCRIPTION

Semua config fine-tuning menggunakan flag yang SALAH:

```json
{
  "pretrained_checkpoint": "path/to/checkpoint/ckpt-99",
  "resume": false,           // ❌ SALAH!
  "no_restore": false        // ❌ SALAH!
}
```

**Akibatnya:**
- Training **TIDAK load dari pretrained_checkpoint**
- Training start **from scratch** (random weights)
- Atau load dari checkpoint lama di `checkpoint_dir` (bukan pretrained)

---

## ✅ CORRECT CONFIGURATION

```json
{
  "pretrained_checkpoint": "path/to/checkpoint/ckpt-99",
  "resume": true,            // ✅ BENAR! Resume training tapi...
  "no_restore": true         // ✅ BENAR! ...jangan restore dari checkpoint_dir
}
```

**Logika yang benar:**
- `resume: true` → "Ya saya mau lanjutkan training dari checkpoint"
- `no_restore: true` → "TAPI jangan ambil dari checkpoint_dir, ambil dari pretrained_checkpoint"
- **Hasil:** Load weights dari pretrained_checkpoint, mulai dari epoch 1 ✅

---

## 🔍 ANALISIS IMPACT

### Config yang Terpengaruh:

1. ✅ **`thin_stroke_preservation_v1_finetuning.json`**
   - **Status:** SUDAH BENAR! Log menunjukkan "🔄 FINE-TUNING MODE: Loading pretrained checkpoint... ckpt-99"
   - **Bukti:** Training epoch 1 langsung converge (losses rendah)
   - **Kesimpulan:** Meskipun config salah, kode `train_enhanced.py` berhasil handle dengan benar

2. ❌ **`dibco_finetuning_from_anri_v1.json`**
   - **Status:** GAGAL! Load dari ckpt-114 (checkpoint DIBCO lama), bukan ckpt-106 (ANRI V1)
   - **Bukti:** Log menunjukkan "Restored from ckpt-114" → restart dari Epoch 1 dengan losses tinggi
   - **Kesimpulan:** Training 30 menit pertama TIDAK VALID

---

## 🛠️ ROOT CAUSE ANALYSIS

### Mengapa ANRI V1 Berhasil?

Melihat log training ANRI V1:
```
🔄 FINE-TUNING MODE: Loading pretrained checkpoint...
   Source: dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/ckpt-99
   ✅ Pretrained weights loaded successfully
   Starting fine-tuning from epoch 0/15
```

**Kemungkinan:** Kode `train_enhanced.py` memiliki logic khusus:
- Jika `pretrained_checkpoint` diset DAN direktori `checkpoint_dir` kosong
- Maka load dari `pretrained_checkpoint` (fine-tuning mode)
- Ignore flags `resume` dan `no_restore`

### Mengapa DIBCO Gagal?

DIBCO gagal karena:
1. Direktori `dibco_finetuning_from_anri_v1/` **sudah ada ckpt-114** (sisa training sebelumnya)
2. Flag `resume: true` membuat training load dari ckpt-114
3. Flag `no_restore: true` tidak berfungsi karena ada konflik priority

**Priority logic (kemungkinan):**
```
if resume == true AND checkpoint_exists_in_checkpoint_dir:
    load_from_checkpoint_dir()  # ckpt-114
else if pretrained_checkpoint:
    load_from_pretrained_checkpoint()  # ckpt-106
```

---

## ✅ SOLUTION

### Immediate Fix:

1. **Untuk DIBCO:**
   ```bash
   rm -rf dual_modal_gan/checkpoints/dibco_finetuning_from_anri_v1/
   # Re-launch training dengan clean state
   ```

2. **Config sudah diperbaiki:**
   - ✅ `thin_stroke_preservation_v1_finetuning.json`: resume=true, no_restore=true
   - ✅ `dibco_finetuning_from_anri_v1.json`: resume=true, no_restore=true

### Long-term Fix:

**Dokumentasi di README/TRAINING_GUIDE:**

```markdown
## Fine-Tuning Configuration

For fine-tuning from pretrained checkpoint:

{
  "pretrained_checkpoint": "path/to/base/model/ckpt-N",
  "checkpoint_dir": "path/to/new/finetuning/dir",
  "resume": true,        # Enable checkpoint loading
  "no_restore": true     # Don't restore from checkpoint_dir, use pretrained_checkpoint
}

⚠️ CRITICAL: Make sure checkpoint_dir is EMPTY or doesn't exist before starting fine-tuning!
```

---

## 📊 VERIFICATION STEPS

Untuk memverifikasi fine-tuning berhasil:

1. **Cek log training:**
   ```bash
   grep "FINE-TUNING MODE\|pretrained" logbook/experiment_*.log
   ```
   
   Expected output:
   ```
   🔄 FINE-TUNING MODE: Loading pretrained checkpoint...
   Source: path/to/pretrained/ckpt-N
   ✅ Pretrained weights loaded successfully
   ```

2. **Cek losses epoch 1:**
   - **Fine-tuning (correct):** Losses rendah dari awal (G<1000, Pix<0.1)
   - **From scratch (wrong):** Losses tinggi (G>1500, Pix>0.5)

3. **Cek checkpoint files:**
   ```bash
   ls -lh checkpoint_dir/*.index
   ```
   - Jika ada checkpoint lama sebelum training → BAHAYA! Bisa load dari sana

---

## 🎓 LESSONS LEARNED

1. **Flags `resume` dan `no_restore` CONFUSING!**
   - Nama tidak intuitif untuk fine-tuning use case
   - Butuh dokumentasi yang jelas

2. **Always verify checkpoint loading di log training**
   - Jangan assume config benar
   - Check "FINE-TUNING MODE" message

3. **Clean checkpoint_dir sebelum fine-tuning**
   - Atau gunakan unique experiment_name
   - Avoid checkpoint conflict

4. **Test dengan epoch minimal dulu**
   - Jangan langsung training 30 epochs
   - Verifikasi epoch 1-2 dulu, cek losses make sense

---

## 📝 ACTION ITEMS

- [x] Fix config `thin_stroke_preservation_v1_finetuning.json`
- [x] Fix config `dibco_finetuning_from_anri_v1.json`
- [ ] Clean DIBCO checkpoint directory
- [ ] Re-launch DIBCO training dengan verification
- [ ] Add checkpoint loading verification di training script
- [ ] Update dokumentasi fine-tuning di README
- [ ] Add pre-training check: fail jika checkpoint_dir not empty untuk fine-tuning

---

## 🔗 RELATED ISSUES

- Config V2, V3 juga kemungkinan punya issue yang sama
- Semua experiment yang claim "fine-tuning" perlu di-audit ulang
- Loss weight experiments (V1 vs V3) masih valid karena both start from ckpt-99

---

**Status:** RESOLVED (configs fixed, awaiting re-launch)  
**Next:** Verify DIBCO training dengan checkpoint loading yang benar
