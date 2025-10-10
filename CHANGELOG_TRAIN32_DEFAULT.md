# ✅ PERUBAHAN: Default Training Script ke train32.py

## 📅 Tanggal: October 11, 2025

## 🎯 Perubahan yang Dilakukan (Option A)

### **SEBELUM:**
```yaml
gan-htr-prod:
  environment:
    - TRAINING_SCRIPT=${TRAINING_SCRIPT:-dual_modal_gan/scripts/train.py}  # Mixed Precision
```

### **SESUDAH:**
```yaml
gan-htr-prod:
  environment:
    - TRAINING_SCRIPT=${TRAINING_SCRIPT:-scripts/train32_production.sh}  # Pure FP32 via train32.py
```

---

## 📊 Status Script Saat Ini

| Service | Default Script | Python Script | Precision |
|---------|---------------|---------------|-----------|
| **gan-htr-prod** | ✅ `train32_production.sh` | ✅ `train32.py` | **Pure FP32** |
| **gan-htr-smoke-test** | ✅ `train32_smoke_test.sh` | ✅ `train32.py` | **Pure FP32** |
| gan-htr-dev | Interactive Shell | N/A | N/A |
| gan-htr-test | Validation Only | N/A | N/A |

---

## ✅ Konfirmasi Verifikasi

### 1. Production Service (gan-htr-prod):
```bash
$ docker-compose config | grep "TRAINING_SCRIPT"
TRAINING_SCRIPT: scripts/train32_production.sh  ✅
```

### 2. Script train32_production.sh memanggil:
```bash
$ grep "poetry run python" scripts/train32_production.sh
poetry run python dual_modal_gan/scripts/train32.py  ✅
```

### 3. Smoke Test Service (gan-htr-smoke-test):
```bash
$ docker-compose config | grep "TRAINING_SCRIPT"
TRAINING_SCRIPT: scripts/train32_smoke_test.sh  ✅
```

---

## 🎯 Impact

### **Semua Service Sekarang Menggunakan train32.py (Pure FP32):**

1. ✅ **Production Training** (`gan-htr-prod`):
   - Script: `train32_production.sh` → calls `train32.py`
   - Mode: Full production training dengan Pure FP32
   - Command: `docker-compose up -d gan-htr-prod`

2. ✅ **Smoke Test** (`gan-htr-smoke-test`):
   - Script: `train32_smoke_test.sh` → calls `train32.py`
   - Mode: Quick 2-epoch validation dengan Pure FP32
   - Command: `docker-compose up -d gan-htr-smoke-test`

---

## 🚀 Quick Start Commands (Updated)

### Production Training (train32.py - Pure FP32):
```bash
docker-compose up -d gan-htr-prod
docker logs -f gan-htr-prod
```

### Smoke Test (train32.py - Pure FP32):
```bash
docker-compose up -d gan-htr-smoke-test
docker logs -f gan-htr-smoke-test
```

### Custom Script (jika masih ingin pakai train.py mixed precision):
```bash
TRAINING_SCRIPT=dual_modal_gan/scripts/train.py docker-compose up -d gan-htr-prod
```

---

## 📝 Alasan Perubahan

**Konsistensi dengan Request User:**
> "saya ingin ketika memulai training dengan docker, saya bisa secara fleksibel menjalankan training dengan menggunakan konfigurasi ini: docRestoration/scripts/train32_smoke_test.sh"

Dan user mengkonfirmasi:
> "apakah kamu sudah menggunakan skrip ini untuk training... docRestoration/dual_modal_gan/scripts/train32.py"

**Solusi:** Semua default service sekarang menggunakan `train32.py` (Pure FP32) untuk konsistensi.

---

## 🔄 Rollback (jika diperlukan)

Jika ingin kembali ke mixed precision (train.py):
```bash
# Edit docker-compose.yml
TRAINING_SCRIPT=${TRAINING_SCRIPT:-dual_modal_gan/scripts/train.py}

# Atau via environment variable (tanpa edit file)
TRAINING_SCRIPT=dual_modal_gan/scripts/train.py docker-compose up -d gan-htr-prod
```

---

## ✅ Status: COMPLETE

- [x] Default production script → `train32_production.sh` (calls `train32.py`)
- [x] Smoke test script → `train32_smoke_test.sh` (calls `train32.py`)
- [x] Verified configuration
- [x] Documented changes
- [x] Ready for deployment

**Semua service sekarang menggunakan train32.py (Pure FP32) sebagai default!** 🎉
