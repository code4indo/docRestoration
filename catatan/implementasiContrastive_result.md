# Implementasi Contrastive Loss - Hasil Eksperimen

## Ringkasan

Dokumen ini melaporkan hasil implementasi dan eksperimen Contrastive Loss pada arsitektur GAN-HTR Dual-Modal untuk pemulihan dokumen historis. Eksperimen membandingkan kinerja model dengan dan tanpa Contrastive Loss untuk mengevaluasi dampaknya terhadap kualitas pemulihan visual dan akurasi pengenalan teks.

## Latar Belakang

Contrastive Loss adalah teknik pembelajaran representasi yang bertujuan untuk:
1. Meminimalkan jarak antara representasi sampel yang mirip (anchor dan positive)
2. Memaksimalkan jarak antara representasi sampel yang berbeda (anchor dan negative)

Dalam konteks GAN-HTR, Contrastive Loss diterapkan pada:
- **Text Encoder**: Untuk mempelajari representasi teks yang lebih diskriminatif
- **Generator**: Untuk menghasilkan gambar yang representasinya konsisten dengan teks target

## Setup Eksperimen

### Konfigurasi Hardware
- **GPU**: NVIDIA RTX A4000 (14GB VRAM)
- **CPU**: Multi-core processor
- **Memory**: Sufficient RAM untuk batch size 4
- **Precision**: Pure FP32 (stabilitas numerik untuk CTC loss)

### Parameter Training
```json
{
  "batch_size": 4,
  "epochs": 30,
  "steps_per_epoch": 100,
  "learning_rate": 0.0002,
  "pixel_loss_weight": 200.0,
  "recognition_feature_loss_weight": 5.0,
  "ctc_loss_weight": 1.0,
  "adv_loss_weight": 3.0,
  "contrastive_loss_weight": 1.0,
  "early_stopping_patience": 15
}
```

### Dataset
- **Path**: `dual_modal_gan/data/dataset_gan.tfrecord`
- **Training Samples**: 4,266
- **Validation Samples**: 473
- **Charset**: 109 karakter (termasuk blank token)

## Eksperimen yang Dilakukan

### 1. Eksperimen Baseline (Contrastive Loss OFF)
- **Deskripsi**: Training tanpa Contrastive Loss
- **Durasi**: 9 epochs (terhenti karena error cuDNN)
- **Output Directory**: `outputs/samples_novelty/contrastive_off/`
- **Metrics File**: `outputs/checkpoints_novelty/contrastive_off/metrics/training_metrics_fp32.json`

### 2. Eksperimen dengan Contrastive Loss (Contrastive Loss ON)
- **Deskripsi**: Training dengan Contrastive Loss aktif
- **Durasi**: 9 epochs (terhenti karena error cuDNN)
- **Output Directory**: `outputs/samples_novelty/contrastive_on/`
- **Metrics File**: `outputs/checkpoints_novelty/contrastive_on/metrics/training_metrics_fp32.json`

## Hasil dan Analisis

### Perbandingan Kinerja Terbaik

#### Contrastive Loss OFF (Baseline)
**Epoch 7 - Best Performance:**
```json
{
  "psnr": 20.264774322509766,
  "ssim": 0.9325624108314514,
  "cer": 0.27310630679130554,
  "wer": 0.45690393447875977
}
```

#### Contrastive Loss ON
**Epoch 8 - Best Performance:**
```json
{
  "psnr": 18.94078826904297,
  "ssim": 0.9197664856910706,
  "cer": 0.29716771841049194,
  "wer": 0.4743603467941284
}
```

### Analisis Metrik

#### 1. Kualitas Visual (PSNR & SSIM)
- **PSNR**: Baseline (20.26) > Contrastive (18.94)
  - Selisih: 1.32 dB (~6.5% lebih baik)
- **SSIM**: Baseline (0.9326) > Contrastive (0.9198)
  - Selisih: 0.0128 (~1.4% lebih baik)

**Kesimpulan**: Model tanpa Contrastive Loss menghasilkan kualitas visual yang sedikit lebih baik.

#### 2. Akurasi Pengenalan Teks (CER & WER)
- **CER**: Baseline (0.2731) < Contrastive (0.2972)
  - Selisih: 0.0241 (~8.8% lebih baik)
- **WER**: Baseline (0.4569) < Contrastive (0.4744)
  - Selisih: 0.0175 (~3.8% lebih baik)

**Kesimpulan**: Model tanpa Contrastive Loss memiliki akurasi pengenalan teks yang lebih baik.

#### 3. Loss Components Analysis

**Training Losses (Epoch Akhir):**

| Component | Contrastive OFF | Contrastive ON | Perubahan |
|-----------|-----------------|---------------|-----------|
| Total Loss | 8.68 | 10.42 | +20.1% |
| Pixel Loss | 0.0227 | 0.0257 | +13.2% |
| Rec. Feature Loss | 0.3197 | 0.3244 | +1.5% |
| Contrastive Loss | 1.3875 | 0.9930 | -28.4% |
| CTC Loss | 299.68 | 299.71 | +0.01% |
| Adv Loss | 0.8479 | 0.8919 | +5.2% |

**Analisis:**
- Contrastive Loss lebih rendah pada eksperimen dengan Contrastive Loss ON, namun ini tidak berkontribusi pada peningkatan kualitas
- Total loss lebih tinggi pada Contrastive Loss ON, menunjukkan ketidakstabilan training
- Pixel loss lebih tinggi pada Contrastive Loss ON, konsisten dengan metrik PSNR yang lebih rendah

### Perbandingan Training Dynamics

#### Stabilitas Training
- **Contrastive OFF**: Lebih stabil, loss menurun secara konsisten
- **Contrastive ON**: Fluktuasi loss lebih tinggi, terutama pada component losses

#### Convergence Speed
- **Contrastive OFF**: Mencapai best performance pada epoch 7
- **Contrastive ON**: Mencapai best performance pada epoch 8
- Tidak ada perbedaan signifikan dalam kecepatan konvergensi

## Masalah yang Ditemukan

### 1. cuDNN RNN Mask Error
Kedua eksperimen terhenti pada epoch 10 karena error:
```
INVALID_ARGUMENT: assertion failed: [You are passing a RNN mask that does not correspond to right-padded sequences, while using cuDNN, which is not supported.]
```

**Root Cause**: Mask padding tidak sesuai dengan format yang diharapkan cuDNN
**Impact**: Eksperimen hanya berjalan 9 epochs dari 30 yang direncanakan
**Solution**: Perlu modifikasi Text Encoder untuk menggunakan `use_cudnn=False`

### 2. Ketidakstabilan dengan Contrastive Loss
- Total loss lebih tinggi dan fluktuatif
- Tidak ada peningkatan kualitas yang signifikan
- Kemungkinan conflict dengan loss components lainnya

## Visualisasi Hasil

### Sample Images
Kedua eksperimen menghasilkan sample images pada epoch 5:
- **Contrastive OFF**: `outputs/samples_novelty/contrastive_off/`
- **Contrastive ON**: `outputs/samples_novelty/contrastive_on/`

### Perbandingan Visual
- Kualitas visual comparable antara kedua eksperimen
- Tidak ada perbedaan signifikan yang dapat diamati secara visual
- Model berhasil memulihkan dokumen dengan noise dan degradasi

## Kesimpulan dan Rekomendasi

### Kesimpulan Utama
1. **Tidak ada manfaat signifikan** dari Contrastive Loss pada arsitektur GAN-HTR Dual-Modal ini
2. **Model baseline (tanpa Contrastive Loss)** menunjukkan kinerja yang lebih baik:
   - PSNR +6.5% lebih tinggi
   - SSIM +1.4% lebih tinggi
   - CER +8.8% lebih baik (lebih rendah)
   - WER +3.8% lebih baik (lebih rendah)
3. **Contrastive Loss menambah kompleksitas** tanpa peningkatan kualitas
4. **Training lebih stabil** tanpa Contrastive Loss

### Rekomendasi

#### Untuk Implementasi Saat Ini:
1. **Gunakan baseline (tanpa Contrastive Loss)** untuk production
2. **Fokus pada optimasi hyperparameter** yang ada:
   - Learning rate scheduling
   - Loss weight tuning
   - Architecture modifications

#### Untuk Penelitian Masa Depan:
1. **Investigasi arsitektur Contrastive Loss yang berbeda**:
   - Different projection heads
   - Alternative contrastive formulations (InfoNCE, Triplet loss)
   - Different temperature parameters

2. **Eksperimen dengan setup yang berbeda**:
   - Larger batch sizes
   - Different augmentation strategies
   - Multi-scale contrastive learning

3. **Perbaikan masalah teknis**:
   - Fix cuDNN compatibility issue
   - Implement proper gradient accumulation
   - Optimize memory usage

### Lessons Learned
1. **Tidak semua teknik SOTA transferable** ke domain spesifik
2. **Pentingnya baseline comparison** sebelum menambah kompleksitas
3. **Stabilitas training lebih penting** dari kompleksitas model
4. **Domain-specific evaluation** crucial (visual + textual metrics)

## File Terkait

### Output Files
- **Sample Images**: `outputs/samples_novelty/contrastive_off/` dan `outputs/samples_novelty/contrastive_on/`
- **Training Metrics**: `outputs/checkpoints_novelty/*/metrics/training_metrics_fp32.json`
- **Log Files**: `logbook/novelty_exp_contrastive_*.log`

### Script Eksperimen
- **Contrastive OFF**: `scripts/novelty_experiments/exp_contrastive_off.sh`
- **Contrastive ON**: `scripts/novelty_experiments/exp_contrastive_on.sh`
- **Training Script**: `dual_modal_gan/scripts/train32.py`

---

*Dibuat pada: 2025-10-13*  
*Versi: 1.0*  
*Status: Final*