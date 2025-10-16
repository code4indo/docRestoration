# Rencana Aksi: Implementasi Bayesian Optimization untuk Bobot Loss

**Tanggal:** 2025-10-13
**Status:** Draft Awal

---

## 🎯 1. Tujuan

Melakukan optimasi bobot *loss* (`pixel_loss_weight`, `rec_feat_loss_weight`, `adv_loss_weight`) untuk arsitektur Baseline GAN-HTR menggunakan metode Bayesian Optimization. Tujuannya adalah menemukan kombinasi bobot yang optimal untuk mencapai keseimbangan terbaik antara kualitas visual (PSNR, SSIM) dan akurasi pengenalan teks (CER, WER), dengan fokus pada peningkatan CER/WER.

## 💡 2. Metodologi

Kami akan menggunakan **Bayesian Optimization** melalui *library* **Optuna** di Python. Metode ini dipilih karena efisiensinya dalam mencari *hyperparameter* optimal pada fungsi objektif yang mahal (seperti *run* training model ML).

### **Mekanisme Kerja Optuna (Bayesian Optimization):**

Optuna bekerja secara iteratif dan cerdas untuk menemukan kombinasi bobot terbaik dengan jumlah percobaan yang minimal. Mekanismenya adalah sebagai berikut:

1.  **Model Probabilistik (Surrogate Model)**: Optuna membangun model statistik (seringkali *Gaussian Process*) dari fungsi objektif (yaitu, skor gabungan metrik validasi kita). Model ini memperkirakan performa dari kombinasi *hyperparameter* yang belum pernah dievaluasi.
2.  **Fungsi Akuisisi (Acquisition Function)**: Berdasarkan model probabilistik tersebut, Optuna menggunakan fungsi akuisisi (misalnya, *Expected Improvement*) untuk memutuskan kombinasi *hyperparameter* mana yang paling menjanjikan untuk dicoba selanjutnya. Fungsi ini menyeimbangkan dua hal:
    *   **Eksplorasi**: Mencoba area baru di ruang pencarian yang berpotensi memiliki performa tinggi.
    *   **Eksploitasi**: Memfokuskan pencarian pada area yang sudah diketahui memiliki performa baik.
3.  **Proses Iteratif**: Proses ini berulang:
    *   Beberapa kombinasi *hyperparameter* awal dievaluasi (seringkali secara acak).
    *   Model probabilistik diperbarui dengan hasil evaluasi tersebut.
    *   Fungsi akuisisi digunakan untuk mengusulkan kombinasi *hyperparameter* terbaik berikutnya.
    *   Kombinasi yang diusulkan dievaluasi.
    *   Langkah 2-4 diulang hingga kriteria berhenti terpenuhi (misalnya, jumlah percobaan (`n_trials`) tercapai).

Dengan mekanisme ini, Optuna dapat menemukan kombinasi bobot optimal jauh lebih efisien dibandingkan metode *brute-force* seperti *grid search*, terutama ketika setiap evaluasi (satu *run* training) membutuhkan biaya komputasi yang tinggi.

## 🚀 3. Ruang Lingkup

*   **Arsitektur Model**: Baseline (Generator satu-pass) yang telah terbukti unggul.
*   **Durasi Training per Percobaan**: 10 epoch (sebagai *smoke test* cepat untuk setiap kombinasi bobot).
*   **Metrik Objektif**: Kombinasi metrik validasi (misalnya, `val_psnr + val_ssim - (CER_weight * val_cer)`) yang akan dimaksimalkan oleh Optuna.
*   **Bobot Loss yang Dioptimalkan**: `pixel_loss_weight`, `rec_feat_loss_weight`, `adv_loss_weight`.
*   **Bobot Loss Tetap**: `contrastive_loss_weight = 0.0` (dinonaktifkan), `ctc_loss_weight = 1.0` (tetap untuk monitoring).

## 🗓️ 4. Agenda Kerja (Langkah-langkah)

### **Fase 1: Persiapan Lingkungan & Skrip (1-2 hari)**

*   **[ ] 4.1. Instalasi Optuna**: Tambahkan `optuna` ke `pyproject.toml` dan instal dependensi menggunakan `poetry install`.
*   **[ ] 4.2. Buat Skrip `objective.py`**: Kembangkan skrip Python baru yang akan menjadi fungsi objektif untuk Optuna. Skrip ini akan:
    *   Menerima `trial` dari Optuna.
    *   Menentukan bobot *loss* yang disarankan oleh `trial`.
    *   Memanggil `train32.py` dengan bobot *loss* yang disarankan (menggunakan `subprocess` atau sejenisnya).
    *   Membaca metrik validasi terbaik (PSNR, CER, SSIM, WER) dari *output* `train32.py` atau log MLflow.
    *   Menghitung dan mengembalikan skor objektif tunggal.
*   **[ ] 4.3. Modifikasi `train32.py` (Opsional, jika diperlukan)**:
    *   Pastikan `train32.py` dapat dengan mudah mengembalikan metrik validasi terbaik dalam format yang dapat dibaca oleh `objective.py` (misalnya, mencetak JSON ke stdout atau menyimpan ke file sementara).
    *   Pastikan `train32.py` menggunakan `mlflow.log_metrics` dengan benar untuk setiap epoch dan metrik validasi.

### **Fase 2: Definisi Ruang Pencarian & Fungsi Objektif (1 hari)**

*   **[ ] 4.4. Definisikan Ruang Pencarian Bobot Loss**: Tentukan rentang nilai yang masuk akal untuk setiap bobot *loss* yang akan dioptimalkan di dalam `objective.py`.
    *   Contoh: `pixel_loss_weight` (10.0 - 150.0), `rec_feat_loss_weight` (10.0 - 100.0), `adv_loss_weight` (0.5 - 5.0).
*   **[ ] 4.5. Definisikan Metrik Objektif**: Tentukan formula skor tunggal yang akan dimaksimalkan oleh Optuna. Contoh: `val_psnr + val_ssim - (CER_weight * val_cer)`. Tentukan `CER_weight` berdasarkan prioritas (misalnya, 100 atau 1000 untuk sangat memprioritaskan CER).

### **Fase 3: Eksekusi Studi Optimasi (2-5 hari, tergantung jumlah percobaan)**

*   **[ ] 4.6. Jalankan Studi Optuna**: Eksekusi skrip utama yang akan menjalankan `optuna.study.optimize()`. Tentukan jumlah percobaan (`n_trials`) yang sesuai (misalnya, 50-100 percobaan).
    *   **Perintah Eksekusi**: `poetry run python scripts/hpo/objective.py > /dev/null 2>&1 &`
*   **[ ] 4.7. Pantau Proses**: Gunakan MLflow UI untuk memantau setiap percobaan yang dijalankan oleh Optuna. Pastikan tidak ada *run* yang gagal dan metrik tercatat dengan benar.
    *   **Indikator Proses Selesai**:
        1.  **Terminasi Proses Python**: Periksa apakah proses `objective.py` masih berjalan dengan `ps aux | grep objective.py`. Jika tidak ada hasil (selain baris `grep` itu sendiri), proses sudah selesai.
        2.  **MLflow UI**: Pantau jumlah *run* yang selesai di eksperimen "HPO_Loss_Weights". Proses selesai ketika jumlah *run* mencapai `n_trials` (saat ini 50) dan tidak ada *run* baru yang muncul. Keberadaan *run* "HPO_Best_Result" juga menandakan studi telah selesai.
        3.  **Direktori Output**: Periksa apakah tidak ada lagi direktori baru yang dibuat atau diperbarui di `dual_modal_gan/outputs/hpo_checkpoints/` dan `dual_modal_gan/outputs/hpo_samples/`.

### **Fase 4: Analisis Hasil & Implementasi (1-2 hari)**

*   **[ ] 4.8. Analisis Hasil Optuna**: Gunakan fitur analisis Optuna (misalnya, `plot_optimization_history`, `plot_param_importances`) dan MLflow untuk mengidentifikasi kombinasi bobot terbaik.
*   **[ ] 4.9. Validasi Bobot Optimal**: Jalankan training *full-epoch* (misalnya, 30-100 epoch) dengan kombinasi bobot terbaik yang ditemukan oleh Optuna pada arsitektur Baseline.
*   **[ ] 4.10. Dokumentasi**: Catat bobot optimal dan hasilnya di logbook dan laporan studi ablasi.

--- End of content ---

## ✅ 5. Pertanggungjawaban

Dokumen ini berfungsi sebagai rencana kerja yang jelas dan dapat dilacak. Setiap langkah akan didokumentasikan melalui *commit* Git, log MLflow, dan catatan di logbook. Hasil dari setiap fase akan dievaluasi sebelum melanjutkan ke fase berikutnya.

---

**Catatan:** Estimasi waktu dapat bervariasi tergantung kompleksitas dan sumber daya komputasi yang tersedia.

## 6. Kendala yang Dihadapi & Solusi

### 6.1. Kendala: Out-Of-Memory (OOM) Error dengan Batch Size 8
*   **Deskripsi**: Saat mencoba meningkatkan `batch_size` dari 4 menjadi 8 untuk mempercepat simulasi, terjadi *error* OOM pada GPU (NVIDIA RTX A4000, 14GB VRAM) selama training `train32.py`.
*   **Solusi**: `batch_size` dikembalikan ke `4`. Ditambahkan variabel lingkungan `TF_GPU_ALLOCATOR=cuda_malloc_async` untuk *subprocess* `train32.py` untuk membantu manajemen memori dan mengurangi fragmentasi.
*   **Status**: Teratasi (dengan kembali ke `batch_size=4` dan penambahan `TF_GPU_ALLOCATOR`).

### 6.2. Kendala: Gagal Mengambil Metrik MLflow (Trial Pruned)
*   **Deskripsi**: Skrip `objective.py` gagal mengambil metrik validasi akhir (`val/psnr`, `val/cer`, `val/ssim`) dari MLflow untuk setiap *trial*, menyebabkan *trial* dipangkas (`Trial Pruned`). Ini terjadi karena pencatatan MLflow bersifat asinkron dan `objective.py` mencoba mengambil metrik terlalu cepat, atau metrik `best_val_` tidak selalu dicatat di *run* singkat.
*   **Solusi**: Diimplementasikan mekanisme coba lagi (*retry mechanism*) dengan penundaan (`time.sleep()`) dan beberapa percobaan untuk mengambil data *run* dari MLflow. Selain itu, pengambilan metrik diubah dari `best_val_` menjadi metrik *final epoch* (`val/psnr`, `val/cer`, `val/ssim`) untuk memastikan ketersediaan data di *run* singkat.
*   **Status**: Sedang diuji (setelah implementasi *retry mechanism* dan perubahan pengambilan metrik).