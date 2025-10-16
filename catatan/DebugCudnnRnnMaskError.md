# Surat Tugas: Debug dan Perbaiki cuDNN RNN Mask Error

**Kepada:** Agen Copilot

**Dari:** [Nama Anda/Gemini CLI Agent]

**Tanggal:** 2025-10-13

---

### 1. Latar Belakang Masalah

Selama menjalankan studi ablasi untuk "Iterative Refinement", agen Copilot melaporkan error `RNN mask dan cuDNN` pada *epoch* 10. Error ini sebelumnya pernah muncul dan diklaim telah diperbaiki dengan mengatur `use_cudnn=False` pada lapisan LSTM di `dual_modal_gan/src/models/text_encoder.py`. Namun, error ini muncul kembali.

### 2. Tujuan Tugas

Mendiagnosis akar masalah dari `cuDNN RNN Mask Error` yang muncul kembali dan mengimplementasikan perbaikan yang stabil untuk memastikan training dapat berjalan hingga selesai.

### 3. Detail Tugas

**A. Konfirmasi dan Analisis Error:**

1.  **Tinjau Log Lengkap:** Dapatkan dan tinjau log lengkap dari run studi ablasi sebelumnya (terutama `logbook/ablation_exp_baseline.log` dan `logbook/ablation_exp_iterative.log`) untuk mendapatkan *traceback* error yang detail. Identifikasi baris kode spesifik yang menyebabkan error.
2.  **Verifikasi Lingkungan:** Pastikan `PYTHONPATH` diatur dengan benar saat skrip training dijalankan dan bahwa versi `dual_modal_gan/src/models/text_encoder.py` yang mengandung `use_cudnn=False` benar-benar dimuat dan digunakan oleh `train32.py`.

**B. Investigasi Akar Masalah:**

1.  **Periksa `text_encoder.py`:** Meskipun `use_cudnn=False` sudah ada, pastikan tidak ada konfigurasi lain yang secara implisit mengaktifkan cuDNN atau ada masalah *masking* lain yang tidak tertangani.
2.  **Periksa `train32.py`:** Analisis bagaimana `text_encoder` digunakan dalam `train_step` dan bagaimana *masking* diterapkan pada input teks. Periksa apakah ada lapisan RNN lain atau penggunaan CTC loss yang mungkin berinteraksi dengan cuDNN atau *masking* secara tidak terduga.
3.  **Kompatibilitas TensorFlow:** Pertimbangkan apakah ada isu kompatibilitas antara versi TensorFlow yang digunakan dengan cuDNN atau penanganan *masking*.

**C. Implementasi dan Verifikasi Perbaikan:**

1.  **Usulkan Perbaikan:** Berdasarkan investigasi, usulkan solusi yang jelas dan terperinci.
2.  **Implementasikan Perbaikan:** Terapkan perubahan kode yang diperlukan.
3.  **Verifikasi dengan Smoke Test:** Setelah perbaikan diimplementasikan, jalankan *smoke test* singkat (misalnya, 1 *epoch*, 10 *steps*) menggunakan `scripts/run_ablation_study.sh` (atau modifikasi sementara untuk *smoke test* saja) untuk memastikan error tidak muncul kembali dan training dapat berjalan.

### 4. Hasil yang Diharapkan

*   Akar masalah `cuDNN RNN Mask Error` teridentifikasi dan terdokumentasi.
*   Perbaikan diimplementasikan dan diverifikasi melalui *smoke test*.
*   Skrip training dapat berjalan hingga selesai tanpa error terkait `RNN mask dan cuDNN`.

---

Terima kasih atas kerja sama Anda.

[Gemini CLI Agent]