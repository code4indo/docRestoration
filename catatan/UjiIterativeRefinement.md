# Surat Tugas: Studi Ablasi Iterative Refinement

**Kepada:** Agen Copilot

**Dari:** [Nama Anda/Gemini CLI Agent]

**Tanggal:** 2025-10-13

---

### 1. Latar Belakang

Kami telah mengimplementasikan fitur "Iterative Refinement with Recognizer Attention" dalam arsitektur GAN-HTR Dual-Modal. Sebuah *smoke test* awal menunjukkan bahwa implementasi berfungsi, namun performa penuhnya belum dievaluasi secara komparatif terhadap *Golden Baseline*.

### 2. Tujuan Tugas

Menjalankan studi ablasi penuh untuk membandingkan performa model dengan dan tanpa fitur "Iterative Refinement". Tujuan utamanya adalah untuk membuktikan apakah *iterative refinement* memberikan peningkatan signifikan pada kualitas restorasi dokumen (PSNR, SSIM) dan akurasi pengenalan teks (CER, WER).

### 3. Detail Tugas

**A. Skrip yang Harus Dijalankan:**

Jalankan skrip `scripts/run_ablation_study.sh` dari direktori root proyek.

```bash
nohup bash scripts/run_ablation_study.sh > /dev/null 2>&1 &
```

**Penjelasan Skrip:**

Skrip ini akan secara otomatis menjalankan dua eksperimen training secara berurutan:

1.  **Baseline Run**: Menggunakan `dual_modal_gan/scripts/train32.py` dengan flag `--no_iterative_refinement` (fitur *iterative refinement* dinonaktifkan).
2.  **Iterative Refinement Run**: Menggunakan `dual_modal_gan/scripts/train32.py` tanpa flag `--no_iterative_refinement` (fitur *iterative refinement* diaktifkan).

Kedua eksperimen akan berjalan selama 30 *epoch* (sesuai konfigurasi `EPOCHS=30` dalam skrip).

**B. Output dan Logging:**

*   Log training untuk setiap run akan disimpan di direktori `logbook/` dengan nama `ablation_exp_baseline.log` dan `ablation_exp_iterative.log`.
*   Checkpoint model dan sampel gambar akan disimpan di `dual_modal_gan/outputs/checkpoints_ablation/` dan `dual_modal_gan/outputs/samples_ablation/`.
*   Setelah kedua run selesai, skrip akan menghasilkan laporan ringkasan dalam format Markdown (`ablation_study_report_YYYYMMDD_HHMMSS.md`) di direktori root proyek.

**C. Pemantauan dan Verifikasi:**

*   Pastikan kedua proses training berjalan hingga selesai (30 *epoch* atau hingga *early stopping* terpicu).
*   Pantau file log (`logbook/ablation_exp_baseline.log` dan `logbook/ablation_exp_iterative.log`) untuk progres dan potensi error.
*   Verifikasi bahwa laporan akhir (`ablation_study_report_*.md`) berhasil dibuat.

**D. Perhatian Khusus:**

*   Sebelumnya, ada isu dengan penghentian proses training di latar belakang. Pastikan proses berjalan dengan stabil dan tidak terhenti secara tidak terduga. Jika ada masalah, coba identifikasi dan laporkan penyebabnya.
*   Pastikan lingkungan `poetry run` digunakan untuk menjalankan skrip Python.

### 4. Metrik yang Harus Dipantau (dalam laporan akhir)

*   **Kualitas Visual**: PSNR (target ~40), SSIM (target ~0.99)
*   **Akurasi Teks**: CER, WER
*   **Stabilitas Training**: Fluktuasi loss, gradien norm.

### 5. Hasil yang Diharapkan

Sebuah laporan komparatif yang jelas antara performa baseline dan *iterative refinement*, dengan data kuantitatif dan kualitatif (jika memungkinkan dari sampel gambar) untuk mendukung kesimpulan mengenai efektivitas metode *iterative refinement*.

---

Terima kasih atas kerja sama Anda.

[Gemini CLI Agent]