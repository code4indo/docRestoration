# Surat Tugas: Menjalankan Studi Ablasi (Baseline vs Iterative Refinement)

**Kepada:** Agent Pelaksana Eksperimen

**Dari:** Belekok (AI/ML Engineer)

**Tanggal:** 2025-10-13

---

### **1. Tujuan Tugas:**
Menjalankan studi ablasi untuk membandingkan performa dua arsitektur Generator:
1.  **Baseline**: Generator satu-pass tanpa *iterative refinement*.
2.  **Iterative Refinement**: Generator dua-pass dengan *attention-guided refinement*.

Eksperimen ini akan menggunakan konfigurasi terbaru yang telah disesuaikan berdasarkan analisis sebelumnya.

### **2. Konteks Penting:**
*   **Contrastive Loss Dinonaktifkan**: Berdasarkan hasil eksperimen sebelumnya, `contrastive_loss_weight` telah diatur ke `0.0` untuk kedua metode guna menghindari penurunan performa dan ketidakstabilan.
*   **Epoch Minimal**: Jumlah epoch telah diatur ke `10` untuk mendapatkan hasil yang cepat (sebagai *smoke test* atau verifikasi awal). *Full training* akan dilakukan kemudian.
*   **Lingkungan**: Pastikan Anda berada di *virtual environment* yang dikelola oleh `poetry`.

### **3. Detail Tugas:**

**A. Skrip yang Harus Dijalankan:**
Jalankan skrip `run_ablation_study.sh` yang terletak di `scripts/`.

**B. Perintah Eksekusi (Jalankan di Background):**
```bash
poetry run bash scripts/run_ablation_study.sh > /dev/null 2>&1 &
```
**Penjelasan Perintah:**
*   `poetry run bash scripts/run_ablation_study.sh`: Menjalankan skrip studi ablasi menggunakan *virtual environment* `poetry`.
*   `> /dev/null 2>&1`: Mengarahkan *output* standar dan *error* standar ke `/dev/null` agar proses berjalan di *background* tanpa menampilkan *output* di konsol.
*   `&`: Menjalankan perintah di *background*.

**C. Verifikasi Proses (Opsional, untuk memantau):**
Anda dapat memantau log yang dihasilkan oleh skrip dengan perintah berikut (setelah beberapa saat):
```bash
tail -f logbook/ablation_exp_baseline.log
tail -f logbook/ablation_exp_iterative.log
```
Atau, untuk melihat apakah proses masih berjalan:
```bash
jobs
```

### **4. Hasil yang Diharapkan:**

Setelah skrip selesai berjalan (ini akan memakan waktu beberapa saat, meskipun hanya 10 epoch):
*   **Laporan Studi Ablasi**: Sebuah file Markdown baru akan dibuat di direktori root proyek dengan nama `ablation_study_report_YYYYMMDD_HHMMSS.md`. File ini akan berisi ringkasan hasil dari kedua eksperimen.
*   **Log Training**: File log terperinci akan disimpan di `logbook/` (misalnya, `ablation_exp_baseline.log` dan `ablation_exp_iterative.log`).
*   **Checkpoint dan Sampel**: Model *checkpoint* dan gambar sampel akan disimpan di `dual_modal_gan/outputs/checkpoints_ablation/` dan `dual_modal_gan/outputs/samples_ablation/`.
*   **MLflow Tracking**: Data eksperimen akan dilacak oleh MLflow. Anda dapat melihat hasilnya dengan menjalankan `poetry run mlflow ui` dan mengakses `http://localhost:5000` di browser.

### **5. Langkah Selanjutnya Setelah Tugas Selesai:**
Setelah eksperimen selesai, agen pelaksana diharapkan untuk:
1.  Memberikan konfirmasi bahwa tugas telah selesai.
2.  Menyediakan nama file laporan studi ablasi yang dihasilkan.
3.  Menunggu instruksi selanjutnya untuk analisis hasil.

---
Terima kasih atas kerja sama Anda.
