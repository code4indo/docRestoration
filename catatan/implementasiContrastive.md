# Surat Tugas: Eksekusi Eksperimen Studi Ablasi Contrastive Loss

**Kepada:** Agen Coding Copilot (Eksekutor)  
**Dari:** Agen AI/ML Strategist  
**Tanggal:** 13 Oktober 2025  
**Status:** 🔴 **UNTUK DILAKSANAKAN**

---

## 1. Latar Belakang & Tujuan

Kita telah berhasil menyelesaikan implementasi teknis dari arsitektur baru yang mengandung **Cross-Modal Contrastive Loss**. Semua file yang diperlukan (`text_encoder.py`, `recognizer.py`, `train32.py`) telah dimodifikasi dan diverifikasi melalui *smoke test*.

Tujuan dari tugas ini adalah untuk **menjalankan studi ablasi secara penuh** guna mengukur dampak dan efektivitas dari *contrastive loss* yang baru kita implementasikan. Hasil dari kedua eksperimen ini akan menjadi data primer untuk membuktikan *novelty* riset kita.

## 2. Deskripsi Tugas

Anda diminta untuk menjalankan dua skrip eksperimen secara **berurutan (satu per satu, tidak paralel)** untuk menghindari konflik sumber daya GPU.

### **Tugas 1: Jalankan Eksperimen Grup Kontrol (`contrastive_off`)**

- **Perintah:**
  ```bash
  /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/scripts/novelty_experiments/exp_contrastive_off.sh
  ```
- **Tujuan:** Menjalankan training penuh (30 epoch) dengan konfigurasi Golden Baseline, di mana *contrastive loss* dinonaktifkan (`--contrastive_loss_weight 0.0`).
- **Hasil yang Diharapkan:** Proses training selesai dengan sukses dan menghasilkan artefak di direktori `dual_modal_gan/outputs/checkpoints_novelty/contrastive_off/`.

### **Tugas 2: Jalankan Eksperimen Grup Perlakuan (`contrastive_on`)**

- **Perintah (jalankan HANYA setelah Tugas 1 selesai):**
  ```bash
  /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/scripts/novelty_experiments/exp_contrastive_on.sh
  ```
- **Tujuan:** Menjalankan training penuh (30 epoch) dengan konfigurasi Golden Baseline dan **mengaktifkan** *contrastive loss* (`--contrastive_loss_weight 1.0`).
- **Hasil yang Diharapkan:** Proses training selesai dengan sukses dan menghasilkan artefak di direktori `dual_modal_gan/outputs/checkpoints_novelty/contrastive_on/`.

## 3. Kriteria Keberhasilan

Tugas ini dianggap selesai jika semua kondisi berikut terpenuhi:

- [ ] Kedua skrip (`exp_contrastive_off.sh` dan `exp_contrastive_on.sh`) berjalan hingga selesai dengan *exit code 0*.
- [ ] Direktori output `.../outputs/checkpoints_novelty/contrastive_off/` berhasil dibuat dan berisi file `metrics/training_metrics_fp32_final.json`.
- [ ] Direktori output `.../outputs/checkpoints_novelty/contrastive_on/` berhasil dibuat dan berisi file `metrics/training_metrics_fp32_final.json`.
- [ ] File log `logbook/novelty_exp_contrastive_off.log` dan `logbook/novelty_exp_contrastive_on.log` telah dibuat.

## 4. Informasi Tambahan

- **Perkiraan Waktu:** Setiap eksperimen akan memakan waktu sekitar 30-60 menit. Total waktu tugas sekitar 1-2 jam.
- **Tujuan Akhir:** Keberhasilan eksekusi ini akan menyediakan data yang diperlukan untuk analisis perbandingan, yang merupakan langkah krusial berikutnya dalam riset ini.

Mohon laksanakan tugas ini dengan cermat. Terima kasih.
