# 🏆 Tugas Delegasi: Eksekusi Kontes Final untuk Optimasi Baseline

**Tanggal:** 12 Oktober 2025  
**Untuk:** Agen Coding Copilot (Eksekutor)  
**Dari:** Agen AI/ML Strategist  
**Status:** ⏳ MENUNGGU EKSEKUSI

---

## 1. Misi Utama

Menjalankan dua eksperimen training terakhir (Kontes Final) untuk secara definitif menentukan konfigurasi hyperparameter terbaik. Model yang dihasilkan dari eksperimen ini akan menjadi **"Golden Baseline"** untuk semua riset novelty di masa depan.

## 2. Latar Belakang & Konteks

-   **Fase 1 (Shortlisting) Selesai:** Kita telah menyelesaikan eksekusi 6 eksperimen *grid search* untuk 10 epoch.
-   **Dua Finalis Teridentifikasi:** Dari hasil tersebut, dua konfigurasi menunjukkan performa yang jauh lebih unggul dari yang lain:
    1.  `p200_r5` (pixel_weight=200, rec_feat_weight=5)
    2.  `p200_r3` (pixel_weight=200, rec_feat_weight=3)
-   **Tujuan Kontes Final:** Meskipun `p200_r5` unggul dalam 10 epoch, kita perlu menjalankan kedua finalis ini untuk durasi yang lebih panjang (30 epoch) untuk memvalidasi performa jangka panjang mereka dan memastikan kita tidak terjebak pada hasil "sprinter".

## 3. Deskripsi Tugas (Job Description)

**Tugas Utama Anda adalah:**
1.  Memodifikasi skrip untuk kedua finalis (`p200_r5` dan `p200_r3`) untuk berjalan selama **30 epoch**.
2.  Menjalankan kedua skrip tersebut secara **BERURUTAN (satu per satu)** untuk menghindari konflik GPU.
3.  Melaporkan kembali setelah kedua eksekusi selesai dengan bukti bahwa file hasil telah dibuat.

## 4. Rincian Pekerjaan & Skrip yang Harus Dijalankan

Lakukan 4 langkah berikut secara berurutan:

**Langkah 1: Modifikasi Skrip `p200_r5`**
Ubah jumlah epoch dari 10 menjadi 30 pada skrip `exp_p200_r5.sh`.

```bash
# Gunakan tool 'replace' dengan parameter berikut:
# file_path: "/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/scripts/grid_search/exp_p200_r5.sh"
# old_string: "--epochs 10 \"
# new_string: "--epochs 30 \"
```

**Langkah 2: Jalankan Eksperimen `p200_r5`**
Jalankan skrip yang telah dimodifikasi. Tunggu hingga proses ini selesai sepenuhnya sebelum melanjutkan ke langkah 3.

```bash
# Gunakan tool 'run_shell_command' dengan perintah:
./scripts/grid_search/exp_p200_r5.sh > /dev/null 2>&1
```

**Langkah 3: Modifikasi Skrip `p200_r3`**
Ubah jumlah epoch dari 10 menjadi 30 pada skrip `exp_p200_r3.sh`.

```bash
# Gunakan tool 'replace' dengan parameter berikut:
# file_path: "/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/scripts/grid_search/exp_p200_r3.sh"
# old_string: "--epochs 10 \"
# new_string: "--epochs 30 \"
```

**Langkah 4: Jalankan Eksperimen `p200_r3`**
Jalankan skrip kedua ini. Tunggu hingga proses ini selesai.

```bash
# Gunakan tool 'run_shell_command' dengan perintah:
./scripts/grid_search/exp_p200_r3.sh > /dev/null 2>&1
```

## 5. Kriteria Keberhasilan & Laporan Akhir

-   **Kriteria Keberhasilan:** Kedua perintah `run_shell_command` di atas berjalan sampai selesai dengan *exit code 0*.
-   **Laporan Akhir:** Setelah Langkah 4 selesai, tugas Anda selesai. Sebagai bukti penyelesaian, laporkan kembali dengan output dari perintah `ls` berikut:

    ```bash
    ls -l /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/outputs/checkpoints_grid_search/p200_r5/metrics/training_metrics_fp32_final.json
    ls -l /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/outputs/checkpoints_grid_search/p200_r3/metrics/training_metrics_fp32_final.json
    ```

## 6. Langkah Selanjutnya (Pasca Eksekusi)

Setelah Anda melaporkan bahwa tugas telah selesai, saya (Agen AI/ML Strategist) akan mengambil alih untuk menganalisis hasil akhir dari kedua eksperimen ini dan secara resmi menentukan konfigurasi "Golden Baseline" kita.

Terima kasih atas bantuan Anda.
