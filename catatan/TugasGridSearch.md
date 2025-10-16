# 📝 Tugas Delegasi: Eksekusi Grid Search untuk Optimasi Baseline

**Tanggal:** 12 Oktober 2025  
**Untuk:** Agen Coding Eksekutor  
**Dari:** Agen AI/ML Strategist  
**Status:** ⏳ MENUNGGU EKSEKUSI

---

## 1. Misi Utama

Melakukan eksekusi serangkaian eksperimen training (*Grid Search*) yang telah disiapkan untuk menemukan kombinasi bobot loss terbaik bagi model restorasi dokumen kita. Misi ini adalah bagian krusial dari tahap **"Optimasi Baseline"** sebelum melanjutkan ke pengembangan fitur baru (novelty).

## 2. Latar Belakang & Konteks

Kita telah berhasil melakukan beberapa langkah awal:
1.  **Analisis Skala Loss:** Sebuah eksperimen singkat telah dijalankan untuk memahami magnitudo alami dari setiap komponen loss (`pixel_loss`, `rec_feat_loss`, `adv_loss`).
2.  **Perencanaan Grid Search:** Berdasarkan analisis tersebut, serangkaian 6 skrip eksperimen telah dibuat di direktori `scripts/grid_search/` untuk menguji berbagai kombinasi bobot loss.
3.  **Eksperimen Pertama:** Eksperimen pertama (`exp_p50_r3.sh`) telah berhasil dijalankan.
4.  **Kegagalan Eksekusi Paralel:** Upaya untuk menjalankan 5 eksperimen sisanya secara paralel **gagal** karena error `CUDA out of memory`. Ini terjadi karena semua skrip mencoba menggunakan GPU yang sama secara serentak.

Oleh karena itu, tugas Anda adalah menjalankan sisa eksperimen ini dengan benar.

## 3. Deskripsi Tugas (Job Description)

**Tugas Utama Anda adalah menjalankan 5 skrip eksperimen yang tersisa secara BERURUTAN (satu per satu).**

Setiap skrip akan menjalankan proses training selama 10 epoch. Anda harus memastikan setiap skrip selesai sepenuhnya (dengan atau tanpa error) sebelum menjalankan skrip berikutnya. Ini sangat penting untuk menghindari konflik sumber daya GPU.

## 4. Skrip yang Harus Dijalankan

Jalankan 5 perintah berikut secara berurutan. Pastikan satu perintah selesai sebelum memulai yang berikutnya. Output dari setiap perintah sengaja dialihkan ke `/dev/null` untuk menjaga kebersihan konsol.

1.  Jalankan eksperimen **p50_r5**:
    ```bash
    ./scripts/grid_search/exp_p50_r5.sh > /dev/null 2>&1
    ```

2.  Setelah yang pertama selesai, jalankan eksperimen **p100_r3**:
    ```bash
    ./scripts/grid_search/exp_p100_r3.sh > /dev/null 2>&1
    ```

3.  Setelah yang kedua selesai, jalankan eksperimen **p100_r5**:
    ```bash
    ./scripts/grid_search/exp_p100_r5.sh > /dev/null 2>&1
    ```

4.  Setelah yang ketiga selesai, jalankan eksperimen **p200_r3**:
    ```bash
    ./scripts/grid_search/exp_p200_r3.sh > /dev/null 2>&1
    ```

5.  Setelah yang keempat selesai, jalankan eksperimen **p200_r5**:
    ```bash
    ./scripts/grid_search/exp_p200_r5.sh > /dev/null 2>&1
    ```

## 5. Kriteria Keberhasilan & Laporan

-   **Kriteria Keberhasilan:** Kelima perintah di atas berjalan sampai selesai.
-   **Laporan Akhir:** Setelah semua eksekusi selesai, laporkan kembali dengan memberikan output dari perintah berikut untuk memverifikasi bahwa semua direktori hasil telah dibuat:
    ```bash
    ls -l dual_modal_gan/outputs/checkpoints_grid_search/
    ```

## 6. Langkah Selanjutnya (Setelah Tugas Anda Selesai)

Setelah Anda menyelesaikan eksekusi, saya (Agen AI/ML Strategist) akan mengambil alih untuk melakukan tugas berikut:
-   Membaca semua file metrik dari setiap direktori hasil.
-   Menganalisis performa (`PSNR`, `CER`, `combined_score`) dari setiap konfigurasi.
-   Menentukan 2-3 konfigurasi terbaik untuk dilanjutkan ke Fase 2 (eksperimen dengan 30 epoch).

Terima kasih atas kerja samanya.
