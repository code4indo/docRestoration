#  handover: Rencana dan Masalah untuk Agen Berikutnya

## 📅 Tanggal: 12 Oktober 2025

### 🎯 Tujuan Proyek

Restorasi dokumen terdegradasi menggunakan Generative Adversarial Network (GAN) dengan tujuan akhir meningkatkan keterbacaan teks oleh model HTR (Handwritten Text Recognition).

---

### 🔴 Masalah yang Dihadapi & Upaya Sebelumnya

Eksperimen kami untuk menggunakan **CTC loss secara end-to-end** sebagai pemandu Generator telah **gagal secara konsisten**. Masalah utamanya adalah **ketidakstabilan gradien**.

1.  **Direct CTC Loss:** Menyebabkan *loss explosion* dan pelatihan gagal.
2.  **Curriculum Learning (On/Off):** Gagal. `CTC_loss` langsung saturasi (mencapai nilai clip `100.0`) saat diaktifkan setelah fase pemanasan.
3.  **Loss Annealing (Ramp-up):** Gagal. Bahkan dengan pengenalan bobot CTC secara bertahap, `CTC_loss` tetap saturasi di `100.0` dan tidak memberikan sinyal belajar yang berguna.

**INSIGHT UTAMA:** Sinyal gradien dari lapisan akhir (logits) model Recognizer yang beku terlalu *chaotic* dan tidak informatif untuk melatih model Generator U-Net secara efektif dari awal. Upaya untuk "menjinakkannya" tidak berhasil.

---

### ✅ Rencana Baru: Pivot ke "Recognition Feature Loss"

Berdasarkan kegagalan di atas, kita telah menyetujui untuk melakukan **pivot strategis**.

**Hipotesis:** Menggunakan representasi fitur dari **lapisan tengah** Recognizer (sebelum lapisan Transformer/RNN) akan memberikan sinyal pemandu yang lebih stabil namun tetap kaya makna.

**Rencana Implementasi:**

1.  **Modifikasi `recognizer.py` (✅ SELESAI):**
    -   Model Recognizer telah dimodifikasi untuk memiliki **dua output**:
        1.  `final_logits`: Untuk memonitor `CTC_loss` (tidak untuk backprop).
        2.  `feature_map`: Output dari blok CNN terakhir, yang akan digunakan untuk loss.
    -   Bug di mana model tidak benar-benar beku (`trainable=False`) juga telah diperbaiki.

2.  **Modifikasi `train32.py` (⏳ LANGKAH BERIKUTNYA):**
    -   **Definisikan Loss Baru:** Buat `recognition_feature_loss` menggunakan `MeanSquaredError` (MSE) untuk membandingkan `feature_map` dari gambar bersih dan gambar hasil generator.
    -   **Update `total_gen_loss`:** Hapus `ctc_loss` dari kalkulasi gradien. `total_gen_loss` akan menjadi kombinasi dari `pixel_loss`, `adversarial_loss`, dan `recognition_feature_loss`.
    -   **Hyperparameter:** Tambahkan argumen baru `--rec_feat_loss_weight`.

3.  **Update `train32_smoke_test.sh`:**
    -   Sesuaikan skrip untuk menjalankan eksperimen dengan hyperparameter baru.

4.  **Validasi:**
    -   Jalankan *smoke test* untuk memvalidasi apakah `recognition_feature_loss` menurun dan `CTC_loss` (sebagai monitor) juga menunjukkan tren penurunan.

---

###  handover Status

-   **Implementasi `recognizer.py` sudah selesai.**
-   **Tugas Anda adalah melanjutkan dengan memodifikasi `train32.py`** sesuai dengan rencana di atas.
