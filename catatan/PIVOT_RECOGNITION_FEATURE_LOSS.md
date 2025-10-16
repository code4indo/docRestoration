# 🎯 PIVOT: Strategi Recognition Feature Loss

## 📅 Tanggal: 12 Oktober 2025

## 🔴 Problem: Kegagalan Konvergensi CTC Loss End-to-End

Serangkaian eksperimen, termasuk yang menggunakan *loss annealing*, telah membuktikan secara konklusif bahwa pendekatan CTC loss end-to-end tidak stabil dan tidak efektif untuk melatih generator kita. 

**Temuan Kunci:**
1.  **Loss Saturasi:** `CTC_loss` secara konsisten mencapai nilai puncaknya (`100.0`) bahkan dengan bobot yang sangat kecil, yang berarti tidak ada sinyal gradien yang berguna.
2.  **Pemanasan Tidak Cukup:** Pemanasan visual (*visual warm-up*) hanya meningkatkan kualitas piksel, tetapi tidak menghasilkan fitur yang cukup berarti bagi Recognizer untuk mencegah ledakan loss.

**Kesimpulan:** Gradien yang mengalir balik dari lapisan akhir Recognizer terlalu *chaotic* dan tidak informatif untuk memandu Generator secara efektif.

---

## 💡 Hipotesis Baru: "The Sweet Spot" Feature Loss

Kita melakukan pivot dari pendekatan yang terbukti gagal ke strategi baru yang lebih canggih dan stabil. Daripada menggunakan sinyal dari lapisan *akhir* Recognizer (logits), kita akan menggunakan sinyal dari lapisan **tengah**.

**Konsep:**
Di dalam arsitektur Recognizer, terdapat sebuah "sweet spot" (kemungkinan besar setelah blok CNN terakhir) di mana gambar telah menjadi representasi fitur tingkat tinggi (guratan, bentuk karakter) tetapi belum menjadi probabilitas akhir yang tidak stabil. Fitur ini secara teoretis merupakan "pemandu" yang ideal: cukup kaya untuk mewakili konten teks, namun cukup stabil untuk tidak menyebabkan ledakan gradien.

---

## ✅ Rencana Aksi Implementasi

1.  **Modifikasi Arsitektur `recognizer.py`:**
    -   Saya akan mengubah fungsi `load_frozen_recognizer`.
    -   Fungsi ini akan dimodifikasi untuk membangun model Keras baru yang membungkus recognizer beku dan menghasilkan **dua output**:
        1.  `final_logits`: Output asli untuk monitoring `CTC_loss`.
        2.  `feature_map`: Output dari lapisan "sweet spot" yang telah diidentifikasi.

2.  **Definisikan Ulang `train32.py`:**
    -   **Loss Baru:** Tambahkan `recognition_feature_loss` (misalnya, `MeanSquaredError`) yang membandingkan `feature_map` dari gambar bersih dan gambar yang dihasilkan generator.
    -   **Update `total_gen_loss`:** Hapus `ctc_loss` dari perhitungan gradien. `total_gen_loss` sekarang akan terdiri dari:
        -   `pixel_loss` (L1/MAE)
        -   `adversarial_loss`
        -   `recognition_feature_loss` (yang baru)
    -   **Hyperparameter Baru:** Tambahkan argumen `--rec_feat_loss_weight` untuk mengontrol bobot loss baru ini.
    -   **Monitoring:** `CTC_loss` tetap dihitung dan ditampilkan di log, tetapi murni sebagai **metrik monitor** untuk mengukur peningkatan keterbacaan teks, bukan untuk backpropagasi.

3.  **Update Skrip Eksperimen:**
    -   Perbarui `train32_smoke_test.sh` untuk menggunakan hyperparameter baru dan menguji strategi ini.

---

## 🎯 Status

-   [ ] **Menunggu Implementasi:** Perubahan arsitektur pada `recognizer.py` dan `train32.py` akan diimplementasikan pada langkah-langkah berikutnya.

Ini adalah pivot strategis yang didasarkan pada bukti empiris. Langkah ini tidak hanya bertujuan untuk memperbaiki masalah, tetapi juga berpotensi menemukan arsitektur baru yang lebih unggul dan menjadi inti dari *novelty* penelitian ini.