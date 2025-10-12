# 📝 LOG: Evolusi ke Strategi CTC Loss Annealing

## 📅 Tanggal: 12 Oktober 2025

## 🔴 Problem: Eksperimen Curriculum Learning (On/Off) Gagal

Eksperimen sebelumnya dengan strategi *Curriculum Learning* sederhana (mengaktifkan/menonaktifkan `ctc_loss`) menunjukkan hasil yang tidak memuaskan.

1.  **Ledakan Loss (Loss Explosion):** Meskipun fase *warm-up* (3 epoch dengan bobot CTC=0) berhasil menjaga `Generator Loss` tetap rendah dan stabil, loss tersebut **meledak hingga >1000** begitu `ctc_loss` diaktifkan dengan bobot `10.0` pada epoch ke-4.
2.  **Saturasi CTC Loss:** Metrik `CTC_loss` langsung mencapai nilai puncaknya (`100.0`) dan tidak bergerak, menandakan bahwa sinyal gradien yang berguna tidak sampai ke generator.
3.  **Bug Pelaporan:** Ditemukan bug dalam implementasi di mana metrik `CTC_loss` yang ditampilkan selama fase *warm-up* tidak menunjukkan `0.0`, meskipun bobot untuk backpropagasi sudah benar.

**Kesimpulan:** "Kejutan" dari perubahan bobot CTC dari `0` ke `10.0` terlalu drastis untuk ditangani oleh model, bahkan setelah pemanasan visual singkat.

---

## ✅ Solusi: Strategi "Loss Annealing" (Pemanasan Bertahap)

Sebagai evolusi dari *Curriculum Learning*, kita akan mengadopsi pendekatan yang lebih halus yang disebut **Loss Annealing**.

**Konsep:** Daripada menggunakan saklar on/off, kita akan menaikkan bobot `ctc_loss` secara **bertahap (ramp-up)** setelah fase *warm-up* selesai. Ini memberi model waktu untuk beradaptasi dengan objektif HTR yang baru tanpa mengorbankan stabilitas.

### Contoh Rencana Implementasi:
- **Epoch 1-10 (Warm-up):** `ctc_weight` = `0.0`
- **Epoch 11:** `ctc_weight` = `1.0`
- **Epoch 12:** `ctc_weight` = `2.0`
- **...**
- **Epoch 20:** `ctc_weight` = `10.0` (mencapai target bobot penuh)

---

## 🛠️ Action Plan

1.  **Perbaiki Bug Pelaporan Metrik:**
    - Di dalam `train32.py`, ubah kondisi kalkulasi `ctc_loss` di dalam `train_step` untuk bergantung pada `ctc_weight` yang aktif, bukan `args.ctc_loss_weight` yang statis. Ini akan memastikan metrik yang ditampilkan di log akurat.

2.  **Implementasi Logika Annealing:**
    - Modifikasi loop training utama di `train32.py`.
    - Tambahkan logika untuk menghitung `current_ctc_weight` secara dinamis pada setiap epoch setelah fase *warm-up* berakhir, dengan menaikkannya secara linear atau bertahap hingga mencapai nilai target.

3.  **Update Skrip Eksperimen:**
    - Perbarui `train32_smoke_test.sh` untuk menguji strategi *annealing* ini dalam skala kecil.

4.  **Manajemen Perubahan:**
    - Lakukan `commit` dan `push` untuk semua perubahan dengan pesan yang jelas, mendokumentasikan transisi ke strategi *annealing*.

---

## 🎯 Status

- [ ] **Menunggu Implementasi:** Perbaikan bug dan logika *loss annealing* akan diimplementasikan pada langkah berikutnya.

Tindakan ini akan menghasilkan model yang lebih kuat dan proses pelatihan yang lebih stabil, yang merupakan praktik standar dalam rekayasa ML untuk masalah optimasi yang kompleks.
