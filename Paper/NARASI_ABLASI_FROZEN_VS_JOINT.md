# Narasi Presentasi: Studi Ablasi Frozen vs Joint Training
**Slide 12 - Seminar Hasil Tesis**

---

## 🎤 Skrip Narasi (2-3 menit)

### Pembukaan
"Baik, sekarang kita masuk ke salah satu temuan paling kritis dari penelitian ini, yaitu **studi ablasi frozen versus joint training**."

---

### Penjelasan Grafik (Sebelah Kiri)
"Mari kita lihat grafik sebelah kiri terlebih dahulu. Grafik ini menunjukkan **trajektori CER** atau Character Error Rate selama 20 epoch pelatihan."

**[Pause - tunjuk garis biru dan merah]**

"Garis **biru** ini adalah frozen recognizer yang kita usulkan, sedangkan garis **merah** adalah joint training."

"Anda bisa lihat dengan jelas perbedaan yang **sangat dramatis**. Frozen recognizer menunjukkan penurunan CER yang **konsisten dan stabil** dari awal hingga akhir. Sedangkan joint training? Sangat fluktuatif, bahkan pada epoch tertentu **CER meroket hingga 69 persen** — ini yang disebut dengan **catastrophic forgetting**."

---

### Penjelasan Tabel (Sebelah Kanan)
"Sekarang mari kita lihat hasil kuantitatifnya di tabel sebelah kanan."

**[Pause - tunjuk baris CER]**

"CER final frozen recognizer: **31.63 persen**. Bandingkan dengan joint training: **42.85 persen**. Artinya frozen jauh lebih baik dalam menjaga kemampuan membaca teks."

**[Pause - tunjuk baris PSNR]**

"PSNR frozen: **23.09 dB**, sedangkan joint hanya **10.71 dB**. Ini menunjukkan kualitas visual frozen juga lebih superior."

**[Pause - tunjuk baris Waktu/epoch]**

"Yang lebih menarik lagi, **waktu pelatihan**. Frozen hanya butuh **85 detik per epoch**, sedangkan joint butuh **630 detik** — itu **7.4 kali lebih lambat**!"

**[Pause - tunjuk baris Mode Collapse]**

"Dan terakhir, frozen **tidak mengalami mode collapse**, sedangkan joint training mengalaminya. Ini sangat fatal karena model bisa menghasilkan output yang seragam dan tidak beragam."

---

### Temuan Kritis (Alert Block)
**[Pause - tunjuk kotak merah]**

"Seperti yang saya singgung tadi, joint training mengalami **catastrophic forgetting**. Ini terjadi karena recognizer yang dilatih bersamaan dengan generator **melupakan** kemampuan dasarnya membaca teks. CER-nya sempat mencapai **69.06 persen** — ini hampir seperti tidak bisa membaca sama sekali!"

---

### Kesimpulan Kuat (Footer)
**[Pause - tunjuk teks footer]**

"Jadi kesimpulannya sangat jelas: **Frozen recognizer unggul di semua aspek** — stabilitas, akurasi, kecepatan, dan efisiensi. Secara statistik, perbedaan ini sangat signifikan dengan **p-value kurang dari 0.001** dan **Cohen's d sebesar 6.23**, yang termasuk kategori efek **sangat besar**."

---

### Penutup
"Temuan ini menjadi salah satu kontribusi utama penelitian kami: **strategi frozen recognizer terbukti secara empiris lebih efektif daripada joint training** dalam konteks restorasi dokumen untuk HTR. Ini bukan hanya soal performa, tapi juga **stabilitas dan efisiensi komputasi**."

**[Transisi]**
"Baik, mari kita lanjut ke studi ablasi selanjutnya..."

---

## 📝 Poin-Poin Kunci

1. **Stabilitas**: Frozen stabil, joint fluktuatif
2. **Akurasi**: CER 31.63% vs 42.85%
3. **Kualitas Visual**: PSNR 23.09 vs 10.71 dB
4. **Efisiensi**: 7.4× lebih cepat
5. **Robustness**: Tidak ada mode collapse
6. **Catastrophic Forgetting**: Joint training kehilangan kemampuan dasar
7. **Validitas Statistik**: p<0.001, Cohen's d=6.23 (very large effect)

---

## 🎯 Tips Presentasi

- **Gunakan pointer laser** untuk menunjuk grafik dan angka spesifik
- **Pause sesaat** setelah menyebutkan angka penting (beri waktu audiens mencerna)
- **Tekankan kontras** antara garis biru (stabil) dan merah (chaos)
- **Suara antusias** saat menyebutkan temuan kritis (catastrophic forgetting)
- **Slow down** saat membaca statistik (p-value, Cohen's d)
- **Eye contact** dengan dewan penguji saat menyimpulkan

---

## ⏱️ Timing Breakdown

- Pembukaan: 10 detik
- Penjelasan Grafik: 40 detik
- Penjelasan Tabel: 50 detik
- Temuan Kritis: 30 detik
- Kesimpulan: 30 detik
- Transisi: 10 detik

**Total: ~2.5 menit**

---

## 🔴 Antisipasi Pertanyaan

**Q: Mengapa frozen recognizer bisa lebih stabil?**  
A: "Karena bobot recognizer tidak berubah selama pelatihan, sehingga sinyal gradien yang diberikan ke generator **konsisten**. Pada joint training, recognizer yang berubah-ubah memberikan sinyal **kontradiktif** ke generator."

**Q: Apakah frozen recognizer tidak kehilangan kesempatan untuk beradaptasi?**  
A: "Benar, tapi hasil empiris menunjukkan bahwa **stabilitas lebih penting daripada adaptabilitas** dalam kasus ini. CTC loss dari frozen recognizer sudah cukup kuat sebagai supervised signal untuk mengarahkan generator."

**Q: Kenapa joint training mengalami catastrophic forgetting?**  
A: "Karena recognizer harus belajar dua tugas sekaligus: (1) membaca teks, dan (2) memberikan feedback ke generator. Kedua tujuan ini **saling berkonflik**, sehingga recognizer melupakan kemampuan dasarnya."

---

_Dokumen ini dibuat untuk mendukung presentasi Slide 12: Studi Ablasi Frozen vs Joint Training_
