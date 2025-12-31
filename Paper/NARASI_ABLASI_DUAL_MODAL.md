# Narasi Presentasi: Studi Ablasi Diskriminator Dual-Modal
**Slide 13 - Seminar Hasil Tesis**

---

## 🎤 Skrip Narasi (2-3 menit)

### Pembukaan
"Selanjutnya, kita akan membahas studi ablasi kedua yang **tidak kalah penting**, yaitu evaluasi kontribusi **diskriminator dual-modal** — salah satu komponen yang awalnya kami klaim sebagai inovasi utama."

**[Pause - serius tapi tetap percaya diri]**

"Dan ini adalah contoh sempurna dari **kejujuran ilmiah**: hasil penelitian tidak selalu sesuai hipotesis awal, tapi justru di situlah kita menemukan **insight yang lebih dalam**."

---

### Penjelasan Tabel Hasil
"Mari kita lihat hasil perbandingan pada tabel ini. Kami menguji dua konfigurasi pada **710 sampel validasi selama 50 epoch**:"

**[Pause - tunjuk baris pertama]**

"**Pertama**, CNN Only — ini adalah diskriminator baseline yang hanya menggunakan cabang visual berbasis Convolutional Neural Network."

**[Pause - tunjuk baris kedua]**

"**Kedua**, Dual-Modal — ini adalah usulan kami yang menambahkan cabang tekstual berbasis BiLSTM untuk evaluasi keterbacaan teks."

---

### Analisis Metrik (Kolom demi Kolom)

**[Pause - tunjuk kolom PSNR]**

"**PSNR**: CNN Only menghasilkan 30.63 dB, Dual-Modal 30.91 dB. Selisihnya? Hanya **+0.28 dB**."

**[Pause - tunjuk kolom SSIM]**

"**SSIM**: CNN 0.9854, Dual-Modal 0.9869. Delta-nya **+0.0015** — secara praktis ini **negligible**."

**[Pause - tunjuk kolom CER]**

"**CER**: Yang paling menarik, CNN 27.12%, Dual-Modal 27.11%. Perbedaannya? **Minus 0.01 persen saja** — praktis **identik**."

---

### Statistik Signifikansi (Baris Terakhir)

**[Pause - tunjuk p-value dengan emphasis]**

"Dan inilah yang paling krusial: **p-value lebih besar dari 0.05**. Secara statistik, ini artinya perbedaan tersebut **TIDAK SIGNIFIKAN**. Dengan kata lain, penambahan cabang tekstual dual-modal **tidak memberikan kontribusi yang berarti**."

**[Pause - biarkan audiens mencerna]**

---

### Implikasi Teoretis (Alert Block Oranye)

**[Transisi ke kotak kiri - nada reflektif]**

"Lalu apa implikasinya? Setelah kami analisis lebih dalam, kami menemukan insight yang sangat penting:"

**[Pause - baca perlahan dengan penekanan]**

"**Kontribusi dual-modal ternyata marginal** karena pengawasan tekstual **sudah dijalankan** oleh frozen recognizer melalui **CTC loss**!"

**[Pause - jelaskan]**

"Artinya begini: frozen recognizer yang kita pakai di generator sudah memberikan supervised signal yang kuat untuk keterbacaan teks. Jadi, menambahkan cabang tekstual **lagi** di diskriminator hanya **redundan** — tidak ada nilai tambah."

**[Gesture: tangan menunjukkan overlap]**

"Ini seperti memasang dua alarm untuk bangun pagi — yang kedua tidak banyak membantu kalau yang pertama sudah efektif."

---

### Prinsip Parsimoni (Example Block Hijau)

**[Transisi ke kotak kanan - nada positif dan actionable]**

"Dari temuan ini, kami mengambil kesimpulan berdasarkan **prinsip parsimoni** dalam sains:"

**[Pause - baca dengan percaya diri]**

"**Diskriminator CNN tunggal DIREKOMENDASIKAN** untuk efisiensi komputasi **tanpa mengorbankan performa**."

**[Pause - jelaskan benefit]**

"Mengapa ini penting?"

1. **Efisiensi**: Lebih sedikit parameter = lebih cepat training
2. **Simplicity**: Arsitektur lebih sederhana = lebih mudah maintain
3. **Same Performance**: Hasilnya identik dengan versi dual-modal

**[Gesture: tangan horizontal menunjukkan kesetaraan]**

"Jadi kita dapat **kesederhanaan tanpa trade-off performa**."

---

### Kejujuran Ilmiah (Transisi Positif)

**[Nada jujur tapi bangga]**

"Saya ingin menekankan: temuan ini **bukan kegagalan**. Ini adalah **kontribusi ilmiah yang valid**."

**[Pause - eye contact dengan dewan penguji]**

"Dalam penelitian, **membuktikan bahwa sesuatu tidak perlu** sama pentingnya dengan **membuktikan bahwa sesuatu perlu**. Kami telah memberikan **bukti empiris** bahwa kompleksitas dual-modal **tidak justified** untuk kasus ini."

**[Pause]**

"Ini sesuai dengan prinsip **Occam's Razor**: *given equal performance, the simpler solution is better*."

---

### Penutup & Transisi

**[Nada menyimpulkan]**

"Jadi, untuk studi ablasi diskriminator, kesimpulannya sangat clear: **frozen recognizer sudah sufficient** untuk supervised signal tekstual. Diskriminator cukup fokus pada evaluasi visual saja."

**[Transisi confident]**

"Temuan ini akan menjadi salah satu rekomendasi utama kami untuk penelitian lanjutan: **gunakan arsitektur yang lebih sederhana**. Dan ini membawa kita ke slide selanjutnya..."

---

## 📝 Poin-Poin Kunci

1. **Hasil Marginal**: Δ PSNR +0.28 dB, Δ SSIM +0.0015, Δ CER -0.01%
2. **Tidak Signifikan**: p > 0.05 secara statistik
3. **Root Cause**: Frozen recognizer sudah memberikan CTC loss (redundansi)
4. **Prinsip Parsimoni**: Simplicity tanpa trade-off performa
5. **Kejujuran Ilmiah**: Negative results are valid scientific findings
6. **Rekomendasi Praktis**: Gunakan CNN single-modal untuk efisiensi
7. **Occam's Razor**: Solusi sederhana lebih baik jika hasilnya sama

---

## 🎯 Tips Presentasi

### Body Language & Delivery
- **Jujur tapi percaya diri** — ini bukan kegagalan, ini discovery
- **Slow down** saat membaca p-value dan statistik
- **Eye contact kuat** saat menjelaskan kejujuran ilmiah
- **Gesture horizontal** saat explain redundansi (overlap)
- **Nod** saat menyebutkan prinsip parsimoni (affirmative)

### Voice Modulation
- **Nada netral** saat membaca angka tabel
- **Nada reflektif** saat explain implikasi teoretis
- **Nada positif** saat discuss prinsip parsimoni
- **Nada bangga** saat defend kejujuran ilmiah

### Timing
- **Pause lebih lama** setelah menyebut p > 0.05 (beri waktu sink in)
- **Slow down** saat jelaskan redundansi (konsep penting)
- **Speed up sedikit** saat transisi ke slide berikutnya

---

## ⏱️ Timing Breakdown

- Pembukaan (Kejujuran Ilmiah): 20 detik
- Penjelasan Tabel & Metrik: 50 detik
- P-value & Signifikansi: 20 detik
- Implikasi Teoretis: 40 detik
- Prinsip Parsimoni: 30 detik
- Kejujuran Ilmiah Defense: 30 detik
- Penutup & Transisi: 15 detik

**Total: ~3 menit**

---

## 🔴 Antisipasi Pertanyaan

### Q1: "Jika dual-modal tidak signifikan, mengapa tetap dimasukkan dalam usulan?"
**A (Jujur & Ilmiah):**  
"Pertanyaan yang sangat bagus, Pak/Bu. Pada saat desain awal, kami **hipotesiskan** bahwa dual-modal akan memberikan kontribusi, berdasarkan literature gap yang kami identifikasi. Namun, **studi ablasi** inilah yang membuktikan bahwa frozen recognizer sudah cukup. Ini adalah **proses ilmiah yang sehat** — kita propose, kita test, dan kita revise based on evidence. Dan temuan ini **valid contribution** karena memberikan guidance untuk future work."

---

### Q2: "Apakah ini berarti penelitian gagal mencapai tujuan?"
**A (Defensive tapi Positif):**  
"Sama sekali tidak, Pak/Bu. Tujuan penelitian kami adalah **mengatasi kesenjangan visual-HTR**, dan itu **tercapai** dengan CER reduction 58.2%. Yang tidak tervalidasi hanya **salah satu komponen hipotesis**. Bahkan, temuan bahwa dual-modal redundant adalah **scientific contribution** yang berharga — kita memberikan **empirical evidence** untuk prinsip parsimoni dalam document restoration. Future researchers sekarang tahu bahwa mereka **tidak perlu** membuang resource untuk dual-modal discriminator."

---

### Q3: "Mengapa tidak menguji ini lebih awal sebelum finalize design?"
**A (Realistis & Metodologis):**  
"Ini memang timing dilemma dalam research, Pak/Bu. Studi ablasi **membutuhkan** model yang sudah trained hingga convergence — kita tidak bisa test ini di early stage. Dan kami sudah follow **best practice**: train model lengkap dulu, baru dissect component by component untuk understand contribution masing-masing. Ini sesuai dengan **DSRM methodology** yang kami pakai — iteration antara design, demonstration, dan evaluation. Temuan ini justru **validates** pentingnya ablation studies sebagai post-hoc analysis."

---

### Q4: "Apakah frozen recognizer di generator dan dual-modal di discriminator tidak beda fungsinya?"
**A (Teknis & Clear):**  
"Betul sekali, Pak/Bu, secara konseptual mereka **berbeda posisi**. Tapi yang kami temukan adalah bahwa **CTC loss dari frozen recognizer** yang di-backpropagate ke generator **sudah memberikan supervised signal tekstual yang sangat kuat**. Sehingga ketika discriminator menambahkan cabang tekstual lagi, sinyal tersebut **tidak menambah informasi baru** yang meaningful. Discriminator cukup fokus pada **adversarial training untuk realism visual** saja, sementara **text readability** sudah dijaga oleh frozen recognizer lewat direct supervision."

---

### Q5: "Apa implikasi praktis untuk ANRI atau deployment?"
**A (Praktis & Value-Oriented):**  
"Implikasi praktisnya sangat positif, Pak/Bu:  
1. **Deployment lebih ringan** — model production bisa pakai CNN single-modal, lebih hemat memory dan compute  
2. **Maintenance lebih mudah** — arsitektur lebih simple, debugging lebih cepat  
3. **Training lebih cepat** — untuk retraining atau fine-tuning di future  
4. **Same quality** — tidak ada pengorbanan performa  

Jadi untuk ANRI, ini **good news** karena solution kami **lebih scalable** untuk production environment."

---

## 💡 Frame Positif untuk Temuan "Negatif"

### Strategi Komunikasi:
1. **Don't apologize** — ini bukan kesalahan
2. **Emphasize scientific rigor** — ablation study yang proper
3. **Highlight practical benefit** — parsimony = efficiency
4. **Connect to broader principles** — Occam's Razor, parsimony
5. **Reframe as contribution** — empirical evidence for simplicity

### Key Phrases untuk Defend:
- "Bukti empiris untuk prinsip parsimoni"
- "Valid scientific contribution"
- "Kejujuran ilmiah"
- "Guidance untuk future research"
- "Efficiency tanpa trade-off"
- "Occam's Razor in practice"

---

## 📊 Visual Cues (Jika Ada Pertanyaan Lanjutan)

Jika ada yang tanya detail teknis, siapkan untuk refer ke:
- **Backup slide**: Distribusi Loss Weights (slide backup 3)
- **Chapter 5**: Section ablasi diskriminator detail
- **Angka konkret**: 17.4M params dual-modal vs CNN only

---

## 🎓 Academic Integrity Note

Slide ini menunjukkan **maturity sebagai researcher**:
- Tidak hide negative results
- Transparent tentang hypothesis yang tidak tervalidasi
- Mampu extract meaningful insight dari "unexpected" findings
- Follow scientific method dengan rigorous

**Ini adalah strength, bukan weakness!**

---

_Dokumen ini dibuat untuk mendukung presentasi Slide 13: Studi Ablasi Diskriminator Dual-Modal_  
_Fokus: Kejujuran ilmiah, prinsip parsimoni, dan defensive strategy untuk temuan "negatif"_
