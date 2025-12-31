# Narasi Presentasi: Studi Ablasi Loss Function & Curriculum Learning
**Slide 14 & 15 - Seminar Hasil Tesis**

---

## 🎤 SLIDE 14: Narasi Optimasi 5 Komponen Loss (2.5 menit)

### Pembukaan
"Setelah memvalidasi komponen arsitektur, sekarang kita evaluasi **kontribusi masing-masing komponen loss function** yang menjadi inti dari sistem optimasi kita."

---

### Penjelasan Tabel Hasil Ablasi Inkremental

**[Tunjuk tabel]**

"Kita melakukan **ablasi inkremental** — artinya menambahkan satu komponen secara bertahap untuk melihat kontribusi masing-masing. Mari kita lihat hasilnya:"

**[Pause - tunjuk baris 1]**

"**Eksperimen 1**: Baseline dengan L1 pixel loss saja. PSNR 24.59 dB, SSIM 0.9614. Ini adalah fondasi dasar rekonstruksi piksel."

**[Pause - tunjuk baris 2]**

"**Eksperimen 2**: Menambahkan adversarial loss. Ini memberikan **kontribusi terbesar** — PSNR naik **+0.27 dB** menjadi 24.86 dB. Komponen adversarial ini yang membuat restorasi terlihat **realistic** dan natural, bukan hanya akurat secara piksel."

**[Pause - tunjuk baris 3]**

"**Eksperimen 3**: Menambahkan perceptual loss dari VGG. Menariknya, PSNR sedikit turun -0.04 dB, tapi SSIM naik menjadi **0.9630** — nilai **tertinggi**. Ini adalah trade-off yang expected: perceptual loss fokus pada **structural similarity**, bukan pixel-wise accuracy."

**[Pause - tunjuk baris 4]**

"**Eksperimen 4**: Ini yang paling krusial — penambahan **CTC loss**. Komponen ini **mengaktifkan kemampuan HTR** dengan CER **29.66 persen**. Inilah yang membedakan sistem kita dari metode restorasi konvensional: kita tidak hanya peduli visual, tapi juga **keterbacaan teks**."

**[Pause - tunjuk baris 5]**

"**Eksperimen 5**: Menambahkan recognition feature loss. Hasilnya? CER malah **naik** 0.5% menjadi 30.16%. Kontribusinya **marginal**, bahkan sedikit kontraproduktif untuk fase awal training."

---

### Analisis Kontribusi Efektif (Kolom Kiri)

**[Transisi ke kolom kiri]**

"Jadi, dari hasil ablasi singkat 15 epoch ini, kita bisa simpulkan kontribusi efektif masing-masing:"

1. **Adversarial**: Champion untuk PSNR (+0.27 dB)
2. **Perceptual**: Optimal untuk structural similarity (SSIM 0.9630)
3. **CTC**: Game changer untuk HTR capability (CER 29.66%)
4. **RecFeat**: Marginal, bahkan sligflightlyht negatif (+0.5% CER degradation)

---

### Distribusi Kontribusi Real (Kolom Kanan)

**[Transisi ke kolom kanan - nada menarik]**

"Tapi ini yang sangat menarik: ketika kita lihat **model produksi 50 epoch**, distribusi kontribusinya **sangat berbeda**!"

**[Pause - baca dengan penekanan]**

- "**CTC mendominasi** dengan **67.5 persen** dari total loss — ini membuktikan bahwa HTR guidance adalah **driver utama** sistem kita"
- "Perceptual 28.6% — signifikan untuk menjaga struktur visual"
- "Adversarial hanya 3.1% — meskipun kecil, dampaknya pada realism tetap penting"
- "Pixel 0.7% — baseline yang necessary tapi kontribusinya kecil"
- "RecFeat **hanya 0.1 persen** — practically negligible!"

**[Gesture: tangan menunjukkan perbedaan ukuran]**

"Mengapa beda? Karena CTC loss punya **magnitude sangat besar** — sekitar 392 dalam nilai raw — jadi meskipun bobotnya kecil (0.15), kontribusinya dominan. Ini adalah hasil dari **inverse scaling** yang kita terapkan untuk menyeimbangkan komponen dengan magnitude berbeda."

---

### Rekomendasi Konfigurasi (Alert Block)

**[Nada conclusive dan actionable]**

"Dari temuan ini, rekomendasi kami sangat jelas:"

**[Pause - baca dengan tegas]**

> "**4 komponen optimal**: Pixel + Adversarial + Perceptual + CTC.  
> RecFeat **dapat dihilangkan** tanpa pengorbanan performa."

"CER-nya malah lebih baik tanpa RecFeat: 29.66% versus 30.16%. Jadi untuk efficiency dan performance, **gunakan 4-component setup**."

**[Transisi]**

"Ini adalah contoh bagaimana **ablation study** bukan hanya validasi, tapi juga **optimization tool** untuk menemukan konfigurasi terbaik."

---

## 🎓 SLIDE 15: Narasi Curriculum Learning (2.5 menit)

### Pembukaan

"Slide terakhir untuk studi ablasi adalah evaluasi **curriculum learning** — salah satu strategi yang kami hipotesiskan akan meningkatkan stabilitas training."

---

### Penjelasan Protokol (Kolom Kiri)

**[Tunjuk kolom kiri]**

"Protokol curriculum learning kita terdiri dari **tiga fase bertahap**:"

**[Pause - jelaskan satu per satu dengan jelas]**

"**Fase 1: Warmup Visual** (epoch 1-10)  
CTC weight = 0. Model fokus **pure visual reconstruction** dulu, membangun kemampuan dasar generator tanpa distraction dari HTR objective."

"**Fase 2: Transisi CTC** (epoch 11-30)  
Weight CTC dinaikkan **bertahap** dari 0 ke 0.15. Gradual introduction untuk menghindari shock pada sistem."

"**Fase 3: Pelatihan Penuh** (epoch 31-50)  
CTC weight konstan di 0.15. Semua komponen bekerja **simultaneous** dengan bobot final."

**[Pause]**

"Hipotesis kami: pembelajaran bertahap ini akan **lebih stabil** dibanding langsung training semua komponen simultaneously dari awal."

---

### Hasil Perbandingan (Kolom Kanan - Tabel)

**[Transisi ke tabel - nada surprised]**

"Tapi hasilnya **mengejutkan**!"

**[Pause - baca metrik satu per satu]**

"**PSNR**: Curriculum 26.02 dB, Non-curriculum **26.16 dB** — Non-curriculum **unggul** +0.14 dB."

"**CER**: Curriculum 28.7%, Non-curriculum **28.6%** — lebih baik tanpa curriculum."

"**Best Epoch**: Curriculum butuh 45 epoch, Non-curriculum cukup **41 epoch** — konvergen **4 epoch lebih cepat**!"

**[Pause - dramatic]**

"Dan yang paling mengejutkan:"

"**CTC Variance**: Curriculum 151.62, Non-curriculum **hanya 4.09** — artinya non-curriculum **37 kali lebih stabil**!"

**[Biarkan angka sink in]**

---

### Analisis Temuan (Alert Block)

**[Nada reflektif]**

"Ini adalah **paradoks**. Curriculum learning dirancang untuk **meningkatkan stabilitas**, tapi ternyata malah **37× lebih tidak stabil** dibanding training simultan!"

**[Pause - explain]**

"Kenapa ini terjadi?"

---

### Interpretasi (Transisi ke Insight)

**[Nada analytical]**

"Setelah analisis mendalam, kami menemukan beberapa faktor:"

1. **Komponen loss sudah optimal balanced**  
   "Kombinasi 5 komponen loss dengan inverse scaling sudah menciptakan **equilibrium stabil**. Tidak perlu phased introduction."

2. **Dataset semi-sintetis terkontrol**  
   "Dataset kita tidak punya variasi ekstrem yang memerlukan **gradual adaptation**. Semua degradasi patterns sudah terdistribusi merata."

3. **Arsitektur robust**  
   "Enhanced U-Net + dual-modal discriminator + frozen recognizer sudah **cukup handal** untuk handle simultaneous optimization dari awal."

**[Gesture: tangan menunjukkan kestabilan]**

"Jadi sistem kita **inherently stable** — tidak butuh training wheels dari curriculum."

---

### Validasi Statistik (Example Block)

**[Tunjuk kotak hijau - nada final verdict]**

"Secara statistik, perbedaan ini **tidak signifikan**:"

- "**p-value = 0.471** untuk PSNR (threshold 0.05)"
- "**Cohen's d = 0.146** — kategori **very small effect**"

**[Pause - conclusive]**

"Artinya, perbedaan numerik yang kita lihat kemungkinan besar **random noise**, bukan efek nyata dari curriculum learning."

---

### Kesimpulan Slide

**[Nada honest & scientific]**

"Jadi kesimpulan untuk curriculum learning: **Tidak memberikan peningkatan signifikan**, bahkan cenderung **menurunkan stabilitas**."

**[Pause - positive spin]**

"Tapi temuan ini justru **good news** untuk praktek!"

**[Explain benefit]**

> "Arsitektur yang handal dapat **belajar secara simultan** dengan stabilitas **lebih tinggi**.  
> Kita **tidak perlu** kompleksitas phased training.  
> Deployment lebih **simple** dan **reproducible**."

**[Pause]**

"Ini sekali lagi memvalidasi prinsip yang konsisten sepanjang penelitian ini:"

**[Conclusive statement]**

> **"Protokol training yang tepat dan arsitektur robust > Kompleksitas strategi training"**

---

## 📊 Timing Breakdown

### Slide 14 (Loss Function):
- Pembukaan: 15 detik
- Penjelasan tabel ablasi: 60 detik
- Analisis kontribusi efektif: 30 detik
- Distribusi kontribusi produksi: 40 detik
- Rekomendasi konfigurasi: 25 detik
**Total: ~2.5 menit**

### Slide 15 (Curriculum Learning):
- Pembukaan: 15 detik
- Protokol 3 fase: 40 detik
- Hasil perbandingan: 45 detik
- Analisis temuan: 35 detik
- Validasi statistik: 25 detik
- Kesimpulan: 30 detik
**Total: ~3 menit**

---

## 🎯 Key Messages per Slide

### Slide 14 Key Takeaways:
1. ✅ **CTC dominates** dengan 67.5% contribution — HTR guidance is the main driver
2. ✅ **4-component setup optimal** — RecFeat dapat dihilangkan (0.1% contribution)
3. ✅ **Inverse scaling works** — menyeimbangkan komponen dengan magnitude 5 orders berbeda
4. ✅ **Ablation reveals efficiency** — bukan hanya validation, tapi optimization tool

### Slide 15 Key Takeaways:
1. ❌ **Curriculum learning NOT beneficial** — paradoxically 37× less stable
2. ✅ **Robust architecture matters more** — dapat handle simultaneous optimization
3. ✅ **Simpler is better** — non-curriculum 4 epochs faster convergence
4. ✅ **Statistical rigor** — p=0.471, d=0.146 (not significant)

---

## 🔴 Antisipasi Pertanyaan

### Q1: "Mengapa RecFeat dipertahankan di model produksi jika kontribusinya minimal?"
**A:**  
"Pertanyaan bagus, Pak/Bu. Model produksi dilatih dengan **5-component full** sebagai baseline experimental. Temuan bahwa RecFeat marginal justru berasal dari **post-hoc ablation study** setelah model produksi selesai. Untuk **future implementation**, kami merekomendasikan 4-component setup. Tapi untuk **reproducibility** hasil utama tesis ini, konfigurasi 5-component tetap didokumentasikan. Ini adalah contoh **iterative scientific process** — kita propose, test, lalu revise based on evidence."

---

### Q2: "Apakah CTC loss 67.5% berarti komponen lain tidak penting?"
**A:**  
"Excellent question, Pak/Bu. **Kontribusi 67.5% itu measured dari weighted loss magnitude**, bukan importance. Semua komponen **necessary**:
- **Adversarial** (3.1%) → krusial untuk realism, tanpa ini output terlihat blur
- **Perceptual** (28.6%) → menjaga structural similarity
- **Pixel** (0.7%) → base reconstruction constraint

Yang kita temukan adalah bahwa **HTR objective** adalah **primary driver**, tapi tetap butuh **complementary components** untuk balance. Analogi: dalam mobil, mesin adalah driver utama, tapi tetap butuh setir, rem, dan suspensi untuk perform optimally."

---

### Q3: "Jika curriculum tidak efektif, mengapa digunakan di model produksi?"
**A:**  
"Penting dicatat, Pak/Bu: model produksi dilatih **sebelum** kita melakukan ablation study curriculum versus non-curriculum. Pada saat design awal, curriculum learning **dihipotesiskan** akan membantu berdasarkan literature. Studi ablasi dilakukan **retrospektif** untuk validate apakah hypothesis ini benar. Ternyata **tidak terbukti signifikan**.

**Tapi ini bukan masalah** karena:
1. Model produksi sudah **achieve all targets** (PSNR 30.91 dB, CER 34.9%)
2. Temuan ablasi memberikan **guidance untuk future work** — gunakan non-curriculum untuk simplicity
3. Ini adalah **valid scientific contribution** — kita provide empirical evidence bahwa curriculum tidak necessary

Jadi curriculum di model produksi bukan kesalahan, melainkan **bagian dari experimental trajectory** yang menghasilkan **learning untuk riset selanjutnya**."

---

### Q4: "Variance CTC 37× lebih tinggi — apakah itu masalah serius?"
**A:**  
"Variance tinggi pada curriculum **bukan catastrophic instability**, Pak/Bu. Ini terjadi karena **phased weight changes** (0 → 0.15) menciptakan **expected fluctuations**. CTC loss mengalami adjustment setiap kali weight berubah di phase transition.

**Yang penting** adalah:
- Model tetap **converge successfully** (PSNR 26.02 dB, CER 28.7%)
- Perbedaan dengan non-curriculum **statistically insignificant** (p=0.471)
- **Best checkpoint** masih tercapai (epoch 45)

Jadi variance tinggi adalah **artifact dari phased training**, bukan indication of failure. Tapi karena **tidak ada benefit** (performa setara), kita simpulkan **unnecessary complexity**. Prinsip parsimoni: **equal performance → choose simpler method**."

---

### Q5: "Apakah hasil ablasi 15 epoch reliable untuk kesimpulan?"
**A:**  
"Valid concern, Pak/Bu. Untuk ablasi loss function, **15 epoch sufficient** untuk measure **relative contribution**, bukan absolute performance. Kita lihat pola kontribusi (adversarial +0.27 dB, RecFeat marginal), yang **konsisten** dengan 50-epoch production run (RecFeat 0.1%).

Untuk curriculum learning ablation, kita gunakan **50 epochs full training** — identik dengan production protocol. Jadi kesimpulan curriculum sangat **robust**.

Prinsipnya: **ablation tracks relative effects**, bukan absolute values. 15 epoch cukup untuk identify component importance, tapi **final performance** tetap require 40-50 epochs untuk convergence penuh."

---

## 💡 Tips Presentasi Khusus

### Untuk Slide 14 (Loss Function):
1. **Gunakan gesture tangan** untuk menunjukkan perbedaan magnitude loss (kecil vs besar)
2. **Pause lebih lama** saat menyebut CTC 67.5% — ini surprising finding
3. **Emphasize "4 optimal components"** dengan jelas — praktical takeaway
4. **Suara confident** saat recommend buang RecFeat (backed by data)

### Untuk Slide 15 (Curriculum Learning):
1. **Tone surprised genuine** saat reveal non-curriculum unggul — ini unexpected
2. **Slow down** saat sebut "37× lebih stabil" — biarkan sink in
3. **Honest tone** saat explain paradox — menunjukkan scientific integrity
4. **Positive spin** di akhir — negative result is valuable finding

---

## 🎓 Academic Framing

Kedua slide ini mendemonstrasikan:

1. **Thorough ablation methodology** — systematic component isolation
2. **Data-driven optimization** — identify unnecessary components
3. **Scientific honesty** — report findings yang tidak sesuai hipotesis
4. **Practical value** — simplify system tanpa sacrifice performance
5. **Rigorous statistics** — p-values, effect sizes, variance analysis

**Message**: Mature researcher tahu bahwa **negative findings sama valuable** dengan positive findings untuk advance the field.

---

## 📈 Visual Cues & Emphasis

### Angka yang Harus Diingat Audiens:

**Slide 14:**
- **67.5%** — CTC dominance
- **0.1%** — RecFeat negligible
- **+0.27 dB** — Adversarial contribution
- **4 components** — optimal setup

**Slide 15:**
- **37×** — stability improvement
- **p=0.471** — not significant
- **4 epochs faster** — convergence speed
- **26.16 vs 26.02 dB** — marginal difference

---

_Dokumen ini mendukung presentasi Slide 14 (Ablasi Loss Function) dan Slide 15 (Ablasi Curriculum Learning)_  
_Fokus: Data-driven insights, honest scientific reporting, practical recommendations_
