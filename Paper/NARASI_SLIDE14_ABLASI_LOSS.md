# Narasi Presentasi: Studi Ablasi Optimasi 5 Komponen Loss Function
**Slide 14 - Seminar Hasil Tesis**

---

## 🎤 Skrip Narasi Lengkap (2.5-3 menit)

### **PEMBUKAAN (15 detik)**

"Slide berikutnya adalah studi ablasi untuk **optimasi lima komponen loss function** — ini adalah jantung dari sistem pembelajaran kita."

**[Pause - transisi ke tabel]**

---

### **PENJELASAN TABEL ABLASI INKREMENTAL (60 detik)**

"Mari kita lihat tabel hasil ablasi inkremental ini. Kita melakukan **ablasi bertahap** — menambahkan satu komponen per eksperimen untuk melihat kontribusi masing-masing."

#### **Eksperimen 1 - Baseline:**
**[Tunjuk baris 1]**

"**Eksperimen pertama**: Kita mulai dengan **pixel loss L1 saja** sebagai baseline. PSNR 24.59 dB, SSIM 0.9614. Ini adalah fondasi dasar — memastikan rekonstruksi piksel yang akurat."

**[Pause singkat]**

#### **Eksperimen 2 - Adversarial:**
**[Tunjuk baris 2 dengan emphasis]**

"**Eksperimen kedua**: Menambahkan **adversarial loss**. Dan inilah yang menarik — PSNR naik menjadi **24.86 dB**, peningkatan **+0.27 dB**."

**[Pause - tunjuk angka yang bold]**

"Ini adalah **kontribusi terbesar** untuk metrik visual! Adversarial loss ini yang membuat hasil restorasi terlihat **realistis dan natural**, bukan hanya akurat secara piksel."

**[Pause]**

#### **Eksperimen 3 - Perceptual:**
**[Tunjuk baris 3]**

"**Eksperimen ketiga**: Menambahkan **perceptual loss dari VGG-19**. Menariknya, PSNR sedikit turun minus 0.04 dB,"

**[Pause - tunjuk SSIM]**

"**TAPI** SSIM naik menjadi **0.9630** — ini nilai **tertinggi** di seluruh eksperimen!"

**[Pause]**

"Ini adalah trade-off yang **expected**. Perceptual loss fokus pada **structural similarity** dan kemiripan perseptual, bukan pixel-wise accuracy. Jadi wajar kalau PSNR turun sedikit, tapi struktur visual lebih baik."

**[Pause]**

#### **Eksperimen 4 - CTC (KRUSIAL):**
**[Tunjuk baris 4 dengan gesture tegas]**

"**Eksperimen keempat**: Ini yang **paling krusial** — menambahkan **CTC loss**."

**[Pause - dramatic]**

"Komponen ini **mengaktifkan kemampuan HTR**. CER pertama kali muncul: **29.66 persen**!"

**[Pause - eye contact dengan audiens]**

"Inilah yang **membedakan** sistem kita dari metode restorasi konvensional. Kita tidak hanya peduli visual, tapi juga **keterbacaan teks** untuk HTR. Ini adalah **supervised signal tekstual** yang memandu generator."

**[Pause]**

#### **Eksperimen 5 - RecFeat:**
**[Tunjuk baris 5 dengan nada reflektif]**

"**Eksperimen kelima**: Menambahkan **recognition feature loss**. CER menjadi 30.16 persen."

**[Pause]**

"Tunggu — ini **naik** 0.5 persen dari eksperimen 4! Artinya, RecFeat **tidak memberikan benefit**, bahkan sedikit **kontraproduktif** untuk fase awal training."

**[Pause - prepare transisi]**

---

### **ANALISIS KONTRIBUSI EFEKTIF (Kolom Kiri, 30 detik)**

**[Transisi ke kolom kiri]**

"Jadi, dari hasil ablasi 15 epoch ini, kita bisa simpulkan kontribusi efektif:"

**[Baca dengan rhythm, satu per satu]**

1. "**Adversarial**: Champion untuk PSNR, plus 0.27 dB"
2. "**Perceptual**: Optimal untuk structural similarity, SSIM 0.9630"
3. "**CTC**: Game changer — aktivasi HTR capability dengan CER 29.66 persen"
4. "**RecFeat**: Marginal, malah sedikit negative, plus 0.5 persen degradation"

**[Pause - prepare big reveal]**

---

### **DISTRIBUSI KONTRIBUSI PRODUKSI (Kolom Kanan, 45 detik)**

**[Transisi ke kolom kanan dengan nada excited]**

"Tapi ini yang **sangat menarik**!"

**[Pause untuk emphasis]**

"Ketika kita lihat **model produksi 50 epoch**, distribusi kontribusinya **sangat berbeda** dari ablasi singkat tadi!"

**[Pause - tunjuk angka CTC]**

"**CTC mendominasi** dengan **67.5 persen** dari total weighted loss!"

**[Pause - biarkan sink in]**

"Ini membuktikan bahwa **HTR guidance adalah driver utama** dari seluruh sistem pembelajaran kita."

**[Pause - lanjut dengan itemize]**

"Yang lainnya:"
- "Perceptual: 28.6 persen — signifikan untuk menjaga struktur visual"
- "Adversarial: hanya 3.1 persen — meskipun kecil, dampaknya pada realism tetap penting"
- "Pixel: 0.7 persen — baseline yang necessary但 kontribusinya minimal"

**[Pause - dramatic untuk poin terakhir]**

"Dan RecFeat?"

**[Pause 2 detik]**

"**Hanya 0.1 persen**!"

**[Gesture: tangan menunjukkan perbedaan 67.5% vs 0.1%]**

"Practically **negligible** — bisa diabaikan!"

**[Pause]**

---

### **PENJELASAN PARADOKS (30 detik)**

**[Nada explaining, pedagogical]**

"Mungkin Anda bertanya: Mengapa distribusinya bisa **sangat berbeda** antara ablasi 15 epoch dengan produksi 50 epoch?"

**[Pause - prepare explanation]**

"Jawabannya ada di **magnitude raw loss**."

**[Gesture: tangan menunjukkan scale]**

"CTC loss punya **magnitude mentah yang sangat besar** — sekitar **392** dalam nilai raw."

"Jadi meskipun bobotnya kecil — hanya **0.15** — tapi kalau dikali dengan magnitude sebesar itu, kontribusi efektifnya menjadi **67.5 persen**!"

**[Pause]**

"Ini adalah hasil dari **inverse scaling** yang kita terapkan — menyeimbangkan komponen dengan magnitude yang berbeda **lima orders of magnitude**!"

**[Gesture: tangan horizontal menunjukkan balance]**

---

### **REKOMENDASI PRAKTIS (Alert Block, 25 detik)**

**[Transisi ke alert block - nada conclusive dan actionable]**

"Dari temuan ini, rekomendasi kami sangat **jelas dan praktis**:"

**[Pause - baca dengan tegas]**

> "**Empat komponen optimal**: Pixel, Adversarial, Perceptual, dan CTC."

**[Pause]**

"RecFeat **dapat dihilangkan** tanpa pengorbanan performa."

**[Tunjuk angka]**

"Lihat — CER tanpa RecFeat: **29.66 persen**. Dengan RecFeat: **30.16 persen**. Malah **lebih buruk**!"

**[Pause]**

"Jadi untuk **efficiency** dan **performance**, gunakan **setup 4 komponen**."

**[Pause]**

---

### **PENUTUP SLIDE (15 detik)**

**[Nada reflective dan transisi]**

"Ini adalah contoh bagaimana **ablation study** bukan hanya untuk validasi, tapi juga sebagai **optimization tool** untuk menemukan konfigurasi terbaik."

**[Pause]**

"Kita telah membuktikan bahwa CTC loss adalah **main driver**, dan RecFeat adalah **redundant component** yang bisa dieliminasi."

**[Pause - prepare transisi]**

"Mari kita lanjut ke studi ablasi berikutnya..."

---

## 📊 **Timing Breakdown Detail**

| Segmen | Durasi | Kata Kunci |
|--------|--------|------------|
| Pembukaan | 15s | "Jantung sistem pembelajaran" |
| Tabel Eks 1 | 10s | "Baseline - pixel L1" |
| Tabel Eks 2 | 15s | "+0.27 dB - champion visual" |
| Tabel Eks 3 | 15s | "SSIM tertinggi - trade-off" |
| Tabel Eks 4 | 20s | "CTC - game changer HTR" |
| Tabel Eks 5 | 10s | "RecFeat - kontraproduktif" |
| Kontribusi Efektif | 30s | "4 bullet points" |
| Distribusi Produksi | 45s | "CTC dominan 67.5%" |
| Penjelasan Paradoks | 30s | "Inverse scaling, magnitude" |
| Rekomendasi | 25s | "4 komponen optimal" |
| Penutup | 15s | "Transition" |
| **TOTAL** | **~2.5 menit** | |

---

## 🎯 **Poin-Poin Kunci yang Harus Diingat**

### **Angka Emas:**
1. **+0.27 dB** → Adversarial contribution (terbesar untuk visual)
2. **0.9630** → SSIM optimal (Perceptual)
3. **29.66%** → CER terbaik (tanpa RecFeat!)
4. **67.5%** → CTC dominance (produksi)
5. **0.1%** → RecFeat negligible

### **Konsep Kunci:**
1. **Ablasi inkremental** → step-by-step component addition
2. **Trade-off PSNR vs SSIM** → perceptual vs pixel accuracy
3. **CTC = main driver** → 67.5% contribution
4. **Inverse scaling** → balance 5 orders of magnitude
5. **4-component optimal** → remove RecFeat

---

## 💡 **Tips Presentasi**

### **Gesture & Body Language:**
1. **Tunjuk tabel** dengan laser pointer saat explain setiap eksperimen
2. **Tangan horizontal** saat jelaskan balance/inverse scaling
3. **Gesture besar vs kecil** untuk kontras 67.5% vs 0.1%
4. **Nod** saat menyebutkan rekomendasi (affirmative)

### **Voice Modulation:**
1. **Excited** saat reveal adversarial +0.27 dB (champion!)
2. **Curious/questioning** saat explain PSNR turun tapi SSIM naik
3. **Emphatic** saat introduce CTC (game changer!)
4. **Reflective** saat explain RecFeat kontraproduktif
5. **Surprised** saat reveal CTC 67.5% (sangat berbeda!)
6. **Conclusive** saat deliver rekomendasi

### **Pause Strategy:**
1. **After numbers**: Biarkan audiens process (2 detik)
2. **Before big reveals**: Build suspense (1-2 detik)
3. **After rhetorical questions**: Give time to think (1 detik)
4. **Between sections**: Clear transition (1 detik)

---

## 🔴 **Antisipasi Pertanyaan**

### **Q1: "Mengapa RecFeat tidak efektif jika di model produksi tetap digunakan?"**
**Jawaban:**
"Pertanyaan yang bagus, Pak/Bu. Model produksi dilatih dengan **full 5-component** sebagai **baseline experimental**. Temuan bahwa RecFeat marginal justru berasal dari **post-hoc ablation study** setelah model produksi selesai.

Untuk **future implementation**, kami merekomendasikan **4-component setup**. Tapi untuk **reproducibility** hasil utama tesis ini, konfigurasi 5-component tetap didokumentasikan.

Ini adalah contoh **iterative scientific process** — kita propose, test, lalu revise based on evidence."

---

### **Q2: "Bagaimana cara menentukan bobot yang optimal untuk setiap komponen?"**
**Jawaban:**
"Excellent question, Pak/Bu. Kami menggunakan **manual empirical tuning** dengan prinsip **inverse scaling**:

\[ w_i \propto \frac{1}{\text{magnitude}_{\text{raw},i}} \]

Artinya, komponen dengan magnitude raw besar dapat bobot kecil, dan sebaliknya. Tujuannya agar **kontribusi efektif** seimbang meskipun magnitude aslinya berbeda 5 orders of magnitude.

Konfigurasi optimal kami: Pixel=50, Adversarial=3, Perceptual=1, CTC=0.15, RecFeat=8 (tapi RecFeat bisa dihilangkan).

Prosesnya iterative — train, evaluate contribution, adjust weights, repeat sampai konvergen stabil."

---

### **Q3: "CER 67.5% dominance — bukankah ini terlalu dominan? Apa dampaknya?"**
**Jawaban:**
"Valid concern, Pak/Bu. CTC 67.5% dominance adalah **measured contribution**, bukan designed weight.

**Dampaknya positif**:
1. ✅ Generator **strongly guided** oleh text readability objective
2. ✅ Alignment visual-HTR terjaga
3. ✅ Hasil akhir CER 34.9% mendekati GT baseline 34.1%

**Tidak ada negative effects** seperti:
- ❌ Visual quality sacrifice (PSNR 30.74 dB, SSIM 0.987 — excellent!)
- ❌ Overfitting ke text (generalisasi tetap baik)
- ❌ Training instability (konvergensi smooth)

Jadi dominance ini **healthy** — CTC adalah **primary objective**, dan komponen lain **complementary** untuk support kualitas visual."

---

### **Q4: "Kenapa ablasi hanya 15 epoch, bukan 50 seperti produksi?"**
**Jawaban:**
"Pertanyaan metodologis yang bagus, Pak/Bu.

**Alasan 15 epoch untuk ablasi**:
1. **Computational efficiency** — 5 eksperimen × 50 epoch = 250 epochs sangat mahal
2. **Relative comparison** — ablasi fokus pada **ranking kontribusi**, bukan absolute performance
3. **Early pattern visible** — pola kontribusi sudah terlihat di fase awal

**Validasi konsistensi**:
- RecFeat minimal di ablasi 15 epoch (30.16% CER) → **consistent** dengan produksi 50 epoch (0.1% contribution)
- Pattern yang sama terdeteksi di kedua durasi

**Absolute performance** (30.91 dB PSNR, 34.9% CER) tetap dari **produksi 50 epoch**, bukan ablasi singkat."

---

### **Q5: "Apakah inverse scaling ini standard practice atau novelty penelitian ini?"**
**Jawaban:**
"Inverse scaling secara prinsip **bukan novel** — banyak penelitian menggunakan weight balancing.

**Yang novel** dari penelitian kami:
1. ✅ **Empirical characterization** untuk domain document restoration + HTR (belum ada di literature)
2. ✅ **Quantitative contribution analysis** (67.5% CTC, 28.6% Perceptual, dll)
3. ✅ **Systematic ablation** dengan 5 komponen bertingkat
4. ✅ **Practical recommendation** (4-component optimal)

Jadi bukan novelty di method-nya, tapi **novelty di empirical findings** untuk domain spesifik paleografi + HTR. Ini memberikan **guidance untuk future researchers** pada domain yang sama."

---

## 📈 **Visual Cues & Emphasis Points**

### **Warna Highlighting (Mental Map untuk Presenter):**
- **Hijau**: Adversarial +0.27 dB (best PSNR)
- **Hijau**: SSIM 0.9630 (best structural)
- **Hijau**: CER 29.66% (best readability, eks 4)
- **Merah**: CER 30.16% (worse dengan RecFeat, eks 5)
- **Emas**: CTC 67.5% (dominan!)
- **Abu-abu**: RecFeat 0.1% (negligible)

### **Gesture Amplifikasi:**
- **Small**: RecFeat 0.1% (jari telunjuk-jempol pinch)
- **Large**: CTC 67.5% (tangan terbuka lebar)
- **Balance**: Inverse scaling (tangan horizontal seperti timbangan)

---

## 🎓 **Academic Framing**

### **Scientific Rigor Points:**
1. ✅ **Systematic methodology** — ablasi inkremental, bukan random
2. ✅ **Quantitative evidence** — semua angka dari empirical evaluation
3. ✅ **Practical outcome** — actionable recommendation (4-component)
4. ✅ **Transparency** — show RecFeat tidak efektif (honest reporting)

### **Contribution to Field:**
- **First ablation study** untuk multi-component loss di GAN-HTR paleografi
- **Empirical characterization** kontribusi masing-masing komponen
- **Simplification guidance** (4 vs 5 components) untuk future work

---

## ✅ **Checklist Pre-Presentasi**

- [ ] Hafal urutan 5 eksperimen (Pixel → Adv → Perc → CTC → RecFeat)
- [ ] Ingat angka kunci: +0.27, 0.9630, 29.66%, 67.5%, 0.1%
- [ ] Latih transisi smooth antara tabel → kiri → kanan → alert
- [ ] Practice pause timing (terutama setelah "67.5%" reveal)
- [ ] Prepare defensive answer untuk Q tentang RecFeat di produksi
- [ ] Review inverse scaling concept (bisa ditanya detail)
- [ ] Siapkan gesture untuk magnitude contrast
- [ ] Test laser pointer untuk tunjuk tabel (jangan kehilangan posisi)

---

**FINAL NOTE**:  
Slide ini adalah **data-driven optimization story**. Bukan hanya "we tried 5 things", tapi "we systematically identified the optimal 4 components with empirical evidence". Frame sebagai **scientific discovery process**, bukan trial-and-error.

**Tone keseluruhan**: Confident, analytical, evidence-based, dengan positive spin untuk "negative" finding RecFeat.

---

_Durasi Total: ~2.5 menit_  
_Complexity: Medium-High (banyak angka, butuh clear explanation)_  
_Key Message: CTC dominates (67.5%), RecFeat redundan (0.1%), 4-component optimal_
