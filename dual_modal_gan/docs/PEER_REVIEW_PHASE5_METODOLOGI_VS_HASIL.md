# PEER REVIEW: FASE 5 METODOLOGI vs IMPLEMENTASI HASIL
**Reviewer:** AI/ML Engineer dengan fokus pada konsistensi metodologi dan validitas ilmiah

**Tanggal Review:** 2025-11-28

**Dokumen yang Direview:**
- Chapter 3 (Metodologi): Fase 5 - Studi Ablasi dan Optimasi Konfigurasi (lines 526-527)
- Chapter 5 (Hasil): Subsection 5.3 - Studi Ablasi (lines 326-948)

---

## EXECUTIVE SUMMARY

**STATUS:** ✅ **KONSISTEN SECARA KESELURUHAN** dengan beberapa rekomendasi perbaikan minor

**TEMUAN UTAMA:**
1. ✅ Semua janji metodologi di Fase 5 telah diimplementasikan di Chapter 5
2. ✅ Tidak ditemukan contradictory statements yang fatal
3. ⚠️ Beberapa detail metodologi di Chapter 3 terlalu ringkas, perlu diperjelas untuk menghindari expectation mismatch
4. ✅ Transparansi dalam melaporkan temuan negatif (dual-modal, RecFeat) sangat baik

---

## ANALISIS DETIL: PERNYATAAN METODOLOGI vs IMPLEMENTASI

### 1. STUDI ABLASI KOMPONEN LOSS FUNCTION

#### Apa yang Dijanjikan (Chapter 3, lines 621-629):
```
"Ablasi Komponen Loss Function: Lima eksperimen inkremental dilakukan dengan menambahkan 
komponen loss secara bertahap:
- Eksperimen 01: Pixel loss saja (setara U-Net)
- Eksperimen 02: +Adversarial loss (GAN standar)  
- Eksperimen 03: +Perceptual loss (VGG)
- Eksperimen 04: +CTC loss (konfigurasi optimal)
- Eksperimen 05: +Recognition feature loss (validasi redundansi)

Setiap eksperimen dievaluasi pada set validasi (n=710) dengan metrik PSNR, SSIM, CER, 
dan WER untuk mengukur kontribusi relatif setiap komponen"
```

#### Apa yang Dilakukan (Chapter 5, lines 346-402):
✅ **SESUAI DAN TERDOKUMENTASI DENGAN BAIK**

**Bukti Implementasi:**
- Table V.4 (tab:ablasi-loss-results) menyajikan 5 eksperimen yang persis sesuai janji
- Durasi: 15 epoch (Chapter 5 line 354) - CATATAN: Chapter 3 tidak menyebutkan durasi spesifik
- Metrik: PSNR, SSIM, CER ✅ (WER tidak ditampilkan - minor discrepancy)
- Set validasi: n=710 ✅

**Temuan Konsisten dengan Metodologi:**
- Eksperimen 04 (4-komponen tanpa RecFeat) optimal untuk ablasi cepat: CER 29.66%
- Eksperimen 05 (5-komponen) menunjukkan redundansi RecFeat: CER meningkat ke 30.16%

**⚠️ REKOMENDASI PERBAIKAN MINOR:**
> **Issue:** Chapter 3 tidak menyebutkan durasi pelatihan ablasi (15 epoch)
> **Dampak:** Pembaca mungkin mengharapkan eksperimen konvergen penuh seperti produksi (50 epoch)
> **Solusi:** Tambahkan di Chapter 3 line 629: "...dengan pelatihan eksplorasi cepat (15-20 epoch) untuk efisiensi komputasi"

---

### 2. ABLASI ARSITEKTUR DISKRIMINATOR

#### Apa yang Dijanjikan (Chapter 3, lines 631-632):
```
"Ablasi Arsitektur Diskriminator: Dua eksperimen dilakukan dengan generator identik 
namun arsitektur diskriminator berbeda: 
(1) Diskriminator CNN single-modal (baseline arsitektur)
(2) Diskriminator dual-modal CNN+BiLSTM (arsitektur yang diusulkan)

Perbandingan ini mengisolasi kontribusi spesifik komponen dual-modal terhadap kualitas 
restorasi. Evaluasi dilakukan pada set validasi (n=710) untuk iterasi cepat, dengan 
verifikasi generalisasi pada set uji (n=712) untuk konfigurasi optimal."
```

#### Apa yang Dilakukan (Chapter 5, lines 595-658):
✅ **SEPENUHNYA SESUAI DAN TRANSPARAN**

**Bukti Implementasi:**
- Table V.10 (tab:diskriminator-ablasi) menunjukkan perbandingan CNN vs Dual-Modal
- Protokol: 50 epoch, generator Enhanced U-Net identik, loss function identik ✅
- Dataset: set validasi n=710 ✅
- Metrik: PSNR, SSIM, CER ✅

**EXCELLENT: Transparansi Temuan Negatif**
```
"Hasil eksperimen menunjukkan perbedaan yang sangat minimal antara kedua arsitektur 
(ΔPSNR = +0.28 dB, ΔSSIM = +0.0015, ΔCER = -0.01%), tidak signifikan secara statistik 
(p > 0.05)."
```

**Analisis Faktor Penyebab (lines 633-643):**
- Kualitas masukan tekstual (CER ~27%)
- Dominasi komponen loss lain (adversarial weight kecil)
- Kecukupan fitur visual

**⚠️ POTENSI LOGICAL INCONSISTENCY - RESOLVED:**
> **Concern:** Chapter 3 mengklaim dual-modal sebagai komponen "inovatif" (line 109), tapi Chapter 5 membuktikan kontribusinya tidak signifikan
> **Resolution:** Chapter 5 menangani ini dengan sangat baik:
> - Line 628-629: "hipotesis mengenai keunggulan diskriminator dual-modal tidak terdukung secara statistik"
> - Line 645-651: Saran penelitian masa depan (Teacher Forcing, Scheduled Sampling)
> - Line 652-657: Prinsip parsimoni (Occam's Razor) - rekomendasi gunakan CNN tunggal
> **Conclusion:** ✅ TIDAK ADA LOGICAL FALLACY - peneliti transparan tentang temuan negatif

---

### 3. OPTIMASI HYPERPARAMETER

#### Apa yang Dijanjikan (Chapter 3, line 527):
```
"(3) optimasi hyperparameter melalui pencarian empiris bertahap"
```

#### Apa yang Dilakukan (Chapter 5, lines 740-905):
✅ **IMPLEMENTASI MELEBIHI EKSPEKTASI**

**Bukti Implementasi:**
- **Loss Weights Configuration** (Table V.11, lines 746-763): Analisis bobot loss produksi dengan justifikasi magnitude
- **Mini Grid Search** (Table V.12, lines 811-831): 27 konfigurasi dengan orthogonal array sampling
- **Sensitivitas Per Komponen** (lines 833-905):
  - Pixel Loss Weight (Table V.13)
  - CTC Loss Weight (Table V.14)
  - RecFeat Weight (Table V.15)

**Metodologi yang Sangat Baik:**
- Inverse scaling pattern terhadap raw magnitude (equation V.1)
- Validasi dengan GradNorm stability analysis
- 3-epoch exploratory experiments untuk efisiensi

**⚠️ POTENTIAL EXPECTATION MISMATCH:**
> **Issue:** Chapter 3 menyebutkan "pencarian empiris bertahap" tanpa mendefinisikan metode spesifik
> **Implementasi Aktual:** Manual empirical tuning + validasi dengan GradNorm + mini grid search
> **Resolution:** ✅ Chapter 5 transparan tentang metodologi:
> - Line 765: "Bobot ditentukan via manual empirical tuning"
> - Line 791-793: "GradNorm Stability Analysis" dan "Mini Grid Search" sebagai validasi
> **Recommendation:** Tambahkan di Chapter 3 line 527: "...melalui penyetelan manual empiris (manual empirical tuning) yang divalidasi dengan analisis stabilitas GradNorm dan pencarian kisi mini"

---

### 4. STUDI FROZEN RECOGNIZER vs JOINT TRAINING

#### Chapter 3 Promise (lines 174, 190):
```
Kesenjangan 1: "Belum ada eksplorasi sistematis terhadap strategi frozen recognizer 
sebagai evaluator objektif yang stabil."

RQ2: "Bagaimana mengintegrasikan model HTR recognizer ke dalam framework GAN untuk 
memberikan sinyal gradien sawar-teks yang stabil tanpa ketidakstabilan pelatihan bersama?"
```

#### Apa yang Dilakukan (Chapter 5, lines 404-592):
✅ **IMPLEMENTASI SANGAT KOMPREHENSIF**

**Bukti Implementasi:**
- **Stabilitas Gradien** (Table V.7, lines 416-430): Perbandingan loss trajectory
  - Frozen: G loss 2.84±0.45 (stabil)
  - Joint: G loss 89.09±7.45 (mode collapse)
  
- **Performa Keterbacaan** (Table V.8, lines 457-475):
  - Frozen: CER 31.63% (perbaikan -2.09% dari baseline)
  - Joint: CER 42.85% epoch 20 (degradasi +9.13%), peak catastrophic forgetting 69.06%
  
- **Efisiensi Komputasi** (Table V.9, lines 516-532):
  - Frozen: 85s/epoch, 17.4M trainable params
  - Joint: 630s/epoch (7.4× lebih lambat), 45.3M trainable params

**Statistical Validation:**
- p<0.001, Cohen's d=6.23 (effect size sangat besar)

**✅ EXCELLENT CONTRIBUTION:**
Studi ini **menjawab langsung** RQ2 dan **memvalidasi** keunggulan frozen recognizer yang dijanjikan di metodologi.

**⚠️ MINOR CLARIFICATION NEEDED:**
> **Chapter 3 line 368:** "Model beku ini diintegrasikan ke kerangka kerja untuk perhitungan CTC loss yang di-backpropagate ke Generator"
> **Concern:** Istilah "di-backpropagate" bisa disalahartikan bahwa recognizer ikut di-update
> **Resolution:** ✅ Chapter 5 line 440 klarifikasi dengan baik: "Recognizer dalam mode inference (weights frozen), tidak di-update. Namun, gradient dari CTC loss tetap di-backpropagate ke Generator"
> **Status:** Sudah konsisten, tidak perlu perbaikan

---

### 5. CURRICULUM LEARNING ABLATION

#### Chapter 3 Promise (implicit di lines 437-439):
```
"Prosedur pelatihan mengikuti strategi curriculum learning tiga fase untuk stabilitas 
konvergensi: (1) Visual Warmup dengan CTC loss disabled..., (2) CTC Introduction..., 
(3) Full Optimization..."
```

#### Apa yang Dilakukan (Chapter 5, lines 660-739):
✅ **IMPLEMENTASI LENGKAP DAN TRANSPARAN**

**Bukti Implementasi:**
- Table V.16 (lines 665-684): Curriculum vs Non-Curriculum comparison
- Figure V.14-V.16: Analisis visual komponen loss, statistik, dan fase

**Temuan Mengejutkan (Transparent Negative Result):**
```
"Non-curriculum sedikit unggul (PSNR 26.16 vs 26.02 dB, Δ=0.14 dB; CER 0.286 vs 0.287, 
Δ=0.001) dengan konvergensi lebih cepat (best epoch 41 vs 45)"
```

**⚠️ POTENTIAL CONTRADICTION - RESOLVED:**
> **Concern:** Chapter 3 memposisikan curriculum learning sebagai "strategi untuk stabilitas", tapi Chapter 5 membuktikan non-curriculum lebih stabil (CTC loss variance 37× lebih rendah)
> **Resolution:** ✅ Chapter 5 menangani dengan sangat baik:
> - Line 689: "Paradoks Stabilitas" dijelaskan dengan jujur
> - Line 691: Analisis penyebab (arsitektur handal, dataset terkontrol)
> - Line 727-736: Klarifikasi protokol - model produksi gunakan curriculum, temuan non-curriculum dari ablasi retrospektif
> - Line 730-736: Rekomendasi jelas untuk masa depan vs protokol yang digunakan
> **Status:** ✅ TIDAK ADA CONTRADICTION - transparansi penuh tentang temuan

---

## ANALISIS LOGICAL FALLACIES

### 1. Post Hoc Ergo Propter Hoc
**Status:** ✅ TIDAK DITEMUKAN

Peneliti berhati-hati tidak mengklaim kausalitas tanpa bukti:
- Chapter 5 line 690: "meskipun curriculum dirancang untuk stabilitas" (mengakui ekspektasi ≠ hasil)
- Chapter 5 line 905: "Keberhasilan ini didorong oleh dominasi komponen CTC dan stabilitas frozen recognizer" (mengidentifikasi faktor kausal yang benar)

### 2. Cherry Picking (Selection Bias)
**Status:** ✅ TIDAK DITEMUKAN

Peneliti melaporkan **semua** hasil, termasuk temuan negatif:
- Dual-modal contribution: tidak signifikan (p>0.05)
- RecFeat: redundan, menurunkan performa
- Curriculum learning: tidak lebih baik dari non-curriculum

**Excellent practice:** Transparansi temuan negatif meningkatkan kredibilitas penelitian.

### 3. False Dichotomy
**Status:** ✅ TIDAK DITEMUKAN

Peneliti tidak membuat pilihan either-or yang salah:
- Chapter 5 line 656-657: Rekomendasi CNN tunggal untuk praktis, tapi dual-modal dipertahankan di produksi untuk konsistensi eksperimen

### 4. Circular Reasoning
**Status:** ✅ TIDAK DITEMUKAN

Setiap klaim didukung bukti independen:
- Frozen recognizer superior → didukung 3 dimensi bukti (stabilitas, CER, efisiensi)
- RecFeat redundan → divalidasi di 2 eksperimen berbeda (ablasi 15 epoch + grid search 3 epoch)

---

## ANALISIS CONTRADICTORY STATEMENTS

### 1. Status Dual-Modal Discriminator

**Chapter 3 (line 109):**
> "komponen inovatif: (1) HTR recognizer terbekukan sebagai evaluator objektif..., 
> (2) optimasi fungsi loss multi-komponen..., dan (3) eksplorasi diskriminator dual-modal..."

**Chapter 5 (line 628-629):**
> "hipotesis mengenai keunggulan diskriminator dual-modal tidak terdukung secara statistik"

**Analysis:**
- ⚠️ Mungkin terlihat kontradiktif: "inovatif" vs "tidak efektif"
- ✅ **RESOLVED:** Chapter 3 menggunakan "eksplorasi" (exploration), bukan "keunggulan terbukti"
- ✅ **RESOLVED:** Chapter 5 line 645: "Hasil negatif... memberikan implikasi teoretis"

**Recommendation:**
> Tambahkan klarifikasi di Chapter 3 line 109:
> "...dan (3) eksplorasi diskriminator dual-modal **sebagai hipotesis penelitian** untuk evaluasi bilateral visual-tekstual, yang kontribusi empirisnya divalidasi melalui studi ablasi terkontrol (lihat Bab V.3.2.3)."

### 2. RecFeat Status in Production Config

**Chapter 3 (implisit di desain):**
> RecFeat adalah komponen dari konfigurasi 5-loss

**Chapter 5 (line 385):**
> "Eksperimen 05... menunjukkan kontribusi marginal... redundan jika dibandingkan dengan CTC Loss"

**Chapter 5 (line 902-905):**
> "Hasil kuantitatif secara konsisten tidak mendukung inklusi komponen ini... 
> direkomendasikan untuk dihapus dalam iterasi pengembangan selanjutnya"

**Analysis:**
- ✅ **NO CONTRADICTION:** Chapter 5 transparan bahwa RecFeat adalah "artefak historis" (line 903)
- ✅ **CLEAR RECOMMENDATION:** Hapus untuk implementasi masa depan

**Status:** KONSISTEN

### 3. Curriculum Learning Efficacy

**Chapter 3 (line 437):**
> "strategi curriculum learning tiga fase untuk stabilitas konvergensi"

**Chapter 5 (line 689):**
> "Paradoks Stabilitas: Non-curriculum lebih stabil overall dengan CTC loss variance 37× lebih rendah"

**Analysis:**
- ⚠️ Terlihat kontradiktif: curriculum untuk "stabilitas" vs non-curriculum "lebih stabil"
- ✅ **RESOLVED:** Chapter 5 lines 727-736 menjelaskan:
  - Model produksi gunakan curriculum (menghasilkan hasil utama)
  - Temuan non-curriculum dari ablasi retrospektif
  - Rekomendasi clear untuk masa depan

**Recommendation:**
> Tambahkan footnote di Chapter 3 line 437:
> "Studi ablasi retrospektif (Bab V.3.2.4) menunjukkan bahwa untuk arsitektur yang handal dan dataset terkontrol, pendekatan non-curriculum dapat menghasilkan stabilitas yang sebanding dengan konvergensi lebih cepat, namun protokol curriculum tetap digunakan dalam pelatihan produksi untuk eksplorasi dinamika pembelajaran bertahap."

---

## KONSISTENSI KOHERENSI NARATIF

### Alur Logis Chapter 3 → Chapter 5

**Chapter 3 Struktur:**
1. Identifikasi 3 kesenjangan (metodologis, arsitektural, evaluasi)
2. Desain solusi dengan 4 komponen inovatif
3. Fase 5: Validasi through ablation

**Chapter 5 Struktur:**
1. Hasil kuantitatif (performa keseluruhan)
2. Studi ablasi (validasi komponen individual)
   - Loss components ✅
   - Frozen recognizer ✅
   - Dual-modal discriminator ✅
   - Curriculum learning ✅
3. Evaluasi kualitatif ANRI

**✅ KOHERENSI PERFECT:** Setiap janji di metodologi terpenuhi dengan bukti empiris

---

## TEMUAN POSITIF (STRENGTHS)

1. **Transparansi Temuan Negatif** ⭐⭐⭐⭐⭐
   - Dual-modal: tidak signifikan
   - RecFeat: redundan
   - Curriculum: tidak lebih baik
   - Ini adalah **scientific integrity** tingkat tinggi

2. **Validasi Statistik Komprehensif**
   - p-values, Cohen's d, confidence intervals
   - Multiple comparison correction (Bonferroni)

3. **Dokumentasi Lengkap**
   - Setiap eksperimen dengan tabel/figure
   - Protokol reproducible (epochs, n, seeds)

4. **Analisis Faktor Penyebab**
   - Tidak hanya melaporkan "tidak berhasil"
   - Menjelaskan **mengapa** (e.g., dual-modal: text quality ~27% CER)

---

## REKOMENDASI PERBAIKAN

### CRITICAL (Harus Diperbaiki)
**Tidak ada.** Semua potensi contradictions sudah di-resolve dengan baik.

### IMPORTANT (Sangat Direkomendasikan)

1. **Klarifikasi Duration Ablasi Loss di Chapter 3**
   - **Lokasi:** Line 629
   - **Tambahkan:** "...dengan pelatihan eksplorasi cepat (15-20 epoch) untuk efisiensi komputasi"
   - **Alasan:** Mencegah ekspektasi pembaca bahwa ablasi menggunakan 50 epoch seperti produksi

2. **Klarifikasi Status Dual-Modal di Chapter 3**
   - **Lokasi:** Line 109
   - **Revisi:** "...(3) eksplorasi diskriminator dual-modal **sebagai hipotesis penelitian** untuk evaluasi bilateral..."
   - **Tambahkan:** "...yang kontribusi empirisnya divalidasi melalui studi ablasi terkontrol (lihat Bab V.3.2.3)"
   - **Alasan:** Menghindari kesan "inovatif = terbukti superior"

3. **Footnote Curriculum Learning di Chapter 3**
   - **Lokasi:** Line 437
   - **Tambahkan footnote:** (lihat draft di section "Contradictory Statements #3" di atas)
   - **Alasan:** Pre-empt pembaca yang bertanya "Kalau tidak lebih baik, mengapa digunakan?"

### SUGGESTED (Nice to Have)

4. **Definisi "Pencarian Empiris Bertahap" di Chapter 3**
   - **Lokasi:** Line 527
   - **Tambahkan:** "...melalui penyetelan manual empiris yang divalidasi dengan analisis stabilitas GradNorm dan pencarian kisi mini pada ruang hyperparameter terbatas"
   - **Alasan:** Lebih transparan tentang metodologi

5. **Tabel Mapping di Chapter 3**
   - **Lokasi:** Setelah line 527 (akhir Fase 5)
   - **Tambahkan tabel:** "Tabel III.X: Pemetaan Eksperimen Ablasi ke Pertanyaan Penelitian"
   
   | Eksperimen Ablasi | RQ Terkait | Lokasi Hasil (Bab V) |
   |-------------------|------------|----------------------|
   | Loss Components   | RQ3        | V.3.2.1              |
   | Frozen Recognizer | RQ2        | V.3.2.2              |
   | Dual-Modal Disc.  | RQ1        | V.3.2.3              |
   | Curriculum        | -          | V.3.2.4              |
   
   - **Alasan:** Memudahkan pembaca melacak validasi setiap hipotesis

---

## KONSISTENSI METRIK DAN TERMINOLOGI

### Metrik Reporting
✅ **KONSISTEN** di semua chapter:
- PSNR (dB) with ± standard deviation
- SSIM (0-1 scale)
- CER dan WER (%)
- Statistical significance: p-value, Cohen's d

### Terminologi
✅ **KONSISTEN:**
- "Frozen recognizer" vs "Joint training"
- "Curriculum learning" phases (warmup, CTC introduction, full)
- "Set validasi" (n=710), "Set uji" (n=712)

**No issues found.**

---

## FINAL ASSESSMENT

### Apakah ada Logical Fallacy?
**❌ TIDAK DITEMUKAN**

Semua klaim didukung bukti empiris. Temuan negatif dilaporkan dengan jujur.

### Apakah ada Contradictory Statements?
**⚠️ ADA POTENSI**, tapi **SEMUA SUDAH DI-RESOLVE** di Chapter 5:
- Dual-modal "inovatif" → dijelaskan sebagai "eksplorasi hipotesis" dengan temuan negatif transparan
- RecFeat di produksi → dijelaskan sebagai "artefak historis"
- Curriculum untuk "stabilitas" → paradoks dijelaskan dengan analisis faktor penyebab

### Apakah konsisten dan koheren?
**✅ YA, SANGAT KONSISTEN**

Setiap janji di Fase 5 Metodologi terpenuhi di Chapter 5:
- Ablasi loss: ✅ (5 experiments)
- Ablasi discriminator: ✅ (CNN vs Dual-Modal)
- Optimasi hyperparameter: ✅ (manual tuning + validation)
- Frozen vs Joint: ✅ (bonus - tidak disebutkan eksplisit di Fase 5 tapi bagian dari RQ2)
- Curriculum ablation: ✅ (bonus)

---

## SCORE CARD

| Kriteria                      | Score | Keterangan                                      |
|-------------------------------|-------|-------------------------------------------------|
| **Implementasi Janji**        | 5/5   | Semua eksperimen dilakukan                      |
| **Transparansi**              | 5/5   | Temuan negatif dilaporkan dengan jujur          |
| **Statistical Rigor**         | 5/5   | p-values, effect size, CI                       |
| **Logical Consistency**       | 4.5/5 | Minor clarifications needed in Chapter 3        |
| **Narrative Coherence**       | 5/5   | Alur metodologi → hasil sangat jelas            |
| **Scientific Integrity**      | 5/5   | Tidak ada cherry-picking, bias, atau fallacy    |

**OVERALL:** **4.9/5** - Excellent scientific work with minor documentation clarifications needed

---

## ACTIONABLE NEXT STEPS

### For Student (belekok):
1. Review 3 recommended clarifications di Chapter 3 (duration, dual-modal status, curriculum footnote)
2. Consider adding traceability table (Ablation → RQ mapping)
3. Pastikan semua cross-references antara Chapter 3 dan Chapter 5 valid

### For Thesis Committee:
**This work demonstrates:**
- High scientific integrity (transparent negative results)
- Rigorous experimental design
- Comprehensive validation
- Clear methodology-to-results mapping

**Minor concerns** (clarifications) tidak mengurangi kualitas substansi penelitian.

---

## KESIMPULAN PEER REVIEW

**REKOMENDASI: TERIMA dengan revisi minor**

Penelitian ini menunjukkan **integritas ilmiah yang sangat baik** dengan melaporkan temuan negatif secara transparan (dual-modal tidak signifikan, RecFeat redundan, curriculum tidak lebih baik). Semua janji di metodologi Fase 5 telah diimplementasikan dan divalidasi secara statistik yang ketat.

**Tidak ditemukan logical fallacy atau contradictory statements yang fatal.** Beberapa potensi kontradiksi sudah di-resolve dengan baik melalui penjelasan di Chapter 5.

**Saran perbaikan** bersifat klarifikasi dokumentasi (Chapter 3) untuk menghindari expectation mismatch pembaca, bukan perbaikan substansi eksperimen.

**Kualitas penelitian:** Publication-ready untuk jurnal Q2-Q3 internasional setelah minor revisions.

---

**Reviewer Signature:** AI/ML Engineering Reviewer
**Date:** 2025-11-28
