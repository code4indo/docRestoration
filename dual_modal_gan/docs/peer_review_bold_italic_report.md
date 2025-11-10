# 📋 PEER REVIEW REPORT: PENGGUNAAN BOLD & ITALIC

## 🔍 STATISTIK ANALISIS
- **Bold (`\textbf{}`):** 181 kali digunakan
- **Italic (`\textit{}`):** 273 kali digunakan
- **Total:** 454 kali kombinasi formatting

---

## ⚠️ MASALAH YANG DITEMUKAN

### 1. ITALIC BERLEBIHAN (KRITIS) ❌

#### A. Istilah Teknis Berulang:
```latex
// PROBLEMATIC:
GAN → \textit{GAN} (diulang 50+ kali)
U-Net → \textit{U-Net} (diulang 30+ kali)
CRNN → \textit{CRNN} (diulang 40+ kali)
CNN → \textit{CNN} (diulang 60+ kali)
```

**Rekomendasi:** Setelah pertama kali dijelaskan, gunakan format normal (tanpa italic)

#### B. KataFungsi yang Ikut Teritalic:
```latex
// PROBLEMATIC:
"untuk \textit{Generator} yang..."
"dan \textit{Diskriminator}..."
"dengan \textit{loss}..."

// SEHARUSNYA:
"untuk Generator..."
"dan Diskriminator..."
"dengan loss..."
```

#### C. Nama Penulis/Institusi:
```latex
// PROBLEMATIC:
Goodfellow \textit{dkk.} (2014)
Souibgui \textit{et al.} (2019)

// SEHARUSNYA:
Goodfellow dkk. (2014)
Souibgui dkk. (2019)
```

---

### 2. BOLD BERLEBIHAN (SEDANG) ⚠️

#### A. Sub-bab Normal (OK):
```latex
\textbf{A. Bleed Through} ✓ [Appropriate]
\textbf{B. Fading} ✓ [Appropriate]
```

#### B. Item List Berlebihan:
```latex
// PROBLEMATIC:
\item \textbf{Generator:} FCN...
\item \textbf{Diskriminator:} FCN...
\item \textbf{Pengenal CRNN:} CNN...
\item \textbf{Fungsi Loss:} L = ...

// SEHARUSNYA:
\item Generator: FCN...
\item Diskriminator: FCN...
\item Pengenal CRNN: CNN...
\item Fungsi Loss: L = ...
```

#### C. Label yang Tidak Penting:
```latex
// PROBLEMATIC:
\textbf{[PLACEHOLDER GAMBAR]}
\textbf{Evolusi Temporal Penelitian}

// SEHARUSNYA:
[PLACEHOLDER GAMBAR]
Evolusi Temporal Penelitian
```

---

## ✅ REKOMENDASI PERBAIKAN

### PRIORITAS 1: Hapus Italic Berlebihan

**TARGET 50% PENGURANGAN (273 → ~135)**

1. **Hapus italic dari:**
   - Nama metode yang sudah dikenal (GAN, U-Net, CNN, CRNN, LSTM, Transformer)
   - Kata sambung dan fungsi (untuk, dan, dengan, pada, dalam)
   - Nama penulis dan institusi
   - Singkatan teknis (PSNR, SSIM, CER, WER, FCN, RNN)

2. **Tetap gunakan italic untuk:**
   - Nama paper/jurnal: "Enhance to Read Better"
   - Istilah bahasa asing yang belum diterjemahkan: "adversarial", "minimax"
   - Nama dataset: "KHATT", "IAM", "DIBCO"
   - First mention dari istilah baru yang perlu penekanan

### PRIORITAS 2: Kurangi Bold Berlebihan

**TARGET 20% PENGURANGAN (181 → ~145)**

1. **Hapus bold dari:**
   - Item list
   - Label placeholder
   - Nama sub-bab tingkat 3 dan seterusnya

2. **Tetap gunakan bold untuk:**
   - Judul utama bab: "BAB II. Tinjauan Pustaka"
   - Judul sub-bab tingkat 1 dan 2
   - Nama kategori penting: "Bleed Through", "Fading"

### PRIORITAS 3: Konsistensi

1. **Buat daftar istilah yang boleh menggunakan italic:**
   - First mention: GAN, U-Net, CRNN, dll
   - subsequent mention: gan, u-net, crnn (normal text)

2. **Penerapan规则 konsisten di seluruh dokumen**

---

## 📊 DAMPAK PERBAIKAN

### Sebelum Perbaikan:
- Total formatting: 454 kali
- Italic: 273 (60% dari total)
- Bold: 181 (40% dari total)

### Setelah Perbaikan (Target):
- Total formatting: ~280 kali
- Italic: ~135 (48% dari total)
- Bold: ~145 (52% dari total)

**Pengurangan:** 38% lebih sedikit formatting

---

## 🎯 KESIMPULAN

**Status:** PEMANISAN BERLEBIHAN (Over-Formatting)

**Prioritas Perbaikan:**
1. **KRITIS:** Kurangi italic berlebihan (target -50%)
2. **SEDANG:** Kurangi bold tidak perlu (target -20%)
3. **RINGAN:** Standardisasi konsistensi

**Benefit:**
- Dokumen lebih mudah dibaca
- Penekanan lebih efektif pada yang penting
- Mengikuti standar penulisan akademik
- Less visual noise

**Effort:** 2-3 jam manual editing
**Impact:** Signifikan meningkatkan kualitas presentasi
