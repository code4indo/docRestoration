# PDF Gabungan Chapter 1-4 + Daftar Pustaka

## 📄 File Output

**File PDF Gabungan:** `main_complete.pdf` (9.5 MB, 104 halaman)

### Struktur PDF Gabungan:

```
main_complete.pdf
├── BAB 1: Pendahuluan (28 halaman)
│   ├── 1.1 Latar Belakang
│   ├── 1.2 Rumusan Masalah
│   ├── 1.3 Pertanyaan Penelitian
│   ├── 1.4 Tujuan Penelitian
│   ├── 1.5 Ruang Lingkup
│   ├── 1.6 Hipotesis Penelitian
│   ├── 1.7 Kebaruan Penelitian
│   └── 1.8 Sistematika Penulisan
│
├── BAB 2: Tinjauan Pustaka (28 halaman)
│   ├── 2.1 Teori Fundamental GAN, U-Net, HTR
│   ├── 2.2 Jenis-Jenis Degradasi Dokumen
│   ├── 2.3 Metode Restorasi Dokumen
│   ├── 2.4 Metrik Evaluasi
│   └── 2.5 Analisis Celah Penelitian
│
├── BAB 3: Metodologi Penelitian (21 halaman)
│   ├── 3.1 Kerangka Kerja Penelitian
│   ├── 3.2 Desain Sistem
│   ├── 3.3 Lingkungan Eksperimen
│   └── 3.4 Linimasa Implementasi
│
├── BAB 4: Desain dan Analisis Sistem (10 halaman)
│   ├── 4.1 Analisis Kebutuhan
│   ├── 4.2 Arsitektur Sistem
│   ├── 4.3 Implementasi Teknis
│   └── 4.4 Pipeline Data dan Pelatihan
│
└── DAFTAR PUSTAKA (36 referensi)
    ├── Document Enhancement & Restoration
    ├── Generative Adversarial Networks
    ├── Loss Functions & Training
    ├── Document Binarization
    ├── Handwritten Text Recognition
    ├── Multimodal Learning
    └── ... (9 kategori total)
```

---

## 📋 File-File Terkait

### Master Document
- **`main_complete.tex`** - Master LaTeX document yang mengintegrasikan semua chapter
- Menggunakan `\documentclass{book}` untuk numerasi chapter otomatis
- Includes semua packages yang diperlukan oleh chapter 1-4

### Chapter Content Files (Content-Only, tanpa LaTeX header)
Digunakan oleh master document melalui `\input{}`:

1. **`chapter1_pendahuluan_content.tex`** (28 KB)
   - Content dari BAB 1: Pendahuluan
   - Latar belakang, rumusan masalah, tujuan, hipotesis

2. **`chapter2_tinjauan_pustaka_content.tex`** (108 KB)
   - Content dari BAB 2: Tinjauan Pustaka
   - Teori dasar, jenis degradasi, metode restorasi, metrik evaluasi

3. **`chapter3_metodologi_content.tex`** (53 KB)
   - Content dari BAB 3: Metodologi Penelitian
   - Kerangka kerja, desain sistem, lingkungan eksperimen

4. **`chapter4_analysis_design_content.tex`** (26 KB)
   - Content dari BAB 4: Desain dan Analisis Sistem
   - Analisis kebutuhan, arsitektur, implementasi teknis

### Bibliography Files
- **`bibliography.bib`** - Full bibliography dengan wrapper (original, untuk reference chapter terpisah)
- **`bibliography_content.bib`** - Bibliography content-only (digunakan di main_complete.pdf)
  - 36 referensi akademik terorganisir dalam 9 kategori
  - Format: `\bibitem{key} Author (Year). Title...`

### Original Chapter Files (Standalone, dengan LaTeX header)
Dapat dicompile sendiri:
- `chapter1_pendahuluan.tex` (363 baris)
- `chapter2_tinjauan_pustaka.tex` (1414 baris)
- `chapter3_metodologi.tex` (676 baris)
- `chapter4_analysis_design.tex` (391 baris)

---

## 🔧 Cara Menggunakan

### 1. **Download PDF Gabungan**
```bash
# PDF sudah tersedia di:
cd dual_modal_gan/docs/
ls -lh main_complete.pdf
```

### 2. **Jika ingin Recompile PDF dari Source**
```bash
cd dual_modal_gan/docs/

# Pastikan semua file ada:
# - main_complete.tex
# - chapter1_pendahuluan_content.tex
# - chapter2_tinjauan_pustaka_content.tex
# - chapter3_metodologi_content.tex
# - chapter4_analysis_design_content.tex
# - bibliography_content.bib

# Compile dengan 2 passes:
pdflatex -interaction=nonstopmode main_complete.tex
pdflatex -interaction=nonstopmode main_complete.tex

# Hasilnya: main_complete.pdf
```

### 3. **Jika ingin Menambah Chapter Tambahan**
Contoh menambah Chapter 5 (Results & Discussion):

```latex
% Di main_complete.tex, tambahkan sebelum Daftar Pustaka:

\chapter{Hasil dan Pembahasan}
\input{chapter5_results_discussion_content.tex}

% Pastikan chapter5_results_discussion_content.tex sudah ada
% (extracted dari chapter5_results_discussion.tex)
```

---

## 📊 Metadata PDF

| Aspek | Nilai |
|-------|-------|
| **Total Halaman** | 104 |
| **Ukuran File** | 9.5 MB |
| **Format** | PDF 1.5 (A4) |
| **Title** | Restorasi Dokumen Terdegradasi Menggunakan GAN dengan Diskriminator Dual-Modal |
| **Subject** | Tesis |
| **Creator** | LaTeX with hyperref |
| **Encryption** | Tidak ada (open format) |

---

## ✅ Verifikasi Isi

Struktur chapter dalam PDF (diverifikasi via `pdftotext`):
- ✓ BAB 1: Pendahuluan (28 hal)
- ✓ BAB 2: Tinjauan Pustaka (28 hal)
- ✓ BAB 3: Metodologi Penelitian (21 hal)
- ✓ BAB 4: Desain dan Analisis Sistem (10 hal)
- ✓ Daftar Pustaka (36 referensi, ~15 hal)

**Total: 104 halaman**

---

## 🎯 Kualitas Output

✅ **Typography & Formatting:**
- Spacing dan indentasi konsisten
- Table of Contents otomatis
- Cross-references valid
- Bibliography entries valid (36 entries)

✅ **Hyperlinks:**
- Hyperref package aktif
- Internal links (chapters, sections)
- External URLs supported

✅ **Language:**
- Bahasa Indonesia (Babel package)
- Hyphenation otomatis
- Special characters properly encoded (UTF-8)

---

## 📝 Catatan Penting

1. **Cleanup Temporary Files (Opsional):**
   ```bash
   # Menghapus file compile temporary
   cd dual_modal_gan/docs/
   rm -f main_complete.{aux,log,out,toc}
   ```

2. **Backup Master Document:**
   ```bash
   cp main_complete.pdf main_complete_backup_$(date +%Y%m%d).pdf
   ```

3. **Update Bibliography:**
   Jika ada perubahan di chapter 2 atau 4:
   - Update bibliography.bib
   - Generate ulang bibliography_content.bib: `tail -n +8 bibliography.bib | head -n -2 > bibliography_content.bib`
   - Recompile main_complete.pdf

---

## 🔍 Troubleshooting

### Jika error "Undefined control sequence"
```bash
# Pastikan semua packages di preamble:
# - tabularx
# - multirow
# - booktabs
# - subfig
# - caption
# - algorithm, algorithmic
# - tikz
# dll
```

### Jika error "Environment XXX undefined"
```bash
# Check main_complete.tex preamble, pastikan semua packages diload
pdflatex -interaction=nonstopmode main_complete.tex > compile.log 2>&1
grep -i "undefined" compile.log
```

### Jika PDF file corrupt
```bash
# Recompile dari awal
rm -f main_complete.pdf
pdflatex -interaction=nonstopmode main_complete.tex
pdflatex -interaction=nonstopmode main_complete.tex
```

---

## 📞 Support

Untuk masalah atau pertanyaan tentang PDF gabungan ini, silakan check:
1. Compile log: `/tmp/main_compile*.log`
2. File structure di `dual_modal_gan/docs/`
3. LaTeX documentation untuk specific packages

---

**Created:** Nov 10, 2025  
**Last Updated:** Nov 10, 2025  
**PDF Generated:** main_complete.pdf (9.5 MB, 104 pages)
