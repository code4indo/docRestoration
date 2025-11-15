# Panduan Bibliografi Terpusat

## ✅ Status Implementasi

**Tanggal**: 15 November 2025  
**Status**: ✓ **SELESAI - Bibliografi Terpusat Aktif**

## 📋 Ringkasan Perubahan

### 1. Konfigurasi Terpusat di `main_tesis.tex`

Bibliografi sekarang dikonfigurasi HANYA di file utama:

```latex
% Di main_tesis.tex (line 38-50)
\usepackage[style=apa,backend=biber]{biblatex}
\usepackage{csquotes}
\addbibresource{bibliography.bib}

\DefineBibliographyStrings{english}{
  andothers       = {dkk.},
  mathesis        = {Tesis Master},
  phdthesis       = {Disertasi},
  and             = {dan},
}
```

### 2. Chapter Files - Konfigurasi Lokal DIHAPUS

Konfigurasi biblatex **telah dihapus** dari:
- ✓ `chapter5_hasil.tex` (baris 62-73 dihapus)
- ✓ `chapter4_analysis_design.tex` (baris 14-26 dihapus)
- ✓ Semua chapter lainnya tidak memiliki konfigurasi biblatex lokal

### 3. File Bibliografi: `bibliography.bib`

Lokasi: `dual_modal_gan/docs/bibliography.bib`

**Referensi yang ditambahkan**:
- ✓ `chen2018gradnorm` - GradNorm untuk adaptive loss balancing
- ✓ `souibgui2021enhance` - Baseline research "Enhance to Read Better"
- ✓ `souibgui2022docentr` - DocEnTr transformer enhancement
- ✓ `kang2021pay` - Multi-scale attention network

Total: **17 entries** di bibliography.bib

## 🚀 Cara Kompilasi

### Metode 1: Script Otomatis (RECOMMENDED)

```bash
cd dual_modal_gan/docs
./compile_with_bibliography.sh main_tesis
```

### Metode 2: Manual

```bash
cd dual_modal_gan/docs

# Pass 1
pdflatex -interaction=nonstopmode main_tesis.tex

# Process bibliography
biber main_tesis

# Pass 2 & 3
pdflatex -interaction=nonstopmode main_tesis.tex
pdflatex -interaction=nonstopmode main_tesis.tex
```

### Metode 3: Makefile (Opsional - Bisa ditambahkan)

```makefile
# Tambahkan ke Makefile jika ada
.PHONY: all clean

all: main_tesis.pdf

main_tesis.pdf: main_tesis.tex bibliography.bib
	pdflatex -interaction=nonstopmode main_tesis.tex
	biber main_tesis
	pdflatex -interaction=nonstopmode main_tesis.tex
	pdflatex -interaction=nonstopmode main_tesis.tex

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.synctex.gz
```

## 📝 Menambahkan Referensi Baru

### 1. Edit `bibliography.bib`

```bibtex
@article{key2025,
  author    = {Penulis, A. and Penulis, B.},
  title     = {Judul Penelitian},
  journal   = {Nama Jurnal},
  year      = {2025},
  volume    = {10},
  number    = {2},
  pages     = {100--120}
}
```

### 2. Gunakan di Chapter

```latex
% Di chapter manapun (misalnya chapter5_hasil.tex)
Penelitian terbaru menunjukkan bahwa... \cite{key2025}
```

### 3. Kompilasi Ulang

```bash
./compile_with_bibliography.sh main_tesis
```

## 🔍 Verifikasi

### Cek Sitasi yang Tidak Ditemukan

```bash
grep "Citation.*undefined" main_tesis.log
```

### Cek Referensi yang Tidak Ditemukan

```bash
grep "LaTeX Warning.*undefined" main_tesis.log
```

### Cek File Bibliography yang Di-generate

```bash
cat main_tesis.bbl | head -50
```

## ⚠️ Troubleshooting

### Problem: "Citation undefined"

**Penyebab**: Referensi belum ada di `bibliography.bib`

**Solusi**:
1. Tambahkan entry ke `bibliography.bib`
2. Jalankan biber: `biber main_tesis`
3. Kompilasi ulang: `pdflatex main_tesis.tex` (2x)

### Problem: "Empty bibliography"

**Penyebab**: Tidak ada `\cite{}` yang digunakan di dokumen

**Solusi**: Pastikan ada minimal 1 `\cite{}` di salah satu chapter

### Problem: Biber warning about locale 'id-ID'

**Status**: ⚠️ **NORMAL - Bukan Error**

Ini hanya warning dari Biber versi lama tentang locale Indonesia. Tidak mempengaruhi hasil akhir.

## 📊 Struktur Akhir

```
dual_modal_gan/docs/
├── main_tesis.tex              ← Konfigurasi biblatex TERPUSAT di sini
├── bibliography.bib            ← Semua referensi di sini (SATU file)
├── chapter1_pendahuluan_content_only.tex
├── chapter2_tinjauan_pustaka_content_only.tex
├── chapter3_metodologi_content_only.tex
├── chapter4_analysis_design_content_only.tex    ← Tidak ada konfigurasi biblatex
├── chapter5_hasil_content_only.tex              ← Tidak ada konfigurasi biblatex
├── chapter6_kesimpulan_content_only.tex
└── compile_with_bibliography.sh ← Script helper
```

## ✅ Keuntungan Sistem Terpusat

1. **Konsistensi**: Satu sumber referensi untuk semua chapter
2. **Maintenance**: Mudah update/tambah referensi
3. **No Duplication**: Tidak ada duplikasi konfigurasi
4. **Portability**: Mudah dipindah atau di-share
5. **APA Style**: Style konsisten di seluruh dokumen

## 📚 Referensi yang Tersedia

Saat ini tersedia 17 referensi mencakup:
- Document Enhancement (GAN-based)
- Generative Adversarial Networks
- Deep Learning Architectures
- Multi-Task Learning
- HTR/OCR Systems

Lihat `bibliography.bib` untuk daftar lengkap.

## 🎯 Next Steps

1. ✓ Sistem bibliografi terpusat sudah aktif
2. ✓ Referensi penting sudah ditambahkan
3. ⏭️ Compile dan verifikasi semua chapter
4. ⏭️ Tambahkan referensi tambahan sesuai kebutuhan chapter

---

**Catatan**: Untuk compile individual chapter (misalnya untuk test), chapter masih bisa di-compile standalone, tetapi bibliografi tidak akan muncul. Untuk hasil final, SELALU compile via `main_tesis.tex`.
