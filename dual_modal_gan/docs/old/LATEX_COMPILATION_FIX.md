# Analisis Masalah Kompilasi LaTeX dan Solusinya

## Masalah Utama yang Ditemukan

### 1. Package Conflict: bookmark vs hyperref
- **Problem**: Package `bookmark` dan `hyperref` keduanya mengatur bookmarks
- **Symptom**: Warning "Option `bookmarks' has already been used"
- **Impact**: Konfigurasi bookmark menjadi tidak konsisten

### 2. Rerun File Check Warning
- **Problem**: File `.out` berubah setelah kompilasi pertama
- **Symptom**: Warning "Rerun to get outlines right"
- **Impact**: Outline/bookmark PDF tidak optimal

### 3. Inkonsistensi Kompilasi
- **Problem**: Setiap bab dikompilasi terpisah dengan file .aux sendiri
- **Symptom**: Numbering dan referensi bisa berbeda antar kompilasi
- **Impact**: PDF yang dihasilkan tidak konsisten

## Solusi yang Diterapkan

### 1. Menghapus Package Bookmark
```latex
% SEBELUM
\usepackage{bookmark}
\usepackage{hyperref}

% SESUDAH
\usepackage{hyperref}
```

### 2. Kompilasi Dua Kali
- Kompilasi pertama: Generate file .out
- Kompilasi kedua: Resolve outline dengan benar

### 3. Menonaktifkan Hyphenation (Pemenggalan Kata)
**Solusi 1: Package-based (Kadang tidak efektif untuk bahasa Indonesia)**
```latex
\usepackage[none]{hyphenat} % Menonaktifkan hyphenation
```

**Solusi 2: Manual Complete Disable (Efektif untuk semua bahasa)**
```latex
% Menonaktifkan hyphenation secara total untuk Bahasa Indonesia
\lefthyphenmin=62
\righthyphenmin=62
\pretolerance=10000
\tolerance=2000
\emergencystretch=10pt
\hbadness=10000
\vbadness=10000
\raggedright
\hyphenpenalty=10000
\exhyphenpenalty=10000
\linepenalty=10000
\binoppenalty=10000
\relpenalty=10000
```

**Keterangan:**
- `\lefthyphenmin=62` dan `\righthyphenmin=62`: Membuat hyphenation tidak mungkin (minimum 62 karakter)
- `\raggedright`: Text alignment tanpa justifikasi penuh
- `\hyphenpenalty=10000`: Penalty maksimal untuk hyphenation
- `\pretolerance=10000`: Lebih memilih whitespace daripada hyphenation

### 4. Konfigurasi Hyperref yang Benar
```latex
\hypersetup{
    pdfencoding=auto,
    pdftitle={Bab 1: Pendahuluan},
    pdfauthor={},
    pdfsubject={Tesis},
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue,
    pdfstartview=FitH,
    bookmarks=true,
    bookmarksopen=true,
    bookmarksnumbered=true
}
```

## Best Practices untuk Kompilasi

1. **Hapus file sementara sebelum kompilasi ulang**:
   ```bash
   rm -f *.aux *.log *.out *.fdb_latexmk *.fls
   ```

2. **Kompilasi dengan urutan yang benar**:
   ```bash
   pdflatex file.tex
   pdflatex file.tex  # Jalankan dua kali
   ```

3. **Gunakan script kompilasi otomatis**:
   ```bash
   # Menggunakan latexmk untuk kompilasi otomatis
   latexmk -pdf file.tex
   ```

## Hasil Setelah Perbaikan

- ✅ Warning package conflict hilang
- ✅ Outline PDF terbentuk dengan benar
- ✅ Kompilasi bersih tanpa warning
- ✅ PDF yang dihasilkan konsisten

## Rekomendasi untuk Masa Depan

1. **Gunakan single main file** yang menggabungkan semua bab daripada kompilasi per bab
2. **Implementasikan sistem build otomatis** dengan makefile atau script
3. **Version control untuk file .aux** jika diperlukan untuk konsistensi referensi silang
4. **Regular cleanup** file temporary untuk menghindari conflict

## PEMBARUAN: PDF DAFTAR PUSTAKA TERPISAH

### Struktur Baru
1. **File stand-alone**: `standalone_references.tex` - khusus untuk PDF daftar pustaka
2. **Chapter tanpa referensi**: `chapter1_pendahuluan.tex` - tidak lagi mengandung referensi
3. **Script kompilasi**: `compile_references.sh` - kompilasi otomatis PDF referensi

### Cara Mengompilasi PDF Daftar Pustaka Terpisah
```bash
# Cara 1: Manual
pdflatex standalone_references.tex
pdflatex standalone_references.tex  # Jalankan dua kali

# Cara 2: Otomatis dengan script
./compile_references.sh
```

### Output yang Dihasilkan
- **Chapter PDF**: `chapter1_pendahuluan.pdf` (tanpa daftar pustaka)
- **References PDF**: `standalone_references.pdf` (hanya daftar pustaka)

### Keuntungan Struktur Terpisah
- ✅ Manajemen dokumentasi lebih modular
- ✅ Tidak ada konflik kompilasi antara chapter dan referensi
- ✅ PDF daftar pustaka bisa diedit independently
- ✅ Ukuran file lebih kecil untuk setiap komponen

---
*Dokumentasi ini dibuat pada 9 Oktober 2025 sebagai referensi untuk troubleshooting kompilasi LaTeX tesis.*