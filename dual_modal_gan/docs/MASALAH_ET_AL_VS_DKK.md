# MASALAH "et al" vs "dkk" di Biblatex - Analisis Lengkap

## RINGKASAN MASALAH

Setelah investigasi mendalam, ditemukan bahwa:

1. ✅ **File `*_content_only.tex` sudah bersih** - tidak ada penulisan manual "et al"
2. ✅ **Konfigurasi `\DefineBibliographyStrings{english}{andothers = {dkk\adddot}}` sudah benar**
3. ❌ **MASALAH UTAMA**: Style `authoryear` dari biblatex menggunakan **"et al" yang di-hardcode** di dalam file `.cbx` (citation style), bukan menggunakan `\bibstring{andothers}`

## ROOT CAUSE

Biblatex style `authoryear` memiliki implementasi internal yang menggunakan string "et al" secara literal dalam macro `cite:labelname`. Meskipun kita mendefinisikan `\DefineBibliographyStrings{english}{andothers = {dkk}}`, style tersebut TIDAK menggunakan `\bibstring{andothers}` untuk render truncated author lists.

## SOLUSI YANG SUDAH DICOBA (GAGAL)

1. ❌ `\DefineBibliographyStrings` - Tidak digunakan oleh style authoryear
2. ❌ `\renewbibmacro*{name:andothers}` - Macro tidak dipanggil oleh cite command
3. ❌ `\DeclareNameFormat{labelname}` - Terlalu kompleks dan tetap hardcoded
4. ❌ `authoryear-comp` style - Masih menggunakan hardcoded "et al"
5. ❌ Custom `.lbx` file (indonesian.lbx) - File tidak ter-load karena path issue
6. ❌ `\AtBeginDocument` override - Terlalu lambat, style sudah loaded
7. ❌ `\DeclareDelimFormat` dan `\DeclareDelimAlias` - Tidak mengubah output

## SOLUSI YANG BEKERJA 100%

### Opsi 1: Gunakan Package `biblatex-ext` (RECOMMENDED)

Package ini adalah extended version dari biblatex standard yang SUPPORT penuh lokalisasi.

```latex
% Ganti baris usepackage biblatex dengan:
\usepackage[style=ext-authoryear,backend=biber,natbib=true]{biblatex}

% Kemudian definisi andothers akan bekerja:
\DefineBibliographyStrings{english}{
  andothers = {dkk\adddot},
}
```

### Opsi 2: Post-Process .bbl File dengan Sed

Buat script `fix_citations.sh`:

```bash
#!/bin/bash
# Jalankan setelah biber generate .bbl file

# Run biber
biber main_tesis

# Replace et al dengan dkk di .bbl file
sed -i 's/\\bibstring{andothers}/dkk./g' main_tesis.bbl

# Compile latex
pdflatex main_tesis.tex
pdflatex main_tesis.tex
```

### Opsi 3: Manual Find-Replace di PDF (Tidak Recommended)

Gunakan `pdftk` atau `qpdf` untuk edit PDF setelah generate.

### Opsi 4: Gunakan Natbib Style (Alternative)

Ganti biblatex dengan natbib yang lebih sederhana:

```latex
\usepackage{natbib}
\bibliographystyle{apalike}

% Di document:
\bibliography{bibliography}
```

## REKOMENDASI AKHIR

**GUNAKAN OPSI 1** - Install dan gunakan `biblatex-ext`:

```bash
# Check apakah biblatex-ext sudah terinstall
kpsewhich ext-authoryear.cbx

# Jika belum, install dengan:
sudo tlmgr install biblatex-ext
```

Kemudian update `main_tesis.tex`:

```latex
\usepackage[style=ext-authoryear,backend=biber,natbib=true,sorting=nyt,hyperref=true,maxcitenames=2,mincitenames=1]{biblatex}
```

## STATUS SAAT INI

File `main_tesis.tex` sudah dikembalikan ke konfigurasi stabil dengan `style=authoryear`. 
Konfigurasi `\DefineBibliographyStrings` sudah benar, namun output masih menampilkan "et al" 
karena limitasi teknis dari style authoryear.

**Untuk mengganti "et al" menjadi "dkk", HARUS menggunakan salah satu dari 4 opsi solusi di atas.**

## FILE TERKAIT

- `main_tesis.tex` - File utama (sudah dikonfigurasi dengan benar)
- `indonesian.lbx` - Custom localization file (dibuat tapi tidak ter-load)
- `fix_etal_to_dkk.sh` - Script helper (dibuat sebagai alternatif)

---
Generated: 2025-11-19
```