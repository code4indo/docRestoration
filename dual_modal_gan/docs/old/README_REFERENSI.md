# Sistem Referensi Terpisah - LaTeX Dokumen Tesis

## Struktur File

```
dual_modal_gan/docs/
├── chapter1_pendahuluan.tex      # Bab 1: Pendahuluan
├── chapter2_tinjauan_pustaka.tex  # Bab 2: Tinjauan Pustaka (asli)
├── chapter2_example.tex           # Contoh Bab 2 dengan referensi terpisah
├── references.tex                 # File referensi utama (dipakai semua bab)
└── README_REFERENSI.md           # File ini
```

## Cara Penggunaan

### 1. Menggunakan File Referensi Terpisah

Setiap file bab dapat mengimport referensi yang sama dengan cara:

```latex
% Di akhir setiap file bab, sebelum \end{document}
% ============================================================
% DAFTAR PUSTAKA
% ============================================================

% Import referensi dari file terpisah
% File references.tex berisi daftar pustaka lengkap untuk seluruh dokumen
\input{references.tex}
```

### 2. Mengcompile File Individual

Untuk mengcompile file individual (misalnya Bab 1 saja):

```bash
# Method 1: menggunakan pdflatex
pdflatex chapter1_pendahuluan.tex
pdflatex chapter1_pendahuluan.tex  # Jalankan 2x untuk referensi

# Method 2: menggunakan xelatex (recommended untuk Bahasa Indonesia)
xelatex chapter1_pendahuluan.tex
xelatex chapter1_pendahuluan.tex
```

### 3. Keuntungan Sistem Ini

#### ✅ **Konsistensi Referensi**
- Satu file `references.tex` untuk seluruh dokumen
- Menghindari duplikasi referensi
- Format konsisten di semua bab

#### ✅ **Manajemen Mudah**
- Tambah referensi baru hanya di satu file
- Edit format referensi hanya di satu tempat
- Reuse referensi di bab manapun

#### ✅ **Modular Development**
- Setiap bab dapat dikerjakan terpisah
- Compile individual bab tanpa dependensi bab lain
- Kolaborasi lebih mudah

#### ✅ **Format Standard**
- Menggunakan APA 7th Edition
- Sudah disesuaikan dengan standar penulisan ilmiah Indonesia
- Alfabetis otomatis

### 4. Format Referensi

File `references.tex` menggunakan format:

```latex
\bibitem{citationkey}
Author, A. A., Author, B. B., \& Author, C. C. (Year).
\textit{Judul Artikel}.
Nama Jurnal, Volume(Issue), Halaman-halaman.
```

### 5. Citation Keys yang Tersedia

- `goodfellow2014` - Original GAN paper
- `erb2019` - ERB-MultiTaskAdversarial
- `degan2021` - DE-GAN
- `docentr2022` - DocEnTr
- `textdiae2022` - Text-DIAE
- `baltrusaitis2019` - Multi-modal learning
- `graves2006` - CTC Loss
- `otsu1979` - Otsu thresholding
- `sauvola2000` - Sauvola thresholding
- `gatos2006` - Document binarization
- `jadhav2022` - Visual quality vs HTR performance

### 6. Menggunakan Referensi dalam Teks

```latex
% Single citation
Menurut Goodfellow et al. (2014), GAN terdiri dari dua jaringan...

% Multiple citation
Beberapa metode seperti DE-GAN (Souibgui \& Kessentini, 2021) dan DocEnTr (Souibgui et al., 2022)...

% In-text citation
...dalam penelitian terkini (Baltrusaitis, Ahuja, \& Morency, 2019).
```

### 7. Integrasi dengan Dokumen Utama

Untuk menggabungkan semua bab menjadi satu dokumen utama:

```latex
% Di main.tex
\documentclass{book}

\begin{document}

\include{chapter1_pendahuluan}
\include{chapter2_tinjauan_pustaka}
\include{chapter3_metodologi}
% ... bab lainnya

% Single reference list di akhir
\input{references.tex}

\end{document}
```

## Troubleshooting

### Problem: Referensi tidak muncul
**Solution:** Jalankan LaTeX dua kali
```bash
pdflatex chapter1_pendahuluan.tex
pdflatex chapter1_pendahuluan.tex
```

### Problem: Citation not found
**Solution:** Pastikan citation key di teks sama dengan di `references.tex`

### Problem: File not found
**Solution:** Pastikan `references.tex` ada di folder yang sama dengan file bab

## Best Practices

1. **Selalu gunakan citation key yang konsisten**
2. **Update `references.tex` jika menambah referensi baru**
3. **Check format APA 7th edition**
4. **Compile dua kali untuk memastikan referensi muncul**
5. **Backup `references.tex` secara berkala**

## Support

Untuk pertanyaan atau issues terkait sistem referensi:
- Check file contoh di `chapter2_example.tex`
- Referensi lengkap ada di `references.tex`
- Format sesuai standar ITB dan APA 7th Edition