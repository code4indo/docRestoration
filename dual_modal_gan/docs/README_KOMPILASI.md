% ============================================================
% PANDUAN KOMPILASI TESIS
% ============================================================

# Cara Menggunakan Sistem Dokumen Tesis

## Struktur File:
1. main_tesis.tex          - File utama (kompilasi ini)
2. Chapter_Lambang.tex     - Daftar Lambang dan Singkatan
3. chapter1_pendahuluan.tex
4. chapter2_tinjauan_pustaka.tex
5. chapter3_metodologi.tex
6. chapter4_analysis_design.tex
7. chapter5_hasil.tex
8. chapter6_kesimpulan.tex

## Opsi Kompilasi:

### OPSI 1: Kompilasi Lengkap (RECOMMENDED)
Gunakan main_tesis.tex untuk menghasilkan dokumen lengkap dengan:
- Daftar Isi otomatis
- Daftar Gambar otomatis
- Daftar Tabel otomatis
- Penomoran halaman konsisten
- Cross-reference antar chapter

Perintah:
```bash
cd dual_modal_gan/docs/
pdflatex main_tesis.tex
pdflatex main_tesis.tex  # Jalankan 2x untuk update references
```

### OPSI 2: Kompilasi Individual Chapter (Untuk Editing)
Setiap chapter bisa dikompilasi sendiri untuk preview cepat.
File chapter individual sudah memiliki \documentclass sendiri.

Perintah:
```bash
cd dual_modal_gan/docs/
pdflatex chapter1_pendahuluan.tex
# atau
pdflatex chapter5_hasil.tex
```

## Catatan Penting:

1. **Penomoran Halaman:**
   - Bagian awal (Daftar Isi, Daftar Gambar, dll): Romawi kecil (i, ii, iii...)
   - Chapter 1-6: Arabic (1, 2, 3...)
   - Setiap chapter individual sudah set halaman awalnya

2. **Cross-Reference:**
   - Antar chapter akan resolved saat kompilasi main_tesis.tex
   - Kompilasi individual mungkin menampilkan "??" untuk ref ke chapter lain

3. **Daftar Pustaka:**
   - Edit bagian DAFTAR PUSTAKA di main_tesis.tex
   - Atau gunakan BibTeX (uncomment baris bibliography)

4. **Halaman Judul:**
   - Edit informasi di main_tesis.tex (Nama, NIM, Universitas, Tahun)

## Troubleshooting:

Jika melihat "??":
- Jalankan pdflatex 2-3 kali
- Pastikan semua file chapter ada di folder yang sama

Jika error "File not found":
- Pastikan nama file chapter sesuai
- Periksa lokasi file gambar yang direferensikan

## File Output:
- main_tesis.pdf         - Dokumen lengkap
- chapter*.pdf           - PDF individual (opsional)
