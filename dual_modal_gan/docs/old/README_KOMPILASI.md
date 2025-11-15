# Panduan Kompilasi Dokumen Tesis

## ✅ Status Kompilasi
**SUKSES** - PDF tesis lengkap (219 halaman) telah berhasil dibuat!

## 📁 Struktur File

### File Utama
- `main_tesis.tex` - Master document (gunakan ini untuk kompilasi lengkap)
- `main_tesis.pdf` - Output PDF final (219 halaman, 21MB)

### File Chapter (Standalone - bisa dikompilasi sendiri)
- `chapter1_pendahuluan.tex`
- `chapter2_tinjauan_pustaka.tex`
- `chapter3_metodologi.tex`
- `chapter4_analysis_design.tex`
- `chapter5_hasil.tex`
- `chapter6_kesimpulan.tex`
- `Chapter_Lambang.tex`

### File Chapter Content-Only (digunakan oleh main_tesis.tex)
- `chapter1_pendahuluan_content_only.tex`
- `chapter2_tinjauan_pustaka_content_only.tex`
- `chapter3_metodologi_content_only.tex`
- `chapter4_analysis_design_content_only.tex`
- `chapter5_hasil_content_only.tex`
- `chapter6_kesimpulan_content_only.tex`
- `Chapter_Lambang_content_only.tex`

**PENTING**: File `*_content_only.tex` adalah ekstraksi otomatis dari file chapter standalone. Jangan edit file ini secara manual. Edit file chapter asli, lalu jalankan ekstraksi ulang.

## 🚀 Cara Kompilasi

### A. Kompilasi Lengkap (Rekomendasi)

```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/docs

# Gunakan script otomatis (3 passes dengan cleanup)
./compile_tesis.sh full

# Atau manual
pdflatex main_tesis.tex
pdflatex main_tesis.tex  # Pass kedua untuk update references
pdflatex main_tesis.tex  # Pass ketiga untuk finalisasi
```

### B. Kompilasi Chapter Individual

```bash
# Gunakan script
./compile_tesis.sh ch1    # Chapter 1
./compile_tesis.sh ch5    # Chapter 5
./compile_tesis.sh ch6    # Chapter 6

# Atau manual
pdflatex chapter1_pendahuluan.tex
pdflatex chapter5_hasil.tex
```

### C. Re-extract Content Setelah Edit Chapter

Jika Anda mengedit file chapter standalone (misalnya `chapter5_hasil.tex`), jalankan:

```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/docs

# Extract semua chapter
for f in chapter1_pendahuluan chapter2_tinjauan_pustaka chapter3_metodologi \
         chapter4_analysis_design chapter5_hasil chapter6_kesimpulan Chapter_Lambang; do
    sed -n '/\\begin{document}/,/\\end{document}/p' ${f}.tex | sed '1d;$d' > ${f}_content_only.tex
done

# Lalu kompilasi ulang
./compile_tesis.sh full
```

## 📋 Fitur Dokumen Final

### Front Matter (Halaman Romawi: i, ii, iii, ...)
1. **Halaman Judul** - Template dengan placeholder untuk nama, NIM, universitas
2. **Daftar Isi** - Auto-generated dari semua chapter
3. **Daftar Gambar** - Auto-generated dari semua `\begin{figure}`
4. **Daftar Tabel** - Auto-generated dari semua `\begin{table}`
5. **Daftar Lambang dan Singkatan** - Dari Chapter_Lambang.tex

### Main Content (Halaman Arabic: 1, 2, 3, ...)
6. **BAB I - Pendahuluan**
7. **BAB II - Tinjauan Pustaka**
8. **BAB III - Metodologi Penelitian**
9. **BAB IV - Perancangan dan Implementasi Sistem**
10. **BAB V - Hasil dan Pembahasan**
11. **BAB VI - Kesimpulan dan Saran**

## ⚙️ Konfigurasi Dokumen

### Format
- **Ukuran Kertas**: A4
- **Font**: Times New Roman (mathptmx)
- **Ukuran Font**: 12pt
- **Spasi**: 1.5 (one-half spacing)
- **Margin**: Kiri 4cm, Kanan/Atas/Bawah 3cm

### Penomoran
- **Section**: Roman numerals (I, II, III, ...)
- **Subsection**: I.1, I.2, ...
- **Subsubsection**: I.1.1, I.1.2, ...
- **Halaman**: 
  - Front matter: i, ii, iii, ...
  - Main content: 1, 2, 3, ...

## 🔧 Troubleshooting

### Masalah: Gambar atau Tabel muncul sebagai "??"

**Penyebab**: Cross-references belum ter-resolve.

**Solusi**: Jalankan kompilasi 2-3 kali
```bash
pdflatex main_tesis.tex
pdflatex main_tesis.tex
```

### Masalah: Error "Can be used only in preamble"

**Penyebab**: File chapter yang di-input masih mengandung `\documentclass` atau `\usepackage`.

**Solusi**: Pastikan menggunakan file `*_content_only.tex` di `main_tesis.tex`, bukan file chapter asli.

### Masalah: Perubahan di chapter tidak muncul di PDF final

**Penyebab**: File `*_content_only.tex` belum di-update.

**Solusi**: Re-extract content dari chapter yang diedit:
```bash
sed -n '/\\begin{document}/,/\\end{document}/p' chapter5_hasil.tex | \
    sed '1d;$d' > chapter5_hasil_content_only.tex
./compile_tesis.sh full
```

### Masalah: File temporary (.aux, .log) menumpuk

**Solusi**: 
```bash
./compile_tesis.sh clean
```

## 📝 Catatan Penting

1. **Jangan edit file `*_content_only.tex` secara manual** - File ini auto-generated
2. **Jalankan kompilasi minimal 2x** untuk resolve cross-references
3. **File chapter standalone** (tanpa `_content_only`) bisa dikompilasi sendiri untuk testing cepat
4. **Edit informasi di halaman judul** - Ganti placeholder `[NAMA MAHASISWA]`, `[NIM]`, dll di `main_tesis.tex`
5. **Konfigurasi bibliography** - Uncomment dan sesuaikan section bibliography di akhir `main_tesis.tex`

## 📊 Hasil Kompilasi Terakhir

```
✅ Kompilasi selesai!
📁 Output: main_tesis.pdf
   Ukuran file: 21M
   Jumlah halaman: 219
```

## 🔄 Workflow Edit-Compile

1. Edit chapter yang diperlukan (misalnya `chapter5_hasil.tex`)
2. Re-extract content: `sed -n '/\\begin{document}/,/\\end{document}/p' chapter5_hasil.tex | sed '1d;$d' > chapter5_hasil_content_only.tex`
3. Kompilasi: `./compile_tesis.sh full`
4. Verifikasi PDF output
