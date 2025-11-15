# 🎓 Quick Start - Kompilasi Tesis

## ✅ Status: READY TO USE

PDF tesis lengkap (**219 halaman, 21MB**) sudah tersedia di:
```
/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/docs/main_tesis.pdf
```

---

## 🚀 Kompilasi Cepat

### Kompilasi Lengkap
```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/docs
./compile_tesis.sh full
```

**Output**: `main_tesis.pdf` (daftar isi + 6 chapter lengkap)

### Kompilasi Per Chapter
```bash
./compile_tesis.sh ch1    # Chapter 1: Pendahuluan
./compile_tesis.sh ch2    # Chapter 2: Tinjauan Pustaka
./compile_tesis.sh ch3    # Chapter 3: Metodologi
./compile_tesis.sh ch4    # Chapter 4: Perancangan
./compile_tesis.sh ch5    # Chapter 5: Hasil
./compile_tesis.sh ch6    # Chapter 6: Kesimpulan
```

---

## 📝 Workflow Edit Chapter

**Scenario**: Anda ingin edit Chapter 5

```bash
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/docs

# 1. Edit file chapter
nano chapter5_hasil.tex   # atau gunakan editor lain

# 2. Re-extract content (PENTING!)
sed -n '/\\begin{document}/,/\\end{document}/p' chapter5_hasil.tex | \
    sed '1d;$d' > chapter5_hasil_content_only.tex

# 3. Kompilasi ulang
./compile_tesis.sh full

# 4. Verifikasi hasil
ls -lh main_tesis.pdf
```

---

## ⚡ One-Liner: Edit → Extract → Compile

```bash
# Edit semua chapter yang berubah, lalu jalankan ini
cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/docs && \
for f in chapter1_pendahuluan chapter2_tinjauan_pustaka chapter3_metodologi \
         chapter4_analysis_design chapter5_hasil chapter6_kesimpulan Chapter_Lambang; do \
    sed -n '/\\begin{document}/,/\\end{document}/p' ${f}.tex | sed '1d;$d' > ${f}_content_only.tex; \
done && \
./compile_tesis.sh full
```

---

## 📋 Checklist Sebelum Submit

- [ ] Edit halaman judul di `main_tesis.tex` (Nama, NIM, Universitas, Tahun)
- [ ] Verifikasi semua gambar tampil dengan benar
- [ ] Cek tidak ada "??" pada nomor gambar/tabel
- [ ] Verifikasi daftar isi, daftar gambar, daftar tabel
- [ ] Cek penomoran halaman (Romawi → Arabic)
- [ ] Review formatting (spasi, margin, font)
- [ ] Tambahkan Daftar Pustaka jika diperlukan

---

## 🔧 Troubleshooting Cepat

| Problem | Solution |
|---------|----------|
| Error "Can be used only in preamble" | Re-extract content: `sed -n '/\\begin{document}/,/\\end{document}/p' chapter5_hasil.tex \| sed '1d;$d' > chapter5_hasil_content_only.tex` |
| Gambar/Tabel muncul "??" | Kompilasi 2-3x: `./compile_tesis.sh full` |
| Perubahan tidak muncul | Re-extract content chapter yang diedit |
| File temporary menumpuk | Cleanup: `./compile_tesis.sh clean` |

---

## 📚 Dokumentasi Lengkap

Lihat `README_KOMPILASI.md` untuk:
- Struktur file detail
- Konfigurasi dokumen
- Troubleshooting lengkap
- Workflow lanjutan

---

## 💡 Tips

1. **Selalu re-extract** setelah edit chapter
2. **Kompilasi 2-3x** untuk resolve references
3. **Test per-chapter** untuk debug cepat
4. **Gunakan script** `compile_tesis.sh` untuk efisiensi
5. **Backup** file chapter sebelum edit besar
