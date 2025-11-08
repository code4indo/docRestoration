# Panduan Konversi Paper ke Bahasa Indonesia

## Status Konversi

### ✅ Sudah Dikonversi:
1. **Packages & Setup** - Ditambahkan `\usepackage[bahasa]{babel}`
2. **Title & Authors** - Sudah dalam Bahasa Indonesia  
3. **Abstract** - Sudah dalam Bahasa Indonesia
4. **Keywords** - Sudah dalam Bahasa Indonesia
5. **Section I (Pendahuluan)** - Sudah sebagian dikonversi

### 🔄 Perlu Dikonversi:
1. Section II (Tinjauan Pustaka) - sebagian besar masih English
2. Section III (Metode yang Diusulkan) - masih English
3. Section IV (Setup Eksperimental) - masih English
4. Section V (Hasil dan Pembahasan) - masih English
5. Section VI (Diskusi) - masih English
6. Section VII (Kesimpulan) - masih English
7. Appendices - masih English
8. References - biarkan dalam format asli (English)

---

## Opsi Konversi

### **Opsi 1: Konversi Manual (Disarankan untuk Quality Control)**

Gunakan find & replace di editor favorit Anda:

**Section Headers**:
- `\section{Introduction}` → `\section{Pendahuluan}`
- `\section{Related Work}` → `\section{Tinjauan Pustaka}`
- `\section{Proposed Method}` → `\section{Metode yang Diusulkan}`
- `\section{Experimental Setup}` → `\section{Setup Eksperimental}`
- `\section{Results and Discussion}` → `\section{Hasil dan Pembahasan}`
- `\section{Discussion}` → `\section{Diskusi}`
- `\section{Conclusion}` → `\section{Kesimpulan}`

**Subsection Headers - Common Patterns**:
- `Framework Overview` → `Gambaran Umum Kerangka Kerja`
- `Generator Architecture` → `Arsitektur Generator`
- `Discriminator Architecture` → `Arsitektur Diskriminator`
- `Loss Function` → `Fungsi Loss`
- `Training Strategy` → `Strategi Training`
- `Datasets` → `Dataset`
- `Evaluation Metrics` → `Metrik Evaluasi`
- `Implementation Details` → `Detail Implementasi`
- `Quantitative Results` → `Hasil Kuantitatif`
- `Qualitative Analysis` → `Analisis Kualitatif`
- `Ablation Studies` → `Studi Ablasi`
- `Failure Case Analysis` → `Analisis Kasus Kegagalan`
- `Limitations` → `Keterbatasan`
- `Future Work` → `Penelitian Mendatang`

**Figure & Table Captions**:
- `Figure` → `Gambar`
- `Table` → `Tabel`
- `[PLACEHOLDER FIGURE X]` → `[PLACEHOLDER GAMBAR X]`

**Common Terms**:
- `our method` → `metode kami`
- `proposed approach` → `pendekatan yang diusulkan`
- `baseline methods` → `metode baseline`
- `state-of-the-art` → `state-of-the-art` (biarkan)
- `ground truth` → `ground truth` (biarkan)
- `training` → `training` (atau `pelatihan`)
- `testing` → `testing` (atau `pengujian`)
- `validation` → `validasi`
- `hyperparameter` → `hyperparameter` (biarkan)
- `optimization` → `optimasi`

---

### **Opsi 2: Gunakan Script Sed (Batch Replace)**

Buat file `convert_to_indonesian.sh`:

```bash
#!/bin/bash

FILE="jatniko.tex"

# Backup original
cp $FILE ${FILE}.backup

# Section headers
sed -i 's/\\section{Introduction}/\\section{Pendahuluan}/g' $FILE
sed -i 's/\\section{Related Work}/\\section{Tinjauan Pustaka}/g' $FILE
sed -i 's/\\section{Proposed Method}/\\section{Metode yang Diusulkan}/g' $FILE
sed -i 's/\\section{Experimental Setup}/\\section{Setup Eksperimental}/g' $FILE
sed -i 's/\\section{Results and Discussion}/\\section{Hasil dan Pembahasan}/g' $FILE
sed -i 's/\\section{Discussion}/\\section{Diskusi}/g' $FILE
sed -i 's/\\section{Conclusion}/\\section{Kesimpulan}/g' $FILE

# Common subsections
sed -i 's/Framework Overview/Gambaran Umum Kerangka Kerja/g' $FILE
sed -i 's/Generator Architecture/Arsitektur Generator/g' $FILE
sed -i 's/Discriminator Architecture/Arsitektur Diskriminator/g' $FILE
sed -i 's/Loss Function/Fungsi Loss/g' $FILE
sed -i 's/Training Strategy/Strategi Training/g' $FILE
sed -i 's/Datasets/Dataset/g' $FILE
sed -i 's/Evaluation Metrics/Metrik Evaluasi/g' $FILE
sed -i 's/Implementation Details/Detail Implementasi/g' $FILE

# Figure/Table
sed -i 's/PLACEHOLDER FIGURE/PLACEHOLDER GAMBAR/g' $FILE
sed -i 's/\\caption{Examples of/\\caption{Contoh /g' $FILE

echo "Conversion complete. Check $FILE"
```

**Jalankan**:
```bash
chmod +x convert_to_indonesian.sh
./convert_to_indonesian.sh
```

---

### **Opsi 3: Saya Buat File Baru yang Lebih Ringkas** ✅ **RECOMMENDED**

Saya akan membuat `jatniko_id.tex` yang:
- Sudah full Bahasa Indonesia
- Struktur sama tapi lebih ringkas (fokus pada inti)
- Semua placeholder tetap ada
- Siap untuk diisi dengan data aktual

---

## Rekomendasi Saya

**Untuk draft cepat → jurnal nasional**: Gunakan **Opsi 3** (saya buatkan file baru)
**Untuk jurnal internasional nanti**: Tetap gunakan `jatniko_english_backup.tex`

### Keuntungan Opsi 3:
✅ Lebih cepat - langsung dapat file lengkap Bahasa Indonesia
✅ Struktur tetap sama dengan English version
✅ Placeholder semua ada
✅ Bisa langsung compile dan lihat hasilnya
✅ Nanti tinggal translate kembali ke English (lebih mudah dari Indonesia → English)

### Workflow yang Disarankan:
1. **Saya buat** `jatniko_id.tex` (Bahasa Indonesia, ~8-10 halaman)
2. **Anda isi** data aktual (angka, gambar, tabel)
3. **Review** dengan pembimbing dalam Bahasa Indonesia
4. **Sempurnakan** struktur dan konten
5. **Translate** ke English untuk jurnal Q1 internasional
6. **Submit** ke jurnal target

---

## Apakah Anda Ingin Saya Lanjutkan dengan Opsi 3?

Jika ya, saya akan membuat `jatniko_id.tex` lengkap dengan:
- ✅ Semua section dalam Bahasa Indonesia
- ✅ Struktur identik dengan versi English
- ✅ Semua placeholder figure/table
- ✅ Notasi matematis sama
- ✅ Citations tetap dalam format English (standard)
- ✅ Siap compile langsung

**Estimasi waktu**: 5-10 menit untuk generate file lengkap.

Konfirmasi jika Anda ingin saya lanjutkan! 🚀
