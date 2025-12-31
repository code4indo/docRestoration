# LAPORAN AUDIT DAFTAR SINGKATAN DAN LAMBANG

## Tanggal: 15 Desember 2025
## File: Chapter_Lambang_content_only.tex, Chapter_Lambang.tex

---

## RINGKASAN EKSEKUTIF

Audit ini dilakukan untuk memverifikasi bahwa nomor halaman pada daftar singkatan dan lambang 
sesuai dengan **halaman pertama kali** singkatan/lambang tersebut digunakan dalam dokumen.

### Hasil Temuan:
- **Total Singkatan:** 62
- **Perlu Update:** 57 singkatan (92%)
- **Tidak Ditemukan dalam Dokumen:** 5 singkatan
- **Total Lambang:** 51
- **Lambang dengan halaman placeholder (75):** 50 (98%)

---

## SINGKATAN YANG PERLU DIPERBARUI

| Singkatan | Halaman Lama | Halaman Baru | Chapter |
|-----------|--------------|--------------|---------|
| ANRI | 1 | 12 | BAB II |
| API | 85 | 66 | BAB III |
| BCE | 30 | 19 | BAB II |
| BiGRU | 25 | 19 | BAB II |
| BiLSTM | 25 | 73 | BAB III |
| CBAM | 35 | 78 | BAB IV |
| CER | 5 | 1 | BAB I |
| cGAN | 20 | 17 | BAB II |
| CLI | 90 | 66 | BAB III |
| CNN | 15 | 2 | BAB I |
| CRNN | 20 | 18 | BAB II |
| CTC | 25 | 2 | BAB I |
| CUDA | 90 | 68 | BAB III |
| cuDNN | 90 | 68 | BAB III |
| DE-GAN | 18 | 1 | BAB I |
| DIBCO | 22 | 13 | BAB II |
| DocEnTr | 19 | 1 | BAB I |
| DRD | 22 | 19 | BAB II |
| DSRM | 50 | 5 | BAB I |
| ERB | 18 | 2 | BAB I |
| FCN | 16 | 18 | BAB II |
| FM | 22 | 19 | BAB II |
| Fps | 22 | 19 | BAB II |
| FR | 60 | 65 | BAB III |
| GAN | 2 | 1 | BAB I |
| GPU | 85 | 35 | BAB II |
| GT | 55 | 20 | BAB II |
| H-DIBCO | 22 | 21 | BAB II |
| HMM | 25 | 30 | BAB II |
| HTR | 3 | 1 | BAB I |
| IAM | 27 | 19 | BAB II |
| ICDAR | 22 | 25 | BAB II |
| IIIT5K | 27 | 25 | BAB II |
| KHATT | 27 | 19 | BAB II |
| LSTM | 25 | 2 | BAB I |
| ML | 2 | 65 | BAB III |
| MLflow | 90 | 62 | BAB III |
| MSE | 30 | 14 | BAB II |
| MSFP | 35 | 78 | BAB IV |
| NFR | 65 | 66 | BAB III |
| OCR | 3 | 21 | BAB II |
| PALM | 27 | 29 | BAB II |
| Pix2Pix | 19 | 17 | BAB II |
| PSNR | 5 | 1 | BAB I |
| RDB | 35 | 78 | BAB IV |
| RMSProp | 85 | 20 | BAB II |
| RNN | 25 | 18 | BAB II |
| SGD | 85 | 89 | BAB IV |
| SNR | 30 | 105 | BAB V |
| SOTA | 5 | 21 | BAB II |
| SSIM | 5 | 1 | BAB I |
| U-Net | 16 | 5 | BAB I |
| VGG | 17 | 38 | BAB II |
| ViT | 19 | 25 | BAB II |
| VOC | 27 | 12 | BAB II |
| WER | 28 | 2 | BAB I |

---

## SINGKATAN YANG TIDAK DITEMUKAN DALAM DOKUMEN

Singkatan berikut **tidak ditemukan** dalam konten chapter dan sebaiknya 
**DIHAPUS** dari daftar singkatan jika memang tidak digunakan:

1. **AI** (Artificial Intelligence) - Tidak digunakan sebagai singkatan mandiri
2. **CWV** (Core Web Vitals) - Tidak ditemukan
3. **MAE** (Mean Absolute Error) - Tidak ditemukan
4. **SDM** (Sumber Daya Manusia) - Tidak ditemukan
5. **SOP** (Standard Operating Procedure) - Tidak ditemukan

---

## STATUS LAMBANG

### Permasalahan:
Hampir semua lambang (50 dari 51) tercantum dengan halaman **75**, yang menunjukkan 
ini adalah **placeholder yang belum diverifikasi**.

### Rekomendasi:
Lambang matematis perlu diverifikasi secara **MANUAL** karena sulit dicari secara otomatis.
Sebagian besar lambang muncul di:
- **BAB IV** (Arsitektur dan Formula Matematis) - halaman ~77-100
- **BAB V** (Analisis Statistik) - halaman ~102-150

---

## TINDAKAN YANG DIPERLUKAN

### Langkah 1: Perbarui Singkatan
File `Chapter_Lambang_content_only_UPDATED.tex` sudah berisi koreksi untuk 57 singkatan.

Untuk menerapkan perubahan:
```bash
cp Chapter_Lambang_content_only_UPDATED.tex Chapter_Lambang_content_only.tex
```

### Langkah 2: Hapus Singkatan yang Tidak Digunakan
Hapus baris berikut dari daftar singkatan:
- AI
- CWV
- MAE
- SDM
- SOP

### Langkah 3: Verifikasi Manual Lambang
Periksa halaman kemunculan pertama untuk setiap lambang di:
- BAB IV (Formula Matematis)
- BAB V (Analisis Statistik)

### Langkah 4: Update Chapter_Lambang.tex
Sinkronisasi perubahan dari `Chapter_Lambang_content_only.tex` ke `Chapter_Lambang.tex`.

### Langkah 5: Re-compile Dokumen
```bash
./sync_all_chapters.sh
# Kemudian kompilasi ulang main_tesis_duplex.tex
```

---

## CATATAN

Estimasi halaman berdasarkan pencarian teks dan perhitungan posisi baris.
Setelah kompilasi ulang, halaman mungkin sedikit bergeser (±1-2 halaman).
Disarankan untuk memverifikasi ulang beberapa entri kunci setelah kompilasi.

---

*Laporan ini dihasilkan oleh audit_singkatan_lambang.py dan update_singkatan_pages.py*
