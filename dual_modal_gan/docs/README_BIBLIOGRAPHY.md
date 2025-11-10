# Panduan Penggunaan Bibliography Bersama

## Lokasi File
File bibliography utama terletak di:
```
dual_modal_gan/docs/bibliography.bib
```

## Cara Penggunaan di Chapter Lain

Untuk menggunakan bibliography bersama di chapter lainnya, tambahkan baris berikut di akhir chapter (sebelum `\end{document}`):

```latex
% Shared bibliography - imported from external file
\input{bibliography.bib}
```

### Contoh di Chapter:

```latex
% ... konten chapter ...

\textbf{Kesimpulan:} Penelitian ini mengintegrasikan GAN dengan frozen recognizer...

% Import bibliography bersama
\input{bibliography.bib}

\end{document}
```

## Struktur Bibliography

File `bibliography.bib` berisi 40+ referensi yang terorganisir dalam kategori:

### 1. Document Enhancement & Restoration (GAN-based)
- `erb2021` - Enhance to Read Better
- `souibgui2021` - DE-GAN
- `textdiae2022` - Text-DIAE
- `docentr2022` - DocEnTr
- `diaz2021` - Transformer-Based HTR

### 2. Generative Adversarial Networks
- `goodfellow2014` - GAN foundational
- `mirza2014` - Conditional GAN
- `isola2017` - Pix2Pix (Image-to-Image)

### 3. Loss Functions & Training
- `graves2006` - CTC Loss
- `johnson2016` - Perceptual Loss
- `chen2018` - GradNorm (Multi-task)

### 4. Image Quality Assessment
- `wang2004` - SSIM
- `thompson2023` - Metrics beyond PSNR

### 5. Document Image Binarization
- `gatos2006` - Adaptive binarization
- `pratikakis2013` - DIBCO 2013
- `pratikakis2019` - DIBCO 2019
- `tensmeyer2017` - Historical document binarization review

### 6. Handwritten Text Recognition (HTR)
- `shi2017` - CRNN-based scene text recognition
- `kiessling2023` - Vision Transformers for document analysis

### 7. Multimodal Learning
- `baltrusaitis2019` - Multimodal ML survey
- `vaswani2023` - Attention mechanisms for multimodal

### 8. Document Conservation
- `neevel1995` - Iron gall ink corrosion
- `krekel1999` - Chemistry of historical inks
- `reilly1993` - Photo preservation
- `porck2000` - Cellulose materials durability

### 9. Advanced Techniques (2024)
- `chen2024` - Diffusion models
- `ni2024` - Perceptual loss in GANs
- `martinez2024` - Multi-scale attention
- `rodriguez2024` - Few-shot learning
- `zhang2024` - Self-supervised learning

## Cara Mereferensi

Gunakan standard LaTeX citation syntax:

```latex
Penelitian Souibgui dkk. (2021)~\cite{erb2021} menunjukkan...

Menurut Gatos~\cite{gatos2006}, metode adaptif binarization...

Studi terbaru~\cite{chen2024} menggunakan diffusion models untuk...
```

## Menambah Referensi Baru

Jika perlu menambah referensi baru:

1. Edit file `bibliography.bib`
2. Tambahkan `\bibitem{key_unik}` dengan format:
   ```latex
   \bibitem{key_unik}
   Penulis (Tahun).
   \textit{Judul penelitian}.
   Jurnal atau Proceeding, Volume(No), halaman.
   ```
3. Simpan file
4. Semua chapter akan otomatis memiliki akses ke referensi baru

## Keuntungan Penggunaan Bibliography Bersama

✅ **Konsistensi** - Semua chapter menggunakan format referensi yang sama
✅ **Efisiensi** - Tidak perlu duplikasi bibliography di setiap chapter
✅ **Maintainability** - Update referensi di satu tempat
✅ **Kolaborasi** - Mudah untuk menambah referensi bersama
✅ **Organisasi** - Referensi dikelompokkan berdasarkan kategori ilmiah

## Chapter yang Sudah Menggunakan Bibliography Bersama

- ✅ Chapter 2 (Tinjauan Pustaka)
- ✅ Chapter 4 (Perancangan dan Implementasi)

## Chapter yang Perlu Update

- ⏳ Chapter 1 (Pendahuluan)
- ⏳ Chapter 3 (Analisis Kebutuhan)
- ⏳ Chapter 5 (Eksperimen dan Hasil)
- ⏳ Chapter 6 (Diskusi dan Kesimpulan)
