# Rencana Aksi Riset (Pivot): Iterative Refinement with Recognizer Attention

**Dokumen:** `perbaikanNovelty.md`  
**Tanggal:** 13 Oktober 2025  
**Status:** 🔴 **UNTUK DILAKSANAKAN**

---

## 1. Latar Belakang

Eksperimen sebelumnya dengan **Cross-Modal Contrastive Loss** telah selesai. Hasilnya menunjukkan bahwa pendekatan tersebut, dalam implementasinya saat ini, tidak memberikan peningkatan performa dibandingkan **Golden Baseline** kita. Ini adalah sebuah temuan riset yang valid dan penting.

Sesuai dengan semangat riset yang iteratif, kita sekarang akan melakukan pivot strategis. Berdasarkan dokumen `catatan/potensiNovelty.md`, kandidat terkuat berikutnya adalah **Ide #1: "Iterative Refinement with Recognizer Attention"**.

Konsep intinya adalah menciptakan sebuah *feedback loop*, di mana Generator tidak hanya bekerja satu arah, tetapi menerima masukan dari Recognizer untuk memperbaiki hasil restorasinya secara iteratif.

## 2. Tujuan

1.  **Implementasi:** Merancang dan mengintegrasikan mekanisme *feedback loop* di mana *attention map* dari Recognizer diumpankan kembali ke Generator.
2.  **Validasi:** Membuktikan secara kuantitatif bahwa mekanisme penyempurnaan iteratif ini mampu menurunkan *error rate* (CER/WER) secara lebih signifikan dibandingkan Golden Baseline.
3.  **Analisis:** Menganalisis bagaimana *feedback loop* ini mempengaruhi kualitas visual dan stabilitas training.

## 3. Hipotesis

Dengan memberikan *attention map* (yang menyorot area-area sulit/tidak jelas) dari Recognizer kembali ke Generator sebagai input tambahan, Generator akan belajar untuk secara cerdas memfokuskan kapasitasnya pada area-area yang paling krusial untuk keterbacaan teks.

**Hipotesis (H$_1$):** Mekanisme *Iterative Refinement* ini akan menghasilkan **penurunan CER/WER yang lebih besar** dibandingkan Golden Baseline, karena proses restorasi secara aktif dipandu oleh kebutuhan dari sistem HTR, bukan hanya oleh optimasi visual.

## 4. Rencana Aksi

### **Fase 1: Perbaikan Teknis & Persiapan** `(Status: ⏳ AKAN DILAKSANAKAN)`
- **1.1.** **Perbaiki Bug cuDNN:** Terapkan fix `use_cudnn=False` pada `text_encoder.py` untuk memastikan stabilitas semua eksperimen di masa depan. Ini adalah prioritas utama sebelum memulai implementasi baru.

### **Fase 2: Implementasi "Iterative Refinement"** `(Status: ⚪ BELUM DIMULAI)`
- **2.1.** **Modifikasi Recognizer:** Menyesuaikan arsitektur `Recognizer` (kemungkinan pada bagian Transformer) untuk dapat **mengekspos *attention scores*** dari *multi-head attention layers*. *Attention map* ini akan digabungkan untuk membentuk satu peta atensi tunggal.
- **2.2.** **Modifikasi Generator:** Menyesuaikan arsitektur `Generator` (U-Net) agar dapat menerima **input multi-channel**. Inputnya tidak lagi hanya `(degraded_image)`, tetapi menjadi `concatenate([degraded_image, attention_map])`.
- **2.3.** **Modifikasi `train_step`:** Merombak alur logika di dalam `train_step` untuk mengimplementasikan proses dua langkah:
    1.  `generated_v1 = generator(degraded_image)`
    2.  `attention_map = recognizer(generated_v1)`
    3.  `input_v2 = concatenate([degraded_image, attention_map])`
    4.  `generated_v2 = generator(input_v2)`
    5.  Semua *loss* (adversarial, pixel, dll.) akan dihitung berdasarkan `generated_v2`.

### **Fase 3: Eksperimen & Analisis** `(Status: ⚪ BELUM DIMULAI)`
- **3.1.** Membuat skrip eksperimen baru untuk arsitektur iteratif ini.
- **3.2.** Menjalankan eksperimen penuh dan membandingkan hasilnya (CER, WER, PSNR, SSIM) secara langsung dengan **Golden Baseline**.

## 5. Metrik Keberhasilan

- Penurunan CER/WER yang signifikan pada model baru dibandingkan Golden Baseline.
- Metrik PSNR/SSIM yang tetap kompetitif atau bahkan meningkat.
