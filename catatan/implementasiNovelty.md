



















d# Rencana Implementasi dan Validasi: Cross-Modal Contrastive Loss

**Dokumen:** `implementasiNovelty.md`  
**Tanggal:** 13 Oktober 2025  
**Status:** Fase 1 (Implementasi Teknis) Selesai

---

## 1. Latar Belakang

Proyek ini bertujuan untuk meningkatkan kualitas restorasi dokumen kuno dengan fokus utama pada peningkatan keterbacaan teks oleh mesin (HTR). Setelah melalui serangkaian optimasi, kita telah berhasil menetapkan sebuah **"Golden Baseline"** (eksperimen `p200_r5`) yang menggunakan kombinasi *adversarial loss*, *pixel loss*, dan *recognition feature loss*.

Untuk mencapai kebaruan (novelty) yang signifikan dan hasil yang layak untuk publikasi Q1, langkah selanjutnya adalah mengimplementasikan ide yang lebih fundamental. Sesuai dokumen `catatan/potensiNovelty.md`, strategi yang paling menjanjikan adalah pengenalan **Cross-Modal Contrastive Loss**.

## 2. Tujuan

1.  **Implementasi:** Mengintegrasikan mekanisme *Cross-Modal Contrastive Loss* ke dalam arsitektur training `train32.py` yang sudah ada.
2.  **Validasi:** Secara ilmiah membuktikan efektivitas dari *contrastive loss* melalui studi ablasi yang terkontrol.
3.  **Publikasi:** Menghasilkan data kuantitatif dan kualitatif yang solid untuk mendukung klaim *novelty* dalam draf jurnal ilmiah.

## 3. Ruang Lingkup

Riset ini berfokus pada implementasi dan evaluasi *Cross-Modal Contrastive Loss*. Ini mencakup:
- Pembuatan model `TextEncoder` baru.
- Modifikasi model `Recognizer` untuk menghasilkan *image embedding*.
- Modifikasi skrip training `train32.py` untuk mengakomodasi logika loss yang baru.
- Evaluasi akan menggunakan metrik yang sudah ada: **PSNR, SSIM, CER, dan WER**.

## 4. Hipotesis

Dengan menambahkan *Cross-Modal Contrastive Loss*, kita memaksa *image embedding* (dari gambar hasil restorasi) untuk memiliki representasi yang mirip dengan *text embedding* (dari transkripsi teks yang benar) di dalam *latent space*. 

**Hipotesisnya adalah:** Pendekatan ini akan mendorong Generator untuk menghasilkan gambar yang tidak hanya bersih secara visual, tetapi juga **secara semantik lebih akurat dan tidak ambigu**. Hal ini akan menghasilkan **penurunan signifikan pada *Character Error Rate* (CER) dan *Word Error Rate* (WER)** dibandingkan dengan Golden Baseline, tanpa mengorbankan (atau bahkan berpotensi meningkatkan) metrik kualitas visual (PSNR dan SSIM).

## 5. Rencana Aksi

Berikut adalah tahapan kerja yang akan dilaksanakan:

### **Fase 1: Implementasi Teknis** `(Status: ✅ SELESAI)`
- **1.1.** Membuat model `TextEncoder` baru di `dual_modal_gan/src/models/text_encoder.py`.
- **1.2.** Memodifikasi `Recognizer` untuk menghasilkan output ketiga berupa *vector embedding* dari gambar.
- **1.3.** Mengintegrasikan `TextEncoder` dan `Contrastive Loss` ke dalam skrip training `train32.py`.
- **1.4.** Melakukan *Smoke Test* untuk memverifikasi bahwa semua komponen baru dapat berjalan tanpa error teknis.

### **Fase 2: Eksperimen & Studi Ablasi** `(Status: ⏳ AKAN DILAKSANAKAN)`
- **2.1.** Membuat skrip eksperimen `exp_contrastive_off.sh` untuk grup kontrol. Skrip ini akan menjalankan konfigurasi Golden Baseline dengan `--contrastive_loss_weight 0.0` selama 30 epoch.
- **2.2.** Membuat skrip eksperimen `exp_contrastive_on.sh` untuk grup perlakuan. Skrip ini akan menjalankan konfigurasi yang sama, namun dengan `--contrastive_loss_weight` yang disetel ke nilai yang masuk akal (misal, `1.0`).
- **2.3.** Menjalankan kedua skrip eksperimen untuk mengumpulkan data training.

### **Fase 3: Analisis Hasil & Penarikan Kesimpulan** `(Status: ⚪ BELUM DIMULAI)`
- **3.1.** Mengumpulkan metrik (PSNR, SSIM, CER, WER) dari kedua eksperimen pada epoch terbaiknya.
- **3.2.** Membuat tabel perbandingan untuk analisis kuantitatif.
- **3.3.** Melakukan analisis kualitatif dengan membandingkan sampel gambar yang dihasilkan secara visual.
- **3.4.** Menarik kesimpulan apakah hipotesis terbukti dan seberapa signifikan dampaknya.

### **Fase 4: Dokumentasi** `(Status: ⚪ BELUM DIMULAI)`
- **4.1.** Membuat laporan akhir riset (`final_report_contrastive_loss.md`) yang merangkum seluruh proses, temuan, dan analisis.
- **4.2.** Menulis draf awal untuk bagian "Metodologi" dan "Hasil" dari jurnal ilmiah berdasarkan temuan ini.

## 6. Metrik Keberhasilan

Keberhasilan implementasi ini akan diukur berdasarkan:
- **Utama:** Penurunan *Character Error Rate* (CER) yang signifikan secara statistik pada grup perlakuan (`contrastive_on`) dibandingkan grup kontrol.
- **Sekunder:** Metrik PSNR dan SSIM yang tetap stabil atau mengalami peningkatan.
- **Tambahan:** Proses training yang tetap stabil dan tidak mengalami divergensi.
