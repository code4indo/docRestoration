# 📝 Rencana Aksi: Optimasi Model Baseline

Dokumen ini menguraikan rencana aksi yang terstruktur dan sistematis untuk misi "Optimalkan Baseline Saat Ini". Tujuannya adalah untuk mencapai performa terbaik dari arsitektur dan strategi training yang ada saat ini sebelum melangkah ke eksplorasi novelty.

---

## 1. Latar Belakang Masalah

Setelah berhasil melakukan pivot dari strategi CTC Loss ke **Recognition Feature Loss**, analisis terhadap hasil training awal menunjukkan sebuah keberhasilan sekaligus tantangan baru:

-   **Keberhasilan:** Model berhasil menurunkan **Character Error Rate (CER)** secara signifikan. Ini membuktikan bahwa *Recognition Feature Loss* adalah pemandu yang efektif untuk meningkatkan keterbacaan teks.
-   **Tantangan:** Kualitas visual gambar hasil restorasi masih sangat rendah, yang tercermin dari skor **PSNR (Peak Signal-to-Noise Ratio)** yang rendah (~15) dan stagnan. Selain itu, `CTC Loss` (yang kini hanya sebagai metrik monitor) tetap tinggi, mengindikasikan bahwa gambar yang dihasilkan masih ambigu bagi model recognizer.

Akar masalahnya adalah konfigurasi hyperparameter (terutama bobot loss) yang belum optimal, sehingga model lebih fokus pada keterbacaan teks dengan mengorbankan kualitas visual.

## 2. Tujuan

Tujuan utama dari fase optimasi ini adalah untuk **menemukan set hyperparameter terbaik yang memaksimalkan kualitas visual (PSNR) dan kualitas keterbacaan teks (CER) secara bersamaan**.

Hasil dari fase ini adalah sebuah **"Baseline Teroptimasi"**: model dengan performa terbaik yang bisa dicapai oleh arsitektur saat ini, yang akan menjadi fondasi dan titik perbandingan untuk eksplorasi novelty di masa depan.

## 3. Ruang Lingkup

Optimasi ini akan berfokus pada hal-hal berikut:

-   **Model:** Arsitektur yang ada saat ini (U-Net Generator, Dual-Modal Discriminator, Frozen Recognizer).
-   **Skrip Training:** `dual_modal_gan/scripts/train32.py`.
-   **Variabel yang Dioptimasi:**
    -   Bobot loss: `pixel_loss_weight`, `rec_feat_loss_weight`, `adv_loss_weight`.
    -   Jadwal Learning Rate (Learning Rate Schedule).
-   **Metrik Evaluasi Utama:** `combined_score` (`PSNR - (bobot_cer * CER)`), `PSNR`, dan `CER`.

Di luar ruang lingkup untuk fase ini adalah perubahan arsitektur model atau penambahan fungsi loss yang fundamental baru.

## 4. Hipotesis

Hipotesis utama yang akan diuji dalam serangkaian eksperimen ini adalah:

> "Dengan **meningkatkan tekanan pada kualitas visual** melalui `pixel_loss_weight` yang lebih tinggi, dan **menstabilkan training tahap akhir** dengan *learning rate scheduler*, kita dapat secara signifikan meningkatkan skor PSNR tanpa mengorbankan perolehan CER yang sudah baik, bahkan berpotensi menurunkannya lebih jauh karena gambar input untuk recognizer menjadi lebih jernih dan tidak ambigu."

---

## 5. Rencana Aksi Terstruktur

#### **Fase 1: Desain Eksperimen Sistematis**

Kita akan beralih dari tebakan ke eksperimen yang terkontrol untuk menguji hipotesis kita.

1.  **Langkah 1.1: Analisis Skala Loss (Titik Awal Ilmiah)**
    -   **Aksi:** Menjalankan training singkat (2-3 epoch) dengan semua bobot loss diatur ke `1.0`.
    -   **Tujuan:** Mengamati magnitudo alami dari setiap komponen loss untuk mendapatkan rasio bobot dasar yang "adil" sebagai titik awal, bukan tebakan.

2.  **Langkah 1.2: Pencarian Terfokus (Focused Grid Search)**
    -   **Aksi:** Menjalankan serangkaian eksperimen dengan variasi pada dua bobot paling penting, berdasarkan hasil Analisis Skala.
        -   **`pixel_loss_weight`:** [50.0, **100.0**, 200.0] (Variasi utama untuk menguji hipotesis)
        -   **`rec_feat_loss_weight`:** [25.0, **50.0**] (Variasi sekunder untuk keseimbangan)
    -   **Tujuan:** Menemukan kombinasi bobot terbaik yang memaksimalkan metrik keberhasilan.

3.  **Langkah 1.3: Implementasi *Learning Rate Scheduler***
    -   **Aksi:** Memodifikasi `train32.py` untuk menambahkan `tf.keras.optimizers.schedules.ExponentialDecay` atau skema serupa pada optimizer.
    -   **Tujuan:** Mengatasi stagnasi performa yang terjadi di akhir training dengan memungkinkan model melakukan *fine-tuning* dengan langkah yang lebih kecil.

#### **Fase 2: Analisis Hasil & Penentuan Baseline Terbaik**

1.  **Metrik Keberhasilan:** Menggunakan `combined_score` (`PSNR - (bobot_cer * CER)`) sebagai metrik utama untuk menentukan "pemenang" dari setiap eksperimen.

2.  **Analisis Trade-off:** Memvisualisasikan hasil (plot PSNR vs. CER) untuk memahami trade-off antara kualitas visual dan keterbacaan teks dari setiap kombinasi bobot.

3.  **Penentuan Baseline:** Kombinasi hyperparameter dari eksperimen terbaik akan ditetapkan sebagai **"Baseline Teroptimasi"** yang baru.

#### **Fase 3: Dokumentasi**

1.  **Membuat Laporan:** Membuat file baru di `@catatan/`, misalnya `OPTIMASI_BASELINE_REPORT.md`.
2.  **Mencatat Hasil:** Laporan akan berisi hasil dari Analisis Skala, tabel hasil dari semua eksperimen, analisis, dan yang terpenting, **hyperparameter final dari "Baseline Teroptimasi"** yang terpilih untuk referensi di masa depan.
