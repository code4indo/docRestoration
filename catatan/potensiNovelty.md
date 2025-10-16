# 💡 Potensi Novelty untuk Riset GAN-HTR

Dokumen ini menguraikan beberapa strategi riset yang berpotensi menghasilkan kebaruan (novelty) yang signifikan untuk publikasi ilmiah tingkat tinggi (Q1). Ide-ide ini melampaui sekadar optimasi hyperparameter dan menyentuh perubahan fundamental pada arsitektur, fungsi loss, dan paradigma training.

---

## 1. Inovasi pada Arsitektur (Interaksi Model)

Fokusnya adalah mengubah cara model-model kita (Generator, Recognizer, Discriminator) berkomunikasi untuk menciptakan sinergi yang lebih dalam.

### Ide Unggulan: "Iterative Refinement with Recognizer Attention"

-   **Konsep:** Mengubah alur informasinya dari satu arah menjadi dua arah, menciptakan sebuah *feedback loop*.
    1.  Generator menghasilkan gambar restorasi versi pertama (`v1`).
    2.  Recognizer mencoba membaca `v1` dan **menghasilkan sebuah *attention map***. *Attention map* ini menyorot area-area di mana recognizer merasa paling "bingung" atau "tidak yakin".
    3.  *Attention map* ini kemudian **diumpankan kembali (fed back)** sebagai input tambahan ke Generator.
    4.  Generator menggunakan informasi ini untuk menghasilkan gambar versi kedua (`v2`), dengan instruksi implisit: "Perbaiki area-area yang membingungkan ini!"

-   **Klaim Novelty:** "Sebuah mekanisme penyempurnaan iteratif (*iterative refinement*) yang dipandu oleh atensi dari modul pengenalan teks. Arsitektur ini memungkinkan modul restorasi untuk secara cerdas memfokuskan kapasitasnya pada area-area yang secara semantik paling menantang, menghasilkan restorasi yang tidak hanya bersih secara visual tetapi juga optimal untuk keterbacaan mesin."

-   **Potensi:** Ide ini sangat intuitif, meniru cara kerja manusia, dan berpotensi besar meningkatkan akurasi pada bagian-bagian dokumen yang paling sulit.

---

## 2. Inovasi pada Fungsi Loss (Jantung dari Novelty)

Fungsi loss mendefinisikan "tujuan" dari model. Mengubah tujuan adalah cara paling langsung untuk mengubah perilaku dan menghasilkan terobosan.

### Ide Unggulan: "Cross-Modal Contrastive Loss"

-   **Konsep:** Terinspirasi dari model SOTA seperti CLIP, kita beralih dari mencocokkan piksel ke mencocokkan **makna**. Tujuannya adalah membuat representasi vektor (embedding) dari gambar hasil restorasi menjadi:
    -   **Dekat** dengan embedding dari teks transkripsi yang benar.
    -   **Jauh** dari embedding dari teks transkripsi yang salah (misalnya, teks dari sampel lain dalam batch yang sama).

-   **Implementasi:**
    1.  Anda memerlukan *text encoder* (misalnya, LSTM atau pre-trained model) untuk mengubah teks menjadi vektor.
    2.  Recognizer (atau bagian darinya) berfungsi sebagai *image encoder*.
    3.  Fungsi loss baru (misalnya, InfoNCE loss) ditambahkan untuk memaksimalkan *cosine similarity* antara pasangan `(gambar_hasil, teks_benar)` dan meminimalkan similarity antara `(gambar_hasil, teks_salah)`.

-   **Klaim Novelty:** "Pengenalan *cross-modal contrastive loss* dalam training GAN untuk restorasi dokumen. Pendekatan ini secara eksplisit memaksa generator untuk menghasilkan gambar yang tidak hanya bersih, tetapi juga **secara semantik tidak ambigu** dan berkorespondensi kuat dengan transkripsi teksnya, secara signifikan mengurangi halusinasi teks yang sering terjadi pada model standar."

-   **Potensi:** Ini adalah lompatan konseptual yang sangat modern dan kuat, berpotensi menjadi inti dari publikasi Q1.

---

## 3. Inovasi pada Paradigma Training

Fokusnya adalah mengubah *bagaimana* model belajar dari data, bukan *apa* yang dipelajari.

### Ide Unggulan: "Curriculum Learning on Degradation"

-   **Konsep:** Jangan melatih model dengan semua jenis kerusakan sekaligus. Buat sebuah "kurikulum" dari mudah ke sulit.
    1.  **Tahap 1 (Mudah):** Latih model hanya pada dokumen dengan kerusakan ringan (sedikit noise, blur ringan).
    2.  **Tahap 2 (Menengah):** Setelah model mahir, lanjutkan training dengan kerusakan yang lebih sulit (bercak tinta, lipatan kertas).
    3.  **Tahap 3 (Sulit):** Lanjutkan dengan kerusakan paling ekstrem (tulisan tangan tumpang tindih, kerusakan akibat air).

-   **Klaim Novelty:** "Sebuah strategi *curriculum learning* yang diatur berdasarkan tingkat kesulitan degradasi. Dengan melatih model secara progresif dari tugas restorasi sederhana ke kompleks, kami menunjukkan konvergensi yang lebih cepat, stabilitas training yang lebih baik, dan kemampuan generalisasi yang superior pada kerusakan dunia nyata yang belum pernah dilihat sebelumnya."

-   **Potensi:** Sangat logis dan praktis, dapat meningkatkan robustisitas dan kecepatan konvergensi model secara signifikan.

---

## Rencana Aksi dan Rekomendasi

Dari ketiga strategi di atas, **Strategi 2 (Cross-Modal Contrastive Loss)** memiliki potensi novelty paling tinggi dan fundamental.

**Rencana Riset yang Disarankan:**
1.  **Optimalkan Baseline:** Selesaikan optimasi model saat ini (dengan *Recognition Feature Loss*) untuk mendapatkan baseline yang kuat.
2.  **Implementasikan Ide Baru:** Pilih salah satu ide di atas (disarankan Contrastive Loss) untuk diimplementasikan.
3.  **Lakukan Studi Ablasi (Ablation Study):** Ini adalah **kunci mutlak** untuk paper Q1. Bandingkan performa `Baseline` vs `Baseline + Ide Baru` untuk membuktikan kontribusi dari ide baru tersebut secara terisolasi.
4.  **Analisis Mendalam:** Lakukan analisis kuantitatif dan kualitatif yang komprehensif untuk menunjukkan keunggulan pendekatan baru dibandingkan metode sebelumnya.
