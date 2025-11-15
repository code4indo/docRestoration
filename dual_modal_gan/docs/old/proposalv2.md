BAB I. PENDAHULUAN

I.1 Latar Belakang
Arsip Nasional RI menyimpan berbagai koleksi arsip bersejarah yang sangat bernilai dan diakui sebagai “Memory of the World Register” dari UNESCO. Dokumen-dokumen ini, yang sebagian besar berupa tulisan tangan kuno dari abad ke-16 hingga ke-18, merupakan representasi jati diri bangsa dan aset budaya yang tak ternilai. Namun, seiring berjalannya waktu dan akibat faktor lingkungan, dokumen-dokumen ini mengalami degradasi fisik, mulai dari yang ringan hingga berat. Kerusakan seperti tembusan tinta (ink bleed-through), pemudaran (fading), noda, dan kerusakan struktural tidak hanya mengancam keutuhan visual, tetapi juga integritas informasi yang terkandung di dalamnya. Kondisi ini menjadi tantangan serius bagi upaya pelestarian digital, sebagaimana diamanatkan oleh UU No. 43 Tahun 2009 tentang Kearsipan.

Degradasi fisik merupakan penghambat utama dalam proses digitalisasi dan transkripsi otomatis. Secara khusus, kualitas gambar yang buruk secara signifikan menurunkan kinerja sistem Handwritten Text Recognition (HTR), yang krusial untuk mengubah arsip tulisan tangan menjadi teks yang dapat dicari dan dianalisis. Akibatnya, tingkat kesalahan pengenalan karakter (Character Error Rate - CER) pada dokumen yang rusak menjadi sangat tinggi, membuat transkripsi otomatis tidak dapat diandalkan dan memaksa lembaga untuk tetap bergantung pada proses transkripsi manual yang lambat dan mahal.

Pendekatan restorasi digital yang ada saat ini dapat dibagi menjadi dua kategori: tradisional dan berbasis deep learning. Metode tradisional seperti thresholding adaptif (misalnya, Sauvola) efektif untuk degradasi sederhana, namun gagal pada kasus yang kompleks seperti bleed-through atau latar belakang yang tidak seragam. Di sisi lain, metode berbasis Generative Adversarial Networks (GAN) menunjukkan potensi besar. Namun, banyak implementasi GAN yang ada, seperti DE-GAN, berfokus secara eksklusif pada peningkatan kualitas visual (diukur dengan PSNR atau SSIM). Peningkatan metrik visual ini, sayangnya, tidak selalu berkorelasi dengan peningkatan keterbacaan teks oleh mesin HTR. Seringkali, proses restorasi yang terlalu agresif justru dapat menghaluskan atau bahkan menghilangkan detail-detail penting dari gurat tulisan, sehingga gambar yang "bersih" secara visual justru lebih sulit dibaca oleh HTR.

Menyadari kesenjangan ini, penelitian ini mengusulkan sebuah kerangka kerja GAN yang dirancang secara fundamental untuk tujuan yang berbeda: restorasi yang berorientasi pada keterbacaan (HTR-Oriented Restoration). Kebaruan utama dari penelitian ini terletak pada desain arsitektur dan fungsi loss yang secara eksplisit mengoptimalkan hasil restorasi untuk memaksimalkan akurasi HTR. Hal ini dicapai melalui dua inovasi utama:
1.  **Diskriminator Dual-Modal:** Sebuah diskriminator yang tidak hanya menilai "keaslian" sebuah gambar, tetapi juga "koherensi" antara konten visual gambar tersebut dengan transkripsi teksnya. Ini memberikan sinyal pelatihan yang lebih kaya daripada diskriminator unimodal standar.
2.  **Fungsi Loss Berorientasi HTR (HTR-Oriented Loss):** Fungsi loss untuk Generator secara langsung memasukkan komponen loss yang dihitung dari model HTR yang sudah terlatih dan dibekukan (frozen). Secara spesifik, CTC (Connectionist Temporal Classification) loss digunakan untuk "menghukum" Generator jika menghasilkan gambar yang menyebabkan model HTR salah membaca, bahkan jika gambar tersebut terlihat bersih secara visual.

Dengan menggabungkan kedua elemen ini, model yang diusulkan dipaksa untuk belajar merestorasi dokumen dengan cara yang tidak hanya mempercantik tampilan visual, tetapi yang terpenting, mempertahankan dan memperjelas struktur karakter tulisan tangan agar dapat ditranskripsi secara akurat. Pendekatan ini diharapkan dapat menghasilkan solusi restorasi yang jauh lebih efektif untuk kebutuhan praktis lembaga kearsipan dan para peneliti.

I.2 Rumusan Masalah
Berdasarkan latar belakang, masalah utama yang dihadapi adalah ketidakmampuan metode restorasi digital yang ada untuk secara simultan meningkatkan kualitas visual dan keterbacaan teks (legibility) pada dokumen historis yang terdegradasi. Masalah ini dapat dirinci menjadi beberapa poin berikut:
1.  **Fokus Berlebihan pada Metrik Visual:** Sebagian besar metode restorasi berbasis GAN, seperti DE-GAN, mengoptimalkan metrik seperti PSNR dan SSIM. Peningkatan ini seringkali tidak menjamin penurunan pada metrik akurasi HTR (CER/WER), bahkan terkadang memperburuknya karena distorsi pada struktur karakter.
2.  **Diskriminator Konvensional yang "Buta Teks":** Diskriminator pada GAN standar hanya mengevaluasi realisme gambar (unimodal). Ia tidak memiliki kemampuan untuk menilai apakah konten teks pada gambar yang direstorasi koheren atau masuk akal, sehingga Generator dapat menghasilkan artefak visual yang terlihat plausibel namun merusak teks.
3.  **Ketiadaan Sinyal "Keterbacaan" Langsung:** Tanpa adanya umpan balik (feedback) langsung yang berkaitan dengan performa HTR selama proses training, Generator tidak memiliki insentif untuk memprioritaskan restorasi detail-detail halus pada tulisan yang krusial bagi model HTR.
4.  **Kelangkaan Dataset Berpasangan (Paired Dataset):** Untuk melatih model restorasi secara terawasi (supervised), dibutuhkan dataset yang berisi pasangan gambar rusak dan versi bersihnya yang identik. Untuk dokumen historis, dataset semacam ini hampir mustahil untuk didapatkan.

Berdasarkan rincian masalah tersebut, pertanyaan penelitian utama dirumuskan sebagai berikut:
**"Bagaimana merancang sebuah kerangka kerja Generative Adversarial Network (GAN) yang mampu merestorasi dokumen historis terdegradasi dengan secara eksplisit mengoptimalkan keterbacaan teks untuk sistem Handwritten Text Recognition (HTR)?"**

Pertanyaan penelitian turunan yang akan dijawab adalah:
a.  Bagaimana arsitektur Diskriminator Dual-Modal dapat dirancang untuk menilai koherensi antara konten visual gambar dan konten semantik teks?
b.  Bagaimana fungsi loss berbasis HTR (CTC Loss) dapat diintegrasikan ke dalam loop training GAN untuk secara langsung memandu Generator menghasilkan gambar yang lebih mudah dibaca oleh mesin?
c.  Bagaimana sebuah pipeline pembuatan dataset sintetis yang terawasi dapat dibangun untuk mengatasi kelangkaan data berpasangan di dunia nyata?

I.3 Tujuan Penelitian
Tujuan utama dari penelitian ini adalah untuk mengembangkan, mengimplementasikan, dan mengevaluasi sebuah kerangka kerja restorasi dokumen berbasis GAN yang secara fundamental dioptimalkan untuk meningkatkan akurasi transkripsi oleh model HTR.

Tujuan spesifik dari penelitian ini adalah:
1.  **Merancang dan Mengimplementasikan Arsitektur GAN dengan Diskriminator Dual-Modal:** Membangun sebuah model Diskriminator yang memiliki dua jalur pemrosesan input paralel (satu untuk gambar, satu untuk teks) yang digabungkan untuk menghasilkan penilaian koherensi antara kedua modalitas tersebut.
2.  **Mengintegrasikan Fungsi Loss Berorientasi HTR:** Mengimplementasikan sebuah fungsi loss gabungan untuk Generator yang mencakup komponen CTC Loss. Loss ini dihitung dengan melewatkan gambar hasil restorasi ke sebuah model Recognizer (HTR) yang bobotnya dibekukan, sehingga memberikan sinyal gradien yang memaksa Generator untuk memprioritaskan keterbacaan.
3.  **Membangun Pipeline Pembuatan Dataset Sintetis:** Mengembangkan dan menerapkan sebuah alur kerja untuk menghasilkan dataset triplet (gambar rusak, gambar bersih, label teks) berkualitas tinggi secara sintetis. Proses ini menggunakan gambar bersih asli dan menerapkan serangkaian augmentasi degradasi yang realistis (berbasis tekstur dari dokumen rusak asli, noda, blur, dll.) untuk menciptakan data training yang efektif.
4.  **Mengevaluasi Performa Model secara Komprehensif:** Menguji kerangka kerja yang diusulkan pada dataset sintetis dan data riil, serta membandingkan performanya dengan metode baseline menggunakan metrik visual (PSNR, SSIM) dan metrik keterbacaan (CER, WER) untuk membuktikan keunggulannya.

I.4 Ruang Lingkup
Agar penelitian ini tetap fokus dan dapat mencapai tujuannya secara efektif, ruang lingkup dan batasan masalah ditetapkan sebagai berikut:
1.  **Objek Penelitian:** Fokus utama adalah restorasi dokumen tulisan tangan kuno (paleografi) dari abad ke-16 hingga ke-18. Dokumen cetak tidak termasuk dalam cakupan penelitian ini.
2.  **Jenis Degradasi:** Degradasi yang ditangani terbatas pada jenis yang paling umum ditemukan, yaitu tembusan tinta (bleed-through), pemudaran (fading), noda, dan efek buram (blur). Kerusakan fisik berat seperti sobekan besar atau bagian yang hilang tidak menjadi fokus utama.
3.  **Arsitektur Model:**
    *   **Generator:** Menggunakan arsitektur **U-Net**.
    *   **Diskriminator:** Menggunakan arsitektur **Dual-Modal** dengan jalur CNN untuk gambar dan jalur Embedding-LSTM untuk teks.
    *   **Recognizer (untuk Loss):** Menggunakan model HTR berbasis **Transformer** yang telah dilatih sebelumnya dan dibekukan.
4.  **Dataset:** Penelitian akan menggunakan dataset yang dibangun secara sintetis dari koleksi data bersih yang sudah ada. Validasi akhir akan dilakukan pada sampel dokumen riil dari koleksi Arsip Nasional.
5.  **Metrik Evaluasi:** Kinerja model akan dievaluasi menggunakan metrik kualitas gambar (PSNR, SSIM) dan, yang lebih penting, metrik akurasi HTR (CER, WER).
6.  **Teknologi:** Implementasi akan dilakukan menggunakan bahasa pemrograman Python dengan framework **TensorFlow/Keras**.

I.5 Hipotesis
Hipotesis utama dari penelitian ini adalah:
*   **H₁ (Hipotesis Alternatif):** Kerangka kerja GAN yang dilatih dengan fungsi loss gabungan yang mencakup komponen HTR-Oriented Loss (CTC Loss) dan menggunakan Diskriminator Dual-Modal akan menghasilkan restorasi dokumen yang menunjukkan **penurunan Character Error Rate (CER) secara signifikan** dibandingkan dengan metode GAN yang hanya menggunakan loss visual, sekaligus mempertahankan atau meningkatkan kualitas visual (PSNR/SSIM).
*   **H₀ (Hipotesis Nol):** Tidak ada perbedaan signifikan dalam penurunan CER antara kerangka kerja yang diusulkan dan metode GAN baseline.

Pengujian hipotesis akan dilakukan dengan membandingkan hasil dari model yang diusulkan terhadap model baseline (misalnya, GAN standar dengan loss L1 dan adversarial saja) pada dataset uji yang sama. Analisis statistik (seperti uji-T) akan digunakan untuk memvalidasi signifikansi dari perbedaan hasil metrik CER dan WER.

I.6 Kebaruan Penelitian
Kebaruan (novelty) dari penelitian ini terletak pada **pendekatan holistik untuk restorasi yang berorientasi pada keterbacaan**, yang dicapai melalui kombinasi spesifik dari tiga elemen yang belum pernah diintegrasikan secara bersamaan dalam penelitian sebelumnya:

1.  **Integrasi Langsung CTC Loss sebagai Sinyal Pelatihan Generator:** Ini adalah inovasi paling krusial. Sementara penelitian lain mungkin menggunakan HTR untuk evaluasi *setelah* training, penelitian ini menggunakan output dari model HTR (secara spesifik, CTC loss) sebagai bagian *langsung* dari fungsi loss Generator. Ini menciptakan loop umpan balik yang memaksa Generator untuk secara aktif menghasilkan gambar yang "disukai" oleh model pengenalan teks.
2.  **Arsitektur Diskriminator Dual-Modal:** Penggunaan diskriminator yang menilai koherensi gambar-teks memberikan tekanan selektif tambahan pada Generator. Generator tidak hanya harus membuat gambar yang terlihat nyata, tetapi juga harus memastikan bahwa teks yang dapat dikenali dari gambar tersebut sesuai. Ini membantu mengurangi artefak yang mungkin terlihat bagus tetapi mengganggu keterbacaan.
3.  **Pipeline Dataset Sintetis yang Terkontrol:** Berbeda dengan pendekatan unsupervised seperti CycleGAN yang bisa menghasilkan restorasi yang tidak terduga, penelitian ini menggunakan pipeline terawasi (supervised) dengan data sintetis. Hal ini memastikan bahwa model dilatih dengan ground truth yang jelas, memungkinkan optimasi yang lebih terarah menuju tujuan restorasi dan keterbacaan.

Kombinasi dari ketiga elemen ini—sinyal loss HTR langsung, pengawasan oleh diskriminator dual-modal, dan dataset sintetis yang terkontrol—merupakan sebuah pendekatan baru yang secara fundamental menggeser tujuan restorasi dari sekadar "memperbaiki gambar" menjadi "membuat teks dapat dibaca".

I.7 Sistematika Pembahasan
Tesis ini akan disusun dengan sistematika sebagai berikut:
*   **Bab I Pendahuluan:** Menguraikan latar belakang, masalah, tujuan, ruang lingkup, hipotesis, dan kebaruan penelitian.
*   **Bab II Tinjauan Pustaka:** Membahas teori dasar mengenai GAN, arsitektur U-Net, model HTR (Transformer dan CTC Loss), serta mengkaji penelitian terkait sebelumnya untuk memposisikan kontribusi penelitian ini.
*   **Bab III Metodologi Penelitian:** Merinci langkah-langkah penelitian, termasuk desain pipeline pembuatan dataset sintetis, arsitektur detail dari Generator, Diskriminator, dan Recognizer, formulasi fungsi loss, serta protokol eksperimen dan metrik evaluasi.
*   **Bab IV Analisis & Desain:** Menyajikan implementasi teknis dari arsitektur dan prosedur training. Bab ini akan menunjukkan bagaimana desain konseptual ditranslasikan ke dalam kode menggunakan TensorFlow, termasuk detail konfigurasi model dan loop training.
*   **Bab V Hasil dan Pembahasan:** Menyajikan hasil eksperimen secara kuantitatif dan kualitatif. Bab ini akan menganalisis perbandingan metrik (PSNR, SSIM, CER, WER) antara model yang diusulkan dan metode baseline, serta membahas implikasi dari temuan tersebut.
*   **Bab VI Kesimpulan & Saran:** Merangkum seluruh temuan penelitian, menjawab pertanyaan penelitian, dan memberikan saran untuk pengembangan di masa depan.
