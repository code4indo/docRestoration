# NARASI PRESENTASI: ARSITEKTUR RESTORASI DOKUMEN DUAL-MODAL GAN-HTR

**Untuk:** Seminar Hasil / Sidang Tesis  
**Durasi:** 3-5 menit  
**Speaker:** Belekok  
**Tanggal:** 2025-11-29

---

## 📊 NARASI VERSI SINGKAT (3 MENIT)

### **[Slide: Framework Overview]**

Baik, sekarang saya akan menjelaskan arsitektur kerangka kerja yang kami kembangkan untuk restorasi dokumen terdegradasi yang berorientasi pada *Handwritten Text Recognition* atau HTR.

**[Tunjuk bagian Input]**

Sistem kami menerima input berupa dokumen yang mengalami degradasi. Bisa berupa noda, blur, atau kerusakan lainnya yang umum dijumpai pada dokumen kuno. Dokumen degradasi inilah yang akan kita perbaiki.

**[Tunjuk Generator]**

Input degradasi ini kemudian diproses oleh komponen pertama, yaitu **Generator**. Kami menggunakan arsitektur U-Net Enhanced versi 2 dengan 21.8 juta parameter. Generator ini memiliki tiga bagian utama:

Pertama, **Encoder** dengan Residual Dense Block dan Channel-Spatial Attention untuk ekstraksi fitur multi-skala. Kedua, **Multi-Scale Fusion** di bottleneck untuk menggabungkan informasi dari berbagai resolusi. Ketiga, **Decoder** dengan Attention Gates yang merekonstruksi gambar hasil restorasi.

Output dari Generator adalah dokumen yang telah diperbaiki, yang kami sebut sebagai *enhanced image*.

**[Tunjuk Discriminator]**

Nah, untuk memastikan hasil restorasi ini realistis dan berkualitas tinggi, kami gunakan komponen kedua yaitu **Discriminator Dual-Modal**. Ini adalah kontribusi utama kami.

Discriminator ini tidak hanya melihat gambar secara visual, tapi juga mempertimbangkan aspek tekstual. Caranya, discriminator memiliki dua cabang:

Cabang pertama, **CNN Branch**, mengekstrak fitur visual dari gambar. Cabang kedua, **BiLSTM Branch**, memproses fitur teks berupa text indices (sequence of integers) yang bisa berasal dari ground truth atau hasil prediksi HTR recognizer.

Kedua cabang ini kemudian digabungkan menggunakan **Bilateral Cross-Modal Attention**. Ini adalah mekanisme yang kami kembangkan untuk memfusikan informasi visual dan tekstual secara efektif. Hasilnya adalah prediksi Real atau Fake dengan probabilitas antara 0 sampai 1.

**[Tunjuk HTR Recognizer]**

Komponen ketiga yang sangat penting adalah **HTR Recognizer** yang bersifat frozen atau tidak dilatih ulang. Recognizer ini punya tiga fungsi krusial:

Pertama, dia memberikan **sinyal CTC gradient** untuk memandu generator menghasilkan gambar yang readable atau bisa dibaca. Kedua, dia mengekstrak **fitur recognition** untuk loss function. Ketiga, dia menghasilkan **predicted text** yang digunakan oleh discriminator dalam mode predicted.

Yang unik, recognizer ini sudah pre-trained dengan CER 33.72% dan tetap frozen selama training. Jadi dia bertindak sebagai proxy untuk mengukur keterbacaan teks.

**[Tunjuk Loss Components]**

Sekarang yang membedakan sistem kami adalah penggunaan **Multi-Component Loss Function** dengan lima komponen yang dioptimasi secara bersamaan.

**Pertama**, L_pixel atau pixel loss menggunakan MAE untuk memastikan kemiripan pixel-level dengan ground truth.

**Kedua**, L_adversarial atau adversarial loss dari discriminator untuk visual realism.

**Ketiga**, L_perceptual menggunakan VGG features untuk semantic consistency di level yang lebih dalam.

**Keempat**, L_rec_feat atau recognition feature loss yang membandingkan fitur HTR dari clean dan enhanced image.

**Kelima**, L_CTC yang merupakan komponen HTR-oriented, memastikan enhanced image tetap readable oleh sistem HTR.

Penting dicatat: empat loss pertama menggunakan clean image sebagai referensi visual, sementara CTC loss menggunakan ground truth text transcription.

**[Tunjuk dashed arrows]**

Di diagram, **dashed arrows** dengan label ∇ adalah representasi backpropagation—gradient dari kelima loss components ini mengalir balik ke generator untuk update parameternya secara bertahap.

Yang penting dicatat, **CTC gradient hanya mengalir ke generator**, tidak ke discriminator. Ini memisahkan optimasi keterbacaan teks dengan optimasi visual realism.

**[Tunjuk Dual Mode Legend]**

Arsitektur kami mendukung dua mode operasi untuk input teks discriminator:

Mode pertama adalah **Ground Truth mode**, di mana discriminator menerima label transkripsi asli.

Mode kedua adalah **Predicted mode**, yang kami gunakan dalam eksperimen final, di mana discriminator menerima hasil argmax dari HTR logits. Mode ini lebih challenging karena discriminator harus membedakan berdasarkan pola teks yang mengandung error recognition.

**[Tunjuk Output]**

Hasil akhir dari sistem ini adalah dokumen yang telah direstorasi dengan kualitas visual tinggi DAN tetap mempertahankan keterbacaan teks untuk sistem HTR.

**[Tunjuk Parameter Count]**

Secara keseluruhan, arsitektur kami memiliki 39.2 juta parameter trainable, dengan Generator 21.8 juta, Discriminator 17.4 juta yang merupakan reduksi 52% dari baseline, dan HTR yang frozen.

**[Kesimpulan]**

Jadi, tiga keunggulan utama arsitektur kami adalah: **Pertama**, dual-modal discriminator dengan bilateral attention yang mempertimbangkan visual dan textual features. **Kedua**, frozen HTR sebagai multi-purpose guide untuk CTC loss, recognition features, dan predicted text. **Ketiga**, multi-component loss dengan adaptive balancing untuk optimasi seimbang.

Terima kasih.

---

## 📊 NARASI VERSI LENGKAP (5-7 MENIT)

### **[Opening - Context]**

Selamat pagi/siang Bapak Ibu penguji dan hadirin sekalian.

Pada bagian ini, saya akan menjelaskan secara detail arsitektur kerangka kerja yang kami kembangkan untuk restorasi dokumen terdegradasi yang berorientasi pada *Handwritten Text Recognition*.

**[Motivasi Problem]**

Sebelumnya, perlu saya sampaikan bahwa dokumen kuno yang kami tangani—khususnya naskah paleografi abad 16 hingga 18—mengalami degradasi kompleks seperti noda, blur, fading, dan kerusakan fisik. Degradasi ini tidak hanya mengganggu secara visual, tapi yang lebih kritis, menurunkan akurasi sistem HTR secara signifikan.

Tantangannya adalah: bagaimana kita merestorasi dokumen ini agar **tidak hanya terlihat bagus secara visual**, tetapi juga **tetap readable atau terbaca dengan baik** oleh sistem HTR? Inilah gap yang kami address.

### **[Slide: Framework Overview - Komponen Utama]**

Baik, mari kita lihat arsitektur kerangka kerja yang kami usulkan.

**[Tunjuk Ground Truth Section]**

Pertama-tama, dalam proses training, kami memiliki dua jenis data ground truth:

Satu, **clean image** yang merupakan target visual yang ingin kita capai. Dua, **ground truth text** yang merupakan transkripsi asli dari dokumen. Kedua informasi ini sangat penting untuk membimbing proses pembelajaran.

**[Tunjuk Input]**

Sistem kami dimulai dengan menerima **dokumen terdegradasi** sebagai input. Ini bisa berupa scan dokumen kuno dengan berbagai jenis degradasi yang saya sebutkan tadi.

### **[Generator - Komponen Inti Pertama]**

**[Tunjuk Generator Box]**

Input degradasi ini masuk ke komponen pertama, yaitu **Generator**. Kami menggunakan arsitektur U-Net Enhanced versi 2 dengan total 21.8 juta parameter.

**[Tunjuk Encoder]**

Bagian **Encoder** menggunakan kombinasi Residual Dense Block dan Channel-Spatial Attention Mechanism atau CBAM. Residual Dense Block memungkinkan reuse fitur multi-level, sehingga informasi detail tidak hilang selama downsampling. Sementara CBAM memberikan attention baik di channel maupun spatial dimension untuk fokus pada area yang penting.

**[Tunjuk Bottleneck]**

Di **bottleneck**, kami implementasikan Multi-Scale Fusion yang menggabungkan informasi dari berbagai skala resolusi. Ini penting untuk menangani degradasi pada berbagai skala—dari noise halus hingga noda besar.

**[Tunjuk Decoder]**

Pada **Decoder**, kami gunakan Attention Gates yang secara selektif meneruskan informasi dari skip connections. Ini memastikan hanya fitur yang relevan yang dikombinasikan untuk rekonstruksi.

**[Tunjuk Generator Output]**

Output dari generator adalah **enhanced image**—dokumen yang telah diperbaiki dan siap dievaluasi.

### **[Discriminator - Komponen Inti Kedua]**

**[Tunjuk Discriminator Box]**

Nah, komponen kedua yang merupakan **kontribusi utama** kami adalah **Discriminator Dual-Modal**.

Berbeda dengan discriminator konvensional yang hanya melihat aspek visual, discriminator kami mempertimbangkan **dua modalitas sekaligus**: visual dan tekstual.

**[Tunjuk CNN Branch]**

**CNN Branch** mengekstrak fitur visual dari gambar menggunakan convolutional layers. Branch ini menangkap informasi spasial seperti struktur, tekstur, dan distribusi pixel.

**[Tunjuk BiLSTM Branch]**

**BiLSTM Branch** memproses informasi tekstual secara sequential. Input text bisa berupa ground truth transcription dalam mode GT, atau predicted text indices hasil argmax dari HTR logits dalam mode predicted.

Yang penting dicatat: HTR mengirimkan **text indices**—yaitu sequence of integers yang merepresentasikan predicted characters, bukan text string atau probabilitas. Discriminator kemudian memproses sequence indices ini melalui embedding layer dan BiLSTM untuk mendeteksi pola sequential text yang natural versus artificial.

BiLSTM dipilih karena kemampuannya menangkap dependensi temporal bidirectional dalam sequence text, yang penting untuk mendeteksi pola keterbacaan.

**[Tunjuk Bilateral Attention]**

Kedua branch ini tidak bekerja independent. Kami menggunakan **Bilateral Cross-Modal Attention** yang kami kembangkan untuk memfusikan informasi visual dan tekstual secara efektif.

Mekanisme ini memungkinkan:
- Visual features di-attend berdasarkan text features
- Text features di-enhance berdasarkan visual context
- Interaksi dua arah ini menghasilkan representasi yang lebih kaya

**[Tunjuk Discriminator Output]**

Output final discriminator adalah probabilitas Real atau Fake dengan nilai antara 0 sampai 1. Nilai mendekati 1 berarti gambar diprediksi sebagai real atau clean, sementara mendekati 0 berarti fake atau generated.

**[Tunjuk Parameter Reduction]**

Yang menarik, meskipun dual-modal, arsitektur discriminator kami hanya 17.4 juta parameter—ini adalah **reduksi 52%** dari discriminator baseline yang 36 juta parameter. Ini menunjukkan efisiensi arsitektur yang kami rancang.

### **[HTR Recognizer - Komponen Ketiga]**

**[Tunjuk HTR Box]**

Komponen ketiga yang sangat krusial adalah **HTR Recognizer**.

**[Tunjuk Frozen Badge]**

Yang unik, recognizer ini bersifat **frozen** atau trainable equals false. Artinya, parameter recognizer tidak diupdate selama training. Mengapa frozen?

Pertama, recognizer ini sudah pre-trained pada domain yang sama dengan CER 33.72%. Kedua, kita ingin recognizer bertindak sebagai **fixed proxy** untuk mengukur keterbacaan, bukan sebagai model yang berubah selama training. Ketiga, ini menghemat computational cost dan mencegah catastrophic forgetting.

**[Tunjuk HTR Components: CNN → Transformer → CTC]**

Recognizer memiliki tiga stage:
1. **CNN Backbone** untuk feature extraction dari image
2. **Transformer** dengan 6 layers dan 8 attention heads untuk sequence modeling
3. **CTC Decoder** untuk menghasilkan sequence prediksi tanpa alignment

**[Tunjuk Tiga Arrow dari HTR]**

Recognizer frozen ini punya **tiga fungsi penting**:

**Fungsi pertama** [tunjuk arrow ke CTC Loss]: Memberikan **CTC gradient** untuk membimbing generator. Enhanced image diproses oleh HTR, hasilnya dibandingkan dengan ground truth, dan gradient mengalir kembali ke generator. Ini memastikan generator belajar menghasilkan gambar yang readable.

**Fungsi kedua** [tunjuk arrow ke Rec Feat Loss]: Mengekstrak **intermediate features** dari layer transformer. Features dari clean dan enhanced image dibandingkan untuk memastikan semantic consistency.

**Fungsi ketiga** [tunjuk arrow ke Discriminator]: Menghasilkan **predicted text** melalui argmax dari CTC logits. Secara spesifik, HTR menghasilkan CTC logits berupa probability distribution per timestep, kemudian kita lakukan argmax untuk mendapatkan **text indices**—sequence of integers [batch, 128] yang represent predicted characters. Text indices inilah yang dikirim ke BiLSTM branch discriminator dalam predicted mode.

### **[Loss Function - Multi-Component]**

**[Tunjuk Loss Container]**

Sekarang, yang membedakan pendekatan kami secara fundamental adalah **Multi-Component Loss Function** dengan lima komponen yang dioptimasi secara bersamaan.

**[Tunjuk setiap loss component satu per satu]**

**Komponen pertama: L_pixel**
Pixel loss menggunakan Mean Absolute Error atau MAE. Ini memastikan kemiripan pixel-by-pixel antara enhanced image dan clean ground truth. Loss ini memberikan baseline reconstruction quality.

**Komponen kedua: L_adversarial**
Adversarial loss dari discriminator. Ini mendorong generator menghasilkan gambar yang tidak bisa dibedakan dari real images oleh discriminator. Loss ini memberikan visual realism.

**Komponen ketiga: L_perceptual**
Perceptual loss menggunakan pre-trained VGG16. Kita ekstrak deep features dari layer block3_conv3 untuk clean dan enhanced images, lalu bandingkan. Loss ini bekerja di semantic level, bukan pixel level, sehingga lebih toleran terhadap small spatial shifts dan mencegah over-smoothing.

**Komponen keempat: L_rec_feat**
Recognition feature loss membandingkan intermediate features dari HTR recognizer. Ini memastikan bahwa tidak hanya output text yang sama, tapi juga internal representation-nya konsisten. Loss ini memberikan semantic text consistency.

**Komponen kelima: L_CTC**
CTC loss adalah komponen HTR-oriented yang membandingkan predicted text dari enhanced image dengan ground truth transcription. **Ini adalah komponen kunci** yang memastikan enhanced image tetap readable oleh sistem HTR.

**[Tunjuk arrow perceptual GT di diagram]**

Penting untuk dicatat bahwa **clean image sebagai ground truth** berperan sebagai referensi untuk empat komponen loss: Pixel loss menggunakan clean image untuk perbandingan langsung. Perceptual loss—yang kita tunjukkan arrow-nya di diagram sebagai contoh—membandingkan VGG features dari clean dan enhanced image. Recognition feature loss membandingkan HTR features. Dan discriminator mengevaluasi clean image sebagai 'real' samples.

Sementara keempat loss tersebut menggunakan ground truth **image**, CTC loss unik karena menggunakan ground truth **text transcription**. Kombinasi inilah yang memastikan hasil restorasi optimal baik secara visual maupun keterbacaan teks.

**[Tunjuk Formula Total Loss]**

Kelima komponen ini dikombinasikan dengan weighted sum:

L_total = λ_pixel × L_pixel + λ_adv × L_adv + λ_perc × L_perc + λ_rec × L_rec + λ_CTC × L_CTC

### **[Backpropagation - Gradient Flow]**

**[Tunjuk dashed arrows di diagram]**

Sekarang, bagaimana total loss ini mengoptimasi generator? Melalui **backpropagation**.

Di diagram, kita bisa lihat **dashed arrows** dengan label nabla atau ∇—ini adalah representasi visual dari **gradient flow** atau backpropagation.

**[Trace arrows satu per satu]**

Ada lima gradient yang mengalir balik ke generator:
- ∇_pixel: gradient untuk pixel-level accuracy
- ∇_adv: gradient dari discriminator untuk visual realism
- ∇_perc: gradient VGG untuk semantic consistency
- ∇_rec: gradient HTR features untuk text consistency
- ∇_CTC: gradient dari CTC untuk text readability

**[Gesture: dari loss area ke generator]**

Kelima gradient ini **dijumlahkan secara weighted** sesuai lambda masing-masing, menghasilkan total gradient yang memberitahu generator: parameter mana yang perlu diubah dan seberapa besar perubahannya.

Melalui Adam optimizer, gradient ini digunakan untuk **update 21.8 juta parameter generator**. Proses ini berulang setiap batch, memandu generator menghasilkan output yang lebih baik secara bertahap.

**[Point to legend]**

Legend di diagram kami explicitly menyatakan: solid arrows represent data flow forward, dashed arrows represent gradient flow backward—ini adalah **visualisasi backpropagation algorithm**.

**[CRITICAL - Tunjuk CTC Gradient Arrow]**

Yang sangat penting untuk dipahami: **CTC gradient hanya mengalir ke generator**, TIDAK ke discriminator.

Mengapa? Karena kita ingin:
- Discriminator fokus pada visual-textual realism
- Generator fokus pada visual quality DAN text readability
- Tidak ada gradient conflict antara dua objective berbeda

Ini adalah **design choice yang deliberate** untuk mencegah discriminator "belajar dari CTC error" yang bisa mengacaukan adversarial training dynamics.

### **[Dual Mode - Text Input]**

**[Tunjuk Legend Text Mode]**

Arsitektur kami mendukung **dua mode operasi** untuk input teks discriminator:

**Mode Ground Truth**: Discriminator menerima label transkripsi asli dari dataset. Ini lebih mudah karena text sempurna tanpa error.

**Mode Predicted**: Discriminator menerima hasil argmax dari HTR logits—yaitu predicted text indices. Mode ini lebih **challenging** karena text mengandung recognition errors (baseline CER 33.72%).

**[Tunjuk Config Verification]**

Dalam eksperimen final kami, kami menggunakan **Predicted Mode**. Kenapa?

Pertama, lebih realistic—discriminator mengevaluasi berdasarkan predicted text pattern yang muncul di real-world scenario.

Kedua, harder task—memaksa generator menghasilkan enhancement yang text-consistent meskipun HTR tidak sempurna.

Ketiga, better generalization—model tidak over-rely pada perfect ground truth yang tidak tersedia saat inference.

### **[Curriculum Learning & Adaptive Balancing]**

**[Tunjuk Curriculum Learning Box]**

Untuk training strategy, kami implementasikan **Curriculum Learning** dengan tiga fase:

**Fase Warmup** (epoch 1-10): Fokus pada visual quality tanpa CTC loss. Generator belajar basic reconstruction dulu.

**Fase Refinement** (epoch 11-40): Gradual CTC ramp-up. CTC weight dinaikkan secara linear dari 0 ke full weight. Generator mulai belajar balance antara visual dan text readability.

**Fase Stabilization** (epoch 41-50): Full CTC weight. Semua loss components bekerja penuh untuk fine-tuning.

**[Tunjuk Adaptive Balancing Box]**

Kami juga gunakan **Adaptive Loss Balancing** untuk mencegah dominasi satu loss component. Sistem secara otomatis adjust weight berdasarkan magnitude loss di setiap batch, memastikan semua components berkontribusi secara seimbang.

### **[Output & Results Preview]**

**[Tunjuk Enhanced Document Output]**

Hasil akhir dari sistem ini adalah dokumen yang:
1. Secara visual **terlihat bersih dan jelas**
2. Secara tekstual **tetap readable oleh HTR**
3. Mempertahankan **semantic content** dari dokumen asli

### **[Novelty & Contributions - Closing]**

**[Tunjuk Novelty Legend]**

Untuk merangkum, **tiga kontribusi utama** arsitektur kami:

**Pertama**: **Bilateral Cross-Modal Attention** untuk dual-modal discriminator yang efektif memfusikan visual dan textual features. Ini adalah novelty di domain document restoration.

**Kedua**: **Frozen HTR sebagai Multi-Purpose Guide** dengan tiga fungsi berbeda—CTC gradient, recognition features, dan predicted text generation. Ini adalah pendekatan baru untuk incorporate HTR knowledge tanpa retraining.

**Ketiga**: **Adaptive Multi-Component Loss** dengan curriculum learning dan dynamic balancing. Ini memastikan optimasi seimbang untuk visual quality DAN text readability.

**[Parameter Efficiency]**

Dan semua ini dicapai dengan **parameter efficiency**—total 39.2M trainable parameters dengan discriminator yang 52% lebih kecil dari baseline.

### **[Closing Statement]**

Jadi, arsitektur yang kami usulkan tidak hanya merestorasi dokumen secara visual, tetapi juga **memastikan hasil restorasi optimal untuk aplikasi downstream HTR**. Ini adalah pendekatan **task-oriented document restoration** yang pertama kali mengintegrasikan dual-modal discrimination dengan frozen HTR guidance.

Terima kasih. Saya siap menjawab pertanyaan Bapak Ibu.

---

## 📋 TALKING POINTS (BULLET FORMAT)

### **Untuk Slide Framework Overview:**

**Input & Ground Truth:**
- Input: Dokumen terdegradasi (noda, blur, fading)
- GT: Clean image + Text transcription
- Target: Restorasi visual + Readability HTR

**Generator (21.8M params):**
- Encoder: RDB + CBAM → Multi-scale features
- Bottleneck: Multi-Scale Fusion
- Decoder: Attention Gates → Selective reconstruction
- Output: Enhanced image [-1, 1]

**Discriminator Dual-Modal (17.4M params, 52% reduction):**
- CNN Branch: Visual features (spatial+texture)
- BiLSTM Branch: Text features (sequential patterns)
- Bilateral Attention: Cross-modal fusion
- Output: Real/Fake probability
- **Input modes**: GT text OR Predicted text (argmax)

**HTR Recognizer (Frozen, CER 33.72%):**
- Tiga fungsi:
  1. CTC gradient → Generator (readability guidance)
  2. Rec features → L_rec (semantic consistency)
  3. Predicted text → Discriminator (mode Pred)
- Frozen = Fixed proxy, no retraining

**Multi-Component Loss (5 components):**
1. L_pixel (MAE): Pixel-level similarity
2. L_adversarial (GAN): Visual realism
3. L_perceptual (VGG): Semantic features
4. L_rec_feat (HTR): Recognition consistency
5. L_CTC (HTR): **Text readability** ← Key contribution
- **Backpropagation**: Dashed arrows (∇) show gradient flow dari loss → generator
- **CRITICAL**: CTC gradient → Generator ONLY (not Disc)

**Training Strategy:**
- Curriculum Learning: Warmup → Refinement → Stabilization
- Adaptive Balancing: Dynamic weight adjustment
- Mode: Predicted (harder, more realistic)

**Novelty:**
1. Bilateral cross-modal attention (first in doc restoration)
2. Frozen HTR multi-purpose guide (novel approach)
3. HTR-oriented multi-component loss (task-specific)
4. Parameter efficiency (52% smaller discriminator)

---

## 💡 TIPS PRESENTASI

### **Untuk Delivery:**

1. **Pace**: 120-150 kata per menit (moderate speed)
2. **Pause**: Setelah setiap komponen utama (2-3 detik)
3. **Emphasis**: Slow down pada novelty points
4. **Pointer**: Gunakan laser pointer untuk trace arrows

### **Antisipasi Pertanyaan:**

**Q: Mengapa discriminator perlu dual-modal?**
A: "Karena document quality tidak hanya visual, tapi juga readability. Dual-modal memastikan enhanced image realistic DAN text-consistent."

**Q: Mengapa HTR frozen, tidak ditraining?**
A: "Tiga alasan: (1) Already pre-trained CER 33.72%, (2) Sebagai fixed proxy untuk readability, (3) Menghemat computational cost dan prevent catastrophic forgetting."

**Q: Predicted mode lebih baik dari GT mode?**
A: "Lebih realistic dan challenging. Discriminator belajar dari predicted text pattern yang mengandung errors, jadi lebih robust untuk real-world scenarios."

**Q: Kontribusi paling signifikan?**
A: "Bilateral cross-modal attention yang memungkinkan discriminator consider both visual and textual aspects simultaneously, supported by frozen HTR multi-purpose guidance."

**Q: Di mana backpropagation-nya di diagram?**
A: "Direpresentasikan oleh **dashed arrows dengan label ∇** (nabla). Ada enam gradient arrows untuk lima loss components. Legend kami explicitly state bahwa dashed=gradient flow backward. Semua gradient ini ultimately flows ke generator untuk update 21.8M parameternya melalui Adam optimizer."

### **Body Language:**

- **Hand gestures**: Trace alur data flow dengan tangan
- **Eye contact**: Scan audience setiap 3-5 detik
- **Posture**: Berdiri tegak, tidak blocking slide
- **Movement**: Minimal, fokus pada pointer

### **Time Management:**

- **Intro (30s)**: Context + Problem
- **Generator (1min)**: Architecture overview
- **Discriminator (1.5min)**: Dual-modal + Bilateral attention
- **HTR (1min)**: Frozen + Three functions
- **Loss (1min)**: 5 components + CTC critique
- **Closing (30s)**: Novelty summary

**Total: 5.5 minutes** (ideal for seminar hasil)

---

**File created:** `PRESENTATION_NARRATION.md`  
**Status:** ✅ Ready for practice & delivery  
**Recommendation:** Practice 3-5x untuk natural flow
