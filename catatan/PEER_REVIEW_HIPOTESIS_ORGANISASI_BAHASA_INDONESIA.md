# PEER REVIEW: Revisi Bahasa Indonesia - Hipotesis & Organisasi Makalah

**Tanggal**: 3 November 2025 08:58 WIB  
**Sections**: 
- I-C - Hipotesis dan Validasi Penelitian
- I-D - Organisasi Makalah  
**Tujuan**: Compliance kaidah ilmiah bahasa Indonesia dan standar IEEE Journal

---

## 📋 RINGKASAN PEER REVIEW

### Kriteria Review
1. ✅ Bahasa Indonesia yang baik dan benar (KBBI)
2. ✅ Istilah asing dicetak miring
3. ✅ Perspektif netral (bukan "kami")
4. ✅ Tanpa detail variabel kode
5. ✅ Bold/italic sesuai standar IEEE
6. ✅ Konsistensi terminologi

---

## ✅ SECTION I-C: HIPOTESIS DAN VALIDASI PENELITIAN

### Revisi yang Dilakukan (18 istilah)

| **Kategori** | **Sebelum (❌)** | **Sesudah (✅)** |
|-------------|----------------|----------------|
| **Istilah Umum** | framework restorasi | kerangka kerja restorasi |
| | loss function | fungsi kehilangan |
| | Character Error Rate (CER) | *Character Error Rate* (CER) |
| | degraded input baseline | masukan terdegradasi *baseline* |
| | effect size | ukuran efek |
| | medium (Cohen's d) | sedang (Cohen's d) |
| | systematic ablation studies | studi ablasi sistematis |
| | test images | citra uji |
| | CER reduction | penurunan CER |
| | degraded | terdegradasi |
| | restored | terestorasi |
| | large effect | efek besar |
| | multi-component loss optimization | optimasi fungsi kehilangan multikomponen |
| **Italic Added** | dual-modal (plain) | *dual-modal* |
| | baseline (plain) | *baseline* |

### Before:
```latex
Penelitian ini memvalidasi hipotesis bahwa framework restorasi dokumen 
dengan diskriminator dual-modal dan optimasi loss function berorientasi 
HTR menghasilkan penurunan Character Error Rate (CER) yang signifikan 
secara statistik dibandingkan dokumen terdegradasi (degraded input 
baseline) dengan effect size minimal medium (Cohen's $d > 0.5$, 
$p < 0.05$). Validasi dilakukan melalui systematic ablation studies 
pada 712 test images untuk membuktikan kontribusi signifikan setiap 
komponen arsitektural. Hasil menunjukkan CER reduction dari 83.4\% 
(degraded) ke 34.9\% (restored) dengan $p<0.001$ dan Cohen's $d=0.85$ 
(large effect), mengkonfirmasi efektivitas kombinasi diskriminator 
dual-modal (CNN+LSTM) dengan HTR-oriented multi-component loss 
optimization.
```

### After:
```latex
Penelitian ini memvalidasi hipotesis bahwa kerangka kerja restorasi 
dokumen dengan diskriminator \textit{dual-modal} dan optimasi fungsi 
kehilangan berorientasi HTR menghasilkan penurunan \textit{Character 
Error Rate} (CER) yang signifikan secara statistik dibandingkan dokumen 
terdegradasi (masukan terdegradasi \textit{baseline}) dengan ukuran 
efek minimal sedang (Cohen's $d > 0.5$, $p < 0.05$). Validasi dilakukan 
melalui studi ablasi sistematis pada 712 citra uji untuk membuktikan 
kontribusi signifikan setiap komponen arsitektural. Hasil menunjukkan 
penurunan CER dari 83.4\% (terdegradasi) ke 34.9\% (terestorasi) dengan 
$p<0.001$ dan Cohen's $d=0.85$ (efek besar), mengkonfirmasi efektivitas 
kombinasi diskriminator \textit{dual-modal} (CNN+LSTM) dengan optimasi 
fungsi kehilangan multikomponen berorientasi HTR.
```

### Key Improvements:
1. ✅ **"framework" → "kerangka kerja"** (padanan Indonesia)
2. ✅ **"loss function" → "fungsi kehilangan"** (konsistensi terminologi)
3. ✅ **"Character Error Rate" → "\textit{Character Error Rate}"** (istilah asing cetak miring)
4. ✅ **"degraded input baseline" → "masukan terdegradasi \textit{baseline}"** (translate + italic)
5. ✅ **"effect size" → "ukuran efek"** (padanan Indonesia)
6. ✅ **"medium" → "sedang"** (dalam konteks statistik)
7. ✅ **"systematic ablation studies" → "studi ablasi sistematis"** (Indonesia)
8. ✅ **"test images" → "citra uji"** (Indonesia)
9. ✅ **"CER reduction" → "penurunan CER"** (Indonesia)
10. ✅ **"degraded/restored" → "terdegradasi/terestorasi"** (konsistensi)
11. ✅ **"large effect" → "efek besar"** (padanan Indonesia)
12. ✅ **"HTR-oriented multi-component loss optimization" → "optimasi fungsi kehilangan multikomponen berorientasi HTR"** (Indonesia)

---

## ✅ SECTION I-D: ORGANISASI MAKALAH

### Revisi yang Dilakukan (24 istilah)

| **Kategori** | **Sebelum (❌)** | **Sesudah (✅)** |
|-------------|----------------|----------------|
| **Perspektif** | yang kami usulkan | yang diusulkan |
| **Istilah Umum** | restorasi gambar | restorasi citra |
| | generator U-Net Enhanced | *generator* U-Net *Enhanced* |
| | diskriminator dual-modal | diskriminator *dual-modal* |
| | frozen recognizer | pengenal beku |
| | fungsi loss | fungsi kehilangan |
| | adaptive balancing | penyeimbangan adaptif |
| | degraded input baseline | masukan terdegradasi *baseline* |
| | clean ground truth reference | referensi *ground truth* bersih |
| | systematic ablation studies | studi ablasi sistematis |
| **Italic Removed** | \textit{GAN} | GAN |
| | \textit{HTR} | HTR |
| | \textit{hyperparameter} | hiperparameter |
| **Italic Added** | generator (plain) | *generator* |
| | Enhanced (plain) | *Enhanced* |
| | dual-modal (plain) | *dual-modal* |
| | dataset (italicized already) | *dataset* ✓ |
| | baseline (plain) | *baseline* |
| | ground truth (plain) | *ground truth* |

### Before:
```latex
Sisa dari makalah ini diatur sebagai berikut: Bagian~II meninjau 
pekerjaan terkait dalam perbaikan dokumen, \textit{GAN} untuk 
restorasi gambar, dan sistem \textit{HTR}. Bagian~III menyajikan 
arsitektur yang kami usulkan termasuk generator U-Net Enhanced, 
diskriminator dual-modal (CNN+LSTM), integrasi frozen recognizer, 
dan fungsi loss yang dioptimalkan dengan adaptive balancing. 
Bagian~IV menjelaskan pengaturan eksperimental, \textit{dataset}, 
metodologi pelatihan, dan strategi optimisasi \textit{hyperparameter}. 
Bagian~V menyajikan hasil komprehensif termasuk perbandingan 
kuantitatif dengan degraded input baseline dan clean ground truth 
reference, systematic ablation studies untuk memvalidasi kontribusi 
setiap komponen, dan analisis kualitatif pada dokumen historis nyata. 
Bagian~VI membahas temuan, implikasi teoretis dan praktis, 
keterbatasan, dan arah masa depan. Akhirnya, Bagian~VII menyimpulkan 
makalah ini dengan rangkuman kontribusi dan dampak penelitian.
```

### After:
```latex
Sisa dari makalah ini diatur sebagai berikut: Bagian~II meninjau 
pekerjaan terkait dalam perbaikan dokumen, GAN untuk restorasi citra, 
dan sistem HTR. Bagian~III menyajikan arsitektur yang diusulkan 
termasuk \textit{generator} U-Net \textit{Enhanced}, diskriminator 
\textit{dual-modal} (CNN+LSTM), integrasi pengenal beku, dan fungsi 
kehilangan yang dioptimalkan dengan penyeimbangan adaptif. Bagian~IV 
menjelaskan pengaturan eksperimental, \textit{dataset}, metodologi 
pelatihan, dan strategi optimisasi hiperparameter. Bagian~V menyajikan 
hasil komprehensif termasuk perbandingan kuantitatif dengan masukan 
terdegradasi \textit{baseline} dan referensi \textit{ground truth} 
bersih, studi ablasi sistematis untuk memvalidasi kontribusi setiap 
komponen, dan analisis kualitatif pada dokumen historis nyata. 
Bagian~VI membahas temuan, implikasi teoretis dan praktis, 
keterbatasan, dan arah masa depan. Akhirnya, Bagian~VII menyimpulkan 
makalah ini dengan rangkuman kontribusi dan dampak penelitian.
```

### Key Improvements:

#### **1. Perspektif Netral**
- ❌ "yang kami usulkan"
- ✅ "yang diusulkan"
- **Alasan**: Hindari perspektif "kami", gunakan netral objektif

#### **2. Italic untuk Istilah Asing Teknis**
- ✅ *generator*, *Enhanced*, *dual-modal*, *baseline*, *ground truth*
- **Alasan**: Istilah asing tanpa padanan Indonesia harus cetak miring

#### **3. Italic Dihapus dari Akronim Umum**
- ❌ \textit{GAN}, \textit{HTR}, \textit{hyperparameter}
- ✅ GAN, HTR, hiperparameter
- **Alasan**: 
  * **GAN** dan **HTR** adalah akronim umum, tidak perlu italic
  * **"hyperparameter" → "hiperparameter"** (sudah diserap KBBI, tidak perlu italic)

#### **4. Translate English → Indonesian**
| Inggris | Indonesia |
|---------|-----------|
| restorasi gambar | restorasi citra |
| frozen recognizer | pengenal beku |
| fungsi loss | fungsi kehilangan |
| adaptive balancing | penyeimbangan adaptif |
| degraded input baseline | masukan terdegradasi *baseline* |
| clean ground truth reference | referensi *ground truth* bersih |
| systematic ablation studies | studi ablasi sistematis |

---

## 📊 STATISTIK REVISI

### **Section I-C (Hipotesis dan Validasi)**:
| Kategori | Jumlah |
|----------|--------|
| Inggris → Indonesia | 13 istilah |
| Italic ditambahkan | 3 (*dual-modal*, *Character Error Rate*, *baseline*) |
| Konsistensi terminologi | 2 (fungsi kehilangan, citra uji) |
| **Total Revisi** | **18 perubahan** |

### **Section I-D (Organisasi Makalah)**:
| Kategori | Jumlah |
|----------|--------|
| Perspektif netral | 1 ("kami usulkan" → "yang diusulkan") |
| Inggris → Indonesia | 8 istilah |
| Italic ditambahkan | 5 (*generator*, *Enhanced*, *dual-modal*, *baseline*, *ground truth*) |
| Italic dihapus | 3 (GAN, HTR, hiperparameter) |
| Konsistensi | 7 (citra, kehilangan, pengenal beku, dll.) |
| **Total Revisi** | **24 perubahan** |

### **GRAND TOTAL: 42 perubahan** (2 sections)

---

## ✅ HASIL KOMPILASI

- **File PDF**: `Paper/main/jatniko_id.pdf`
- **Ukuran**: 10 MB (10,474,722 bytes)
- **Halaman**: 31
- **Timestamp**: 3 November 2025 08:58 WIB
- **Status**: ✅ Berhasil dikompilasi tanpa error

---

## 🎯 COMPLIANCE CHECKLIST

### Kaidah Ilmiah Bahasa Indonesia (KBBI)
- ✅ Semua istilah umum menggunakan bahasa Indonesia baku
- ✅ Istilah asing teknis dicetak miring (*dual-modal*, *generator*, *Enhanced*, *baseline*, *ground truth*, *Character Error Rate*)
- ✅ Akronim **tidak** dicetak miring (GAN, HTR, CER, CNN, LSTM)
- ✅ Kata serapan **tidak** dicetak miring (hiperparameter)
- ✅ Konsistensi "kehilangan" untuk "loss"
- ✅ Konsistensi "citra" untuk "gambar/image"
- ✅ Perspektif netral objektif (bukan "kami")

### Standar IEEE Journal
- ✅ **Perspektif netral** ("yang diusulkan" vs "yang kami usulkan")
- ✅ **Italic hanya untuk istilah asing teknis**, bukan akronim
- ✅ **Konsistensi terminologi** di seluruh section
- ✅ **Tidak ada detail implementasi berlebihan**
- ✅ **Format statistik benar** (Cohen's $d$, $p$-value)

---

## 📝 PANDUAN ITALIC vs NON-ITALIC

### **✅ ITALIC (Istilah Asing Teknis)**:
- *dual-modal* (konsep arsitektur asing)
- *generator* (istilah teknis ML)
- *Enhanced* (nama arsitektur spesifik)
- *baseline* (istilah teknis evaluasi)
- *ground truth* (istilah teknis ML)
- *Character Error Rate* (metrik spesifik)
- *dataset* (istilah teknis ML)

### **❌ TIDAK ITALIC (Akronim Umum)**:
- GAN (Generative Adversarial Network)
- HTR (Handwritten Text Recognition)
- CER (Character Error Rate - hanya kepanjangan yang italic)
- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- PSNR, SSIM, WER (akronim standar)

### **❌ TIDAK ITALIC (Kata Serapan KBBI)**:
- hiperparameter (sudah diserap)
- algoritma, data, parameter, sistem, optimal

---

## 🔍 RASIONALISASI ITALIC

### **Mengapa *generator* italic tapi GAN tidak?**
1. **GAN**: Akronim umum yang sudah dikenal luas di ML community
2. **generator**: Istilah teknis spesifik yang merujuk komponen arsitektur (belum ada padanan Indonesia baku)

### **Mengapa *baseline* italic?**
- Tidak ada padanan Indonesia yang presisi dalam konteks evaluasi ML
- KBBI tidak memiliki definisi untuk konteks "baseline experiment"
- Istilah teknis yang perlu dipertahankan keasliannya

### **Mengapa hiperparameter tidak italic?**
- Kata "hiper-" sudah diserap dalam KBBI (hipertensi, hiperbola)
- "Parameter" sudah diserap dalam KBBI
- Gabungan "hiperparameter" sudah digunakan dalam literatur Indonesia

---

## 📋 PERBANDINGAN BEFORE/AFTER LENGKAP

### **I-C: Hipotesis dan Validasi Penelitian**

#### Before (Banyak Inggris + Tidak Konsisten):
> Penelitian ini memvalidasi hipotesis bahwa **framework** restorasi dokumen dengan diskriminator **dual-modal** (tidak italic) dan optimasi **loss function** berorientasi HTR menghasilkan penurunan **Character Error Rate** (tidak italic) yang signifikan secara statistik dibandingkan dokumen terdegradasi (**degraded input baseline**) dengan **effect size** minimal **medium**. Validasi dilakukan melalui **systematic ablation studies** pada 712 **test images**. Hasil menunjukkan **CER reduction** dari 83.4% (**degraded**) ke 34.9% (**restored**) dengan $p<0.001$ dan Cohen's $d=0.85$ (**large effect**), mengkonfirmasi efektivitas kombinasi diskriminator **dual-modal** dengan **HTR-oriented multi-component loss optimization**.

#### After (Indonesia Baku + Konsisten):
> Penelitian ini memvalidasi hipotesis bahwa **kerangka kerja** restorasi dokumen dengan diskriminator ***dual-modal*** (cetak miring) dan optimasi **fungsi kehilangan** berorientasi HTR menghasilkan penurunan ***Character Error Rate*** (cetak miring) yang signifikan secara statistik dibandingkan dokumen terdegradasi (**masukan terdegradasi *baseline***) dengan **ukuran efek** minimal **sedang**. Validasi dilakukan melalui **studi ablasi sistematis** pada 712 **citra uji**. Hasil menunjukkan **penurunan CER** dari 83.4% (**terdegradasi**) ke 34.9% (**terestorasi**) dengan $p<0.001$ dan Cohen's $d=0.85$ (**efek besar**), mengkonfirmasi efektivitas kombinasi diskriminator ***dual-modal*** dengan **optimasi fungsi kehilangan multikomponen** berorientasi HTR.

**Transformasi**: 13 istilah Inggris → Indonesia + 3 italic ditambahkan

---

### **I-D: Organisasi Makalah**

#### Before (Perspektif "Kami" + Istilah Asing Tidak Konsisten):
> Bagian~II meninjau pekerjaan terkait dalam perbaikan dokumen, **\textit{GAN}** (italic berlebihan) untuk restorasi **gambar**, dan sistem **\textit{HTR}** (italic berlebihan). Bagian~III menyajikan arsitektur yang **kami usulkan** (perspektif tidak netral) termasuk **generator U-Net Enhanced** (tidak italic), diskriminator **dual-modal** (tidak italic), integrasi **frozen recognizer**, dan **fungsi loss** yang dioptimalkan dengan **adaptive balancing**. Bagian~IV menjelaskan pengaturan eksperimental, \textit{dataset}, metodologi pelatihan, dan strategi optimisasi **\textit{hyperparameter}** (italic + Inggris). Bagian~V menyajikan hasil komprehensif termasuk perbandingan kuantitatif dengan **degraded input baseline** dan **clean ground truth reference**, **systematic ablation studies** untuk memvalidasi kontribusi setiap komponen.

#### After (Perspektif Netral + Italic Konsisten + Indonesia):
> Bagian~II meninjau pekerjaan terkait dalam perbaikan dokumen, **GAN** (akronim tidak italic) untuk restorasi **citra**, dan sistem **HTR** (akronim tidak italic). Bagian~III menyajikan arsitektur **yang diusulkan** (perspektif netral) termasuk ***generator*** (italic) U-Net ***Enhanced*** (italic), diskriminator ***dual-modal*** (italic), integrasi **pengenal beku**, dan **fungsi kehilangan** yang dioptimalkan dengan **penyeimbangan adaptif**. Bagian~IV menjelaskan pengaturan eksperimental, \textit{dataset}, metodologi pelatihan, dan strategi optimisasi **hiperparameter** (Indonesia, tidak italic). Bagian~V menyajikan hasil komprehensif termasuk perbandingan kuantitatif dengan **masukan terdegradasi *baseline*** dan **referensi *ground truth* bersih**, **studi ablasi sistematis** untuk memvalidasi kontribusi setiap komponen.

**Transformasi**: 8 istilah Inggris → Indonesia + 1 perspektif netral + 5 italic ditambahkan + 3 italic dihapus

---

## ✅ KESIMPULAN

Revisi berhasil dilakukan dengan:

### **Section I-C (18 perubahan)**:
1. ✅ 13 istilah Inggris → Indonesia
2. ✅ 3 istilah asing dicetak miring
3. ✅ Konsistensi terminologi (fungsi kehilangan, citra uji)

### **Section I-D (24 perubahan)**:
1. ✅ 1 perspektif dinetralkan ("kami usulkan" → "yang diusulkan")
2. ✅ 8 istilah Inggris → Indonesia
3. ✅ 5 istilah asing dicetak miring (*generator*, *Enhanced*, *dual-modal*, *baseline*, *ground truth*)
4. ✅ 3 italic berlebihan dihapus (GAN, HTR, hiperparameter)
5. ✅ Konsistensi terminologi di seluruh section

### **TOTAL: 42 perubahan** dalam 2 sections

Paper sekarang **memenuhi standar penulisan ilmiah bahasa Indonesia** dan **IEEE Journal guidelines** untuk publikasi Q1, dengan:
- ✅ Perspektif netral konsisten
- ✅ Italic hanya untuk istilah asing teknis (bukan akronim)
- ✅ Terminologi Indonesia yang konsisten
- ✅ Format yang sesuai standar IEEE

---

**Catatan**: Perubahan ini pada **Section I-C dan I-D**. Untuk konsistensi penuh di seluruh paper, terapkan pola yang sama terutama:
1. "kehilangan" untuk "loss"
2. "citra" untuk "gambar/image"
3. "pengenal beku" untuk "frozen recognizer"
4. Italic hanya untuk istilah asing teknis (bukan akronim)
5. Perspektif netral (hindari "kami")
