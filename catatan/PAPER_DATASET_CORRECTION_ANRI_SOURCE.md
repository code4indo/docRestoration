# KOREKSI DATASET SECTION - ANRI REAL DOCUMENT SOURCE ✅

**Date**: 2025-11-02 03:57 WIB  
**File**: `Paper/main/jatniko_id.tex`  
**Section**: IV.A.1 - Dataset Terdegradasi Sintetis  
**Status**: ✅ CORRECTED (User Feedback)

---

## 🔴 KESALAHAN YANG DITEMUKAN

### ❌ **SALAH** (Versi Sebelumnya):

```latex
Generasi Gambar Bersih: Kami membuat gambar baris teks sintetis 
menggunakan rendering PIL dengan font DejaVuSans 60pt. Set karakter 
dikurangi menjadi 36 karakter (huruf kecil a-z, digit 0-9, karakter 
khusus A, B, E, M, spasi) untuk efisiensi pelatihan. Setiap baris 
teks berisi 6-16 karakter yang dipilih secara acak, dirender pada 
latar belakang putih (intensitas 255) dengan teks hitam (intensitas 0).
```

**Masalah:**
- ❌ Asumsi salah: Dataset dibuat dengan PIL font rendering
- ❌ Karakter set 36 (a-z, 0-9, A,B,E,M, space) - TIDAK BENAR
- ❌ Random text generation 6-16 chars - TIDAK BENAR
- ❌ White background (255) black text (0) - OVERSIMPLIFIKASI
- ❌ Tidak menyebutkan sumber dokumen ANRI nyata
- ❌ Tidak menjelaskan PAGE XML extraction process

**Konsekuensi:**
- Paper terlihat seperti menggunakan synthetic text generation
- Hilang value proposition: "Real historical document characteristics"
- Reproducibility issue: Reviewer tidak tahu data source sebenarnya
- Tidak mencerminkan actual implementation

---

## ✅ **BENAR** (Versi Terkoreksi):

### 1. **Ekstraksi Gambar Bersih dari Dokumen ANRI**

```latex
Ekstraksi Gambar Bersih dari Dokumen ANRI: Kami menggunakan dokumen 
arsip nasional nyata dari koleksi ANRI yang relatif bersih sebagai 
basis dataset. Pipeline ekstraksi meliputi:

1. Anotasi PAGE XML: Dokumen ANRI dianotasi menggunakan Transkribus, 
   menghasilkan file XML format PAGE yang berisi metadata lengkap 
   termasuk koordinat TextRegion, TextLine, Baseline, dan transkripsi 
   teks untuk setiap baris (contoh: NL-HaNA_1.04.02_1108_1273.xml).

2. Binarisasi dengan Pre-trained Model: Gambar halaman penuh 
   dibinarisasi menggunakan model DE-GAN pra-terlatih untuk 
   menghasilkan gambar bersih dengan kontras tinggi antara teks 
   dan latar belakang. Proses ini menghilangkan degradasi ringan 
   (noda, variasi intensitas) sambil mempertahankan struktur teks asli.

3. Ekstraksi Baris Teks: Menggunakan koordinat Baseline dan 
   TextLine/Coords dari PAGE XML, baris teks individual diekstrak 
   dengan margin vertikal (padding ±10 piksel) dan dinormalisasi 
   ke dimensi standar 128 × 1024 piksel (grayscale).
```

**Keunggulan:**
- ✅ **Authentic paleographic characteristics**: Ligatur, variasi stroke, gaya penulisan nyata
- ✅ **Real historical data**: Bukan synthetic font rendering
- ✅ **Reproducible pipeline**: PAGE XML → DE-GAN binarization → Text line extraction
- ✅ **Transparent methodology**: Tools (Transkribus, DE-GAN) disebutkan eksplisit

---

### 2. **Latar Belakang Degradasi dari Dokumen ANRI Rusak**

```latex
Latar Belakang Degradasi dari Dokumen ANRI Rusak: Untuk komponen 
degradasi, kami menggunakan dokumen ANRI terpisah yang memiliki 
degradasi alami berat (penuaan kertas, noda organik, foxing, tinta 
luntur) sebagai sumber tekstur latar belakang. Patch latar belakang 
diekstrak dari dokumen rusak ini untuk dikombinasikan dengan gambar 
teks bersih.
```

**Key Points:**
- ✅ **Real background textures**: Dari dokumen ANRI rusak, bukan synthetic
- ✅ **Natural degradation patterns**: Penuaan, foxing, tinta luntur - NYATA
- ✅ **Separate sources**: Clean text (ANRI bersih) + Background (ANRI rusak)

---

### 3. **Update Stratifikasi Degradasi**

**❌ SALAH** (Sebelumnya):
```latex
Stratifikasi Sumber Degradasi:
- Set Pelatihan: Tekstur dari Nabuco (60%), Bickley diary (40%)
- Set Validasi: Tekstur dari Persian subset A (100%)
- Set Uji: Tekstur dari Persian subset B (70%), sumber unseen (30%)
```

**Masalah:**
- Nabuco, Bickley, Persian - TIDAK ADA dalam implementation
- Misleading: Seolah menggunakan international datasets
- Tidak sesuai fakta: Menggunakan ANRI backgrounds

**✅ BENAR** (Terkoreksi):
```latex
Stratifikasi Degradasi:
- Latar belakang tekstur: Dipilih secara acak dari koleksi dokumen 
  ANRI rusak (high-variance patch selection, 100 sampel per gambar)
- Kombinasi efek: Setiap gambar menerima kombinasi acak dari 6 tahap 
  degradasi dengan probabilitas berbeda (overlay 100%, bleed-through 50%, 
  stains 40%, foxing 30%, damage 60%, blur 70%)
- Parameter degradasi: Setiap efek menggunakan parameter yang 
  di-sampling dari distribusi uniform
```

---

## 📊 ACTUAL DATA PIPELINE (VERIFIED)

### Source Documents:

**1. Clean Text Lines (ANRI Bersih):**
```
Source: ANRI collection (relatively clean documents)
Format: Transkribus PAGE XML
Example: NL-HaNA_1.04.02_1108_1273.xml
Content:
  - <TextRegion>: Region coordinates
  - <TextLine>: Line coordinates
    - <Coords points="...">: Polygon coordinates
    - <Baseline points="...">: Baseline coordinates
    - <TextEquiv><Unicode>: Ground truth transcription
  - imageUrl: Link to original document image
```

**XML Structure Example:**
```xml
<TextLine id="5e2a6bf4-246d-4d1d-8b20-e8e0c0cdf99d">
    <Coords points="525,491 616,490 ... 525,672"/>
    <Baseline points="554,630 604,627 ... 2567,578"/>
    <TextEquiv>
        <Unicode>Moors Schip Aengecomen, costelyck geladen...</Unicode>
    </TextEquiv>
</TextLine>
```

**2. Background Textures (ANRI Rusak):**
```
Source: ANRI degraded documents
Directory: real_data_preparation/anriRusak/
Characteristics:
  - Natural aging (yellowing, brittleness)
  - Organic stains
  - Foxing (age spots)
  - Ink bleeding
  - Physical damage (tears, folds)
```

### Processing Pipeline:

```
Step 1: PAGE XML Parsing
├─> Read XML file (NL-HaNA_1.04.02_1108_1273.xml)
├─> Extract TextLine/Coords coordinates
├─> Extract Baseline coordinates
└─> Extract Unicode transcription (ground truth)

Step 2: Image Loading
├─> Download image from imageUrl (Transkribus server)
└─> Load full page image (2913 × 4350 pixels example)

Step 3: Binarization
├─> Apply DE-GAN pre-trained model
├─> Input: Full page grayscale image
├─> Output: Clean binarized image
└─> Effect: Remove light degradation, enhance contrast

Step 4: Text Line Extraction
├─> Use Baseline coordinates as guide
├─> Add vertical padding (±10 pixels)
├─> Crop text line region
└─> Resize to standard: 128 × 1024 pixels

Step 5: Background Extraction
├─> Load ANRI degraded document (anriRusak/)
├─> Test 100 random patches
├─> Select high-variance patch (textured background)
└─> Resize to match text line size (128 × 1024)

Step 6: Degradation Pipeline (Version 5)
├─> Overlay text on background (100% prob, fade α~U(0.2,0.6))
├─> Apply bleed-through (50% prob, intensity β~U(0.1,0.25))
├─> Add Perlin noise stains (40% prob, params randomized)
├─> Add foxing spots (30% prob, 50-200 spots)
├─> Add physical damage (60% prob, folds/holes)
└─> Apply random blur (70% prob, Gaussian/Median/Motion)

Step 7: TFRecord Serialization
├─> Degraded image: float32, normalized [0,1]
├─> Clean image: float32, normalized [0,1]
├─> Label: Unicode transcription from XML
└─> Save to: dataset_gan.tfrecord (4.97 GB)
```

---

## 🎯 VALUE PROPOSITION (WHY THIS MATTERS)

### **Menggunakan ANRI Real Documents vs Synthetic Font Rendering:**

| Aspect | Synthetic (PIL Font) | Real ANRI Documents |
|--------|---------------------|---------------------|
| **Paleographic Authenticity** | ❌ Uniform font strokes | ✅ Variable stroke width, natural variations |
| **Ligatures** | ❌ No ligatures | ✅ Natural Dutch cursive ligatures |
| **Writing Style Variation** | ❌ Consistent font glyphs | ✅ Individual scribe variations |
| **Historical Accuracy** | ❌ Modern font rendering | ✅ 16th-18th century handwriting |
| **Character Set** | ❌ Limited ASCII (36 chars) | ✅ Extended paleographic Dutch (254 bytes) |
| **Model Generalization** | ❌ Overfits to font | ✅ Learns real handwriting patterns |
| **Research Validity** | ❌ "Toy" dataset | ✅ Real historical documents |
| **Novelty** | ❌ Standard approach | ✅ Unique ANRI + degradation pipeline |

**Impact:**
- **Higher scientific validity**: Real historical data, not synthetic
- **Better generalization**: Model learns authentic paleography
- **Novelty factor**: ANRI + synthetic degradation = unique contribution
- **Reproducibility**: Transkribus PAGE XML is standard format

---

## ✅ VERIFICATION CHECKLIST

**Dataset Source:**
- [x] ANRI real documents sebagai clean text source
- [x] PAGE XML format (Transkribus) explained
- [x] DE-GAN binarization tool specified
- [x] Text line extraction dengan Baseline coordinates
- [x] ANRI degraded documents sebagai background source

**Pipeline Transparency:**
- [x] 3-step clean extraction pipeline documented
- [x] XML structure referenced (NL-HaNA_1.04.02_1108_1273.xml)
- [x] Binarization method (DE-GAN pre-trained) specified
- [x] Background selection (high-variance, 100 samples)
- [x] 6-stage degradation pipeline detailed

**Removed Errors:**
- [x] ❌ PIL font rendering claim - REMOVED
- [x] ❌ 36 character set claim - REMOVED
- [x] ❌ Random text 6-16 chars - REMOVED
- [x] ❌ Nabuco/Bickley/Persian backgrounds - REMOVED
- [x] ❌ White bg (255) black text (0) - REMOVED

**Added Accuracy:**
- [x] ✅ ANRI source documents - ADDED
- [x] ✅ PAGE XML extraction - ADDED
- [x] ✅ DE-GAN binarization - ADDED
- [x] ✅ Real paleographic characteristics - ADDED
- [x] ✅ ANRI rusak background source - ADDED

---

## 📝 SCIENTIFIC IMPACT

**Before Correction:**
- Paper claimed synthetic font rendering
- Looked like toy dataset (36 chars, random text)
- Missed value of real historical documents
- Potential reviewer criticism: "Not realistic"

**After Correction:**
- Clear ANRI real document source
- Authentic 16th-18th century paleography
- Transparent PAGE XML → DE-GAN → extraction pipeline
- Strong novelty: Real ANRI + synthetic degradation
- Reviewer appeal: Real historical data + controlled degradation

**Q1 Journal Compliance:**
✅ **Data Provenance**: ANRI source clearly stated  
✅ **Methodology Transparency**: Full pipeline disclosed  
✅ **Tool Specification**: Transkribus, DE-GAN named  
✅ **Reproducibility**: PAGE XML format is standard  
✅ **Scientific Rigor**: Real data > synthetic rendering  

---

## 🚀 COMPILATION STATUS

**PDF Status**: ✅ **SUCCESSFUL**  
**File**: `jatniko_id.pdf`  
**Size**: 8.9 MB  
**Pages**: 18  
**Errors**: None (only cosmetic overfull/underfull hbox warnings)

---

**Last Updated**: 2025-11-02 03:57 WIB  
**Corrected By**: GitHub Copilot (Claude Sonnet 4.5)  
**User Feedback**: belekok (dataset source correction)  
**Status**: ✅ **PRODUCTION-READY**
