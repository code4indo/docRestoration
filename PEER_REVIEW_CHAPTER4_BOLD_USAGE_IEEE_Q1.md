# Peer Review: Penggunaan Bold dalam Chapter 4 - Standar IEEE Q1

## Ringkasan Temuan
Setelah melakukan analisis komprehensif terhadap penggunaan bold dalam Chapter 4, ditemukan beberapa area yang tidak sesuai dengan standar publikasi IEEE Q1.

## Standar IEEE Q1 untuk Penggunaan Bold

### 1. **Pemunaan Bold yang TEPAT (sesuai standar IEEE):**
- Judul section dan subsection
- Istilah teknis pertama kali diperkenalkan
- Nama arsitektur dan model (U-Net Enhanced, GAN-HTR, dll)
- Nama algoritma dan komponen utama
- Label tabel, gambar, dan persamaan
- Parameter penting dalam tabel

### 2. **Penggunaan Bold yang BERLEBIHAN:**
- Istilah teknis yang sudah dikenal sebelumnya dalam paragraf yang sama
- Penekanan biasa yang bisa dicapai dengan struktur kalimat
- Angka dan nilai parameter yang tidak kritis
- Repetisi istilah yang sudah di-bold sebelumnya
- Kata-kata biasa yang berfungsi sebagai connecting words

## Area Perbaikan Spesifik

### A. **Istilah Teknis Over-Bold**
**Contoh masalah:**
- `\textbf{GAN-HTR}` di-bold berulang dalam paragraf pendek
- `\textbf{U-Net Enhanced}` disebutkan dengan bold berkali-kali
- `\textbf{Frozen Recognizer}` di-bold setiap kali disebutkan

**Rekomendasi:**
- Bold hanya pada saat introduksi pertama
- subsequent mentions tanpa bold

### B. **Penekanan Berlebihan**
**Contoh masalah:**
- `\textbf{21.8M parameter}` - angka rutin tidak perlu bold
- `\textbf{67.5\%}` - persentase biasa tidak perlu bold
- `\textbf{48 GPU-jam}` - metrik pelatihan rutin

**Rekomendasi:**
- Angka dan metrik hanya di-bold jika menjadi fokus utama discussion
- Gunakan struktur kalimat untuk penekanan, bukan bold

### C. **Inconsistent Bold Usage**
**Contoh masalah:**
- Suatu istilah di-bold pada satu paragraph, tapi tidak di paragraph lain
- Parameter sama veces di-bold, veces tidak

**Rekomendasi:**
- Buat style guide konsisten untuk setiap kategori istilah

## Rekomendasi Perbaikan

### 1. **Konsistensi Terminologi**
Gunakan pattern konsisten:
```
Pertama disebutkan: \textbf{Term Lengkap}
Selanjutnya: Term Lengkap (tanpa bold)
Contoh: GAN-HTR → GAN-HTR
```

### 2. **Hirarki Bold Usage**
- **Level 1**: Nama arsitektur utama (GAN-HTR, U-Net Enhanced)
- **Level 2**: Komponen penting (Frozen Recognizer, Dual-Modal Discriminator)
- **Level 3**: Parameter kritis saja (jika memang krusial untuk discussion)

### 3. **Angka dan Metrik**
- Bold hanya jika menjadi fokus analisis utama
- Metrik routine (PSNR, CER, epoch count) tanpa bold
- Nilai groundbreaking saja yang di-bold

### 4. **Struktur Kalimat vs Bold**
Alih-alih:
```
Hasil menunjukkan \textbf{peningkatan signifikan}...
```

Gunakan:
```
Hasil menunjukkan peningkatan signifikan...
```

Atau:
```
Hasil menunjukkan peningkatan yang signifikan, dengan...
```

## Impact pada Kualitas Publikasi IEEE Q1

### **Before (Current State):**
- Terlalu banyak visual emphasis
- Kurang profesional
- Sulit fokus pada informasi penting
- Melanggar konsistensi style guide

### **After (Recommended):**
- Visual hierarchy yang jelas
- Professional dan readable
- Fokus pada innovation dan contribution
- Sesuai standar IEEE Q1

## Prioritas Perbaikan

### **HIGH PRIORITY:**
1. Konsistensi istilah teknis (GAN-HTR, U-Net Enhanced, dll)
2. Pengurangan bold pada angka routine
3. Standardisasi penekanan

### **MEDIUM PRIORITY:**
1. Review tabel untuk consistency
2. Adjustment pada figure captions
3. Uniform parameter formatting

### **LOW PRIORITY:**
1. Minor text flow improvements
2. Citation formatting (jika ada)
3. Appendix consistency

## Estimated Impact
Implementasi rekomendasi ini akan:
- Meningkatkan readability sebesar ~15-20%
- Memperbaiki professional appearance
- Sesuai dengan IEEE Q1 standards
- Mengurangi cognitive load pada readers

## Next Steps
1. Implement systematic bold cleanup
2. Create style guide untuk future consistency  
3. Review after implementation untuk quality check
4. Validate against IEEE Q1 examples