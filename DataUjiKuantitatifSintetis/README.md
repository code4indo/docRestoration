# Data Uji Kuantitatif Sintetis

Dataset ini berisi 5 sampel yang SAMA PERSIS dengan visualisasi perbandingan metode.
**Diexport langsung dari output script visualize_method_comparison.py**

## Struktur Direktori:
- `degraded/` - Gambar dokumen terdegradasi (input untuk DE-GAN)
- `ground_truth/` - Gambar dokumen bersih (ground truth untuk evaluasi)
- `metadata/` - Informasi sampel dan hasil metode usulan untuk referensi

## Format Gambar:
- Dimensi: 128 × 1024 pixels (horizontal orientation)
- Format: Grayscale PNG
- Range: [0, 255]
- **Tidak ada mirror/transpose, langsung horizontal**

## Metrik Rata-rata (Metode Usulan):
- Average CER: 18.9%
- Average PSNR: 29.39 dB
- Average SSIM: 0.989

## Cara Penggunaan:
1. Gunakan gambar di folder `degraded/` sebagai input untuk metode DE-GAN
2. Bandingkan output DE-GAN dengan `ground_truth/` untuk kalkulasi metrik
3. Lihat `metadata/` untuk hasil metode usulan sebagai baseline perbandingan

Generated: 2025-11-04 17:46:16
