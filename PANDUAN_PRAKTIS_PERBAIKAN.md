# PANDUAN PRAKTIS PERBAIKAN BAHASA INGGRIS KE INDONESIA
*Contoh konkret antes dan sesudah untuk diterapkan dalam dokumen*

## CONTOH PERBAIKAN KALIMAT DARI DOKUMEN CHapter 2

### Contoh 1: Kalimat tentang Deep Learning
**SEBELUM:**
```
" emergence of deep learning represents a paradigm shift in computer vision and document image processing."
```

**SESUDAH:**
```
" kemunculan pembelajaran mendalam menandai perubahan paradigma dalam visi komputer dan pemrosesan citra dokumen."
```

### Contoh 2: Kalimat tentang CNN dan fitur
**SEBELUM:**
```
"Convolutional Neural Network (CNN) ability for hierarchical feature learning from raw pixels opened new possibilities for document restoration."
```

**SESUDAH:**
```
" kemampuan Jaringan Saraf Tiruan Konvolusi untuk pembelajaran hierarkis fitur dari piksel mentah membuka kemungkinan baru untuk restorasi dokumen."
```

### Contoh 3: Kalimat tentang Generator dan Discriminator
**SEBELUM:**
```
"GAN framework involves two neural networks: Generator (G) that tries to generate realistic examples, and Discriminator (D) that tries to distinguish between real and generated examples."
```

**SESUDAH:**
```
" kerangka kerja Jaringan Adversarial Generatif melibatkan dua jaringan saraf: Pembangkit (G) yang mencoba menghasilkan contoh realistis, dan Pencari Kebenaran (D) yang mencoba membedakan antara contoh nyata dan contoh yang dihasilkan."
```

### Contoh 4: Kalimat tentang PSNR dan kualitas
**SEBELUM:**
```
"Previous research shows that U-Net achieved PSNR 32 dB on document binarization, significant improvement compared to standard autoencoder which only achieved 28 dB."
```

**SESUDAH:**
```
" Penelitian sebelumnya menunjukkan bahwa Arsitektur U-Net mencapai RSDP 32 dB pada binerisasi dokumen, peningkatan signifikan dibandingkan autoencoder standar yang hanya mencapai 28 dB."
```

### Contoh 5: Kalimat tentang HTR dan OCR
**SEBELUM:**
```
"Handwritten Text Recognition (HTR) using Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) loss has become the de facto standard for line-level HTR."
```

**SESUDAH:**
```
" Pengenalan Teks Tulisan Tangan (PTT) menggunakan Jaringan Saraf Konvolusional Berulang (JSKB) dengan kerugian Klasifikasi Temporal Koneksionis (KTK) telah menjadi standar de facto untuk PTT tingkat baris."
```

### Contoh 6: Kalimat tentang evaluasi dan metrik
**SEBELUM:**
```
"Performance evaluation includes both visual quality metrics (PSNR, SSIM) and functional readability metrics (CER, WER) using pretrained HTR models."
```

**SESUDAH:**
```
" Evaluasi kinerja mencakup metrik kualitas visual (RSDP, IMS) dan metrik keterbacaan fungsional (TKH, TKK) menggunakan model PTT pra-terlatih."
```

### Contoh 7: Kalimat tentang training dan optimization
**SEBELUM:**
```
"Training process involves optimizing multi-component loss function combining adversarial loss, reconstruction loss, and text-aware CTC loss through backpropagation."
```

**SESUDAH:**
```
" Proses pelatihan melibatkan optimisasi fungsi kerugian multi-komponen yang menggabungkan kerugian adversarial, kerugian rekonstruksi, dan kerugian KTK yang sadar-teks melalui rambatan balik."
```

### Contoh 8: Kalimat tentang multi-modal dan discriminator
**SEBELUM:**
```
"Dual-modal discriminator evaluates both visual modality through convolutional network and textual modality through sequential processing to provide comprehensive supervision."
```

**SESUDAH:**
```
" Pencari kebenaran dual-modal mengevaluasi modalitas visual melalui jaringan konvolusional dan modalitas tekstual melalui pemrosesan sekuensial untuk memberikan supervisi komprehensif."
```

### Contoh 9: Kalimat tentang frozen recognizer
**SEBELUM:**
```
"Frozen recognizer provides consistent text-aware gradients while maintaining training stability through joint optimization of multiple loss components."
```

**SESUDAH:**
```
" Pengenal beku memberikan gradien yang sadar-teks konsisten sambil menjaga stabilitas pelatihan melalui optimisasi bersama dari beberapa komponen kerugian."
```

### Contoh 10: Kalimat tentang document enhancement
**SEBELUR:**
```
"Document enhancement aims to improve visual quality while preserving text readability for downstream Handwritten Text Recognition applications."
```

**SESUDAH:**
```
" Peningkatan kualitas dokumen bertujuan untuk memperbaiki kualitas visual sambil menjaga keterbacaan teks untuk aplikasi Pengenalan Teks Tulisan Tangan downstream."
```

## REPLACE RULES BERDASARKAN KONTEKS

### 1. Machine Learning & AI Terms
```python
# Dictionary untuk replacement
replacement_dict = {
    'deep learning': 'pembelajaran mendalam',
    'artificial intelligence': 'kecerdasan buatan',
    'machine learning': 'pembelajaran mesin',
    'neural network': 'jaringan saraf tiruan',
    'convolutional neural network': 'jaringan saraf tiruan konvolusi',
    'cnn': 'JST-K',
    'recurrent neural network': 'jaringan saraf berulang',
    'rnn': 'JST-B',
    'long short-term memory': 'jaringan saraf dengan memori jangka panjang',
    'lstm': 'JST-PJ',
    'transformer': 'arsitektur pengubah',
    'attention': 'mekanisme perhatian',
    'self-attention': 'perhatian-diri',
    'multi-head attention': 'perhatian-multi-kepala'
}
```

### 2. GAN & Architecture Terms
```python
replacement_dict.update({
    'generative adversarial network': 'jaringan adversarial generatif',
    'gan': 'JAG',
    'generator': 'pembangkit',
    'discriminator': 'pencari kebenaran',
    'encoder': 'enkoder',
    'decoder': 'dekoder',
    'autoencoder': 'autoenkoder',
    'u-net': 'arsitektur u-net',
    'patchgan': 'pencari kebenaran patch',
    'end-to-end': 'ujung-ke-ujung'
})
```

### 3. Evaluation Metrics
```python
replacement_dict.update({
    'peak signal-to-noise ratio': 'rasio sinyal terhadap derau puncak',
    'psnr': 'RSDP',
    'structural similarity index': 'indeks kemiripan struktural',
    'ssim': 'IMS',
    'character error rate': 'tingkat kesalahan karakter',
    'cer': 'TKH',
    'word error rate': 'tingkat kesalahan kata',
    'wer': 'TKK',
    'f-measure': 'pengukuran-f',
    'drd': 'distorsi resiprok jarak'
})
```

### 4. HTR & Text Recognition
```python
replacement_dict.update({
    'handwritten text recognition': 'pengenalan teks tulisan tangan',
    'htr': 'PTT',
    'optical character recognition': 'pengenalan optik karakter',
    'ocr': 'POK',
    'convolutional recurrent neural network': 'jaringan saraf konvolusional berulang',
    'crnn': 'JSKB',
    'connectionist temporal classification': 'klasifikasi temporal koneksionis',
    'ctc': 'KTK',
    'text recognition': 'pengenalan teks',
    'character recognition': 'pengenalan karakter'
})
```

### 5. Image & Document Processing
```python
replacement_dict.update({
    'image': 'citra',
    'document image': 'citra dokumen',
    'pixel': 'piksel',
    'resolution': 'resolusi',
    'enhancement': 'peningkatan',
    'restoration': 'restorasi',
    'binarization': 'binerisasi',
    'degradation': 'degradasi',
    'denoising': 'penghilangan derau',
    'blur': 'kekaburan',
    'noise': 'derau',
    'feature': 'fitur',
    'edge': 'tepi',
    'texture': 'tekstur',
    'patch': 'bagian'
})
```

### 6. Training & Optimization
```python
replacement_dict.update({
    'training': 'pelatihan',
    'testing': 'pengujian',
    'validation': 'validasi',
    'learning': 'pembelajaran',
    'optimizing': 'optimisasi',
    'optimization': 'optimisasi',
    'backpropagation': 'rambatan balik',
    'gradient': 'gradien',
    'loss function': 'fungsi kerugian',
    'loss': 'kerugian',
    'frozen': 'beku',
    'joint training': 'pelatihan bersama',
    'fine-tuning': 'penalaan halus',
    'pre-training': 'pra-pelatihan',
    'epoch': 'epoch',
    'batch': 'kelompok',
    'regularization': 'regularisasi'
})
```

## CHECKLIST PERBAIKAN

### ✅ Checklist Kata yang WAJIB Diganti:
- [ ] deep learning → pembelajaran mendalam
- [ ] CNN → JST-K (jaringan saraf tiruan konvolusi)
- [ ] GAN → JAG (jaringan adversarial generatif)
- [ ] generator → pembangkit
- [ ] discriminator → pencari kebenaran
- [ ] PSNR → RSDP
- [ ] SSIM → IMS
- [ ] CER → TKH
- [ ] WER → TKK
- [ ] HTR → PTT
- [ ] OCR → POK
- [ ] training → pelatihan
- [ ] optimization → optimisasi
- [ ] loss function → fungsi kerugian
- [ ] enhancement → peningkatan
- [ ] document → dokumen
- [ ] image → citra
- [ ] quality → kualitas
- [ ] performance → kinerja

### ✅ Checklist Nama yang BOLEH Dipertahankan:
- [ ] Nama penelitian: ERB-MultiTask, DocEnTr, Text-DIAE
- [ ] Nama arsitektur: U-Net, ResNet, VGG
- [ ] Nama dataset: IAM, KHATT, DIBCO
- [ ] Nama konferensi: CVPR, ICCV, ECCV
- [ ] Nama journals: Pattern Recognition, IEEE T-PAMI

### ✅ Checklist Singkatan:
- [ ] Berikan penjelasan pada первый упоминание
- [ ] Gunakan singkatan secara konsisten
- [ ] Pastikan singkatan jelas dan tidak ambigu

## STEPS PRAKTIS UNTUK PERBAIKAN

### Step 1: Identifikasi
```
1. Baca dokumen secara keseluruhan
2. Highlight semua kata bahasa Inggris
3. Identifikasi apakah termasuk:
   - Istilah teknis yang harus diganti
   - Nama proper yang boleh dipertahankan
   - Singkatan yang perlu dijelaskan
```

### Step 2: Replacement
```
1. Gunakan dictionary replacement di atas
2. Pastikan konteks tetap logis
3. Periksa konsistensi penggunaan
4. Tambah penjelasan untuk singkatan baru
```

### Step 3: Review
```
1. Baca ulang dokumen setelah replacement
2. Pastikan alur baca masih lancar
3. Periksa tidak ada salto logika
4. Validasi konsistensi terminologi
```

---
*Panduan ini dibuat berdasarkan dokumen Chapter 2 yang sedang редактируется*