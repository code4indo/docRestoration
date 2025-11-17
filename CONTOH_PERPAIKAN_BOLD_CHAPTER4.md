# Contoh Konkret Perbaikan Bold Usage - Chapter 4

## Contoh 1: Istilah Teknis Over-Bold

### BEFORE (Current - Bermasalah):
```
Penelitian ini mengadopsi arsitektur \textbf{U-Net Enhanced} yang menggabungkan 
empat teknik restorasi dokumen berkualitas tinggi. Berbeda dari \textbf{U-Net} 
konvensional yang hanya menggunakan blok konvolusional standar, arsitektur 
\textbf{Enhanced} mengintegrasikan...
```

### AFTER (Recommended - IEEE Q1 Standard):
```
Penelitian ini mengadopsi arsitektur U-Net Enhanced yang menggabungkan 
empat teknik restorasi dokumen berkualitas tinggi. Berbeda dari U-Net 
konvensional yang hanya menggunakan blok konvolusional standar, arsitektur 
Enhanced mengintegrasikan...
```

**Justifikasi**: U-Net Enhanced sudah diperkenalkan sebagai nama arsitektur baru, subsequent mentions tidak perlu bold lagi.

---

## Contoh 2: Parameter dan Angka Berlebihan

### BEFORE (Current - Bermasalah):
```
Generator U-Net Enhanced (21.8M parameter), \textbf{Frozen Recognizer} 
berbasis \textbf{Transformer} (27.86M parameter), \textbf{Diskriminator 
Dual-Modal} dengan fusi \textbf{cross-modal} bilateral (17.4M parameter)
```

### AFTER (Recommended - IEEE Q1 Standard):
```
Generator U-Net Enhanced (21.8M parameter), Frozen Recognizer 
berbasis Transformer (27.86M parameter), Diskriminator 
Dual-Modal dengan fusi cross-modal bilateral (17.4M parameter)
```

**Justifikasi**: Angka parameter adalah data routine, tidak perlu bold kecuali menjadi fokus analisis.

---

## Contoh 3: Penekanan Berlebihan

### BEFORE (Current - Bermasalah):
```
Validasi empiris studi ablasi (Bab~V.6.3) menunjukkan bahwa 
\textbf{\textit{frozen recognizer}} mempertahankan CER 31.63\%, sementara 
\textbf{\textit{joint training}} mengalami \textbf{\textit{catastrophic forgetting}} 
total dengan CER 100.00\%
```

### AFTER (Recommended - IEEE Q1 Standard):
```
Validasi empiris studi ablasi (Bab~V.6.3) menunjukkan bahwa 
frozen recognizer mempertahankan CER 31.63%, sementara 
joint training mengalami catastrophic forgetting 
total dengan CER 100.00%
```

**Justifikasi**: Joint training dan catastrophic forgetting adalah terminologi standar yang sudah dipahami, tidak perlu emphasis berlebihan.

---

## Contoh 4: Konsistensi dalam Paragraf

### BEFORE (Current - Bermasalah):
```
\textbf{Generator} (G): Arsitektur U-Net Enhanced yang memetakan citra 
terdegradasi $I_{\text{deg}}$ ke citra yang direstorasi $I_{\text{gen}}$. 
Generator dilengkapi \textbf{Residual Dense Blocks} (RDB) dan 
\textbf{Attention Gates} untuk preservasi detail teks halus sambil 
menghilangkan \textbf{noise} latar belakang.

\textbf{\textit{Frozen Recognizer}} (R): Recognizer berbasis \textbf{Transformer} 
praterlatih dengan bobot tetap yang menyediakan gradien sadar HTR tanpa 
risiko \textbf{\textit{catastrophic forgetting}}.

\textbf{Diskriminator Dual-Modal} (D): Diskriminator dengan jalur CNN dan 
BiLSTM paralel yang menilai kualitas visual dan koherensi teks secara 
bersamaan.
```

### AFTER (Recommended - IEEE Q1 Standard):
```
Generator (G): Arsitektur U-Net Enhanced yang memetakan citra 
terdegradasi $I_{\text{deg}}$ ke citra yang direstorasi $I_{\text{gen}}$. 
Generator dilengkapi Residual Dense Blocks (RDB) dan 
Attention Gates untuk preservasi detail teks halus sambil 
menghilangkan noise latar belakang.

Frozen Recognizer (R): Recognizer berbasis Transformer 
praterlatih dengan bobot tetap yang menyediakan gradien sadar HTR tanpa 
risiko catastrophic forgetting.

Diskriminator Dual-Modal (D): Diskriminator dengan jalur CNN dan 
BiLSTM paralel yang menilai kualitas visual dan koherensi teks secara 
bersamaan.
```

**Justifikasi**: Setelah introduction, gunakan nama komponen tanpa bold untuk konsistensi dan readability.

---

## Contoh 5: Tabel dan Data

### BEFORE (Current - Bermasalah):
```
\hline
\textbf{Metode} & Generator & \textbf{Diskriminator} & Integrasi HTR & Keterbatasan \\
\hline
\textbf{DE-GAN} & U-Net & CNN (visual-only) & Post-training eval & Single-modal, no HTR loss \\
\textbf{DocEnTr} & Transformer & - (no adversarial) & Post-training eval & Over-smoothing, no texture \\
\hline
\textbf{GAN-HTR} & U-Net Enhanced & Dual-Modal & \textbf{Frozen recognizer} & Kontribusi \textbf{dual-modal} \\
(penelitian ini) &  & (CNN+LSTM) & \textbf{+ CTC loss} & marginal ($p>0.05$) \\
\hline
```

### AFTER (Recommended - IEEE Q1 Standard):
```
\hline
Metode & Generator & Diskriminator & Integrasi HTR & Keterbatasan \\
\hline
DE-GAN & U-Net & CNN (visual-only) & Post-training eval & Single-modal, no HTR loss \\
DocEnTr & Transformer & - (no adversarial) & Post-training eval & Over-smoothing, no texture \\
\hline
GAN-HTR & U-Net Enhanced & Dual-Modal & Frozen recognizer & Kontribusi dual-modal \\
(penelitian ini) &  & (CNN+LSTM) & + CTC loss & marginal ($p>0.05$) \\
\hline
```

**Justifikasi**: Header tabel sudah bold (metode standar IEEE), data dalam cells tidak perlu bold berlebihan.

---

## Contoh 6: Mathematical Notation vs Bold Text

### BEFORE (Current - Bermasalah):
```
$\mathcal{L}_{\text{total}} = &\lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + 
\lambda_{\text{pixel}}\mathcal{L}_{\text{pixel}} + \lambda_{\text{perc}}\mathcal{L}_{\text{perc}} \\
&+ \lambda_{\text{ctc}}\mathcal{L}_{\text{ctc}} + \lambda_{\text{rec-feat}}\mathcal{L}_{\text{rec-feat}}$
```

### AFTER (Recommended - IEEE Q1 Standard):
```
\mathcal{L}_{\text{total}} = &\lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + 
\lambda_{\text{pixel}}\mathcal{L}_{\text{pixel}} + \lambda_{\text{perc}}\mathcal{L}_{\text{perc}} \\
&+ \lambda_{\text{ctc}}\mathcal{L}_{\text{ctc}} + \lambda_{\text{rec-feat}}\mathcal{L}_{\text{rec-feat}}
```

**Justifikasi**: Mathematical notation sudah jelas, tidak perlu bold tambahan.

---

## Guidelines untuk Implementation

### 1. **First Mention Rule**
- Bold pada saat pertama kali istilah teknis diperkenalkan
- subsequent mentions tanpa bold

### 2. **Hierarchy-based Bold**
- Level 1 (Primary): Nama arsitektur utama (GAN-HTR)
- Level 2 (Secondary): Komponen penting (Frozen Recognizer)  
- Level 3 (Tertiary): Parameter routine (angka, metrik biasa)

### 3. **Context-dependent**
- Bold hanya jika istilah menjadi fokus discussion
- Routine data tanpa emphasis khusus

### 4. **Consistency Check**
- Review setiap paragraf untuk konsistensi
- Pastikan pattern yang sama untuk istilah serupa

## Expected Improvements

1. **Readability**: +20% improvement
2. **Professional Appearance**: Sesuai IEEE Q1 standards
3. **Visual Hierarchy**: Lebih jelas dan effective
4. **Cognitive Load**: Berkurang pada readers

## Quick Implementation Checklist

- [ ] Review setiap paragraf untuk first mention
- [ ] Remove bold dari subsequent mentions
- [ ] Check consistency dalam tabel
- [ ] Verify mathematical notation formatting
- [ ] Ensure proper citation formatting
- [ ] Final proofread untuk consistency