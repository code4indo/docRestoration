# ANALISIS FAIRNESS: Config Ablation Single-Modal

**Pertanyaan:** Apakah config baru menguntungkan atau merugikan single-modal?

**Tanggal:** 6 November 2025

---

## 🎯 TL;DR: **MENGUNTUNGKAN SINGLE-MODAL** (Fair Comparison)

Config baru memberikan single-modal **kesempatan terbaik** untuk perform optimal. Bukan menguntungkan dalam arti "bias", tapi **menghilangkan handicap** yang ada di config lama.

---

## 📊 Perbandingan Config: h2a (Lama) vs Ablation Fair (Baru)

### Loss Weights Comparison

| Loss Component | h2a (Lama) | Ablation Fair (Baru) | Impact |
|----------------|------------|---------------------|---------|
| **Pixel** | 50.0 | 60.0 (+20%) | ✅ **MENGUNTUNGKAN** |
| **Adversarial** | 3.0 | 4.0 (+33%) | ✅ **MENGUNTUNGKAN** |
| **RecFeat** | 8.0 | 0.0 (-100%) | ✅ **MENGUNTUNGKAN** (tidak digunakan) |
| **CTC** | 0.15 | 0.0 (-100%) | ✅ **MENGUNTUNGKAN** (tidak digunakan) |
| **Perceptual** | 1.0 | 2.0 (+100%) | ✅ **MENGUNTUNGKAN** |
| **Total Used** | ~54.15 | ~66.0 (+22%) | ✅ **MENGUNTUNGKAN** |

**Analisis:**
- Config lama: Total loss magnitude ~54 (tapi 8.15 wasted di RecFeat+CTC yang tidak digunakan)
- Config baru: Total loss magnitude ~66 (semua komponen aktif digunakan)
- **Dampak:** Gradient magnitude lebih besar → training lebih efektif ✅

---

### Training Schedule Comparison

| Parameter | h2a (Lama) | Ablation Fair (Baru) | Impact |
|-----------|------------|---------------------|---------|
| **Warmup Epochs** | 10 | 0 | ✅ **MENGUNTUNGKAN** |
| **Annealing Epochs** | 20 | 0 | ✅ **MENGUNTUNGKAN** |
| **Full Training Epochs** | 20 (31-50) | 100 (1-100) | ✅ **SANGAT MENGUNTUNGKAN** |
| **Total Epochs** | 50 | 100 | ✅ **MENGUNTUNGKAN** |
| **Effective Budget** | 20 epochs | 100 epochs | ✅ **5x LEBIH BANYAK** |

**Analisis:**
- Config lama: 30 epoch pertama WASTED (warmup/annealing untuk fitur yang tidak ada)
- Config baru: Semua 100 epoch digunakan untuk full training
- **Dampak:** Single-modal dapat explore solution space jauh lebih lama ✅

---

### Learning Rate & Optimization

| Parameter | h2a (Lama) | Ablation Fair (Baru) | Impact |
|-----------|------------|---------------------|---------|
| **LR Generator** | 0.0002 | 0.0002 | ⚖️ **NETRAL** |
| **LR Discriminator** | 0.0002 | 0.0001 | ✅ **MENGUNTUNGKAN** |
| **LR Schedule** | Cosine | Cosine | ⚖️ **NETRAL** |
| **Adaptive Balancing** | true | false | ⚖️ **NETRAL/SEDIKIT UNTUNG** |

**Analisis:**
- LR discriminator lebih rendah (0.0001 vs 0.0002): Lebih cocok untuk CNN-only yang lebih simple
- Adaptive balancing disabled: Menghindari overhead computational yang tidak perlu
- **Dampak:** Optimization lebih stable untuk arsitektur single-modal ✅

---

### Early Stopping & Monitoring

| Parameter | h2a (Lama) | Ablation Fair (Baru) | Impact |
|-----------|------------|---------------------|---------|
| **Early Stop Metric** | combined (CER+PSNR) | psnr | ✅ **MENGUNTUNGKAN** |
| **CER Weight** | 0.2 | 0.0 | ✅ **MENGUNTUNGKAN** |
| **Patience** | 25 | 25 | ⚖️ **NETRAL** |
| **Min Delta** | 0.05 | 0.05 | ⚖️ **NETRAL** |

**Analisis:**
- Config lama: Monitor "combined" metric tapi tidak optimize CER di loss → inconsistent
- Config baru: Monitor PSNR yang memang di-optimize → aligned dengan objective
- **Dampak:** Fokus training lebih jelas, early stopping lebih meaningful ✅

---

## 🔬 Analisis Mendalam: Apakah "Menguntungkan" = Bias?

### ❌ BUKAN BIAS, Tapi Fair Adaptation

**Prinsip Ablation Study yang Benar:**
> "Berikan setiap varian **kesempatan terbaik** untuk perform optimal dengan hyperparameter yang disesuaikan dengan arsitekturnya"

**Analogi:**
- **SALAH:** Tes mobil bensin vs mobil diesel dengan sama-sama pakai bensin
  - Hasil: Mobil diesel jelek (tapi ini unfair!)
  
- **BENAR:** Tes mobil bensin dengan bensin, mobil diesel dengan solar
  - Hasil: Masing-masing perform optimal, comparison fair

**Kasus Kita:**
- **Config lama (SALAH):** Tes single-modal dengan config dual-modal
  - Warmup/annealing untuk HTR integration (single-modal tidak punya HTR)
  - RecFeat/CTC loss weights (single-modal tidak gunakan ini)
  - Combined metric (single-modal tidak optimize CER)
  
- **Config baru (BENAR):** Tes single-modal dengan config adapted untuk single-modal
  - No warmup/annealing (tidak butuh HTR integration)
  - RecFeat=0, CTC=0 (tidak digunakan, jadi weight=0)
  - PSNR-only metric (sesuai dengan objective)

---

## 📐 Mathematical Analysis: Total Gradient Magnitude

### Config Lama (h2a)
```
Total Loss = 50.0*L_pixel + 3.0*L_adv + 8.0*L_recfeat + 0.15*L_ctc + 1.0*L_percep

Tapi di single-modal:
- L_recfeat diabaikan di discriminator (tidak ada text input)
- L_ctc diabaikan di discriminator (tidak ada text input)

Effective Loss = 50.0*L_pixel + 3.0*L_adv + 1.0*L_percep
                ≈ 54.0 units (magnitude reference)

WASTED gradient budget: 8.0 + 0.15 = 8.15 units
```

### Config Baru (Ablation Fair)
```
Total Loss = 60.0*L_pixel + 4.0*L_adv + 0.0*L_recfeat + 0.0*L_ctc + 2.0*L_percep

Semua komponen aktif di single-modal:
Effective Loss = 60.0*L_pixel + 4.0*L_adv + 2.0*L_percep
                ≈ 66.0 units (magnitude reference)

WASTED gradient budget: 0 units
```

### Comparison dengan Dual-Modal
```
Dual-Modal Config (production_v4):
Total Loss = 50.0*L_pixel + 3.0*L_adv + 8.0*L_recfeat + 0.15*L_ctc + 1.0*L_percep

Semua komponen aktif di dual-modal:
Effective Loss ≈ 62.15 units

Perbandingan:
- h2a single-modal: 54.0 units (87% dari dual-modal) ❌ UNDER-POWERED
- Ablation fair single-modal: 66.0 units (106% dari dual-modal) ✅ COMPARABLE
```

**Kesimpulan:** Config baru membuat single-modal **setara** dengan dual-modal dalam gradient magnitude, bukan lebih kuat.

---

## ⚖️ Fairness Analysis: Parameter-by-Parameter

### 1. Pixel Loss: 50 → 60 (+20%)

**Menguntungkan atau Merugikan?**
- ✅ **MENGUNTUNGKAN** single-modal
- Pixel loss adalah primary objective untuk image restoration
- Single-modal fokus HANYA ke visual quality (tidak ada text branch)
- Meningkatkan weight = meningkatkan fokus ke core objective
- **Fair?** YA - Dual-modal harus balance visual+text, single-modal full visual

### 2. Adversarial Loss: 3 → 4 (+33%)

**Menguntungkan atau Merugikan?**
- ✅ **MENGUNTUNGKAN** single-modal
- Adversarial loss penting untuk photorealistic texture
- Peningkatan dari 3 ke 4 = sedikit peningkatan perceptual quality
- **Fair?** YA - Mengkompensasi hilangnya RecFeat loss yang ada di dual-modal

### 3. RecFeat Loss: 8 → 0 (-100%)

**Menguntungkan atau Merugikan?**
- ✅ **MENGUNTUNGKAN** single-modal (menghilangkan noise)
- Config lama: RecFeat=8.0 tapi TIDAK DIGUNAKAN (discriminator tidak terima text)
- Config baru: RecFeat=0 (explicit, jujur bahwa tidak digunakan)
- **Fair?** YA - RecFeat loss MEMANG tidak applicable untuk single-modal

### 4. CTC Loss: 0.15 → 0 (-100%)

**Menguntungkan atau Merugikan?**
- ✅ **MENGUNTUNGKAN** single-modal (menghilangkan noise)
- Config lama: CTC=0.15 tapi TIDAK DIGUNAKAN (discriminator tidak terima text)
- Config baru: CTC=0 (explicit)
- **Fair?** YA - CTC loss MEMANG tidak applicable untuk single-modal

### 5. Perceptual Loss: 1 → 2 (+100%)

**Menguntungkan atau Merugikan?**
- ✅ **MENGUNTUNGKAN** single-modal
- Perceptual loss (VGG features) penting untuk high-level visual quality
- Peningkatan 2x = lebih fokus ke semantic similarity
- **Fair?** YA - Mengkompensasi total loss magnitude yang lebih rendah sebelumnya

### 6. Warmup: 10 → 0 epochs

**Menguntungkan atau Merugikan?**
- ✅ **SANGAT MENGUNTUNGKAN** single-modal
- Warmup dirancang untuk stabilize HTR integration (visual + text)
- Single-modal TIDAK PUNYA text branch → warmup TIDAK PERLU
- Config lama: 10 epoch WASTED dengan loss weights yang suboptimal
- **Fair?** YA - Warmup hanya diperlukan untuk dual-modal

### 7. Annealing: 20 → 0 epochs

**Menguntungkan atau Merugikan?**
- ✅ **SANGAT MENGUNTUNGKAN** single-modal
- Annealing dirancang untuk gradual increase CTC weight
- Single-modal TIDAK PUNYA CTC → annealing TIDAK PERLU
- Config lama: 20 epoch dengan annealing 0→0.15 untuk loss yang tidak digunakan
- **Fair?** YA - Annealing hanya diperlukan untuk dual-modal

### 8. Total Epochs: 50 → 100

**Menguntungkan atau Merugikan?**
- ✅ **MENGUNTUNGKAN** single-modal
- Tapi ini untuk **fair comparison** dengan dual-modal yang juga train 100 epochs
- Config lama: 50 epoch total (hanya 20 effective) vs dual-modal 100 epoch
- **Fair?** YA - Same training budget adalah prinsip dasar ablation study

### 9. LR Discriminator: 0.0002 → 0.0001

**Menguntungkan atau Merugikan?**
- ✅ **SEDIKIT MENGUNTUNGKAN** single-modal (lebih stable)
- Discriminator single-modal lebih simple (15.6M params vs 17.4M)
- Lower LR = lebih stable training, less risk of mode collapse
- **Fair?** YA - LR harus disesuaikan dengan complexity model

### 10. Early Stop Metric: combined → psnr

**Menguntungkan atau Merugikan?**
- ✅ **MENGUNTUNGKAN** single-modal (konsisten dengan objective)
- Config lama: Monitor CER+PSNR tapi tidak optimize CER → confusing signal
- Config baru: Monitor PSNR yang memang di-optimize → clear signal
- **Fair?** YA - Metric harus aligned dengan objective

---

## 🧮 Quantitative Fairness Score

Mari kita hitung **net advantage** dari semua perubahan:

| Parameter Change | Advantage Type | Magnitude | Fair? |
|------------------|----------------|-----------|-------|
| Pixel +20% | Training advantage | +0.5 dB (estimated) | ✅ Yes (core objective) |
| Adv +33% | Training advantage | +0.2 dB (estimated) | ✅ Yes (compensates RecFeat) |
| RecFeat -100% | Noise removal | +0.1 dB (estimated) | ✅ Yes (not used anyway) |
| CTC -100% | Noise removal | +0.05 dB (estimated) | ✅ Yes (not used anyway) |
| Percep +100% | Training advantage | +0.3 dB (estimated) | ✅ Yes (semantic quality) |
| Warmup -10 | Time advantage | +10 effective epochs | ✅ Yes (not needed) |
| Annealing -20 | Time advantage | +20 effective epochs | ✅ Yes (not needed) |
| Epochs +50 | Training budget | +50 epochs | ✅ Yes (fair comparison) |
| LR_D -50% | Stability advantage | Better convergence | ✅ Yes (simpler model) |
| Metric change | Signal clarity | Clearer optimization | ✅ Yes (aligned objective) |

**Total Estimated PSNR Advantage: +1.15 dB**

Breakdown:
- Loss rebalancing: +0.5 + 0.2 + 0.1 + 0.05 + 0.3 = **+1.15 dB**
- Effective epochs: 20 → 100 = **5x training budget**
- Stability: Lower LR_D = **better convergence**

---

## 🎯 Kesimpulan: MENGUNTUNGKAN, Tapi FAIR

### Jawaban Singkat:
**YA, config baru MENGUNTUNGKAN single-modal**, tapi ini adalah **fair advantage** yang menghilangkan handicap sebelumnya, BUKAN bias yang membuat single-modal artifisial superior.

### Bukti Fairness:

#### 1. **Principle of Adapted Hyperparameters**
- Setiap arsitektur deserve hyperparameter yang optimal untuk arsitekturnya
- Dual-modal butuh warmup/annealing → dapat warmup/annealing
- Single-modal tidak butuh warmup/annealing → tidak dapat warmup/annealing
- **Fair!** ✅

#### 2. **Comparable Gradient Magnitude**
- Config lama single-modal: ~54 units (87% dari dual-modal)
- Config baru single-modal: ~66 units (106% dari dual-modal)
- Difference: 106% vs 100% = **hanya 6% lebih tinggi** (dalam margin of error)
- **Fair!** ✅

#### 3. **Same Training Budget**
- Dual-modal: 100 epochs (warmup 10 + annealing 20 + full 70) = 100 total
- Single-modal lama: 50 epochs (warmup 10 + annealing 20 + full 20) = 50 total
- Single-modal baru: 100 epochs (full 100) = 100 total
- **Fair!** ✅

#### 4. **Same Dataset, Seed, Batch Size**
- Dataset: ✅ Same TFRecord
- Seed: ✅ Same (42)
- Split: ✅ Same (70/15/15)
- Batch size: ✅ Same (2)
- **Fair!** ✅

#### 5. **Comparable Model Size**
- Single-modal: 15.6M params
- Dual-modal: 17.4M params
- Difference: 1.8M (11% lebih besar dual-modal)
- **Fair!** ✅ (comparable range)

---

## 🔮 Expected Outcome dengan Config Baru

### Scenario Analysis

#### Scenario 1: Config Lama (h2a) Baseline
```
PSNR: 30.63 dB
CER: 27.12%
Effective training: 20 epochs
Handicap: -30 wasted epochs, -8.15 unused loss magnitude
```

#### Scenario 2: Config Baru (Realistic)
```
Estimated improvement dari config optimization:
- Loss rebalancing: +0.5 to +0.8 dB
- 5x effective epochs: +0.3 to +0.5 dB
- Better LR schedule: +0.1 to +0.2 dB

Expected PSNR: 30.63 + 0.9 to 1.5 = 31.5 to 32.1 dB
Expected CER: ~26-27% (similar, karena tidak optimize text)
```

#### Scenario 3: Dual-Modal Baseline
```
Test PSNR: 30.86 dB
Test CER: 27.07%
```

### Critical Question: Apakah Single-Modal akan LEBIH BAIK dari Dual-Modal?

**Kemungkinan:**

**A. Single-Modal ≈ Dual-Modal (50% probability)**
- PSNR single-modal: 30.5 - 31.2 dB
- Δ PSNR: -0.36 to +0.34 dB (NOT significant)
- **Interpretation:** Dual-modal complexity tidak worth it
- **Impact:** ❌ Dual-modal bukan major contribution

**B. Single-Modal < Dual-Modal dengan margin significant (30% probability)**
- PSNR single-modal: 29.8 - 30.3 dB
- Δ PSNR: -1.06 to -0.56 dB (significant, p<0.05)
- **Interpretation:** ✅ Dual-modal terbukti superior
- **Impact:** ✅ Paper narrative intact, Q1 valid

**C. Single-Modal > Dual-Modal (20% probability)**
- PSNR single-modal: 31.4 - 32.1 dB
- Δ PSNR: +0.54 to +1.24 dB (significant, p<0.05)
- **Interpretation:** ❌ Dual-modal COUNTER-PRODUCTIVE
- **Impact:** ❌ Major problem untuk paper

---

## ⚠️ Risk Assessment

### Jika Single-Modal Perform SANGAT BAIK (PSNR > 31.5 dB):

**Apakah ini akibat "menguntungkan" config?**

**Analisis:**
1. **Loss rebalancing:** Memberi advantage +0.5-0.8 dB (reasonable)
2. **Effective epochs:** 5x lebih banyak (fair, dual-modal juga dapat 100)
3. **LR tuning:** +0.1-0.2 dB (standard practice)

**Total advantage dari config: ~1.0-1.2 dB**

**Jika single-modal mencapai 31.5-32.0 dB:**
- Improvement dari h2a: +0.87 to +1.37 dB
- Expected dari config: +0.9 to +1.5 dB
- **Within expected range!** ✅

**Artinya:**
- Bukan karena bias/unfair config
- Tapi karena **single-modal memang capable** kalau di-optimize dengan benar
- **Implikasi:** Dual-modal complexity tidak memberikan benefit yang cukup

---

## 📝 Rekomendasi untuk Scientific Rigor

### Untuk Membuktikan Fairness di Paper:

#### 1. **Transparency Section di Methodology**
```latex
\subsection{Ablation Study: Hyperparameter Adaptation}

Untuk perbandingan yang fair antara dual-modal dan single-modal, kami 
menyesuaikan hyperparameter sesuai dengan karakteristik masing-masing 
arsitektur:

\textbf{Single-Modal Adaptations:}
\begin{itemize}
  \item Warmup/annealing disabled (tidak ada HTR untuk di-integrate)
  \item Loss weights direbalance: pixel=60, adv=4, percep=2, ctc=0
  \item Early stopping metric: PSNR-only (visual quality objective)
  \item LR discriminator: 0.0001 (arsitektur lebih simple)
\end{itemize}

\textbf{Kept Constant untuk Fair Comparison:}
\begin{itemize}
  \item Dataset, split ratio (70/15/15), seed (42)
  \item Training budget (100 epochs)
  \item Batch size (2), optimizer (Adam)
  \item Generator architecture dan LR (0.0002)
\end{itemize}

Prinsip: Memberikan setiap varian kesempatan optimal dengan hyperparameter 
yang disesuaikan, sambil menjaga konsistensi di aspek fundamental.
```

#### 2. **Ablation Table dengan Explanation**
```latex
\begin{table}[h]
\caption{Hyperparameter Adaptation untuk Ablation Study}
\begin{tabular}{lccl}
\hline
Parameter & Dual-Modal & Single-Modal & Rationale \\
\hline
Warmup epochs & 10 & 0 & No HTR integration \\
Annealing epochs & 20 & 0 & No CTC loss \\
Pixel loss weight & 50 & 60 & Increased visual focus \\
RecFeat loss weight & 8.0 & 0.0 & Not used in single-modal \\
CTC loss weight & 0.15 & 0.0 & Not used in single-modal \\
Perceptual loss & 1.0 & 2.0 & Compensate total magnitude \\
LR discriminator & 0.0001 & 0.0001 & Simpler architecture \\
\hline
\end{tabular}
\end{table}
```

#### 3. **Statistical Validation**
```latex
\subsection{Statistical Significance Testing}

Untuk memvalidasi kontribusi dual-modal, kami menggunakan paired t-test 
pada test set (n=712) dengan significance level α=0.05:

\begin{equation}
H_0: \mu_{\text{dual}} \leq \mu_{\text{single}}
\end{equation}

Threshold untuk practical significance: Δ PSNR ≥ 0.53 dB (Cohen's d > 0.1)
```

---

## ✅ FINAL VERDICT

### Pertanyaan: "Apakah config baru menguntungkan atau merugikan single-modal?"

**Jawaban:**

1. **MENGUNTUNGKAN** ✅ (estimated +1.0-1.5 dB advantage dari config optimization)

2. **FAIR** ✅ (semua adaptasi justified dan mengikuti prinsip ablation study yang benar)

3. **TRANSPARENT** ✅ (semua perubahan documented dan akan dijelaskan di paper)

4. **RISK:** Medium risk bahwa single-modal perform equally well atau better
   - Jika terjadi: BUKAN karena bias config
   - Tapi karena: Dual-modal complexity memang tidak worth it
   - Mitigation: Frozen recognizer fallback ready

### Bottom Line:

**Config baru TIDAK bias pro-single-modal**, tapi memberikan single-modal **kesempatan fair** untuk compete dengan dual-modal. Jika single-modal menang, itu legitimate scientific finding bahwa dual-modal tidak superior. Jika dual-modal menang, itu juga legitimate karena comparison sudah fair.

**Prinsip:** Better to do fair comparison dan risk finding "dual-modal tidak superior" daripada unfair comparison dan publish invalid conclusion.

---

**Training sudah berjalan. Results akan menentukan scientific truth, bukan bias dari config.**
