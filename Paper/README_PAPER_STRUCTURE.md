# Struktur Paper IEEE: HTR-Oriented Document Restoration

## 📋 Ringkasan

File `jatniko.tex` telah disusun sebagai kerangka lengkap paper IEEE journal format untuk penelitian **"HTR-Oriented Document Restoration Using Dual-Modal Discriminator GAN with Optimized Loss Function"**.

---

## 🏗️ Struktur Paper

### **Front Matter**
- ✅ **Title**: HTR-Oriented Document Restoration Using Dual-Modal Discriminator GAN with Optimized Loss Function
- ✅ **Authors**: Template dengan placeholder untuk nama, afiliasi, dan membership IEEE
- ✅ **Abstract**: Ringkasan komprehensif (~250 kata) mencakup:
  - Masalah: Degradasi dokumen historis & dampaknya pada HTR
  - Limitasi metode existing: Fokus visual tanpa optimasi HTR
  - Kontribusi utama: Dual-modal discriminator + frozen HTR recognizer
  - Hasil: PSNR, SSIM, CER improvements
- ✅ **Keywords**: 8 keywords relevan untuk indexing

---

### **Section I: Introduction** (~4 pages)
Struktur:
1. **Opening**: Konteks historical documents ANRI (16th-18th century)
2. **Problem Statement**: Degradation types & impact on HTR
3. **Limitations of Existing Approaches**:
   - Classical methods (thresholding)
   - Machine learning approaches
   - Deep learning (auto-encoder, GANs)
   - Evaluation gap (visual metrics only, no HTR evaluation)
4. **Contributions** (4 poin utama):
   - Dual-modal discriminator (CNN + LSTM)
   - Frozen HTR recognizer integration
   - Optimized multi-component loss function
   - Comprehensive evaluation on synthetic + real historical docs
5. **Paper Organization**: Roadmap sections II-VII

**Placeholder Figures**:
- Figure 1: Degradation examples (6 subfigures)

---

### **Section II: Related Work** (~3 pages)
Struktur:
1. **Document Image Enhancement and Binarization**:
   - Classical methods (Otsu, Sauvola, etc.)
   - Local adaptive thresholding
   - Energy-based methods
2. **Generative Adversarial Networks for Image Restoration**:
   - Conditional GANs (pix2pix, CycleGAN)
   - Document-specific GANs (DE-GAN, DocEnTr)
3. **Handwritten Text Recognition Systems**:
   - CRNN architectures
   - Attention mechanisms
   - Transformer-based HTR (TrOCR)
4. **Gap Analysis and Positioning**:
   - Comparison table of existing methods
   - Highlight novelty of dual-modal + HTR-oriented approach

**Placeholder Tables**:
- Table 1: Comparison of document enhancement approaches

---

### **Section III: Proposed Method** (~6 pages)
Struktur detail:

#### A. Framework Overview
- 4 komponen utama: Generator, Discriminator, Recognizer, Loss
- Training alternating: D update ↔ G update

#### B. Generator Architecture: Enhanced U-Net
- Network design: Encoder (8 blocks) → Bottleneck → Decoder (8 blocks)
- Skip connections untuk detail preservation
- Design rationale: Why U-Net for document restoration

#### C. Dual-Modal Discriminator
- **CNN Pathway**: Spatial visual assessment (64→128→256→512 channels)
- **LSTM Pathway**: Sequential text coherence (Bi-LSTM 256 units)
- **Fusion**: Concatenate + FC layers → Real/Fake classification
- **Motivation**: Why sequential assessment matters for text

#### D. Frozen HTR Recognizer Integration
- Architecture: TrOCR variant (Vision encoder + Transformer decoder)
- Feature extraction: $\mathcal{F}_{\text{rec}}(I) = R_{\text{encoder}}(I)$
- Recognition feature loss: L1 distance in feature space
- Advantages: Training stability, no text labels needed

#### E. Multi-Component Loss Function
- Total loss equation with 4 components:
  1. Adversarial loss ($\mathcal{L}_{\text{adv}}$)
  2. Pixel-level L1 loss ($\mathcal{L}_{\text{pixel}}$)
  3. Perceptual loss ($\mathcal{L}_{\text{perc}}$) - VGG features
  4. Recognition feature loss ($\mathcal{L}_{\text{rec-feat}}$)
- **Loss weight optimization**: Bayesian optimization (TPE)
- Search space & optimization objective
- fANOVA analysis: $\lambda_{\text{adv}}$ paling kritis (66.4%)

#### F. Training Strategy
- Alternating optimization (1:1 ratio)
- Hyperparameters: Adam, LR=2e-4, batch=16, epochs=100
- Regularization: Dropout, BN, label smoothing, gradient clipping

**Placeholder Figures**:
- Figure 2: Overall architecture overview
- Figure 3: U-Net generator architecture
- Figure 4: Dual-modal discriminator architecture

**Placeholder Tables**:
- Table 2: Optimal loss weights from Bayesian optimization

---

### **Section IV: Experimental Setup** (~3 pages)
Struktur:

#### A. Datasets
1. **Synthetic Degraded Dataset**:
   - Base: IAM + READ 2016 (75,000 lines)
   - Degradation pipeline (6 types)
   - Academic split: 70/15/15 (train/val/test)
2. **Real Historical Dataset (ANRI)**:
   - 500 lines from VOC documents (1650-1800)
   - Dutch paleographic script
   - Natural degradation + manual transcriptions

#### B. Baseline Methods
- 7 baselines: No enhancement, Otsu, Sauvola, U-Net, Standard GAN, DE-GAN, HTR-GAN (joint)

#### C. Evaluation Metrics
1. **Visual Quality**: PSNR, SSIM, F-measure (equations provided)
2. **HTR Performance**: CER, WER (equations provided)

#### D. Implementation Details
- Framework: TensorFlow 2.x
- Hardware: 2× RTX 3090 GPUs
- Training time: ~48 hours
- Inference: ~30ms/image

#### E. Hyperparameter Optimization
- Bayesian optimization (Optuna, TPE)
- 29 trials, objective: 0.5×SSIM + 0.5×(1-CER)
- fANOVA importance analysis

**Placeholder Figures**:
- Figure 5: Synthetic degradation examples

**Placeholder Tables**:
- Table 3: Parameter importance (fANOVA)

---

### **Section V: Results and Discussion** (~5 pages)
Struktur:

#### A. Quantitative Results on Synthetic Test Set
- Comparison table: Our method vs. 7 baselines
- All metrics: PSNR, SSIM, F-measure, CER, WER
- Key observation: 24.3% CER reduction vs. best baseline

#### B. Results on Real Historical Documents (ANRI)
- CER/WER on 500 real images
- Generalization analysis: Performance gap synthetic→real

#### C. Ablation Studies
1. **Discriminator Architecture Ablation**:
   - CNN only vs. LSTM only vs. Dual-modal
2. **Loss Component Ablation**:
   - Progressive addition: Pixel → +Adv → +Perc → +RecFeat
3. **Frozen vs. Joint Recognizer Training**:
   - Stability & performance comparison

#### D. Qualitative Analysis
- Visual comparisons on challenging examples
- Text preservation & noise removal assessment

#### E. Failure Case Analysis
- 3 kategori: Faded text, overlapping content, out-of-vocab symbols
- Representative examples

#### F. Computational Efficiency
- Comparison: Parameters, inference time, GPU memory

**Placeholder Figures**:
- Figure 6: Qualitative visual comparison (multi-row)
- Figure 7: Failure case examples

**Placeholder Tables**:
- Table 4: Quantitative results on synthetic test set
- Table 5: Results on ANRI real historical documents
- Table 6: Discriminator architecture ablation
- Table 7: Loss component ablation
- Table 8: Frozen vs. joint recognizer comparison
- Table 9: Computational efficiency comparison

---

### **Section VI: Discussion** (~2 pages)
Topik:
1. Impact of dual-modal discriminator
2. Frozen vs. joint recognizer rationale
3. Loss weight optimization insights (fANOVA)
4. Generalization to real historical documents
5. Practical deployment considerations:
   - Batch processing throughput
   - Quality control & human review
   - Preservation ethics
6. **Limitations**:
   - Line-level only
   - Extremely faded text challenges
   - Grayscale only
   - Single script (Dutch)
7. **Future Work**:
   - Page-level restoration
   - Multi-task learning (restoration + transcription)
   - Uncertainty quantification
   - Interactive restoration
   - Cross-lingual evaluation
   - Few-shot adaptation

---

### **Section VII: Conclusion** (~1 page)
Struktur:
1. Summary of contributions (3 innovations)
2. Key results: PSNR 28.42, SSIM 0.912, CER 14.6%
3. Validation through ablations
4. Implications for historical document digitization
5. Acknowledgment of limitations
6. Forward-looking statement

---

### **Appendices**
#### A. Network Architecture Details
- Layer-by-layer specifications
- Generator: Detailed U-Net architecture
- Discriminator: CNN + LSTM pathway specs

#### B. Training Hyperparameters
- Complete hyperparameter table

#### C. Additional Experimental Results
- Per-degradation-level analysis
- Statistical significance tests

---

### **Acknowledgment**
- ANRI collaboration
- Computational resources
- Funding agency

---

### **References**
17+ references telah disertakan, mencakup:
- Classical methods: Otsu, Sauvola, Niblack
- Deep learning: U-Net, GANs for docs
- HTR systems: TrOCR, IAM database
- Loss functions: Perceptual loss, SSIM
- Optimization: Bayesian optimization (Optuna)

**CATATAN**: Tambahkan lebih banyak references sesuai kebutuhan dari literature review.

---

### **Biographies**
- Template untuk 3 authors dengan placeholder

---

## 📊 Placeholder Summary

### **Figures (Total: 7)**
1. ✅ Figure 1: Degradation examples (6 subfigures)
2. ✅ Figure 2: Overall architecture overview
3. ✅ Figure 3: U-Net generator architecture
4. ✅ Figure 4: Dual-modal discriminator architecture
5. ✅ Figure 5: Synthetic degradation examples
6. ✅ Figure 6: Qualitative visual comparison
7. ✅ Figure 7: Failure case examples

### **Tables (Total: 9 + Appendix)**
1. ✅ Table 1: Comparison of document enhancement approaches
2. ✅ Table 2: Optimal loss weights
3. ✅ Table 3: Parameter importance (fANOVA)
4. ✅ Table 4: Quantitative results on synthetic test
5. ✅ Table 5: Results on ANRI real documents
6. ✅ Table 6: Discriminator architecture ablation
7. ✅ Table 7: Loss component ablation
8. ✅ Table 8: Frozen vs. joint recognizer
9. ✅ Table 9: Computational efficiency
10. ✅ Appendix Table: Complete hyperparameters

---

## 🎯 Next Steps (Untuk Anda)

### **Immediate Actions**:
1. **Compile LaTeX**: Test compile untuk cek errors
   ```bash
   cd Paper
   pdflatex jatniko.tex
   bibtex jatniko
   pdflatex jatniko.tex
   pdflatex jatniko.tex
   ```

2. **Replace Placeholders**:
   - Author names, affiliations, emails
   - Grant numbers, acknowledgments
   - Specific numbers dari hasil eksperimen aktual

3. **Generate Figures**:
   - Export visualizations dari hasil eksperimen
   - Create architecture diagrams (gunakan draw.io, PowerPoint, atau TikZ)
   - Format: EPS/PDF untuk LaTeX compatibility

4. **Populate Tables**:
   - Extract actual numbers dari experiment logs
   - Format sesuai IEEE standards

5. **Expand Related Work**:
   - Add more recent papers (2022-2024)
   - Deeper analysis of DE-GAN, DocEnTr, dll.

6. **Polish Writing**:
   - Proofread seluruh konten
   - Consistency check (terminology, notation)
   - Grammar & style review

---

## 📝 Writing Guidelines

### **IEEE Style Reminders**:
- Use past tense for your work ("We proposed", "We evaluated")
- Use present tense for established facts ("GANs are", "The discriminator assesses")
- Avoid first person singular ("we" OK, "I" not OK)
- Be concise but complete
- Define acronyms on first use
- Number equations, figures, tables consecutively
- Refer to sections as "Section III", figures as "Fig. 2", tables as "Table I" or "Table 1"

### **Technical Writing**:
- Prioritize clarity over brevity
- Use mathematical notation consistently
- Provide intuition before equations
- Support claims with evidence (figures, tables, citations)
- Acknowledge limitations honestly

---

## 🔍 Quality Checklist

Before submission:
- [ ] All placeholders replaced
- [ ] All figures generated and inserted
- [ ] All tables filled with actual data
- [ ] All references cited in text
- [ ] Equations numbered and referenced
- [ ] Consistent notation throughout
- [ ] No orphaned/widow lines
- [ ] Proper figure/table placement
- [ ] Abstract ≤250 words
- [ ] Page limit met (check journal requirements)
- [ ] Copyright form prepared
- [ ] Supplementary materials ready (code, datasets)

---

## 📚 Resources

### **LaTeX Tips**:
- Use `\cite{}` for references
- Use `\ref{}` for cross-references
- Use `\label{}` after captions for figures/tables
- Compile 3x after adding references (LaTeX → BibTeX → LaTeX → LaTeX)

### **IEEE Templates**:
- Official IEEEtran class documentation: http://www.michaelshell.org/tex/ieeetran/
- IEEE Author Center: https://journals.ieeeauthorcenter.ieee.org/

---

## 💡 Tips dari Profesor

1. **Novelty harus jelas**: Section I (Contributions) sudah highlight 4 novelty points. Pastikan di Section III dijelaskan MENGAPA choices ini better than baseline.

2. **Ablation studies krusial**: Table 6-8 membuktikan setiap komponen contributes. Ini penting untuk reviewer skeptis.

3. **Real-world validation**: ANRI dataset (Section IV-A, Table 5) membuktikan practical applicability, bukan hanya synthetic evaluation.

4. **Honest about limitations**: Section VI-E menunjukkan scientific integrity.

5. **Reproducibility**: Appendices A-B provide enough detail untuk reproduce.

---

## ✅ Validation Status

**Struktur**: ✅ Complete & systematic
**Flow**: ✅ Logical progression Introduction → Method → Experiments → Results → Discussion → Conclusion
**Completeness**: ✅ Semua section standar IEEE journal ada
**Novelty**: ✅ Clearly articulated (dual-modal disc + frozen HTR + optimized loss)
**Evidence**: ✅ Placeholders untuk comprehensive evaluation
**Academic rigor**: ✅ Ablations, baselines, statistical tests planned

---

## 🎓 Target Journal

Paper ini cocok untuk:
1. **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)** - Q1, IF ~24
2. **Pattern Recognition** - Q1, IF ~8
3. **IEEE Transactions on Image Processing (TIP)** - Q1, IF ~10
4. **International Journal on Document Analysis and Recognition (IJDAR)** - Q2, IF ~2-3

Pilih berdasarkan:
- Scope alignment (TPAMI: broader ML/CV, IJDAR: document-specific)
- Timeline (TPAMI: slower review, IJDAR: faster)
- Impact vs. acceptance rate trade-off

---

**Good luck with your Q1 journal submission! 🚀**
