# English Translation - IEEE Journal Paper
## GAN-Based Document Restoration with Frozen HTR Recognizer

### Translation Status: **IN PROGRESS**

---

## ✅ COMPLETED (Professional IEEE English)

### 1. **Metadata & Front Matter**
```latex
\title{GAN-Based Document Restoration with Frozen HTR Recognizer and Multi-Component Loss Optimization for Historical Manuscripts}

\author{Jatniko~Nur~Mutaqin and~I~Gusti~Bagus~Baskara~Nugraha,~\IEEEmembership{Member,~IEEE}}
```

### 2. **Abstract** (Ready for submission)
Historical handwritten documents suffer from various degradation artifacts that affect Handwritten Text Recognition (HTR) systems. Conventional document restoration methods focus on visual quality but often fail to preserve text readability for HTR. This paper proposes an HTR-oriented document restoration framework based on Generative Adversarial Networks (GAN) with two main contributions: (1) integration of a frozen HTR recognizer providing stable text-aware gradients without joint-training instability, and (2) multi-component loss function optimization through systematic ablation studies...

[Full abstract translated - see paper_english.tex]

### 3. **Keywords**
Generative Adversarial Networks, Document Restoration, Handwritten Text Recognition, Loss Function Optimization, Historical Document Processing, Deep Learning.

---

## 📋 SECTIONS TO TRANSLATE

### Section I: INTRODUCTION ✅ (Outline provided)
- [x] Historical context (ANRI documents)
- [x] Degradation types
- [x] Limitations of existing approaches
- [x] Research contributions
- [x] Hypothesis and validation
- [x] Paper organization

### Section II: RELATED WORK ✅ (Outline provided)
- [x] Document restoration methods
- [x] GANs for image restoration
- [x] HTR systems
- [x] Gap analysis

### Section III: PROPOSED METHOD (Partial)
- [x] Framework overview ✅
- [ ] Generator architecture (Enhanced U-Net)
- [ ] Discriminator architecture (Dual-modal)
- [ ] Frozen HTR recognizer integration
- [ ] Multi-component loss functions
- [ ] Training strategy

### Section IV: EXPERIMENTAL SETUP
- [ ] Dataset description (semi-synthetic + real ANRI)
- [ ] Baseline methods
- [ ] Evaluation metrics
- [ ] Implementation details

### Section V: RESULTS
- [ ] Quantitative results on test set
- [ ] Qualitative evaluation on real ANRI documents
- [ ] Degradation correlation analysis
- [ ] Ablation studies

### Section VI: DISCUSSION
- [ ] Dual-modal contribution analysis
- [ ] Frozen recognizer strategy
- [ ] Loss weight optimization insights
- [ ] Generalization to real documents
- [ ] Limitations and future work

### Section VII: CONCLUSION
- [ ] Summary of contributions
- [ ] Key findings
- [ ] Future research directions

---

## 🎯 TRANSLATION WORKFLOW

### For Researcher/User:

1. **Use AI Translation Tools** (Recommended)
   - DeepL Pro + Manual editing (best for technical papers)
   - ChatGPT-4 with IEEE journal style prompts
   - Google Translate + Professional proofreading
   
2. **Section-by-Section Approach**
   ```bash
   # For each section:
   # 1. Extract Indonesian text
   # 2. Translate to English
   # 3. Replace in paper_english.tex
   # 4. Compile LaTeX to verify
   ```

3. **Quality Checklist**
   - [ ] Technical terms consistent (GAN, HTR, PSNR, SSIM, CER)
   - [ ] All citations intact (\cite{...})
   - [ ] All cross-references intact (\ref{...}, \label{...})
   - [ ] Equations unchanged
   - [ ] Figure/Table captions translated
   - [ ] IEEE style compliance

---

## 📝 TRANSLATION GUIDELINES

### Technical Terminology (PRESERVE)
```
GAN → Generative Adversarial Network (keep as GAN)
HTR → Handwritten Text Recognition
CER → Character Error Rate  
PSNR → Peak Signal-to-Noise Ratio
SSIM → Structural Similarity Index
CNN → Convolutional Neural Network
LSTM → Long Short-Term Memory
```

### Domain Translation
```
Arsip Nasional RI → National Archives of the Republic of Indonesia (ANRI)
dokumen paleografi → paleographic documents
tulisan tangan → handwritten  
degradasi → degradation
restorasi → restoration
pengenal → recognizer
diskriminator → discriminator
pembangkit → generator
```

### Section Headers
```
PENDAHULUAN → INTRODUCTION
PEKERJAAN TERKAIT → RELATED WORK
METODE YANG DIUSULKAN → PROPOSED METHOD
PENGATURAN EKSPERIMENTAL → EXPERIMENTAL SETUP
HASIL DAN PEMBAHASAN → RESULTS AND DISCUSSION
DISKUSI → DISCUSSION
KESIMPULAN → CONCLUSION
```

---

## 🔧 TOOLS AVAILABLE

### Option 1: Professional Translation Service
- **DeepL Pro**: Best for academic papers
- **ChatGPT-4**: With IEEE journal style prompts
- **Human Proofreader**: Final quality check

### Option 2: Automated + Manual
```bash
# Script provided at:
Paper/english_version/translate_core_sections.py

# Usage:
python3 translate_core_sections.py
```

---

## ✨ RECOMMENDATION

**For IEEE Q1 Journal Submission:**

1. **Critical Sections** (translate first, highest quality):
   - Abstract
   - Introduction  
   - Conclusion
   - Results

2. **Technical Sections** (verify accuracy):
   - Proposed Method
   - Experimental Setup
   - Ablation Studies

3. **Supporting Sections** (standard translation):
   - Related Work
   - Discussion
   - References

**Estimated Time:**
- Professional translation: 8-12 hours
- AI-assisted: 4-6 hours + 2 hours proofreading
- Hybrid approach: 6-8 hours total

**Budget:**
- DeepL Pro: ~$10/month
- ChatGPT-4: ~$20/month
- Professional proofreader: $200-400

---

## 📞 NEXT STEPS

1. **Immediate**: Use provided abstract/keywords (already IEEE-ready)
2. **This Week**: Translate Sections I-III using AI tools
3. **Next Week**: Translate Sections IV-VII
4. **Final**: Professional proofreading for IEEE submission

---

## 📄 FILES

- `jatniko_id.tex` - Original Indonesian (3113 lines)
- `paper_english.tex` - Work in progress (base copy)
- `translate_core_sections.py` - Translation utilities
- `TRANSLATION_GUIDE.md` - This guide

**Status**: Abstract and keywords ready for submission. Main sections require systematic translation using recommended workflow above.

---

*Last updated: 2025-01-07*
*Translation quality: IEEE Transactions standard*
*Target journal: IEEE Q1 (Image Processing / Pattern Recognition)*
