# Quick Reference: Placeholder Filling Guide

## 🎯 Prioritas Pengisian

### **CRITICAL (Harus diisi sebelum submission)**

#### 1. Author Information (Lines ~370-380)
```latex
\author{[Your Name],~\IEEEmembership{Student Member,~IEEE,}
```
**Action**: 
- Replace `[Your Name]` dengan nama lengkap Anda
- Replace `[Supervisor Name]` dan `[Co-Supervisor Name]`
- Update affiliations: `[Your Department]`, `[Your University]`, `[City, Country]`
- Add email addresses

#### 2. Abstract Numbers (Lines ~390-410)
**Saat ini**: `PSNR of [XX.XX] dB`, `SSIM of [0.XX]`, `reduces CER by [XX]%`

**Action**: Extract dari experiment results
- File: `dual_modal_gan/docs/chapter5_results_discussion.tex` Table hasil HPO
- Atau dari MLflow runs terbaik
- Atau dari validation metrics training terbaru

**Contoh**:
```
PSNR of 28.42 dB
SSIM of 0.912
reduces CER by 24.3%
```

---

### **HIGH PRIORITY (Untuk validitas teknis)**

#### 3. Table 4: Quantitative Results (Line ~950)
**Current**: Placeholder values

**Data Source**: 
```bash
# Extract dari evaluation results
cat dual_modal_gan/logs/evaluation_synthetic_test.json
# Atau dari MLflow
mlflow ui
```

**Format**:
| Method | PSNR↑ | SSIM↑ | F-measure↑ | CER↓ | WER↓ |
|--------|-------|-------|------------|------|------|
| Degraded | 12.34 | 0.423 | 0.512 | 67.3 | 89.2 |
| Ours | **28.42** | **0.912** | **0.921** | **14.6** | **31.2** |

#### 4. Table 5: ANRI Real Results (Line ~980)
**Data Source**: 
```bash
# Inference results on ANRI dataset
ls -lh outputs/anri_evaluation_*.json
```

#### 5. Table 6-8: Ablation Studies (Lines ~1000-1050)
**Data Source**: 
- Discriminator ablation: Compare runs dengan discriminator variants
- Loss ablation: Progressive training dengan different loss combinations
- Frozen vs Joint: Compare dengan baseline HTR-GAN

**Tip**: Jika belum punya data, lakukan quick ablation:
```bash
# Train with CNN-only discriminator
poetry run python dual_modal_gan/scripts/train_enhanced.py \
  --config configs/ablation_cnn_only.json \
  --epochs 20

# Train with LSTM-only discriminator  
poetry run python dual_modal_gan/scripts/train_enhanced.py \
  --config configs/ablation_lstm_only.json \
  --epochs 20
```

---

### **MEDIUM PRIORITY (Untuk kelengkapan visual)**

#### 6. Figure 1: Degradation Examples (Line ~440)
**Action**: 
```bash
# Buat montage dari degraded images
cd data/synthetic_degraded/examples
# Pilih 6 gambar representatif untuk subfigures a-f
```

**Tools**:
- ImageMagick: `montage *.png -geometry 200x100 -tile 3x2 degradation_examples.png`
- Python: `matplotlib.pyplot.subplots(2,3)`
- PowerPoint/Keynote untuk layout manual

#### 7. Figure 2: Overall Architecture (Line ~510)
**Tool Rekomendasi**:
- **Draw.io** (https://app.diagrams.net/): Gratis, web-based
- **TikZ** (LaTeX native): Lebih profesional tapi time-consuming
- **PowerPoint**: Quick & dirty

**Konten**:
```
Input (Degraded) 
    ↓
Generator (U-Net) 
    ↓
Generated Image
    ├──→ Dual-Modal Discriminator (CNN || LSTM) → Real/Fake
    └──→ Frozen HTR Recognizer → Recognition Features
         ↓
    Multi-Component Loss (Adv + L1 + Perc + RecFeat)
```

#### 8. Figure 3: U-Net Architecture (Line ~540)
**Reference**: 
- Original U-Net paper: https://arxiv.org/abs/1505.04597
- Copy & adapt diagram, cite properly

#### 9. Figure 4: Dual-Modal Discriminator (Line ~580)
**Konten**:
```
Input: Concat(Degraded, Generated/Real) [64×512×2]
    ↓
┌─────────────────────┬──────────────────────┐
│   CNN Path          │    LSTM Path         │
│   Conv 64           │    Reshape to seq    │
│   Conv 128          │    Bi-LSTM 256       │
│   Conv 256          │    Sequential        │
│   Conv 512          │    features          │
│   Spatial features  │                      │
└─────────────────────┴──────────────────────┘
              ↓ Concatenate
         FC 512 → FC 256 → FC 1 (Sigmoid)
              ↓
         Real (1) / Fake (0)
```

#### 10. Figure 5: Synthetic Degradation Examples (Line ~900)
**Similar to Figure 1**, tapi fokus pada degradation pipeline stages:
- (a) Clean GT
- (b) + Background texture
- (c) + Gaussian noise
- (d) + Blur + All combined

#### 11. Figure 6: Qualitative Results (Line ~1075)
**Format**: Multi-row comparison
```
Row 1: Degraded Input (3 examples side-by-side)
Row 2: Sauvola result
Row 3: Standard GAN result
Row 4: DE-GAN result
Row 5: HTR-GAN result
Row 6: Ours (proposed)
Row 7: Ground Truth
```

**Pilih challenging examples** yang highlight kelebihan method:
- Thin strokes preservation
- Character connectivity
- Noise removal without text loss

#### 12. Figure 7: Failure Cases (Line ~1090)
**Konten**: 3 subfigures showing limitations
- (a) Extremely faded → hallucination artifact
- (b) Overlapping stamp → incomplete separation
- (c) Paleographic abbreviation → misreconstruction

**Tip**: Honest failure cases menunjukkan scientific integrity

---

### **LOW PRIORITY (Nice to have)**

#### 13. Additional Tables in Appendix
- Table 10: Hyperparameter configuration (Line ~1330) - Already mostly filled
- Per-degradation-level analysis - Generate dari experiments

#### 14. References Expansion
**Current**: 17 references

**Add more**:
- Recent 2023-2024 papers on document restoration
- GAN architectures (StyleGAN, Pix2Pix, CycleGAN)
- HTR systems (CRNN papers, Attention-based OCR)
- Bayesian optimization papers
- DIBCO/H-DIBCO challenge papers

**Format**:
```bibtex
\bibitem{example2024}
A. Author, B. Author, and C. Author, ``Title of paper,'' 
\textit{Journal Name}, vol. X, no. Y, pp. ZZ--ZZ, Month Year.
```

---

## 📊 Data Extraction Scripts

### Script 1: Extract Best Model Metrics
```bash
#!/bin/bash
# extract_best_metrics.sh

# Find best checkpoint based on validation CER
best_ckpt=$(ls -t dual_modal_gan/checkpoints/*/best_model/ckpt-* | head -1)

# Run evaluation
poetry run python dual_modal_gan/scripts/evaluate_model.py \
  --checkpoint "$best_ckpt" \
  --test_data data/synthetic_test \
  --output_file metrics_synthetic.json

# Pretty print results
cat metrics_synthetic.json | jq '{PSNR, SSIM, CER, WER}'
```

### Script 2: Generate Comparison Table Data
```bash
#!/bin/bash
# compare_baselines.sh

methods=("degraded" "otsu" "sauvola" "unet" "standard_gan" "degan" "htr_gan" "ours")

for method in "${methods[@]}"; do
  echo "Evaluating $method..."
  poetry run python dual_modal_gan/scripts/evaluate_baseline.py \
    --method "$method" \
    --test_data data/synthetic_test \
    --output_file "results/baseline_${method}.json"
done

# Aggregate results into table
poetry run python scripts/aggregate_results_table.py \
  --input_dir results \
  --output_latex tables/table4_synthetic_results.tex
```

### Script 3: Ablation Study Automation
```bash
#!/bin/bash
# run_ablations.sh

# Discriminator ablation
configs=("cnn_only" "lstm_only" "dual_modal")
for cfg in "${configs[@]}"; do
  poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --config "configs/ablation_disc_${cfg}.json" \
    --epochs 20 \
    --output_dir "ablations/discriminator_${cfg}"
done

# Loss ablation
losses=("pixel_only" "pixel_adv" "pixel_adv_perc" "pixel_adv_perc_rec")
for loss in "${losses[@]}"; do
  poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --config "configs/ablation_loss_${loss}.json" \
    --epochs 20 \
    --output_dir "ablations/loss_${loss}"
done
```

---

## 🎨 Figure Generation Tips

### Using Python/Matplotlib:
```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create professional figure for paper
fig = plt.figure(figsize=(7, 4))  # IEEE 2-column width
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Subfigures a-f for degradation types
for i in range(6):
    ax = fig.add_subplot(gs[i//3, i%3])
    # Load and display image
    img = load_image(f"examples/degradation_{i}.png")
    ax.imshow(img, cmap='gray')
    ax.set_title(f"({chr(97+i)}) {degradation_types[i]}")
    ax.axis('off')

plt.savefig('fig1_degradation_examples.pdf', 
            bbox_inches='tight', dpi=300)
```

### Using Draw.io:
1. Open https://app.diagrams.net/
2. Template: Engineering → Flowchart/Architecture
3. Export: File → Export as → PDF (for LaTeX)
4. Include: `\includegraphics[width=\columnwidth]{fig2_architecture.pdf}`

### Using TikZ (in LaTeX):
```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[scale=0.8]
  \node[draw, rectangle] (gen) at (0,0) {Generator};
  \node[draw, rectangle] (disc) at (3,1) {Discriminator};
  \node[draw, rectangle] (rec) at (3,-1) {Recognizer};
  
  \draw[-latex] (gen) -- (disc);
  \draw[-latex] (gen) -- (rec);
\end{tikzpicture}
\caption{Simplified architecture}
\end{figure}
```

---

## ✅ Quality Assurance Checklist

Before considering paper "complete":

### Content Completeness:
- [ ] All `[PLACEHOLDER]` text replaced
- [ ] All `[XX.XX]` numbers filled with actual data
- [ ] All author info updated
- [ ] All 7 figures created and inserted
- [ ] All 9 main tables filled
- [ ] All references cited in text exist in bibliography
- [ ] All equations numbered and referenced

### Technical Accuracy:
- [ ] Numbers in abstract match numbers in results section
- [ ] All metrics defined before use
- [ ] Consistent notation (e.g., $I_{\text{gen}}$ everywhere)
- [ ] All claims supported by evidence (figures/tables/citations)
- [ ] Ablation study results logical (incremental improvement)

### LaTeX Formatting:
- [ ] Compile without errors: `pdflatex jatniko.tex`
- [ ] All figures appear in correct locations
- [ ] All tables fit within column width
- [ ] No overfull/underfull hbox warnings (or minimized)
- [ ] Equations don't overflow margins
- [ ] References formatted correctly

### IEEE Style Compliance:
- [ ] Title capitalization correct
- [ ] Author names with proper IEEE membership
- [ ] Abstract ≤ 250 words
- [ ] Keywords relevant and specific
- [ ] Section/Figure/Table numbering consistent
- [ ] References follow IEEE format

### Writing Quality:
- [ ] No grammatical errors
- [ ] Consistent verb tense (past for your work, present for established)
- [ ] Clear and concise sentences
- [ ] Logical flow between sections
- [ ] Transitions between paragraphs
- [ ] Active voice preferred

---

## 🚀 Submission Workflow

### Pre-submission (1-2 weeks before):
1. **Internal review**: Share with advisor/colleagues
2. **Spell check**: `aspell check jatniko.tex`
3. **Plagiarism check**: Use Turnitin/iThenticate
4. **Reproducibility**: Package code/data for reviewers

### Submission preparation:
1. **PDF generation**: Final compile
2. **Source files**: Package .tex + figures + .bib
3. **Supplementary materials**: 
   - Code repository link (GitHub/GitLab)
   - Dataset access instructions
   - Pre-trained model weights
4. **Cover letter**: Explain novelty and significance

### Target Journal Selection:
Based on your work, consider:

**Tier 1 (IF > 10)**:
- IEEE TPAMI (IF ~24): Broader audience, slower review (~6-12 months)
- IEEE TIP (IF ~10): Image processing focus

**Tier 2 (IF 5-10)**:
- Pattern Recognition (IF ~8): Good fit, faster review (~4-6 months)
- IEEE Access (IF ~4): Open access, fast review (~2-3 months)

**Tier 3 (IF 2-5)**:
- IJDAR (IF ~2-3): Document-specific, very relevant
- Journal of Visual Communication (IF ~3)

**Recommendation**: Start with Pattern Recognition or IEEE TIP. If rejected, revise based on reviews and submit to IJDAR.

---

## 🔧 Troubleshooting

### LaTeX won't compile:
```bash
# Check specific error
pdflatex jatniko.tex | grep -A5 "^!"

# Clean auxiliary files
rm -f jatniko.aux jatniko.log jatniko.out

# Recompile
pdflatex jatniko.tex
```

### Figures not appearing:
```latex
% Check path
\graphicspath{{./figures/}{./images/}}

% Use correct extension
\includegraphics[width=\columnwidth]{fig1.pdf}  % Not .png
```

### References not showing:
```bash
# Full compile sequence
pdflatex jatniko.tex
bibtex jatniko
pdflatex jatniko.tex  
pdflatex jatniko.tex
```

### Table too wide:
```latex
% Scale down
\begin{table*}[t]
\centering
\scalebox{0.9}{  % 90% size
\begin{tabular}{...}
...
\end{tabular}
}
\end{table*}
```

---

## 📚 Additional Resources

### Writing:
- IEEE Author Center: https://journals.ieeeauthorcenter.ieee.org/
- "How to Write a Good Paper": https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/

### LaTeX:
- Overleaf templates: https://www.overleaf.com/gallery/tagged/ieee
- TikZ examples: https://texample.net/tikz/

### Figures:
- Matplotlib IEEE style: https://github.com/garrettj403/SciencePlots
- Draw.io templates: https://www.diagrams.net/blog/network-diagrams

### Metrics:
- SSIM implementation: `scikit-image`
- CER/WER calculation: `jiwer` library
- Statistical tests: `scipy.stats`

---

## 🎯 Final Checklist Before Submission

**72 hours before deadline**:
- [ ] All placeholders filled
- [ ] PDF compiles without errors
- [ ] Internal review completed
- [ ] All figures high-resolution (300 DPI minimum)

**48 hours before deadline**:
- [ ] Final proofreading pass
- [ ] Supplementary materials packaged
- [ ] Cover letter drafted

**24 hours before deadline**:
- [ ] Format check against journal requirements
- [ ] File size < limit (usually 10-20 MB)
- [ ] Author agreement forms signed

**Submission day**:
- [ ] Upload PDF + source files
- [ ] Submit cover letter
- [ ] Suggest reviewers (if required)
- [ ] Declare conflicts of interest
- [ ] Submit!

---

**Semoga sukses dengan publikasi jurnal Q1! 🎉📄🚀**
