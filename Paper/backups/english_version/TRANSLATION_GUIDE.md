# Professional English Translation - IEEE Paper
## Translation Status

### ✅ COMPLETED SECTIONS:
1. **Title & Metadata**: "GAN-Based Document Restoration with Frozen HTR Recognizer and Multi-Component Loss Optimization for Historical Manuscripts"

2. **Abstract** (Lines ~1140-1180): Fully translated - HTR-oriented restoration framework, frozen recognizer strategy, multi-component loss

3. **Keywords**: "Generative Adversarial Networks, Document Restoration, Handwritten Text Recognition, Loss Function Optimization, Historical Document Processing, Deep Learning"

### 🔄 IN PROGRESS:
All main sections need systematic translation following the pattern provided by previous subagent:
- Section I: PENDAHULUAN → INTRODUCTION (partial)
- Section II: PEKERJAAN TERKAIT → RELATED WORK (partial)  
- Section III: METODE → PROPOSED METHOD (partial)
- Section IV: PENGATURAN EKSPERIMENTAL → EXPERIMENTAL SETUP
- Section V: HASIL DAN PEMBAHASAN → RESULTS AND DISCUSSION
- Section VI: DISKUSI → DISCUSSION  
- Section VII: KESIMPULAN → CONCLUSION

### 📋 KEY TRANSLATION PRINCIPLES:
1. **Technical Terms (KEEP English)**:
   - Generative Adversarial Networks (GAN)
   - Handwritten Text Recognition (HTR)
   - Character Error Rate (CER)
   - Word Error Rate (WER)
   - Peak Signal-to-Noise Ratio (PSNR)
   - Structural Similarity Index (SSIM)
   - Convolutional Neural Network (CNN)
   - Long Short-Term Memory (LSTM)
   - Transformer
   - Attention mechanism
   - Deep learning

2. **Domain-Specific (translate context)**:
   - "Arsip Nasional Republik Indonesia" → "National Archives of the Republic of Indonesia (ANRI)"
   - "dokumen paleografi" → "paleographic documents"
   - "tulisan tangan" → "handwritten"
   - "degradasi" → "degradation"
   - "restorasi" → "restoration"

3. **Section Headers Translation**:
   ```
   PENDAHULUAN → INTRODUCTION
   PEKERJAAN TERKAIT → RELATED WORK
   METODE YANG DIUSULKAN → PROPOSED METHOD
   PENGATURAN EKSPERIMENTAL → EXPERIMENTAL SETUP
   HASIL DAN PEMBAHASAN → RESULTS AND DISCUSSION
   DISKUSI → DISCUSSION
   KESIMPULAN → CONCLUSION
   PENGHARGAAN → ACKNOWLEDGMENTS
   ```

### 🎯 NEXT STEPS:
Due to file length (3113 lines), professional translation requires:
1. Section-by-section replacement using multi_replace_string_in_file
2. Verify LaTeX compilation after each section
3. Final proofreading for consistency

**Note**: Base file `paper_english.tex` currently contains Indonesian text copied from `main/jatniko_id.tex`. Systematic translation in progress.

**Current Approach**: Using multi-stage replacement to convert Indonesian → English while preserving LaTeX structure.
