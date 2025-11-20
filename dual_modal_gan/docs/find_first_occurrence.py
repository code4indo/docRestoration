#!/usr/bin/env python3
import re
import sys

# List of terms to search (singkatan)
singkatan = [
    "AI", "ANRI", "API", "BCE", "BiGRU", "BiLSTM", "CBAM", "CER", "cGAN", "CLI",
    "CNN", "CRNN", "CTC", "CUDA", "cuDNN", "CWV", "DE-GAN", "DIBCO", "DocEnTr",
    "DRD", "DSRM", "ERB", "FCN", "FM", "Fps", "FR", "GAN", "GPU", "GT",
    "H-DIBCO", "HMM", "HTR", "IAM", "ICDAR", "IIIT5K", "KHATT", "LSTM", "MAE",
    "ML", "MLflow", "MSE", "MSFP", "NFR", "OCR", "PALM", "Pix2Pix", "PSNR",
    "RDB", "RMSProp", "RNN", "SDM", "SGD", "SNR", "SOP", "SOTA", "SSIM",
    "UNESCO", "U-Net", "VGG", "ViT", "VOC", "WER"
]

# Lambang symbols (escaped for regex)
lambang = [
    r"\\alpha", r"\\beta", r"\\lambda", r"\\theta", r"\\sigma", r"\\mu",
    r"\\epsilon", r"\\Delta", r"\\pi", r"\\nabla", r"\\partial", r"\\sum",
    r"\\prod", r"\\int", "argmax", r"\\sim", r"\\mathcal\{L\}", r"\\mathcal\{B\}",
    r"\\mathbb\{E\}", r"\$G\$", r"\$D\$", r"\$x\$", r"\$y\$", r"\$z\$",
    r"\$d\$", r"\$n\$", r"\$p\$", r"\$r\$", r"\$R\^2\$", r"O\(n\)"
]

# Files to search in order
chapters = [
    "chapter1_pendahuluan_content_only.tex",
    "chapter2_tinjauan_pustaka_content_only.tex",
    "chapter3_metodologi_content_only.tex",
    "chapter4_analysis_design_content_only.tex",
    "chapter5_hasil_content_only.tex",
    "chapter6_kesimpulan_content_only.tex"
]

# Approximate starting page for each chapter
chapter_start_pages = {
    "chapter1_pendahuluan_content_only.tex": 1,
    "chapter2_tinjauan_pustaka_content_only.tex": 10,
    "chapter3_metodologi_content_only.tex": 50,
    "chapter4_analysis_design_content_only.tex": 80,
    "chapter5_hasil_content_only.tex": 120,
    "chapter6_kesimpulan_content_only.tex": 170
}

results = {}

# Search for each term
for term in singkatan + lambang:
    found = False
    for chapter in chapters:
        try:
            with open(chapter, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple search - check if term appears
                if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                    results[term] = chapter_start_pages[chapter]
                    found = True
                    break
        except FileNotFoundError:
            continue
    
    if not found:
        results[term] = ""  # Not found

# Print results
print("SINGKATAN:")
for term in singkatan:
    page = results.get(term, "")
    print(f"{term}: {page}")

print("\nLAMBANG:")
for term in lambang:
    page = results.get(term, "")
    print(f"{term}: {page}")

