#!/usr/bin/env python3
"""
Generate Updated Chapter_Lambang_content_only.tex
=================================================
Skrip ini mengupdate halaman pada daftar singkatan dan lambang
berdasarkan kemunculan pertama di dokumen.
"""

import re
import os
from collections import OrderedDict

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
LAMBANG_FILE = os.path.join(DOCS_DIR, "Chapter_Lambang_content_only.tex")
OUTPUT_FILE = os.path.join(DOCS_DIR, "Chapter_Lambang_content_only_UPDATED.tex")

# Mapping halaman chapter dari TOC yang sudah diparsing
CHAPTER_START_PAGES = {
    'BAB I': 1,
    'BAB II': 11,
    'BAB III': 62,
    'BAB IV': 77,
    'BAB V': 102,
    'BAB VI': 151,
}

# File chapter
CHAPTERS = [
    ("chapter1_pendahuluan_content_only.tex", 'BAB I'),
    ("chapter2_tinjauan_pustaka_content_only.tex", 'BAB II'),
    ("chapter3_metodologi_content_only.tex", 'BAB III'),
    ("chapter4_analysis_design_content_only.tex", 'BAB IV'),
    ("chapter5_hasil_content_only.tex", 'BAB V'),
    ("chapter6_kesimpulan_content_only.tex", 'BAB VI'),
]

def get_chapter_lengths():
    """Hitung perkiraan panjang setiap chapter dalam halaman."""
    lengths = {}
    for filename, chapter in CHAPTERS:
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [l for l in f.readlines() if l.strip() and not l.strip().startswith('%')]
            # Perkiraan halaman: ~45 baris content per halaman
            lengths[chapter] = max(1, len(lines) // 45)
        except:
            lengths[chapter] = 10
    return lengths

def find_first_occurrence(term, strict=True):
    """
    Temukan kemunculan pertama dari term di seluruh dokumen.
    Returns: (chapter, estimated_page)
    """
    chapter_lengths = get_chapter_lengths()
    cumulative_pages = {}
    running_total = CHAPTER_START_PAGES['BAB I']
    
    for chapter in ['BAB I', 'BAB II', 'BAB III', 'BAB IV', 'BAB V', 'BAB VI']:
        cumulative_pages[chapter] = running_total
        running_total = CHAPTER_START_PAGES.get(chapter.replace('BAB ', ''), running_total) or running_total
    
    for filename, chapter_name in CHAPTERS:
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                # Skip komentar LaTeX
                if line.strip().startswith('%'):
                    continue
                
                # Build pattern based on strict mode
                if strict:
                    # Word boundary search
                    pattern = r'(?<![A-Za-z])' + re.escape(term) + r'(?![A-Za-z])'
                else:
                    pattern = re.escape(term)
                
                if re.search(pattern, line):
                    # Estimasi halaman: start_page + (line_number / 45)
                    start_page = CHAPTER_START_PAGES[chapter_name]
                    estimated_page = start_page + (i // 45)
                    return (chapter_name, estimated_page)
        except FileNotFoundError:
            continue
    
    return None

# Singkatan yang perlu dicari
SINGKATAN_LIST = [
    "AI", "ANRI", "API", "BCE", "BiGRU", "BiLSTM", "CBAM", "CER", "cGAN", "CLI",
    "CNN", "CRNN", "CTC", "CUDA", "cuDNN", "CWV", "DE-GAN", "DIBCO", "DocEnTr",
    "DRD", "DSRM", "ERB", "FCN", "FM", "Fps", "FR", "GAN", "GPU", "GT",
    "H-DIBCO", "HMM", "HTR", "IAM", "ICDAR", "IIIT5K", "KHATT", "LSTM", "MAE",
    "ML", "MLflow", "MSE", "MSFP", "NFR", "OCR", "PALM", "Pix2Pix", "PSNR",
    "RDB", "RMSProp", "RNN", "SDM", "SGD", "SNR", "SOP", "SOTA", "SSIM",
    "UNESCO", "U-Net", "VGG", "ViT", "VOC", "WER"
]

def main():
    print("=" * 80)
    print("GENERATE UPDATED DAFTAR SINGKATAN DAN LAMBANG")
    print("=" * 80)
    print()
    
    # Find first occurrence for each singkatan
    updates = {}
    for term in SINGKATAN_LIST:
        result = find_first_occurrence(term)
        if result:
            chapter, page = result
            updates[term] = page
            print(f"  {term}: halaman {page} ({chapter})")
        else:
            print(f"  {term}: TIDAK DITEMUKAN")
            updates[term] = None
    
    print()
    print("=" * 80)
    print("UPDATING FILE...")
    print("=" * 80)
    
    # Read original file
    with open(LAMBANG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update each singkatan
    changes = 0
    for term, new_page in updates.items():
        if new_page is None:
            continue
        
        # Pattern: TERM & Description & OLD_PAGE \\
        pattern = rf'^({re.escape(term)}\s*&\s*.+?&\s*)\d+(\s*\\\\)'
        replacement = rf'\g<1>{new_page}\2'
        
        new_content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if n > 0:
            content = new_content
            changes += 1
    
    # Write to output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Updated {changes} singkatan")
    print(f"📄 Output saved to: {OUTPUT_FILE}")
    print()
    print("CATATAN: File ini adalah preview. Review sebelum menerapkan ke file asli.")
    print("Untuk lambang matematis, perlu verifikasi manual karena sulit dicari secara otomatis.")

if __name__ == "__main__":
    main()
