#!/usr/bin/env python3
"""
Script untuk menambahkan short caption pada semua caption panjang di LaTeX.
Mengubah \caption{long} menjadi \caption[short]{long}
"""

import re
import sys

def generate_short_caption(long_caption):
    """Generate versi pendek dari caption panjang"""
    
    # Dictionary mapping untuk pola umum
    short_versions = {
        r'Hasil Kuantitatif pada Set Uji Semi-Sintetis.*': 'Hasil kuantitatif pada set uji semi-sintetis',
        r'Trajektori metrik validasi dengan interval.*': 'Trajektori metrik validasi selama 50 epoch',
        r'Perbandingan visual metode restorasi pada.*': 'Perbandingan visual metode restorasi',
        r'Studi Ablasi Inkremental Komponen Fungsi.*': 'Studi ablasi inkremental komponen fungsi loss',
        r'Perbandingan trajektori.*frozen recognizer.*joint training.*': 'Perbandingan trajektori loss frozen vs joint training',
        r'Perbandingan Character Error Rate.*frozen.*joint.*': 'Perbandingan CER frozen vs joint training',
        r'Perbandingan kualitas visual.*frozen.*joint.*': 'Perbandingan PSNR frozen vs joint training',
        r'Perbandingan efisiensi komputasi.*frozen.*joint.*': 'Perbandingan efisiensi komputasi frozen vs joint',
        r'Perbandingan multidimensi.*frozen.*joint.*diagram radar.*': 'Diagram radar perbandingan frozen vs joint training',
        r'Analisis mendalam komponen loss:.*curriculum learning.*': 'Analisis komponen loss curriculum vs non-curriculum',
        r'Analisis statistik:.*distribusi PSNR.*': 'Analisis statistik curriculum learning',
        r'Analisis fase curriculum learning:.*jadwal.*': 'Analisis fase curriculum learning',
        r'Analisis scientific lanjutan:.*PSNR dengan confidence.*': 'Analisis scientific curriculum learning',
        r'Evolusi bobot GradNorm sepanjang.*': 'Evolusi bobot GradNorm',
        r'Perbandingan metrik PSNR, SSIM, dan CER.*GradNorm.*': 'Perbandingan metrik GradNorm vs manual',
        r'Validasi optimalitas bobot.*GradNorm.*': 'Validasi optimalitas bobot loss dengan GradNorm',
        r'Mekanisme.*inverse scaling.*tuning manual.*': 'Mekanisme inverse scaling pada bobot loss',
        r'Restorasi kualitatif pada dokumen ANRI.*abad ke-17.*': 'Restorasi kualitatif dokumen ANRI autentik',
    }
    
    # Cek apakah ada pola yang cocok
    for pattern, short in short_versions.items():
        if re.match(pattern, long_caption, re.DOTALL | re.IGNORECASE):
            return short
    
    # Fallback: ambil 60 karakter pertama dan bersihkan
    if len(long_caption) > 60:
        short = long_caption[:60]
        # Cari kata terakhir yang utuh
        last_space = short.rfind(' ')
        if last_space > 30:
            short = short[:last_space]
        return short.strip()
    
    return long_caption

def process_latex_file(input_file, output_file):
    """Proses file LaTeX dan tambahkan short caption"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern untuk menangkap \caption{...} dengan nested braces
    # Menggunakan recursive approach untuk handle nested brackets
    def find_captions(text):
        """Find all caption commands with proper brace matching"""
        results = []
        i = 0
        while i < len(text):
            # Look for \caption{
            if text[i:i+9] == r'\caption{':
                # Check if it's already short caption
                if i > 0 and text[i-1] == ']':
                    # Already has short caption, skip
                    i += 9
                    continue
                
                start = i
                i += 9  # Move past \caption{
                
                # Find matching closing brace
                brace_count = 1
                caption_start = i
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                
                if brace_count == 0:
                    caption_text = text[caption_start:i-1]
                    results.append((start, i, caption_text))
            else:
                i += 1
        
        return results
    
    captions = find_captions(content)
    
    # Replace from end to start to preserve positions
    modified_content = content
    modifications = 0
    
    for start, end, caption_text in reversed(captions):
        # Skip if caption is short
        if len(caption_text) < 80:
            continue
        
        # Generate short caption
        short_caption = generate_short_caption(caption_text)
        
        # Build replacement
        replacement = f'\\caption[{short_caption}]{{{caption_text}}}'
        
        # Replace in content
        modified_content = modified_content[:start] + replacement + modified_content[end:]
        modifications += 1
    
    # Tulis ke output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ Selesai!")
    print(f"   Total caption ditemukan: {len(captions)}")
    print(f"   Caption dimodifikasi: {modifications}")
    print(f"   Output: {output_file}")

if __name__ == "__main__":
    input_file = "chapter5_hasil.tex"
    output_file = "chapter5_hasil_with_short_captions.tex"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"📝 Memproses {input_file}...")
    process_latex_file(input_file, output_file)
