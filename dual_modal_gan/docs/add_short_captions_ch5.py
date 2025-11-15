#!/usr/bin/env python3
"""
Script untuk menambahkan short caption ke semua caption panjang di chapter5_hasil.tex
"""

import re
import sys

def process_chapter5_captions(input_file, output_file):
    """Process chapter 5 and add short captions"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary of short captions
    replacements = [
        # Line 278 - SOTA comparison
        (
            r'\\caption\{Perbandingan Indikatif dengan Metode \\textit\{State-of-the-Art\} Berdasarkan Literatur\}',
            r'\\caption[Perbandingan dengan metode state-of-the-art]{Perbandingan Indikatif dengan Metode \\textit{State-of-the-Art} Berdasarkan Literatur}'
        ),
        
        # Line 350 - Ablasi loss table
        (
            r'\\caption\{Studi Ablasi Inkremental Komponen Fungsi \\textit\{Loss\}',
            r'\\caption[Studi ablasi komponen loss]{Studi Ablasi Inkremental Komponen Fungsi \\textit{Loss}'
        ),
        
        # Line 487 - Loss trajectory comparison table
        (
            r'\\caption\{Perbandingan Trajektori \\textit\{Loss\} antara \\textit\{Frozen Recognizer\} dan \\textit\{Joint Training\}\}',
            r'\\caption[Perbandingan loss frozen vs joint training]{Perbandingan Trajektori \\textit{Loss} antara \\textit{Frozen Recognizer} dan \\textit{Joint Training}}'
        ),
        
        # Line 500 - Loss trajectory figure
        (
            r'\\caption\{Perbandingan trajektori \\textit\{loss\} antara \\textit\{frozen recognizer\} dan \\textit\{joint training\} selama 20 \\textit\{epoch\}\. \(a\) Generator \\textit\{loss\}',
            r'\\caption[Trajektori loss frozen vs joint training]{Perbandingan trajektori \\textit{loss} antara \\textit{frozen recognizer} dan \\textit{joint training} selama 20 \\textit{epoch}. (a) Generator \\textit{loss}'
        ),
        
        # Line 514 - Readability comparison table
        (
            r'\\caption\{Perbandingan Metrik Keterbacaan Teks antara \\textit\{Frozen Recognizer\} dan \\textit\{Joint Training\}\}',
            r'\\caption[Metrik keterbacaan frozen vs joint]{Perbandingan Metrik Keterbacaan Teks antara \\textit{Frozen Recognizer} dan \\textit{Joint Training}}'
        ),
        
        # Line 536 - CER comparison figure
        (
            r'\\caption\{Perbandingan Character Error Rate \(CER\) antara \\textit\{frozen recognizer\} dan \\textit\{joint training\} selama 20 \\textit\{epoch\}\. \\textit\{Frozen recognizer\} mempertahankan CER',
            r'\\caption[Perbandingan CER frozen vs joint]{Perbandingan Character Error Rate (CER) antara \\textit{frozen recognizer} dan \\textit{joint training} selama 20 \\textit{epoch}. \\textit{Frozen recognizer} mempertahankan CER'
        ),
        
        # Line 545 - PSNR comparison figure
        (
            r'\\caption\{Perbandingan kualitas visual \(PSNR\) antara \\textit\{frozen recognizer\} dan \\textit\{joint training\}\. \\textit\{Frozen recognizer\} mempertahankan PSNR',
            r'\\caption[Perbandingan PSNR frozen vs joint]{Perbandingan kualitas visual (PSNR) antara \\textit{frozen recognizer} dan \\textit{joint training}. \\textit{Frozen recognizer} mempertahankan PSNR'
        ),
        
        # Line 564 - Computational efficiency table
        (
            r'\\caption\{Perbandingan Efisiensi Komputasi antara \\textit\{Frozen Recognizer\} dan \\textit\{Joint Training\}\}',
            r'\\caption[Efisiensi komputasi frozen vs joint]{Perbandingan Efisiensi Komputasi antara \\textit{Frozen Recognizer} dan \\textit{Joint Training}}'
        ),
        
        # Line 594 - Computational efficiency figure
        (
            r'\\caption\{Perbandingan efisiensi komputasi antara \\textit\{frozen recognizer\} dan \\textit\{joint training\}\. \(a\) Waktu pelatihan per \\textit\{epoch\}:',
            r'\\caption[Efisiensi komputasi frozen vs joint]{Perbandingan efisiensi komputasi antara \\textit{frozen recognizer} dan \\textit{joint training}. (a) Waktu pelatihan per \\textit{epoch}:'
        ),
        
        # Line 621 - Radar diagram
        (
            r'\\caption\{Perbandingan multidimensi antara \\textit\{frozen recognizer\} dan \\textit\{joint training\} menggunakan diagram radar\. Lima dimensi evaluasi',
            r'\\caption[Diagram radar frozen vs joint]{Perbandingan multidimensi antara \\textit{frozen recognizer} dan \\textit{joint training} menggunakan diagram radar. Lima dimensi evaluasi'
        ),
        
        # Line 646 - Curriculum learning comparison
        (
            r'\\caption\{Perbandingan Performa Curriculum Learning vs Non-Curriculum Learning\}',
            r'\\caption[Performa curriculum vs non-curriculum]{Perbandingan Performa Curriculum Learning vs Non-Curriculum Learning}'
        ),
        
        # Line 827 - Loss weights configuration
        (
            r'\\caption\{Konfigurasi Loss Weights Production Training dengan Analisis Magnitude\}',
            r'\\caption[Konfigurasi loss weights production]{Konfigurasi Loss Weights Production Training dengan Analisis Magnitude}'
        ),
        
        # Line 1011 - Inverse scaling mechanism
        (
            r'\\caption\{Mekanisme \\textit\{inverse scaling\} yang muncul dari tuning manual bobot komponen \\textit\{loss\}\. Panel kiri menunjukkan',
            r'\\caption[Mekanisme inverse scaling pada loss]{Mekanisme \\textit{inverse scaling} yang muncul dari tuning manual bobot komponen \\textit{loss}. Panel kiri menunjukkan'
        ),
        
        # Line 1032 - Grid search results
        (
            r'\\caption\{Hasil Mini Grid Search: Top 5 Konfigurasi \(3 Epoch Ablation\)\}',
            r'\\caption[Hasil mini grid search top 5]{Hasil Mini Grid Search: Top 5 Konfigurasi (3 Epoch Ablation)}'
        ),
        
        # Line 1059 - Pixel weight sensitivity
        (
            r'\\caption\{Analisis Sensitivitas Pixel Weight \(Mini Grid Search\)\}',
            r'\\caption[Sensitivitas pixel weight]{Analisis Sensitivitas Pixel Weight (Mini Grid Search)}'
        ),
        
        # Line 1079 - CTC weight impact
        (
            r'\\caption\{Analisis Dampak CTC Weight Terhadap Kualitas Visual\}',
            r'\\caption[Dampak CTC weight pada kualitas visual]{Analisis Dampak CTC Weight Terhadap Kualitas Visual}'
        ),
        
        # Line 1099 - RecFeat weight analysis
        (
            r'\\caption\{Analisis RecFeat Weight: Ablation vs Production Training\}',
            r'\\caption[Analisis RecFeat weight]{Analisis RecFeat Weight: Ablation vs Production Training}'
        ),
        
        # Line 1268 - ANRI performance
        (
            r'\\caption\{Performa Restorasi pada Dokumen ANRI Berdasarkan Tingkat Kontras Masukan \(n=15\)\}',
            r'\\caption[Performa restorasi dokumen ANRI]{Performa Restorasi pada Dokumen ANRI Berdasarkan Tingkat Kontras Masukan (n=15)}'
        ),
        
        # Line 1333 - Expert validation
        (
            r'\\caption\{Hasil Validasi Ahli Paleografi pada Dokumen ANRI \(n=15\)\}',
            r'\\caption[Validasi ahli paleografi]{Hasil Validasi Ahli Paleografi pada Dokumen ANRI (n=15)}'
        ),
        
        # Line 1386 - Degradation correlation
        (
            r'\\caption\{Analisis korelasi tingkat degradasi pada \\textit\{set\} validasi \(n=710\): \(a\) SNR',
            r'\\caption[Korelasi tingkat degradasi]{Analisis korelasi tingkat degradasi pada \\textit{set} validasi (n=710): (a) SNR'
        ),
        
        # Line 1690 - Hypothesis evaluation
        (
            r'\\caption\{Ringkasan Evaluasi Komponen Hipotesis Penelitian\}',
            r'\\caption[Evaluasi komponen hipotesis]{Ringkasan Evaluasi Komponen Hipotesis Penelitian}'
        ),
        
        # Line 1785 - Failure distribution
        (
            r'\\caption\{Distribusi Pola Kegagalan pada \\textit\{Test Set\} \(\$n\$=712\)\}',
            r'\\caption[Distribusi pola kegagalan]{Distribusi Pola Kegagalan pada \\textit{Test Set} ($n$=712)}'
        ),
    ]
    
    # Apply replacements
    modified_count = 0
    for pattern, replacement in replacements:
        old_content = content
        content = re.sub(pattern, replacement, content, count=1)
        if content != old_content:
            modified_count += 1
            print(f"✓ Modified caption: {pattern[:50]}...")
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ Total {modified_count} captions modified")
    print(f"✓ Output saved to: {output_file}")
    
    return modified_count

if __name__ == "__main__":
    input_file = "chapter5_hasil.tex"
    output_file = "chapter5_hasil.tex"
    
    # Create backup
    import shutil
    backup_file = "chapter5_hasil_backup_shortcaptions.tex"
    shutil.copy(input_file, backup_file)
    print(f"✓ Backup created: {backup_file}\n")
    
    # Process
    modified = process_chapter5_captions(input_file, output_file)
    
    if modified > 0:
        print(f"\n✓ Successfully modified {modified} captions in {input_file}")
        print("✓ Next step: Run ./compile_tesis.sh full")
    else:
        print("\n✗ No captions were modified!")
        sys.exit(1)
