#!/usr/bin/env python3
"""
Professional Indonesian to English translator for IEEE journal paper
Preserves LaTeX structure and technical terminology
"""

import re
import sys

# Translation dictionary for common academic phrases
TRANSLATIONS = {
    # Section headers
    "Pendahuluan": "Introduction",
    "Pekerjaan Terkait": "Related Work",
    "Metodologi": "Methodology",
    "Metodologi Penelitian": "Research Methodology",
    "Pengaturan Eksperimental": "Experimental Setup",
    "Hasil": "Results",
    "Hasil dan Analisis": "Results and Analysis",
    "Diskusi": "Discussion",
    "Kesimpulan": "Conclusion",
    "Keterbatasan dan Arah Penelitian Masa Depan": "Limitations and Future Research Directions",
    
    # Common phrases
    "Bagian": "Section",
    "Gambar": "Fig.",
    "Tabel": "Table",
    "Persamaan": "Equation",
    "sebagaimana": "as",
    "sebagaimana ditunjukkan": "as shown",
    "seperti yang ditunjukkan": "as shown",
    "dapat dilihat": "can be seen",
    "menunjukkan bahwa": "shows that",
    "mengindikasikan bahwa": "indicates that",
    "mengonfirmasi bahwa": "confirms that",
    "memvalidasi bahwa": "validates that",
    
    # Technical terms
    "Arsip Nasional Republik Indonesia": "National Archives of the Republic of Indonesia",
    "fungsi kehilangan": "loss function",
    "kehilangan": "loss",
    "pengenal": "recognizer",
    "penggenal": "recognizer",
    "diskriminator": "discriminator",
    "pembangkit": "generator",
    "studi ablasi": "ablation study",
    "set pelatihan": "training set",
    "set validasi": "validation set",
    "set uji": "test set",
    "himpunan pelatihan": "training set",
    "himpunan validasi": "validation set",
    "himpunan uji": "test set",
    "citra": "image",
    "tulisan tangan": "handwritten",
    "dokumen historis": "historical document",
    "dokumen paleografi": "paleographic document",
    "restorasi dokumen": "document restoration",
    "peningkatan dokumen": "document enhancement",
    "degradasi": "degradation",
    "terdegradasi": "degraded",
    "goresan": "stroke",
    "ligatur": "ligature",
    "konvergensi": "convergence",
    "stabilitas pelatihan": "training stability",
    "kualitas visual": "visual quality",
    "keterbacaan": "readability",
    "akurasi pengenalan": "recognition accuracy",
}

def translate_line(line):
    """Translate a single line while preserving LaTeX structure"""
    # Don't translate comments
    if line.strip().startswith('%'):
        return line
    
    # Don't translate LaTeX commands (but translate their content)
    # This is simplified - full implementation would need more sophisticated parsing
    
    # Apply translations
    result = line
    for indo, eng in sorted(TRANSLATIONS.items(), key=lambda x: -len(x[0])):
        # Case-sensitive replacement
        result = result.replace(indo, eng)
    
    return result

def main():
    input_file = "Paper/main/jatniko_id.tex"
    output_file = "Paper/english_version/paper_english_auto.tex"
    
    print(f"Translating {input_file} to {output_file}...")
    print("Note: This is a basic translation. Manual review required for IEEE submission.")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for line_num, line in enumerate(f_in, 1):
                    translated = translate_line(line)
                    f_out.write(translated)
                    
                    if line_num % 500 == 0:
                        print(f"  Processed {line_num} lines...")
        
        print(f"\n✓ Translation complete: {output_file}")
        print("⚠ IMPORTANT: This is automated translation. Professional review needed!")
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
