#!/bin/bash

# Script untuk kompilasi dokumen tesis
# Lokasi: dual_modal_gan/docs/compile_tesis.sh

set -e  # Exit on error

DOCS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DOCS_DIR"

echo "=================================================="
echo "  Kompilasi Dokumen Tesis"
echo "=================================================="
echo ""

# Fungsi untuk kompilasi lengkap
compile_full() {
    echo "� Sinkronisasi semua chapter..."
    echo ""
    
    # Sync all chapters first
    ./sync_all_chapters.sh || {
        echo "❌ Error saat sinkronisasi chapter. Jalankan manual:"
        echo "   ./sync_all_chapters.sh"
        exit 1
    }
    
    echo ""
    echo "�📄 Mengkompilasi dokumen lengkap (main_tesis.tex)..."
    echo ""
    
    # First pass
    echo "▶️  Pass 1/4: Kompilasi awal..."
    pdflatex -interaction=nonstopmode main_tesis.tex > /dev/null 2>&1 || {
        echo "❌ Error pada pass 1. Jalankan manual untuk melihat error:"
        echo "   pdflatex main_tesis.tex"
        exit 1
    }
    
    # Run biber for bibliography processing
    echo "▶️  Pass 2/4: Memproses bibliography dengan biber..."
    biber main_tesis > /dev/null 2>&1 || {
        echo "⚠️  Warning: biber gagal. Melanjutkan tanpa bibliography..."
    }
    
    # Second pass (untuk update references)
    echo "▶️  Pass 3/4: Update references..."
    pdflatex -interaction=nonstopmode main_tesis.tex > /dev/null 2>&1
    
    # Third pass (untuk finalisasi)
    echo "▶️  Pass 4/4: Finalisasi dokumen..."
    pdflatex -interaction=nonstopmode main_tesis.tex > /dev/null 2>&1
    
    # Cleanup auxiliary files
    echo "🧹 Membersihkan file temporary..."
    rm -f *.aux *.log *.toc *.lof *.lot *.out *.synctex.gz 2>/dev/null
    
    echo ""
    echo "✅ Kompilasi selesai!"
    echo "📁 Output: main_tesis.pdf"
    echo ""
    
    # Show file info
    if [ -f "main_tesis.pdf" ]; then
        SIZE=$(du -h main_tesis.pdf | cut -f1)
        echo "   Ukuran file: $SIZE"
        PAGES=$(pdfinfo main_tesis.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
        if [ ! -z "$PAGES" ]; then
            echo "   Jumlah halaman: $PAGES"
        fi
    fi
}

# Fungsi untuk kompilasi chapter individual
compile_chapter() {
    CHAPTER=$1
    echo "📄 Mengkompilasi chapter individual: $CHAPTER"
    echo ""
    
    if [ ! -f "$CHAPTER" ]; then
        echo "❌ File tidak ditemukan: $CHAPTER"
        exit 1
    fi
    
    echo "▶️  Kompilasi $CHAPTER..."
    pdflatex -interaction=nonstopmode "$CHAPTER" > /dev/null 2>&1 || {
        echo "❌ Error. Jalankan manual untuk melihat error:"
        echo "   pdflatex $CHAPTER"
        exit 1
    }
    
    # Second pass
    pdflatex -interaction=nonstopmode "$CHAPTER" > /dev/null 2>&1
    
    # Cleanup
    BASENAME="${CHAPTER%.tex}"
    rm -f "${BASENAME}.aux" "${BASENAME}.log" "${BASENAME}.out" "${BASENAME}.synctex.gz" 2>/dev/null
    
    echo ""
    echo "✅ Kompilasi selesai!"
    echo "📁 Output: ${BASENAME}.pdf"
    echo ""
}

# Fungsi cleanup semua file temporary
cleanup_all() {
    echo "🧹 Membersihkan semua file temporary..."
    rm -f *.aux *.log *.toc *.lof *.lot *.out *.synctex.gz 2>/dev/null
    echo "✅ Cleanup selesai!"
}

# Main menu
case "${1:-help}" in
    full|lengkap)
        compile_full
        ;;
    
    chapter1|ch1)
        compile_chapter "chapter1_pendahuluan.tex"
        ;;
    
    chapter2|ch2)
        compile_chapter "chapter2_tinjauan_pustaka.tex"
        ;;
    
    chapter3|ch3)
        compile_chapter "chapter3_metodologi.tex"
        ;;
    
    chapter4|ch4)
        compile_chapter "chapter4_analysis_design.tex"
        ;;
    
    chapter5|ch5)
        compile_chapter "chapter5_hasil.tex"
        ;;
    
    chapter6|ch6)
        compile_chapter "chapter6_kesimpulan.tex"
        ;;
    
    clean|cleanup)
        cleanup_all
        ;;
    
    help|--help|-h|*)
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  full, lengkap    - Kompilasi dokumen lengkap (main_tesis.pdf)"
        echo "  chapter1, ch1    - Kompilasi Chapter 1"
        echo "  chapter2, ch2    - Kompilasi Chapter 2"
        echo "  chapter3, ch3    - Kompilasi Chapter 3"
        echo "  chapter4, ch4    - Kompilasi Chapter 4"
        echo "  chapter5, ch5    - Kompilasi Chapter 5"
        echo "  chapter6, ch6    - Kompilasi Chapter 6"
        echo "  clean, cleanup   - Hapus file temporary"
        echo "  help, --help     - Tampilkan help ini"
        echo ""
        echo "Contoh:"
        echo "  $0 full          # Kompilasi lengkap"
        echo "  $0 ch5           # Kompilasi Chapter 5 saja"
        echo "  $0 clean         # Cleanup file temporary"
        ;;
esac
