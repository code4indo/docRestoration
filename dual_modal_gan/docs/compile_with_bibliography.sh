#!/bin/bash
# Script untuk kompilasi LaTeX dengan bibliografi terpusat
# Usage: ./compile_with_bibliography.sh [main_tesis|chapter5_hasil|chapter4_analysis_design]

set -e

FILE=${1:-main_tesis}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================="
echo "Kompilasi LaTeX dengan Bibliografi Terpusat"
echo "File: $FILE.tex"
echo "========================================="

# First pass
echo "[1/4] Kompilasi LaTeX (Pass 1)..."
pdflatex -interaction=nonstopmode "$FILE.tex" > /dev/null 2>&1 || true

# Run biber
echo "[2/4] Memproses bibliografi dengan Biber..."
biber "$FILE" 2>&1 | grep -E "(INFO|WARN|ERROR)" | tail -10

# Second pass
echo "[3/4] Kompilasi LaTeX (Pass 2)..."
pdflatex -interaction=nonstopmode "$FILE.tex" > /dev/null 2>&1 || true

# Final pass
echo "[4/4] Kompilasi LaTeX (Pass 3 - Final)..."
pdflatex -interaction=nonstopmode "$FILE.tex" 2>&1 | tail -20

echo ""
echo "========================================="
echo "✓ Kompilasi selesai!"
echo "Output: $FILE.pdf"
echo "========================================="

# Check for errors
if grep -q "LaTeX Warning: There were undefined references" "$FILE.log" 2>/dev/null; then
    echo "⚠ WARNING: Ada referensi yang tidak terdefinisi"
    grep "LaTeX Warning.*undefined" "$FILE.log" | head -10
fi

if grep -q "LaTeX Warning: Citation.*undefined" "$FILE.log" 2>/dev/null; then
    echo "⚠ WARNING: Ada sitasi yang tidak terdefinisi"
    grep "Citation.*undefined" "$FILE.log" | head -10
fi

# Show statistics
echo ""
echo "Statistik:"
echo "- Total halaman: $(pdfinfo "$FILE.pdf" 2>/dev/null | grep Pages | awk '{print $2}')"
echo "- Ukuran file: $(ls -lh "$FILE.pdf" | awk '{print $5}')"
echo ""
