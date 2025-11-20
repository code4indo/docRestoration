#!/bin/bash
# ============================================================================
# COMPILE THESIS WITH AUTOMATIC "ET AL" → "DKK" REPLACEMENT
# ============================================================================
# Script ini mengatasi keterbatasan biblatex yang hardcode "et al"
# dengan cara post-process file .bbl yang dihasilkan biber
# 
# WORKFLOW:
# 1. pdflatex: Generate .aux file
# 2. biber: Generate .bbl file dari bibliography.bib
# 3. sed: Ganti "et al" → "dkk" di file .bbl
# 4. pdflatex (2x): Final compilation dengan .bbl yang sudah diganti
#
# USAGE: ./compile_with_dkk.sh
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "KOMPILASI TESIS DENGAN DKK (bukan et al)"
echo "=========================================="

# Clean previous build artifacts
echo "[1/6] Membersihkan file cache..."
rm -f main_tesis.aux main_tesis.bbl main_tesis.bcf main_tesis.blg main_tesis.run.xml main_tesis.toc main_tesis.lof main_tesis.lot main_tesis.log main_tesis.out

# First pdflatex: Generate .aux file
echo "[2/6] Running pdflatex (pass 1 - generate .aux)..."
pdflatex -interaction=nonstopmode main_tesis.tex > /dev/null

# Run biber: Generate .bbl file
echo "[3/6] Running biber (generate .bbl)..."
biber main_tesis

# CRITICAL STEP: Replace "et al" with "dkk" in .bbl file
echo "[4/6] Replacing 'et al' → 'dkk' in .bbl file..."
if [ -f main_tesis.bbl ]; then
    # Count before replacement
    BEFORE_COUNT=$(grep -o "et al\." main_tesis.bbl | wc -l)
    echo "    Found $BEFORE_COUNT instances of 'et al.' in .bbl"
    
    # Replace all instances
    sed -i 's/et al\./dkk./g' main_tesis.bbl
    
    # Count after replacement
    AFTER_COUNT=$(grep -o "dkk\." main_tesis.bbl | wc -l)
    echo "    Replaced with $AFTER_COUNT instances of 'dkk.'"
else
    echo "ERROR: main_tesis.bbl not found!"
    exit 1
fi

# Second pdflatex: Incorporate .bbl changes
echo "[5/6] Running pdflatex (pass 2 - incorporate .bbl)..."
pdflatex -interaction=nonstopmode main_tesis.tex > /dev/null

# Third pdflatex: Finalize references
echo "[6/6] Running pdflatex (pass 3 - finalize)..."
pdflatex -interaction=nonstopmode main_tesis.tex > /dev/null

echo ""
echo "=========================================="
echo "KOMPILASI SELESAI!"
echo "=========================================="

# Verify results
if [ -f main_tesis.pdf ]; then
    PAGES=$(pdfinfo main_tesis.pdf | grep Pages | awk '{print $2}')
    echo "✓ PDF generated: $PAGES pages"
    
    # Extract text and count occurrences
    pdftotext main_tesis.pdf - | grep -o "dkk\." | wc -l > /tmp/dkk_count.txt
    pdftotext main_tesis.pdf - | grep -o "et al\." | wc -l > /tmp/etal_count.txt
    
    DKK_COUNT=$(cat /tmp/dkk_count.txt)
    ETAL_COUNT=$(cat /tmp/etal_count.txt)
    
    echo "✓ Citation check:"
    echo "  - 'dkk' instances: $DKK_COUNT"
    echo "  - 'et al' instances: $ETAL_COUNT"
    
    if [ "$ETAL_COUNT" -eq 0 ] && [ "$DKK_COUNT" -gt 0 ]; then
        echo "✓ SUCCESS: All citations use 'dkk' (Indonesian standard)"
    elif [ "$ETAL_COUNT" -gt 0 ]; then
        echo "⚠ WARNING: Still found $ETAL_COUNT instances of 'et al'"
        echo "  This may be from:"
        echo "  - Manually typed 'et al' in text (not citations)"
        echo "  - Bibliography entries with 'et al' in title/abstract"
    fi
else
    echo "✗ ERROR: PDF not generated"
    exit 1
fi

echo "=========================================="
echo "File output: main_tesis.pdf"
echo "=========================================="
