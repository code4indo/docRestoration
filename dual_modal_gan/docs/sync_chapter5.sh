#!/bin/bash
# Script untuk sync content dari chapter5_hasil.tex ke chapter5_hasil_content_only.tex
# Usage: ./sync_chapter5.sh

set -e

DOCS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DOCS_DIR"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          🔄 SYNC chapter5_hasil.tex → content_only.tex          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if source file exists
if [ ! -f "chapter5_hasil.tex" ]; then
    echo "❌ ERROR: chapter5_hasil.tex not found!"
    exit 1
fi

# Find \begin{document} line
BEGIN_LINE=$(grep -n '\\begin{document}' chapter5_hasil.tex | head -1 | cut -d: -f1)

# Find \end{document} line
END_LINE=$(grep -n '\\end{document}' chapter5_hasil.tex | tail -1 | cut -d: -f1)

if [ -z "$BEGIN_LINE" ] || [ -z "$END_LINE" ]; then
    echo "❌ ERROR: No \\begin{document} or \\end{document} found!"
    exit 1
fi

# Calculate content lines (exclude \begin{document} and \end{document})
START_LINE=$((BEGIN_LINE + 1))
CONTENT_END_LINE=$((END_LINE - 1))

echo "📍 Detection result:"
echo "   \\begin{document} at line: $BEGIN_LINE"
echo "   \\end{document} at line:   $END_LINE"
echo "   Content range: $START_LINE-$CONTENT_END_LINE"
echo ""

# Backup content_only file
if [ -f "chapter5_hasil_content_only.tex" ]; then
    BACKUP_FILE="chapter5_hasil_content_only.tex.backup_$(date +%Y%m%d_%H%M%S)"
    cp chapter5_hasil_content_only.tex "$BACKUP_FILE"
    echo "✓ Backup created: $BACKUP_FILE"
fi

# Extract content dynamically
sed -n "${START_LINE},${CONTENT_END_LINE}p" chapter5_hasil.tex > chapter5_hasil_content_only.tex

echo "✓ Content extracted and synced"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STATISTICS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Source: chapter5_hasil.tex ($(wc -l < chapter5_hasil.tex) lines)"
echo "  Target: chapter5_hasil_content_only.tex ($(wc -l < chapter5_hasil_content_only.tex) lines)"
echo ""
echo "✅ SYNC COMPLETE!"
echo ""
echo "Next steps:"
echo "  1. Compile: pdflatex main_tesis.tex"
echo "  2. Biber:   biber main_tesis"
echo "  3. Compile: pdflatex main_tesis.tex (2x)"
echo "  4. View:    evince main_tesis.pdf"
echo ""
