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

# Backup content_only file
if [ -f "chapter5_hasil_content_only.tex" ]; then
    BACKUP_FILE="chapter5_hasil_content_only.tex.backup_$(date +%Y%m%d_%H%M%S)"
    cp chapter5_hasil_content_only.tex "$BACKUP_FILE"
    echo "✓ Backup created: $BACKUP_FILE"
fi

# Extract content (line 125 to 2066, excluding \begin{document} and \end{document})
# Line 125 is \begin{document}, we start from line 126
# Line 2067 is \end{document}, we end at line 2066
sed -n '125,2066p' chapter5_hasil.tex > chapter5_hasil_content_only.tex

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
