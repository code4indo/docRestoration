#!/bin/bash
# Script untuk memverifikasi semua citations tersedia di bibliography.bib
# Usage: ./verify_all_citations.sh

set -e

DOCS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIB_FILE="$DOCS_DIR/bibliography.bib"

cd "$DOCS_DIR"

echo "========================================="
echo "Verifikasi Completeness Bibliografi"
echo "========================================="
echo ""

# Extract all citation keys from all tex files
echo "[1/3] Mengekstrak semua citation dari chapter files..."
ALL_CITATIONS=$(grep -ohP '\\cite\{\K[^}]+' *_content_only.tex 2>/dev/null | tr ',' '\n' | sort -u)
TOTAL_CITATIONS=$(echo "$ALL_CITATIONS" | wc -l)
echo "✓ Total unique citations ditemukan: $TOTAL_CITATIONS"
echo ""

# Extract all bib entries
echo "[2/3] Mengekstrak semua entries dari bibliography.bib..."
ALL_BIB_ENTRIES=$(grep -oP '@\w+\{\K[^,]+' "$BIB_FILE" | sort)
TOTAL_ENTRIES=$(echo "$ALL_BIB_ENTRIES" | wc -l)
echo "✓ Total bibliography entries: $TOTAL_ENTRIES"
echo ""

# Check for missing citations
echo "[3/3] Memeriksa citation yang missing..."
MISSING_COUNT=0
MISSING_LIST=""

while IFS= read -r cite_key; do
    if ! echo "$ALL_BIB_ENTRIES" | grep -q "^${cite_key}$"; then
        echo "❌ MISSING: $cite_key"
        MISSING_LIST="${MISSING_LIST}${cite_key}\n"
        ((MISSING_COUNT++))
    fi
done <<< "$ALL_CITATIONS"

echo ""
echo "========================================="
if [ $MISSING_COUNT -eq 0 ]; then
    echo "✅ BERHASIL! Semua citations tersedia!"
    echo "========================================="
    echo ""
    echo "📊 Statistik:"
    echo "  - Total citations digunakan: $TOTAL_CITATIONS"
    echo "  - Total entries di bibliography: $TOTAL_ENTRIES"
    echo "  - Missing citations: 0"
    echo ""
    echo "✓ Sistem bibliografi terpusat berfungsi dengan baik"
    exit 0
else
    echo "⚠️  PERHATIAN: $MISSING_COUNT citation(s) tidak ditemukan!"
    echo "========================================="
    echo ""
    echo "Citations yang missing:"
    echo -e "$MISSING_LIST"
    echo ""
    echo "Tambahkan entries ini ke bibliography.bib"
    exit 1
fi
