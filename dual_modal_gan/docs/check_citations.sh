#!/bin/bash
# Script untuk verifikasi sitasi dan bibliography
# Usage: ./check_citations.sh

set -e

DOCS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DOCS_DIR"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║            🔍 AUDIT SITASI DAN BIBLIOGRAPHY                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Extract all citations used
echo "📊 Mengekstrak sitasi dari dokumen..."
grep -h "\\cite" *.tex *_content_only.tex 2>/dev/null | \
    grep -oP '\\cite[a-z]*\{[^}]+\}' | \
    grep -oP '\{[^}]+\}' | \
    tr -d '{}' | \
    tr ',' '\n' | \
    sort -u > /tmp/citations_used.txt

CITATIONS_COUNT=$(wc -l < /tmp/citations_used.txt)
echo "   ✓ Ditemukan $CITATIONS_COUNT citation keys unik"

# Extract all bib entries
echo ""
echo "📚 Mengekstrak entries dari bibliography.bib..."
grep -P "^@\w+\{" bibliography.bib | \
    grep -oP '\{[^,]+' | \
    tr -d '{' | \
    sort -u > /tmp/bib_entries.txt

BIB_COUNT=$(wc -l < /tmp/bib_entries.txt)
echo "   ✓ Ditemukan $BIB_COUNT entries dalam bibliography"

# Check for missing citations
echo ""
echo "🔎 Memeriksa sitasi yang hilang..."
MISSING=$(comm -23 /tmp/citations_used.txt /tmp/bib_entries.txt)

if [ -z "$MISSING" ]; then
    echo "   ✅ SEMUA SITASI ADA DI BIBLIOGRAPHY!"
    echo ""
else
    echo "   ❌ DITEMUKAN SITASI YANG HILANG:"
    echo "$MISSING" | sed 's/^/      - /'
    echo ""
    echo "   Tambahkan entries berikut ke bibliography.bib"
    exit 1
fi

# Citation statistics per chapter
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 STATISTIK SITASI PER CHAPTER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TOTAL=0
for file in chapter*_content_only.tex; do
    if [ -f "$file" ]; then
        count=$(grep -o "\\cite" "$file" 2>/dev/null | wc -l)
        TOTAL=$((TOTAL + count))
        printf "   %-45s : %3d sitasi\n" "$file" "$count"
    fi
done

echo "   ─────────────────────────────────────────────────────────────────"
printf "   %-45s : %3d sitasi\n" "TOTAL" "$TOTAL"

# Check hyperref configuration
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔗 KONFIGURASI HYPERREF"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if grep -q "hyperref=true" main_tesis.tex; then
    echo "   ✅ biblatex hyperref: AKTIF"
else
    echo "   ⚠️  biblatex hyperref: TIDAK AKTIF"
    echo "      Tambahkan 'hyperref=true' ke opsi biblatex"
fi

if grep -q "\\usepackage{hyperref}" main_tesis.tex; then
    echo "   ✅ hyperref package: DIMUAT"
else
    echo "   ❌ hyperref package: TIDAK DITEMUKAN"
fi

# Check package loading order
BIBLATEX_LINE=$(grep -n "usepackage.*biblatex" main_tesis.tex | cut -d: -f1)
HYPERREF_LINE=$(grep -n "usepackage{hyperref}" main_tesis.tex | cut -d: -f1)

if [ "$BIBLATEX_LINE" -lt "$HYPERREF_LINE" ]; then
    echo "   ✅ Urutan package: BENAR (biblatex → hyperref)"
else
    echo "   ⚠️  Urutan package: SALAH (hyperref sebelum biblatex)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ AUDIT SELESAI!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Ringkasan:"
echo "  • Citation keys: $CITATIONS_COUNT"
echo "  • Bibliography entries: $BIB_COUNT"
echo "  • Total sitasi dalam dokumen: $TOTAL"
echo "  • Missing citations: 0"
echo ""
echo "Untuk kompilasi lengkap:"
echo "  ./compile_tesis.sh full"
echo ""
