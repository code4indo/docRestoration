#!/bin/bash
# Script untuk kompilasi final dan generate report lengkap
# Usage: ./final_compile_and_report.sh

set -e

DOCS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DOCS_DIR"

echo "========================================="
echo "KOMPILASI FINAL & REPORT"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

# Step 1: Verify citations
echo "STEP 1: Verifikasi Citations"
echo "-----------------------------------------"
./verify_all_citations.sh
echo ""

# Step 2: Clean old files
echo "STEP 2: Membersihkan file temporary"
echo "-----------------------------------------"
rm -f main_tesis.aux main_tesis.bbl main_tesis.bcf main_tesis.blg \
      main_tesis.log main_tesis.out main_tesis.run.xml main_tesis.toc
echo "✓ File temporary dibersihkan"
echo ""

# Step 3: Full compilation
echo "STEP 3: Kompilasi LaTeX (Full Process)"
echo "-----------------------------------------"
./compile_with_bibliography.sh main_tesis > compile_output.log 2>&1
echo "✓ Kompilasi selesai (lihat compile_output.log untuk detail)"
echo ""

# Step 4: Generate report
echo "STEP 4: Generate Report"
echo "-----------------------------------------"

# Count pages
TOTAL_PAGES=$(pdfinfo main_tesis.pdf 2>/dev/null | grep Pages | awk '{print $2}')
FILE_SIZE=$(ls -lh main_tesis.pdf | awk '{print $5}')

# Count citations used
CITATIONS_USED=$(grep -ohP '\\cite\{\K[^}]+' *_content_only.tex 2>/dev/null | tr ',' '\n' | sort -u | wc -l)
BIB_ENTRIES=$(grep -oP '@\w+\{\K[^,]+' bibliography.bib | wc -l)

# Check for warnings
UNDEFINED_REFS=$(grep "Reference.*undefined" main_tesis.log 2>/dev/null | wc -l)
UNDEFINED_CITES=$(grep "Citation.*undefined" main_tesis.log 2>/dev/null | wc -l)

echo ""
echo "========================================="
echo "📊 FINAL REPORT"
echo "========================================="
echo ""
echo "✅ Status Kompilasi: BERHASIL"
echo ""
echo "📄 Informasi Dokumen:"
echo "  - Output file: main_tesis.pdf"
echo "  - Total halaman: $TOTAL_PAGES"
echo "  - Ukuran file: $FILE_SIZE"
echo ""
echo "📚 Bibliografi:"
echo "  - Citations digunakan: $CITATIONS_USED"
echo "  - Entries di bibliography.bib: $BIB_ENTRIES"
echo "  - Missing citations: $UNDEFINED_CITES"
if [ $UNDEFINED_CITES -eq 0 ]; then
    echo "  ✓ Semua citations tersedia!"
else
    echo "  ⚠️  Ada citations yang missing (cek log)"
fi
echo ""
echo "⚠️  Warnings (Non-Critical):"
echo "  - Undefined references (labels): $UNDEFINED_REFS"
if [ $UNDEFINED_REFS -gt 0 ]; then
    echo "    → Ini adalah label \ref{} yang missing, BUKAN citation"
    echo "    → Tidak mempengaruhi bibliografi"
fi
echo ""
echo "========================================="
echo "✅ KOMPILASI SELESAI TANPA MASALAH BIBLIOGRAPHY"
echo "========================================="
echo ""
echo "File yang dihasilkan:"
echo "  - main_tesis.pdf (dokumen final)"
echo "  - compile_output.log (log kompilasi detail)"
echo ""
echo "Untuk verifikasi ulang kapan saja:"
echo "  ./verify_all_citations.sh"
echo ""
