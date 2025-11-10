#!/bin/bash
# Quick Reference - PDF Gabungan Chapter 1-4 + Daftar Pustaka
# Generated: Nov 10, 2025

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     PDF GABUNGAN CHAPTER 1-4 + DAFTAR PUSTAKA                 ║"
echo "║                   QUICK REFERENCE                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

echo "📄 FILE OUTPUT:"
echo "   └─ main_complete.pdf (9.5 MB, 104 halaman)"
echo ""

echo "📊 STRUKTUR:"
echo "   ├─ BAB 1: Pendahuluan (28 hal)"
echo "   ├─ BAB 2: Tinjauan Pustaka (28 hal)"
echo "   ├─ BAB 3: Metodologi Penelitian (21 hal)"
echo "   ├─ BAB 4: Desain dan Analisis Sistem (10 hal)"
echo "   └─ Daftar Pustaka (36 referensi, ~15 hal)"
echo ""

echo "📋 FILE YANG TERSEDIA:"
echo ""

echo "   Master Document & Content Files:"
ls -lh main_complete.tex chapter*_content.tex 2>/dev/null | awk '{printf "   • %-40s %6s\n", $9, $5}'
echo ""

echo "   Bibliography:"
ls -lh bibliography*.bib 2>/dev/null | awk '{printf "   • %-40s %6s\n", $9, $5}'
echo ""

echo "   Dokumentasi:"
ls -lh COMBINED_PDF_README.md 2>/dev/null | awk '{printf "   • %-40s %6s\n", $9, $5}'
echo ""

echo "✅ STATUS VERIFIKASI:"
if [ -f main_complete.pdf ]; then
    pages=$(pdfinfo main_complete.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
    echo "   ✓ PDF compiled successfully (${pages} pages)"
else
    echo "   ✗ PDF not found"
fi

if [ -f bibliography_content.bib ]; then
    refs=$(grep -c "^\\\\bibitem" bibliography_content.bib)
    echo "   ✓ Bibliography with $refs references"
else
    echo "   ✗ Bibliography not found"
fi

if [ -f main_complete.tex ]; then
    echo "   ✓ Master document ready"
fi

echo ""
echo "🔧 CARA MENGGUNAKAN:"
echo "   1. Download: main_complete.pdf"
echo "   2. Open dengan PDF reader (Adobe Reader, Foxit, dll)"
echo ""

echo "📖 DOKUMENTASI LENGKAP:"
echo "   Lihat: COMBINED_PDF_README.md"
echo ""

echo "═══════════════════════════════════════════════════════════════"
