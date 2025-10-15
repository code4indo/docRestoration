#!/bin/bash

# Script untuk mengompilasi daftar pustaka ke PDF terpisah
# Usage: ./compile_references.sh
# Fitur: Tanpa hyphenation (pemenggalan kata di akhir baris)

echo "🔄 Mengompilasi PDF daftar pustaka terpisah (tanpa hyphenation)..."

# Clean up old files
echo "🧹 Membersihkan file sementara..."
rm -f standalone_references.aux standalone_references.log standalone_references.out

# Compile pertama kali
echo "📝 Kompilasi pertama..."
pdflatex standalone_references.tex

# Compile kedua untuk outline sempurna
echo "📝 Kompilasi kedua untuk outline..."
pdflatex standalone_references.tex

# Check hasil
if [ -f "standalone_references.pdf" ]; then
    echo "✅ PDF daftar pustaka berhasil dibuat: standalone_references.pdf"
    echo "📄 Ukuran file: $(du -h standalone_references.pdf | cut -f1)"
else
    echo "❌ Gagal membuat PDF daftar pustaka"
    exit 1
fi