#!/usr/bin/env bash
# Quick Command Reference - PDF Gabungan
# Location: dual_modal_gan/docs/

cd "$(dirname "$0")" 2>/dev/null || exit 1

cat << 'HELP'

╔════════════════════════════════════════════════════════════════╗
║  QUICK COMMAND REFERENCE - PDF Gabungan Chapter 1-4           ║
╚════════════════════════════════════════════════════════════════╝

📁 LOKASI FILES:
   /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/dual_modal_gan/docs/

═══════════════════════════════════════════════════════════════════

📄 DOWNLOAD PDF GABUNGAN:
   
   Langsung buka file di folder:
   $ main_complete.pdf  (9.5 MB, 104 halaman)

═══════════════════════════════════════════════════════════════════

🔍 VERIFIKASI STRUKTUR PDF:
   
   $ ./check_combined_pdf.sh
   
   Atau manual:
   $ pdfinfo main_complete.pdf
   $ pdftotext main_complete.pdf - | head -50

═══════════════════════════════════════════════════════════════════

🛠️  RECOMPILE PDF (Jika ada perubahan):
   
   $ cd dual_modal_gan/docs/
   $ rm -f main_complete.pdf
   $ pdflatex -interaction=nonstopmode main_complete.tex
   $ pdflatex -interaction=nonstopmode main_complete.tex
   $ ls -lh main_complete.pdf

═══════════════════════════════════════════════════════════════════

📝 EDIT CHAPTER:
   
   1. Edit file content yang diinginkan:
      $ vim chapter2_tinjauan_pustaka_content.tex
      
   2. Recompile PDF:
      $ pdflatex -interaction=nonstopmode main_complete.tex
      $ pdflatex -interaction=nonstopmode main_complete.tex

═══════════════════════════════════════════════════════════════════

📚 EDIT DAFTAR PUSTAKA:
   
   1. Edit bibliography items:
      $ vim bibliography_content.bib
      
   2. Recompile:
      $ pdflatex -interaction=nonstopmode main_complete.tex
      $ pdflatex -interaction=nonstopmode main_complete.tex

═══════════════════════════════════════════════════════════════════

➕ TAMBAH CHAPTER 5 & 6:
   
   Lihat dokumentasi:
   $ cat OPSI_TAMBAH_CHAPTER5_6.md
   
   Quick steps:
   1. Extract content dari chapter5 & chapter6
   2. Edit main_complete.tex (tambah \input{chapter5_*.tex})
   3. Recompile

═══════════════════════════════════════════════════════════════════

📖 DOKUMENTASI:
   
   Dokumentasi lengkap:
   $ cat COMBINED_PDF_README.md
   
   Ringkasan:
   $ cat RINGKASAN_PDF_GABUNGAN.txt

═══════════════════════════════════════════════════════════════════

🔧 CHECK COMPILE ERROR:
   
   $ tail -50 /tmp/main_compile.log
   $ grep -i "error\|undefined" /tmp/main_compile.log

═══════════════════════════════════════════════════════════════════

🗑️  CLEANUP TEMPORARY FILES:
   
   $ rm -f main_complete.{aux,log,out,toc}
   $ rm -f main_complete.bbl
   $ rm -f chapter*.bbl chapter*.blg

═══════════════════════════════════════════════════════════════════

📊 FILE LISTING:
   
   $ ls -lh main_complete.*
   $ du -sh main_complete.pdf
   $ wc -l main_complete.tex

═══════════════════════════════════════════════════════════════════

📋 STATUS CHECK LENGKAP:
   
   $ ./check_combined_pdf.sh
   
   Menampilkan:
   • PDF file size dan halaman
   • Bibliography references count
   • Master document status
   • Links & cross-references

═══════════════════════════════════════════════════════════════════

📤 EXPORT / BACKUP:
   
   $ cp main_complete.pdf main_complete_backup_v1.pdf
   $ tar -czf pdf_gabungan_backup.tar.gz main_complete.pdf *.tex *.bib

═══════════════════════════════════════════════════════════════════

✉️  SHARE / SEND TO REVIEWER:
   
   $ scp main_complete.pdf user@reviewer:/path/to/folder/
   
   Atau upload ke:
   • Google Drive
   • Dropbox
   • OneDrive
   • Email

═══════════════════════════════════════════════════════════════════

UNTUK BANTUAN LEBIH LANJUT:

   ① COMBINED_PDF_README.md    - Dokumentasi komprehensif
   ② OPSI_TAMBAH_CHAPTER5_6.md - Panduan menambah chapter
   ③ RINGKASAN_PDF_GABUNGAN.txt - Ringkasan lengkap
   ④ check_combined_pdf.sh     - Verification script

═══════════════════════════════════════════════════════════════════

CONTACT / QUESTIONS:

   Check file-file di atas atau gunakan:
   $ pdftotext main_complete.pdf - | less
   (untuk preview content)

═══════════════════════════════════════════════════════════════════

HELP

echo "Generated: $(date)"
echo ""
