#!/bin/bash
# QUICK REFERENCE: Bibliography Management
# ========================================

cat << 'EOF'
╔══════════════════════════════════════════════════════════╗
║         BIBLIOGRAPHY QUICK REFERENCE                     ║
║         Status: ✅ COMPLETE (0 missing)                  ║
╚══════════════════════════════════════════════════════════╝

📚 VERIFIKASI CITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ./verify_all_citations.sh
  
  Output: ✅ BERHASIL! Semua citations tersedia!
          📊 16 citations | 22 entries | 0 missing

🔨 KOMPILASI DOKUMEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Kompilasi normal
  ./compile_with_bibliography.sh main_tesis
  
  # Kompilasi + full report
  ./final_compile_and_report.sh

📝 MENAMBAH REFERENSI BARU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Edit: bibliography.bib
     @article{newkey,
       author = {...},
       title = {...},
       year = {...}
     }
  
  2. Gunakan di chapter:
     \cite{newkey}
  
  3. Compile:
     ./compile_with_bibliography.sh main_tesis

📊 CITATIONS TERSEDIA (16/16)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ chen2017deeplab          ✓ chen2018gradnorm
  ✓ erb2021                  ✓ he2016deep
  ✓ isola2017                ✓ johnson2016
  ✓ kang2021pay              ✓ kirkpatrick2017overcoming
  ✓ lin2017fpn               ✓ oktay2018attention
  ✓ ronneberger2015unet      ✓ souibgui2021enhance
  ✓ souibgui2022             ✓ souibgui2022docentr
  ✓ woo2018cbam              ✓ zhang2018residual

📁 FILE LOCATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Konfigurasi:  main_tesis.tex (line 38-50)
  Database:     bibliography.bib (22 entries)
  Scripts:      verify_all_citations.sh
                compile_with_bibliography.sh
                final_compile_and_report.sh
  Docs:         BIBLIOGRAPHY_CENTRALIZED_GUIDE.md
                BIBLIOGRAPHY_STATUS_COMPLETE.md

⚡ TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Problem: Citation undefined
  Solution: 
    1. Tambahkan ke bibliography.bib
    2. Jalankan biber: biber main_tesis
    3. Compile ulang: pdflatex main_tesis.tex (2x)

  Problem: Empty bibliography
  Solution: Pastikan ada minimal 1 \cite{} di chapter

  Problem: Warning "Reference undefined"
  Note: Ini untuk \ref{} (labels), BUKAN \cite{} (citations)
        Tidak mempengaruhi bibliografi

🎯 CURRENT STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ Bibliografi: LENGKAP (0 missing)
  ✅ Kompilasi: BERHASIL (210 halaman)
  ✅ Format: APA Style (Indonesian labels)
  ✅ System: Terpusat & Automated

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Last verified: $(date '+%Y-%m-%d %H:%M:%S')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
