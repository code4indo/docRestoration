================================================================================
LATEX SUBMISSION PACKAGE - COMPILATION INSTRUCTIONS
================================================================================

Paper Title: HTR-Oriented Document Restoration Using Generative Adversarial 
             Networks with Frozen Recognizer and Optimized Loss Functions

Authors: Jatniko Nur Mutaqin, I Gusti Bagus Baskara Nugraha

================================================================================
PACKAGE CONTENTS
================================================================================

1. jatniko_en_final.tex          - Main LaTeX source file
2. IEEEtran.cls                  - IEEE Transaction document class
3. images_for_paper/             - Directory containing all figures (20 files)
   - Degradation examples (6 PNG files)
   - Architecture diagrams (3 PNG files)
   - Analysis results (9 PDF files)
   - Author photos (2 image files)

================================================================================
COMPILATION INSTRUCTIONS
================================================================================

METHOD 1: Using pdflatex (Recommended)
---------------------------------------
Run the following commands in sequence:

    pdflatex jatniko_en_final.tex
    pdflatex jatniko_en_final.tex

Note: Run pdflatex TWICE to resolve all cross-references and citations.

METHOD 2: Using latexmk (Alternative)
--------------------------------------
    latexmk -pdf jatniko_en_final.tex

METHOD 3: Using TeX editor (e.g., TeXstudio, Overleaf)
-------------------------------------------------------
1. Open jatniko_en_final.tex in your TeX editor
2. Ensure the compiler is set to pdfLaTeX
3. Click "Build" or "Compile"
4. Run compilation twice for proper references

================================================================================
SYSTEM REQUIREMENTS
================================================================================

Required LaTeX Packages:
- graphicx (for images)
- amsmath, amssymb (for mathematical equations)
- cite (for citations)
- subfig (for subfigures)
- IEEEtran (document class - included in package)
- hyperref (for hyperlinks and references)
- url (for URL formatting)
- array, multirow, booktabs (for tables)
- xcolor (for colors)

These packages are typically included in standard LaTeX distributions:
- TeX Live (Linux/Mac)
- MiKTeX (Windows)
- Overleaf (online)

================================================================================
IMPORTANT NOTES
================================================================================

1. All image paths use RELATIVE paths (images_for_paper/...)
2. Bibliography is embedded in the .tex file (no separate .bib file needed)
3. Keep the folder structure intact:
   main_document_LaTeX/
   ├── jatniko_en_final.tex
   ├── IEEEtran.cls
   └── images_for_paper/
       └── (20 image files)

4. Expected output: jatniko_en_final.pdf (approximately 15-20 pages)

================================================================================
TROUBLESHOOTING
================================================================================

Problem: "File not found" error for images
Solution: Ensure images_for_paper/ folder is in the same directory as .tex file

Problem: Missing package errors
Solution: Install missing packages using your LaTeX distribution's package manager
          - TeX Live: tlmgr install <package-name>
          - MiKTeX: automatic installation on first use

Problem: Bibliography not showing
Solution: Run pdflatex twice (second run resolves citations)

Problem: Cross-references showing "??"
Solution: Run pdflatex twice (second run resolves references)

================================================================================
VERIFICATION
================================================================================

After successful compilation, verify:
✓ PDF generated: jatniko_en_final.pdf
✓ All figures appear correctly
✓ All citations are resolved (no [?] marks)
✓ All cross-references work (no ?? marks)
✓ Page count: approximately 15-20 pages
✓ File size: approximately 5-8 MB

================================================================================
CONTACT INFORMATION
================================================================================

For questions or issues regarding compilation, please contact the corresponding
author through the journal submission system.

Document Class: IEEEtran (IEEE Transactions format)
Compiler: pdfLaTeX
Encoding: UTF-8

Last Updated: November 17, 2025
================================================================================
