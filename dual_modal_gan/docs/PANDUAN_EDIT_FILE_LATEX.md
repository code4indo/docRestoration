# 📝 PANDUAN EDIT FILE LATEX - MANA YANG HARUS DI-EDIT?

**Date:** 2025-11-15  
**Status:** ✅ VALIDATED STRUCTURE

---

## 🎯 TL;DR - Quick Answer

### ❌ **JANGAN EDIT FILE INI:**
```
chapter5_hasil.tex          ← STANDALONE, hanya untuk testing individual
chapter4_analysis_design.tex
chapter1_pendahuluan.tex
chapter2_tinjauan_pustaka.tex
chapter3_metodologi.tex
chapter6_kesimpulan.tex
```

### ✅ **EDIT FILE INI:**
```
chapter5_hasil_content_only.tex          ← EDIT INI untuk Chapter 5
chapter4_analysis_design_content_only.tex
chapter1_pendahuluan_content_only.tex
chapter2_tinjauan_pustaka_content_only.tex
chapter3_metodologi_content_only.tex
chapter6_kesimpulan_content_only.tex
```

---

## 🏗️ Struktur Dokumen Tesis

### Master Document
```
main_tesis.tex
├─ \input{chapter1_pendahuluan_content_only.tex}
├─ \input{chapter2_tinjauan_pustaka_content_only.tex}
├─ \input{chapter3_metodologi_content_only.tex}
├─ \input{chapter4_analysis_design_content_only.tex}
├─ \input{chapter5_hasil_content_only.tex}          ← INI YANG DI-INCLUDE
├─ \input{chapter6_kesimpulan_content_only.tex}
└─ \printbibliography (centralized)
```

### File Architecture

```
docs/
│
├── main_tesis.tex                    🎯 MASTER (compile ini untuk full thesis)
│   └─> INCLUDES semua *_content_only.tex files
│
├── chapter5_hasil.tex                ⚠️  STANDALONE (testing only)
│   ├─ Punya preamble sendiri
│   ├─ Punya bibliography config sendiri (outdated)
│   └─ \input{chapter5_hasil_content_only.tex}
│
├── chapter5_hasil_content_only.tex   ✅ EDIT INI!
│   ├─ PURE CONTENT, no preamble
│   ├─ Digunakan oleh main_tesis.tex
│   └─ Digunakan oleh chapter5_hasil.tex (untuk testing)
│
└── bibliography.bib                  📚 CENTRALIZED (managed by main_tesis.tex)
```

---

## 📋 Aturan Emas Edit File

### Rule #1: Edit Content Files
**SELALU edit file `*_content_only.tex`** untuk:
- Menambah/edit teks
- Menambah/edit tabel
- Menambah/edit gambar
- Menambah/edit equations
- Menambah/edit references `\ref{}`
- Menambah/edit citations `\cite{}`

**File yang AKTIF digunakan:**
- ✅ `chapter5_hasil_content_only.tex` ← **EDIT INI**
- ✅ `chapter4_analysis_design_content_only.tex`
- ✅ `chapter1_pendahuluan_content_only.tex`
- ✅ dst...

### Rule #2: Jangan Edit Standalone Files
**JANGAN edit file `chapter*.tex`** (tanpa `_content_only`) karena:
- ❌ Hanya untuk testing individual chapters
- ❌ Punya duplicate preamble dan config
- ❌ TIDAK di-include di `main_tesis.tex`
- ❌ Berpotensi konflik dengan centralized system

**File yang TIDAK digunakan di final thesis:**
- ❌ `chapter5_hasil.tex` ← **JANGAN EDIT INI**
- ❌ `chapter4_analysis_design.tex`
- ❌ `chapter1_pendahuluan.tex`
- ❌ dst...

### Rule #3: Centralized Configuration
**Edit di `main_tesis.tex`** untuk:
- Package imports
- Bibliography configuration
- Document geometry
- Custom commands
- Global formatting

**JANGAN duplicate config** di individual chapter files.

---

## 🔍 Cara Verify File Mana yang Digunakan

### Method 1: Check main_tesis.tex
```bash
grep '\\input{' main_tesis.tex
```

**Output:**
```
\input{chapter1_pendahuluan_content_only.tex}
\input{chapter2_tinjauan_pustaka_content_only.tex}
\input{chapter3_metodologi_content_only.tex}
\input{chapter4_analysis_design_content_only.tex}
\input{chapter5_hasil_content_only.tex}         ← INI YANG DIGUNAKAN
\input{chapter6_kesimpulan_content_only.tex}
```

### Method 2: Check File Structure
```bash
head -20 chapter5_hasil.tex
# Jika ada \documentclass, \begin{document} → STANDALONE (testing only)

head -20 chapter5_hasil_content_only.tex
# Jika langsung content (section, text, dll) → CONTENT FILE (used by main)
```

---

## ✅ Workflow Edit Content

### Step 1: Identifikasi File yang Benar
```bash
# Untuk Chapter 5:
vim chapter5_hasil_content_only.tex  # ← EDIT INI

# Untuk Chapter 4:
vim chapter4_analysis_design_content_only.tex

# dst...
```

### Step 2: Edit Content
Edit file `*_content_only.tex`:
- Tambah text, tabel, gambar
- Update references
- Update citations
- **JANGAN tambah preamble, packages, atau \begin{document}**

### Step 3: Compile Full Thesis
```bash
cd /path/to/docs
./compile_with_bibliography.sh main_tesis
# atau
pdflatex main_tesis.tex
biber main_tesis
pdflatex main_tesis.tex
pdflatex main_tesis.tex
```

### Step 4: Verify Output
```bash
# Check missing references
grep 'Reference.*undefined' main_tesis.log

# Check missing citations
grep 'Citation.*undefined' main_tesis.log

# Quick verification
./verify_all_citations.sh
```

---

## 🚨 Common Mistakes dan Cara Avoid

### Mistake #1: Edit Wrong File
**Problem:**
```bash
# SALAH - edit standalone file
vim chapter5_hasil.tex  # ❌ changes tidak muncul di main_tesis.pdf
```

**Solution:**
```bash
# BENAR - edit content file
vim chapter5_hasil_content_only.tex  # ✅ changes muncul di main_tesis.pdf
```

### Mistake #2: Duplicate Bibliography Config
**Problem:**
```latex
% Di chapter5_hasil.tex (SALAH)
\usepackage[style=apa,backend=biber]{biblatex}  % ❌ duplicate config
\addbibresource{bibliography.bib}
```

**Solution:**
```latex
% Di chapter5_hasil_content_only.tex (BENAR)
% No preamble, no packages, no biblatex config
% Just content: \section, text, tables, \cite{}, dll
```

Config bibliography **HANYA** di `main_tesis.tex` (centralized).

### Mistake #3: Testing Individual Chapter Wrong
**Problem:**
```bash
# Compile standalone file yang outdated
pdflatex chapter5_hasil.tex  # ❌ using outdated config
```

**Solution:**
```bash
# Option A: Compile full thesis (recommended)
pdflatex main_tesis.tex

# Option B: Update standalone file (if needed for testing)
# Tapi ingat: changes tetap harus di *_content_only.tex
```

---

## 📊 File Usage Matrix

| File Type | Purpose | Edit? | Compiled by main_tesis? |
|-----------|---------|-------|-------------------------|
| `main_tesis.tex` | Master document | Config only | ✅ YES (main file) |
| `*_content_only.tex` | Pure content | ✅ **YES** | ✅ YES (included) |
| `chapter*.tex` | Standalone testing | ❌ NO | ❌ NO (not included) |
| `bibliography.bib` | Citations database | ✅ YES | ✅ YES (centralized) |

---

## 🎯 Summary - What Should You Edit?

### For Chapter 5 Updates:

```
✅ EDIT THIS:
   chapter5_hasil_content_only.tex
   
✅ COMPILE THIS:
   main_tesis.tex (or use compile_with_bibliography.sh)
   
✅ CHECK THIS:
   main_tesis.pdf
   
❌ DON'T EDIT:
   chapter5_hasil.tex (outdated standalone)
```

### Quick Check Command:
```bash
# Verify yang file mana yang di-include:
grep 'chapter5' main_tesis.tex

# Output should be:
# \input{chapter5_hasil_content_only.tex}  ← INI YANG DIGUNAKAN
```

---

## 📞 Troubleshooting

### Q: "Saya sudah edit chapter5_hasil.tex tapi changes tidak muncul?"
**A:** Karena file itu TIDAK di-include di `main_tesis.tex`. Edit `chapter5_hasil_content_only.tex` instead.

### Q: "Kenapa ada 2 file (chapter5_hasil.tex dan chapter5_hasil_content_only.tex)?"
**A:** 
- `chapter5_hasil.tex`: Standalone file untuk testing individual chapter (optional)
- `chapter5_hasil_content_only.tex`: Content file yang di-include di main thesis (AKTIF)

### Q: "Apakah saya perlu maintain kedua file?"
**A:** **TIDAK.** Hanya maintain `*_content_only.tex`. File standalone (chapter*.tex) optional dan bisa diabaikan.

### Q: "Bagaimana jika saya butuh test individual chapter?"
**A:** 
```bash
# Option 1: Compile full thesis (recommended)
pdflatex main_tesis.tex

# Option 2: Use standalone file (must be updated first)
# Tapi changes tetap HARUS di *_content_only.tex
pdflatex chapter5_hasil.tex  # will input chapter5_hasil_content_only.tex
```

---

## ✅ Verification Checklist

Sebelum commit changes, pastikan:

- [ ] Edit dilakukan di file `*_content_only.tex`
- [ ] Compile `main_tesis.tex` (bukan standalone file)
- [ ] Check `main_tesis.pdf` untuk verify changes
- [ ] Run `./verify_all_citations.sh` → 0 missing
- [ ] Check `main_tesis.log` untuk warnings
- [ ] Bibliography references resolved (0 undefined)
- [ ] Label references resolved (0 undefined)

---

## 🎓 Best Practices

1. **Always edit `*_content_only.tex` files**
2. **Always compile `main_tesis.tex` for final output**
3. **Never duplicate config** (packages, biblatex, etc) in content files
4. **Use centralized bibliography** (managed by main_tesis.tex)
5. **Verify changes** in `main_tesis.pdf`, not standalone PDFs
6. **Run verification scripts** after major edits
7. **Keep standalone files optional** - focus on content files

---

**Last Updated:** 2025-11-15  
**Validated With:** main_tesis.tex structure analysis  
**Status:** ✅ Production-ready workflow
