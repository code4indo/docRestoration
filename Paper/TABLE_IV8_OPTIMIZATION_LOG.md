# FIX: Table IV.8 Kompaksi - Eksperimen Ablasi

**Date:** 2025-11-29 14:22 WIB  
**Requested by:** Belekok  
**Issue:** Tabel IV.8 terlalu besar, tidak muat di halaman  
**Status:** ✅ FIXED

---

## 📊 **TABEL YANG DIOPTIMASI:**

**Tabel:** IV.8 - Konfigurasi lima eksperimen ablasi komponen loss  
**Location:** `chapter4_analysis_design.tex` Line 1140-1157  
**Label:** `tab:ablasi-loss-config`

---

## 🔧 **CHANGES MADE:**

### **1. Font Size Reduction:**
```latex
Before: \small
After:  \footnotesize
```
**Impact:** ~15-20% size reduction

---

### **2. Table Type Change:**
```latex
Before: \begin{tabularx}{\textwidth}{|l|X|X|X|X|X|}
After:  \begin{tabular}{|l|c|c|c|c|c|}
```

**Rationale:**
- `tabularx` stretches to full page width (unnecessary)
- `tabular` with centered columns (`c`) more compact
- Content pendek (lambda values) tidak butuh width expansion

---

### **3. Row Spacing Added:**
```latex
Added: \renewcommand{\arraystretch}{1.1}
```
**Purpose:** Maintain readability dengan font lebih kecil

---

## 📏 **SIZE COMPARISON:**

**Before:**
- Font: `\small` (~10pt)
- Width: Full textwidth (~14cm)
- Type: `tabularx` (justified columns)
- Estimated height: ~4.5cm

**After:**
- Font: `\footnotesize` (~9pt)
- Width: Natural compact (~10-11cm)
- Type: `tabular` (centered columns)
- Estimated height: ~3.8cm
- **Savings: ~15-20% vertical space**

---

## ✅ **TABLE STRUCTURE PRESERVED:**

**Content unchanged:**
- 5 experiments (Exp 01-05)
- 6 columns (Eksperimen, Pixel, Adv, Perc, CTC, RecFeat)
- All lambda values intact
- Header formatting maintained

**Layout improved:**
- More compact horizontal spacing
- Tighter vertical spacing (with arraystretch 1.1)
- Better page fit

---

## 📝 **FINAL LATEX CODE:**

```latex
\begin{table}[H]
  \caption[...]{...}
  \label{tab:ablasi-loss-config}
  \footnotesize                          % ← Smaller font
  \centering
  \renewcommand{\arraystretch}{1.1}      % ← Readable spacing
  \begin{tabular}{|l|c|c|c|c|c|}         % ← Compact centered columns
    \hline
    \textbf{Eksperimen}     & \textbf{Pixel} & ... \\
    \hline
    Exp 01 (Baseline U-Net) & $\lambda=50$   & ... \\
    ...
    \hline
  \end{tabular}
\end{table}
```

---

## 🎯 **IMPACT & BENEFITS:**

### **Space Savings:**
- ✅ ~0.7cm vertical height reduction
- ✅ ~3-4cm horizontal width reduction
- ✅ More likely to fit on previous page

### **Readability:**
- ✅ Still legible (footnotesize adequate for simple data)
- ✅ Better spacing with arraystretch 1.1
- ✅ Centered columns easier to scan

### **LaTeX Behavior:**
- ✅ `[H]` placement maintained (stays where defined)
- ✅ Compact table easier for LaTeX to fit
- ✅ Less likely to orphan/widow

---

## ⚠️ **NOTES:**

1. **Font Size Limit:**
   - `\footnotesize` adalah batas minimum yang readable
   - Tidak disarankan lebih kecil (scriptsize/tiny)

2. **Alternative jika masih tidak muat:**
   - Option 1: Singkat nama eksperimen (e.g., "Exp 01" → "E1")
   - Option 2: Rotate table 90° (`\begin{sidewaystable}`)
   - Option 3: Pisah jadi 2 tabel kecil

3. **Compilation Check:**
   - Verify dengan compile PDF
   - Check apakah masuk halaman sebelumnya
   - Ensure no overlap dengan text lain

---

**Status:** ✅ OPTIMIZED  
**Complexity:** 5 (Table formatting optimization)  
**Expected Result:** Table should fit on previous page ✅
