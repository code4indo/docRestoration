# Narasi Presentasi (UPDATED): Studi Ablasi Curriculum Learning
**Slide 15 - Seminar Hasil Tesis (REVISED)**

---

## 🎤 Skrip Narasi Ringkas (~2 menit) - FOKUS EFISIENSI

### **PEMBUKAAN (10 detik)**

"Slide terakhir untuk studi ablasi adalah evaluasi **curriculum learning** — strategi pembelajaran bertahap yang kami hipotesiskan akan meningkatkan efisiensi convergence."

---

### **PROTOKOL 3 FASE (20 detik)**

**[Tunjuk kolom kiri]**

"Protokol curriculum learning terdiri dari **tiga fase**: Warmup visual tanpa CTC, transisi bertahap, dan pelatihan penuh. "

**[Pause]**

"Hipotesis: phased activation akan mempercepat convergence."

---

### **HASIL MENGEJUTKAN (40 detik)**

**[Tunjuk tabel - nada factual]**

"Tapi hasilnya **bertolak belakang**:"

**[Baca metrik dengan emphasis pada bold]**

- "PSNR dan CER: **Comparable** — perbedaan marginal, tidak signifikan"
- "**Best Epoch**: Curriculum butuh 49 epoch, non-curriculum cukup **45 epoch**"
- "**Convergence**: Non-curriculum **4 epoch lebih cepat**!"

**[Pause - tunjuk row Convergence]**

"Artinya, phased activation **bukan mempercepat**, malah **memperlambat** convergence!"

---

### **TEMUAN KUNCI - EFFICIENCY (30 detik)**

**[Tunjuk alert block kiri]**

"Temuan kunci:"

1. "**4 epoch lebih cepat** — penghematan waktu komputasi ~6 menit per training"
2. "**Implementasi lebih simple** — no phased scheduling, langsung full CTC weight dari awal"

**[Pause - gesture simplicity]**

"Dua benefit praktis: **faster convergence** dan **simpler code**."

**[Tunjuk example block kanan]**

"Secara statistik: **p = 0.471**, tidak ada perbedaan signifikan dalam performa akhir."

---

### **ANALISIS PARADOX (25 detik)**

**[Nada explaining]**

"Mengapa curriculum **lebih lambat** padahal dirancang untuk efficiency?"

**[Pause]**

"**Root cause**: Arsitektur kita sudah **cukup robust**. Kombinasi frozen recognizer, dual-modal discriminator, dan optimized loss balance membuat sistem mampu optimize **semua objektif dari awal** tanpa memerlukan warmup bertahap."

**[Gesture: direct path vs winding path]**

"**Direct activation** (non-curriculum) = straight path to optimum. **Phased activation** = detour yang unnecessary."

---

### **KESIMPULAN & PRACTICAL IMPLICATION (20 detik)**

**[Baca footer dengan nada conclusive]**

"Kesimpulan: Curriculum learning **tidak memberikan benefit**."

**[Pause - positive practical spin]**

"Tapi ini **good news** untuk **praktisi**!"

**[Enumerate dengan jari]**

"Untuk implementasi sistem serupa:"

1. "**Skip phased scheduling** — gunakan full CTC weight dari epoch 1"
2. "**Hemat waktu** training — 4 epoch = ~6 menit per run, significant untuk iterative development"
3. "**Kode lebih simple** — no need untuk phase transition logic"

**[Pause]**

"**Simplicity is efficiency**."

---

### **PENUTUP (10 detik)**

**[Nada reflektif]**

"Prinsip yang konsisten: **Robust architecture design** lebih penting daripada **complex training strategies**."

**[Pause - transisi]**

"Sekarang mari kita lihat validasi pada dokumen ANRI autentik..."

---

## ⏱️ **Timing Total: ~2 menit**

| Segmen | Durasi | Focus |
|--------|--------|-------|
| Pembukaan | 10s | Hypothesis |
| Protokol 3 Fase | 20s | Quick overview |
| Hasil Mengejutkan | 40s | Data presentation |
| Temuan Kunci | 30s | **Efficiency benefits** ⭐ |
| Analisis Paradox | 25s | Why curriculum slower |
| Kesimpulan | 20s | **Practical implications** ⭐ |
| Penutup | 10s | Transition |
| **TOTAL** | **~2 min** | |

---

## 🎯 **Angka Kunci (Hafal!)** - UPDATED

1. **49 vs 45 epochs** → 4 epochs difference
2. **4 epochs faster** → Core benefit
3. **~6 minutes saved** → Per training run (@85s/epoch)
4. **p=0.471** → Not significant
5. **Cohen's d=0.146** → Very small effect

---

## 💡 **Key Messages (REVISED):**

### **OLD Focus** (Misleading):
❌ "37× lebih stabil" → Stability difference

### **NEW Focus** (Accurate):
✅ **"4 epoch lebih cepat"** → Convergence efficiency  
✅ **"Implementasi lebih simple"** → Practical benefit  
✅ **"Direct activation works better"** → Architectural robustness

---

## 🎤 **Voice Tone Updates:**

- **Factual** (bukan surprised) saat reveal data → "bertolak belakang dengan hipotesis"
- **Practical** saat explain benefits → "6 menit per run, significant untuk iterative development"
- **Confident** saat conclude → "robust architecture > complex strategies"
- **Positive** saat deliver practical implication → "simplicity is efficiency"

---

## 📊 **Updated Key Gestures:**

### **Convergence Speed**:
- **Tangan horizontal** → smooth line untuk non-curriculum
- **Tangan zigzag up** → winding path untuk curriculum
- **Direct vs indirect** gesture

### **Simplicity**:
- **One finger** → direct activation (simple)
- **Three fingers** → phased activation (complex)
- **Contrast complexity**

### **Efficiency**:
- **Clock gesture** → time savings
- **Thumbs up** → practical benefit

---

## 🔴 **Antisipasi Pertanyaan (UPDATED):**

### **Q1: "Kenapa curriculum lebih lambat jika dirancang untuk efficiency?"**
**A (30 detik):**
> "Excellent question! Curriculum learning memang terbukti efektif di **banyak domain** lain, terutama untuk architecture yang **belum mature** atau dataset yang **highly heterogeneous**.
> 
> Tapi sistem kita punya **tiga keunggulan** yang membuat phased approach unnecessary:
> 1. **Frozen recognizer** sudah pre-trained dan stable
> 2. **Dual-modal discriminator** provides strong supervision dari awal
> 3. **Loss balance** sudah optimized secara empiris
> 
> Jadi arsitektur sudah **cukup robust** untuk handle multi-objective optimization **simultaneously from epoch 1**. Phase transition malah jadi **unnecessary overhead**."

---

### **Q2: "Apakah 4 epoch difference signifikan?"**
**A (25 detik):**
> "Secara **absolute**, 4 epoch = ~6 menit per training run. Ini memang tidak besar untuk **single run**.
> 
> Tapi dalam **research context** dengan iterative experimentation:
> - Misal 20× experiments = **2 hours saved**
> - Plus **code simplicity**: no phase transition logic → easier debugging
> - Plus **cognitive simplicity**: one less hyperparameter to tune
> 
> **Cumulative benefit** significant untuk **development velocity**."

---

### **Q3: "Apakah finding ini applicable untuk domain lain?"**
**A (20 detik):**
> "Temuan ini **specific** untuk setup kami: frozen recognizer + dual-modal GAN + optimized loss.
> 
> **General lesson**: Jika arsitektur **sudah robust** dan **loss balance** optimal, curriculum mungkin **unnecessary**.
> 
> **Recommendation**: For new systems, **start simple** (non-curriculum), only add complexity (curriculum) jika ada **empirical evidence** of benefit."

---

### **Q4: "Bagaimana dengan stability yang disebutkan di chapter?"**
**A (35 detik):**
> "Good catch! Memang ada diskusi tentang variability difference (σ = 151 vs 4).
> 
> **Clarification**: Nilai tersebut adalah **total variance across all 50 epochs**. Tapi saat kita analisis **per-phase**:
> - Phase 1 (warmup): curriculum σ = 0.4 (CTC off)
> - Phase 3 (convergence): **both ≈ 2.7** (same!)
> 
> High total variance adalah **artifact** dari phase transition, bukan **inherent instability**. At convergence, **both equally stable**.
> 
> Jadi focus kita bukan stability, tapi **convergence efficiency**: non-curriculum **4 epochs faster** dengan **equal final performance**."

---

## ✅ **Checklist Pre-Presentasi (UPDATED):**

- [ ] Hafal urutan 3 fase (tapi jangan dwell - quick overview)
- [ ] **FOCUS**: 4 epochs faster + simpler implementation
- [ ] Ingat angka: 49 vs 45, p=0.471, d=0.146
- [ ] Practice explaining WHY curriculum slower (robust architecture)
- [ ] Prepare answer untuk stability clarification question
- [ ] Gesture: direct path vs winding path
- [ ] **Key message**: "Simplicity is efficiency"

---

## 🎯 **One-Liner Takeaway (UPDATED):**

### **OLD** (Misleading):
❌ "Robust architecture beats phased training — 37× more stable!"

### **NEW** (Accurate):
✅ **"Robust architecture enables direct optimization — 4 epochs faster, simpler implementation!"**

---

## 📌 **Critical Success Factors:**

### **1. Don't Overemphasize Stability**
- Stability claim is nuanced (phase-dependent)
- Focus on **clearer metric**: convergence speed

### **2. Emphasize Practical Benefits**
✓ Time savings (4 epochs)  
✓ Code simplicity (no phase logic)  
✓ Equal performance (no trade-off)

### **3. Frame Positively**
NOT: "Curriculum failed"  
BUT: "Architecture is robust enough to skip complexity"

### **4. Academic Honesty**
- Acknowledge curriculum works in other contexts
- Our finding is **context-specific** (robust architecture)
- General lesson: **simplicity first**, add complexity only if needed

---

## 🎓 **Academic Framing (UPDATED):**

### **Hypothesis**:
> "Curriculum learning will improve convergence efficiency"

### **Result**:
> "**Not supported**. Non-curriculum converges 4 epochs faster with equal performance."

### **Interpretation**:
> "When architecture is **sufficiently robust**, phased training is **unnecessary complexity**. Direct multi-objective optimization from epoch 1 is **more efficient**."

### **Contribution**:
> "Empirical evidence that **architectural robustness** can eliminate need for curriculum strategies in document restoration + HTR domain."

---

**Durasi**: ~2 menit (same as before, but **better focused**)  
**Tone**: Factual, practical, positive  
**Message**: **Efficiency through simplicity** when architecture is robust

---

_Catatan: Narasi ini menghindari misleading stability claim dan fokus pada **convergence efficiency** dan **practical implementation benefits** yang lebih defensible dan valuable._
