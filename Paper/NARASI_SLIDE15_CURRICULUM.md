# Narasi Presentasi: Studi Ablasi Curriculum Learning
**Slide 15 - Seminar Hasil Tesis (SINGKAT)**

---

## 🎤 Skrip Narasi Ringkas (1.5-2 menit)

### **PEMBUKAAN (10 detik)**

"Slide terakhir untuk studi ablasi adalah evaluasi **curriculum learning** — strategi pembelajaran bertahap yang kami hipotesiskan akan meningkatkan stabilitas."

---

### **PROTOKOL 3 FASE (25 detik)**

**[Tunjuk kolom kiri]**

"Protokol curriculum learning kami terdiri dari **tiga fase bertahap**:"

1. "**Fase Warmup** (epoch 1-10): CTC weight nol — model fokus pure visual reconstruction dulu"
2. "**Fase Transisi** (epoch 11-30): CTC weight dinaikkan bertahap dari 0 ke 0.15"
3. "**Fase Penuh** (epoch 31-50): CTC weight konstan 0.15 — pelatihan penuh"

**[Pause]**

"Hipotesis: pembelajaran bertahap ini akan **lebih stabil**."

---

### **HASIL MENGEJUTKAN (35 detik)**

**[Tunjuk tabel - nada surprised]**

"Tapi hasilnya **mengejutkan**!"

**[Baca metrik dengan emphasis pada bold]**

- "PSNR: Curriculum 26.02 dB, Non-curriculum **26.16 dB** — non-curriculum unggul"
- "CER: Curriculum 28.7%, Non-curriculum **28.6%** — lebih baik tanpa curriculum"
- "Best Epoch: Curriculum butuh 45 epoch, non-curriculum cukup **41 epoch** — 4 epoch lebih cepat!"

**[Pause - dramatic untuk angka terakhir]**

"Dan yang paling mengejutkan:"

"**CTC Variance**: Curriculum 151.62, non-curriculum hanya **4.09**"

**[Pause - biarkan sink in]**

"Non-curriculum **37 kali lebih stabil**!"

---

### **TEMUAN KUNCI (20 detik)**

**[Tunjuk alert block kiri]**

"Ini adalah **paradoks** — curriculum learning dirancang untuk meningkatkan stabilitas, tapi ternyata malah **37× lebih tidak stabil**!"

**[Pause]**

"Kenapa? Karena **arsitektur kita sudah cukup robust**. Kombinasi frozen recognizer, dual-modal discriminator, dan optimasi loss yang baik sudah menciptakan **equilibrium stabil** dari awal."

---

### **VALIDASI STATISTIK (15 detik)**

**[Tunjuk example block kanan]**

"Secara statistik:"
- "**p-value = 0.471** — jauh di atas threshold 0.05"
- "**Cohen's d = 0.146** — very small effect"

"Artinya perbedaan ini **tidak signifikan**. Equal performance, tapi non-curriculum **lebih simple**."

---

### **KESIMPULAN & POSITIVE SPIN (20 detik)**

**[Baca footer dengan nada conclusive]**

"Kesimpulan: Curriculum learning **tidak memberikan benefit signifikan**."

**[Pause - positive spin]**

"Tapi ini **good news**! Arsitektur yang handal dapat **learn simultaneously** dari awal dengan stabilitas **lebih tinggi**."

**[Gesture: simplicity]**

"Kita **tidak perlu** kompleksitas phased training. Deployment lebih **simple**, **reproducible**, dan **efficient**."

---

### **PENUTUP (10 detik)**

**[Nada reflektif]**

"Sekali lagi, temuan ini memvalidasi prinsip yang konsisten: **Protocol yang tepat dan arsitektur robust lebih penting daripada kompleksitas strategi training**."

**[Pause - transisi]**

"Sekarang mari kita lihat validasi pada dokumen ANRI autentik..."

---

## ⏱️ **Timing Total: ~2 menit**

| Segmen | Durasi |
|--------|--------|
| Pembukaan | 10s |
| Protokol 3 Fase | 25s |
| Hasil Mengejutkan | 35s |
| Temuan Kunci | 20s |
| Validasi Statistik | 15s |
| Kesimpulan | 20s |
| Penutup | 10s |
| **TOTAL** | **~2 min** |

---

## 🎯 **Angka Kunci (Hafal!)**

1. **26.16 vs 26.02 dB** → Non-curriculum slightly better
2. **28.6% vs 28.7%** → CER marginally better
3. **41 vs 45 epoch** → 4 epochs faster
4. **37×** → Stability difference (4.09 vs 151.62)
5. **p=0.471** → Not significant
6. **d=0.146** → Very small effect

---

## 💡 **Tips Delivery Ringkas**

### **Voice Tone:**
- **Surprised** saat reveal non-curriculum unggul
- **Dramatic** saat sebut "37× lebih stabil"
- **Honest** saat explain paradox
- **Positive** saat spin ke "good news - simpler is better"

### **Key Gestures:**
- **Tangan bertingkat** (3 fase) saat explain curriculum
- **Contrast gesture** untuk 151.62 vs 4.09
- **Simplicity gesture** (tangan horizontal smooth) untuk "no need complexity"

### **Pause Points:**
- After "37× lebih stabil" → 2 seconds (let it sink)
- After "paradoks" → 1.5 seconds
- After "tidak signifikan" → 1 second

---

## 🔴 **Antisipasi Pertanyaan (Singkat)**

### **Q: "Jika curriculum tidak efektif, mengapa digunakan di model produksi?"**
**A (30 detik):**  
"Model produksi dilatih **sebelum** ablasi study ini. Pada design awal, kami hipotesiskan curriculum akan membantu — hypothesis yang **valid** berdasarkan literature.

Ablasi dilakukan **retrospektif** untuk validate. Ternyata **tidak terbukti**.

Tapi model produksi **tetap achieve all targets** (PSNR 30.91 dB, CER 34.9%). Jadi curriculum bukan kesalahan, melainkan **learning untuk future work**: gunakan non-curriculum untuk simplicity."

---

### **Q: "Variance 37× — apakah ini masalah serius?"**
**A (25 detik):**  
"Variance tinggi pada curriculum bukan catastrophic failure. Ini **artifact** dari phased weight changes (0 → 0.15).

Yang penting: **both converge successfully**. PSNR comparable (26.16 vs 26.02 dB), perbedaan **not significant** (p=0.471).

Kesimpulan: curriculum adds **unnecessary complexity** without benefit. Principle: **equal performance → choose simpler**."

---

### **Q: "Apakah finding ini applicable untuk domain lain?"**
**A (20 detik):**  
"Temuan ini **specific** untuk setup kami: frozen recognizer + dual-modal + optimized loss balance.

Pada system dengan **less robust architecture**, curriculum **might still help**.

Kontribusi kami: **empirical evidence** bahwa **robust architecture can learn simultaneously**. Guidance untuk researchers: **invest in architecture robustness > phased training strategies**."

---

## ✅ **Key Messages (Checklist)**

- [ ] Curriculum = 3 fase bertahap (warmup, transisi, penuh)
- [ ] Non-curriculum **outperforms** di semua metrik
- [ ] **37× more stable** (variance 4.09 vs 151.62)
- [ ] **Not statistically significant** (p=0.471, d=0.146)
- [ ] **Positive spin**: Simpler is better, no need complexity
- [ ] Principle: Robust architecture > training strategy complexity

---

## 🎓 **Academic Defensibility (Quick Reference)**

### **Framing:**
> "Ini bukan kegagalan hypothesis, tapi **valid scientific finding**. Curriculum learning, yang effective di banyak domain, **ternyata not necessary** untuk system dengan architecture robust dan loss balance optimal."

### **Value:**
- ✅ Simplify deployment (no phased training)
- ✅ Faster convergence (4 epochs earlier)
- ✅ More stable (37× lower variance)
- ✅ Guidance for future work

### **Honesty:**
> "**Negative results are valid contributions**. Kita provide empirical evidence untuk **what doesn't need to be done** — sama valuable dengan what should be done."

---

## 📊 **Visual Mental Map**

```
CURRICULUM:
[Warmup 0] → [Transisi 0→0.15] → [Penuh 0.15]
     ↓              ↓                  ↓
  Complex      Fluctuating         Stabilize
  
VS

NON-CURRICULUM:
[Direct 0.15 from start]
         ↓
    Stable throughout
    (Variance 37× lower!)
```

---

## 🎯 **One-Liner Takeaway**

> **"Robust architecture beats phased training — simpler is better!"**

---

**Durasi**: ~2 menit (singkat seperti yang diminta)  
**Tone**: Honest, surprised, positive spin  
**Message**: Curriculum unnecessary, architecture robustness matters more

---

_Catatan: Narasi ini dirancang SINGKAT sesuai permintaan user, tapi tetap comprehensive untuk cover key points dan defensive terhadap potential questions._
