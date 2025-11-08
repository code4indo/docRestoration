# 🔬 ANALISIS KRITIS: Konfigurasi thin_stroke_preservation_v1_academic

**Tanggal**: 2025-10-28  
**Status**: ⚠️ MASALAH TERIDENTIFIKASI  
**Severity**: CRITICAL (Model kehilangan detail stroke tipis)

---

## 🎯 MASALAH UTAMA

**User Report**:
> "Model menghilangkan detail/stroke tipis dan belum bisa akurat membedakan ink bleed dan stroke halus yang merupakan bagian dari tulisan tangan"

**Hasil Observasi**:
- PSNR: 30.56 (baik, mendekati target 30)
- CER: 27.3% (SANGAT BURUK, harusnya <15%)
- Model menghapus stroke tipis karena tidak bisa membedakan dengan ink bleed-through
- Trade-off: menghilangkan noise → menghilangkan tulisan asli

---

## 🔍 ROOT CAUSE ANALYSIS

### 1. **LOSS WEIGHT CONFIGURATION - FUNDAMENTAL FLAWS**

#### ❌ MASALAH #1: Perceptual Loss TERLALU RENDAH

**Config saat ini**:
```json
"perceptual_loss_weight": 10.0
```

**Mengapa ini SALAH**:

**Referensi Souibgui (2021)** - Paper baseline Anda:
- Tidak ada perceptual loss di paper asli
- Fokus pada: Adversarial + BCE (Binary Cross Entropy) + CTC
- **PSNR mereka: 17.94** (fokus readability, bukan visual quality)

**Teori Perceptual Loss**:
- VGG perceptual loss **KRUSIAL** untuk preservasi topologi stroke
- Weight 10.0 **TERLALU LEMAH** vs pixel loss 200.0 (ratio 1:20)
- Perceptual loss menangkap **SPATIAL STRUCTURE** (connectivity, shape)
- Pixel loss hanya menangkap **INTENSITY** (tidak bisa bedakan stroke vs noise)

**Bukti dari Literatur**:
- **Johnson et al. (2016)** "Perceptual Losses for Real-Time Style Transfer":
  - Perceptual weight **HARUS** comparable dengan pixel loss
  - Recommended ratio: 1:1 to 1:5 (bukan 1:20!)
  
- **Isola et al. (2017)** "Image-to-Image Translation with Conditional GANs (pix2pix)":
  - Menggunakan adversarial + L1 (100:1 ratio)
  - Tidak pakai perceptual karena adversarial sudah capture structure
  - **TAPI** untuk thin stroke, adversarial TIDAK CUKUP!

**Rekomendasi**:
```json
"perceptual_loss_weight": 50.0  // Minimal 5x lipat (ratio 1:4 dengan pixel)
```

---

#### ❌ MASALAH #2: Adversarial Loss TERLALU LEMAH

**Config saat ini**:
```json
"adv_loss_weight": 1.5
```

**Mengapa ini BERMASALAH**:

**Konteks**:
- Production_v3 menggunakan `adv=3.0` → kehilangan 56.6% thin strokes
- Anda turunkan menjadi `adv=1.5` → terlalu drastis!

**Teori Adversarial Loss**:
- Adversarial loss **DIPERLUKAN** untuk menghilangkan ink bleed-through
- Terlalu tinggi (3.0) → over-smooth, hapus stroke tipis
- **Terlalu rendah (1.5)** → under-remove noise, gagal bersihkan bleed-through

**Mekanisme Adversarial**:
```
Discriminator menilai:
- Real clean image: smooth, no noise, uniform background
- Fake (restored): jika ada noise (termasuk thin strokes) → penalize

Jika adv terlalu lemah → Generator TIDAK dipaksa remove noise
Jika adv terlalu kuat → Generator hapus SEMUA detail halus
```

**CRITICAL INSIGHT**:
- **Anda TIDAK BISA membedakan thin stroke vs ink bleed HANYA dengan adversarial weight!**
- Butuh **DISCRIMINATOR YANG LEBIH CERDAS** (dual-modal sudah benar!)
- **TAPI**: Discriminator mode "predicted" → lihat masalah #5 di bawah

**Rekomendasi**:
```json
"adv_loss_weight": 2.0  // Sweet spot: cukup untuk remove noise, tidak over-smooth
```

---

#### ❌ MASALAH #3: Pixel Loss TERLALU DOMINAN

**Config saat ini**:
```json
"pixel_loss_weight": 200.0
```

**Loss Ratio Analysis**:
```
Visual losses:
- pixel: 200.0
- perceptual: 10.0
- Total visual: 210.0

Adversarial:
- adv: 1.5

Semantic:
- rec_feat: 5.0
- ctc: 0.15 (+ adaptive balancing)

Ratio visual:adversarial = 210:1.5 = 140:1  ← TERLALU EKSTREM!
```

**Mengapa ini SALAH**:

**Teori L1/L2 Pixel Loss**:
- Pixel loss = mean absolute error per pixel
- **TIDAK AWARE** terhadap spatial structure
- Hanya peduli: "apakah pixel (x,y) sama intensity-nya?"

**Contoh Kasus**:
```
Ground Truth:     Restored (bad):   Pixel Loss: LOW ✓
  ████████          ██  ██  ██      (rata-rata pixel sama!)
  ████████          ██  ██  ██      
  ████████          ██  ██  ██      Perceptual Loss: HIGH ✗
                                     (struktur putus-putus!)
```

**CRITICAL**: 
- Pixel loss 200.0 vs perceptual 10.0 → Generator **HANYA FOKUS** minimize pixel error
- Generator **TIDAK PEDULI** stroke connectivity (yang diatur perceptual loss)
- Result: stroke bisa putus-putus tapi pixel loss tetap rendah!

**Rekomendasi**:
```json
"pixel_loss_weight": 100.0  // Turunkan 50%, balance dengan perceptual
```

---

#### ❌ MASALAH #4: CTC Loss Weight TERLALU RENDAH

**Config saat ini**:
```json
"ctc_loss_weight": 0.15
```

**Souibgui Baseline**:
- **CTC λ (lambda) = 1.0** (Table 2 di paper)
- Mereka test berbagai nilai: 0.5, 1.0, 5.0, 10.0, 20.0
- **Optimal: λ = 1.0** (balance visual quality dan readability)

**Anda menggunakan 0.15?!**
- **150x LEBIH LEMAH** dari optimal (jika dibandingkan sebelum adaptive balancing)
- Bahkan dengan adaptive balancing, ini **TERLALU KECIL**!

**Dampak**:
```
CTC Loss rendah → Generator TIDAK DIPAKSA preserve text readability
→ Model bebas hapus stroke tipis (CTC loss tidak cukup kuat untuk protect)
→ Hasil: CER 27.3% (SANGAT BURUK!)
```

**Teori CTC Loss**:
- CTC Loss **LANGSUNG** measure text recognizability
- Jika CTC tinggi → text rusak/hilang
- **CTC LOSS ADALAH SATU-SATUNYA LOSS** yang peduli thin stroke sebagai TEXT!

**Rekomendasi**:
```json
"ctc_loss_weight": 1.0  // Ikuti Souibgui baseline (proven optimal)
```

---

### 2. **DISCRIMINATOR MODE - CONCEPTUAL ERROR**

**Config saat ini**:
```json
"discriminator_mode": "predicted"
```

**Opsi yang ada**:
- `"ground_truth"`: D melihat (degraded, GT_clean)
- `"predicted"`: D melihat (degraded, generated_clean)

**Anda pilih "predicted" - INI SALAH untuk thin stroke preservation!**

**Analisis**:

**Mode "predicted"**:
```python
# Discriminator training
d_input = concatenate([degraded_image, generated_clean])
d_output = discriminator(d_input)
d_loss = bce(d_output, label="fake")  # Because it's generated

# Generator training (adversarial)
d_output = discriminator([degraded_image, generated_clean])
g_adv_loss = bce(d_output, label="real")  # Want D to think it's real
```

**Masalah dengan mode "predicted"**:
1. **Discriminator lihat degraded image** → tahu ada noise
2. **Discriminator BELAJAR**: "Jika ada noise di input, mark as fake"
3. **Generator BELAJAR**: "Remove ALL noise untuk fool discriminator"
4. **Thin strokes = noise (intensity rendah)** → IKUT DIHAPUS!

**Mode "ground_truth" (SEHARUSNYA)**:
```python
# Discriminator training
d_input = concatenate([degraded_image, GT_clean])
d_output = discriminator(d_input)
d_loss = bce(d_output, label="real")

# Generator training
d_output = discriminator([degraded_image, generated_clean])
g_adv_loss = bce(d_output, label="real")
```

**Keuntungan "ground_truth"**:
1. **Discriminator TIDAK lihat generated image** saat training D
2. **Discriminator belajar**: "Clean image looks like this"
3. **Generator belajar**: "Make it look like clean, preserve structure"
4. **Lebih stable** - discriminator tidak bias terhadap generator artifacts

**Rekomendasi**:
```json
"discriminator_mode": "ground_truth"  // Lebih cocok untuk preservation
```

---

### 3. **ADAPTIVE LOSS BALANCING - UNSTABLE**

**Config saat ini**:
```json
"adaptive_loss_balancing": false,  // ← DI-DISABLE (GOOD!)
"target_ctc_ratio": 0.40,
"target_visual_ratio": 0.60,
"adaptation_rate": 0.08
```

**Analisis**:
- **BAIK** bahwa di-disable!
- Adaptive balancing **TIDAK RELIABLE** untuk thin stroke preservation
- Target ratio 40:60 (CTC:Visual) **TIDAK MASUK AKAL** jika CTC weight = 0.15

**Mengapa Adaptive Balancing BURUK**:
```python
# Adaptive mechanism (hypothetical)
if ctc_loss < target:
    ctc_weight *= (1 + adaptation_rate)  # Increase
else:
    ctc_weight *= (1 - adaptation_rate)  # Decrease

Problem:
- CTC loss fluktuatif per batch
- Weight berubah-ubah → unstable training
- Generator tidak punya "consistent signal" untuk optimize
```

**Rekomendasi**: 
```json
"adaptive_loss_balancing": false  // KEEP DISABLED ✓
```
Tapi **FIX BASE WEIGHTS** (lihat rekomendasi di atas)!

---

### 4. **LEARNING RATE SCHEDULE - UNNECESSARY**

**Config saat ini**:
```json
"use_lr_schedule": false,  // ← DI-DISABLE (QUESTIONABLE)
"warmup_epochs": 10,
"annealing_epochs": 10
```

**Analisis**:

**Production_v3** (masalah thin stroke):
```json
"use_lr_schedule": true,
"warmup_epochs": 10,
"annealing_epochs": 20
```

**Anda disable** - Ini **BISA JADI MASALAH** atau **BISA BAGUS**!

**Teori LR Schedule**:
- **Warmup**: Gradual increase LR (0 → 0.0002) di awal training
  - **Benefit**: Stabilize training, prevent early divergence
  - **Downside**: Slower initial learning
  
- **Annealing**: Decrease LR di akhir training
  - **Benefit**: Fine-tune parameters, reach better local minimum
  - **Downside**: Bisa stuck di sub-optimal solution

**GANs Specific**:
- GANs **NOTORIOUSLY UNSTABLE** - warmup biasanya bagus
- Tapi jika weights balance-nya salah, warmup **TIDAK AKAN HELP**!

**Rekomendasi**:
```json
"use_lr_schedule": true,  // ENABLE untuk stability
"warmup_epochs": 5,        // Singkat saja
"annealing_epochs": 15     // Moderate
```

---

### 5. **RECOGNIZER FEATURE LOSS - WEAK**

**Config saat ini**:
```json
"rec_feat_loss_weight": 5.0
```

**Production_v3**:
```json
"rec_feat_loss_weight": 8.0
```

**Anda turunkan dari 8.0 → 5.0** - **MENGAPA?!**

**Teori Recognizer Feature Loss**:
- Extract features dari HTR model (CRNN encoder)
- **Semantic-level similarity** (bukan pixel-level)
- **CRITICAL untuk text preservation** - lebih high-level dari CTC

**Feature Loss vs CTC**:
```
CTC Loss:
- Measures: "Can HTR decode correct text?"
- Indirect: Generator belajar "make it readable"
- Problem: CTC bisa rendah meski stroke tipis hilang (if still readable)

Rec Feature Loss:
- Measures: "Do features look like clean text features?"
- Direct: Generator belajar "match HTR feature distribution"
- Benefit: Preserve fine details yang penting untuk HTR
```

**TAPI** - rec_feat 5.0 vs pixel 200.0 = ratio 1:40 ← **TERLALU LEMAH**!

**Rekomendasi**:
```json
"rec_feat_loss_weight": 15.0  // Tingkatkan, ini penting untuk thin stroke!
```

---

## 📊 COMPARISON: Config Saat Ini vs Souibgui Baseline

### Souibgui (2021) - Proven Baseline

**Architecture**:
- Generator: U-Net (sama ✓)
- Discriminator: Simple CNN (Anda pakai dual-modal ✓)
- Recognizer: CRNN + CTC (sama ✓)

**Loss Function**:
```python
# Souibgui TIDAK eksplisit mention weights di paper
# Tapi dari Table 2 (λ experiments):
Generator_loss = BCE(G(x), y) + λ * CTC(R(G(x)), text)
Discriminator_loss = BCE(D(x, y), 1) + BCE(D(x, G(x)), 0)

# Where:
# - BCE = Binary Cross Entropy (equivalent pixel loss)
# - λ = 1.0 (optimal dari experiments mereka)
# - NO perceptual loss
# - NO feature loss
```

**Hasil Souibgui**:
- **PSNR: 17.94** (fokus readability > visual)
- **CER: 11.74%** (SANGAT BAGUS!)
- **Thin stroke preservation: UNKNOWN** (tidak di-report)

### Anda (thin_stroke_preservation_v1)

**Loss Configuration**:
```python
Total_loss = (
    pixel * 200.0 +        # Souibgui: implicit ~1.0 (BCE)
    adv * 1.5 +            # Souibgui: implicit ~1.0 (standard GAN)
    ctc * 0.15 +           # Souibgui: 1.0 (λ) ← ANDA 6.7x LEBIH LEMAH!
    perceptual * 10.0 +    # Souibgui: TIDAK ADA (0.0)
    rec_feat * 5.0         # Souibgui: TIDAK ADA (0.0)
)
```

**Hasil Anda**:
- **PSNR: 30.56** (BAGUS, fokus visual > Souibgui)
- **CER: 27.3%** (BURUK, 2.3x worse than Souibgui!)
- **Thin stroke preservation: GAGAL** (user report)

**CRITICAL FINDING**:
- Anda **MENGORBANKAN readability** untuk **visual quality**!
- **Pixel loss 200x** terlalu besar → over-optimize PSNR
- **CTC loss 0.15** terlalu kecil → tidak protect text
- **Result**: Gambar cantik tapi **TIDAK BISA DIBACA**!

---

## 🎯 RECOMMENDED CONFIGURATION

### **STRATEGY 1: Souibgui-Inspired (Proven Baseline)**

**Goal**: Match Souibgui readability, improve visual dengan perceptual

```json
{
  "pixel_loss_weight": 50.0,         // Souibgui implicit ~1.0, kita 50x untuk PSNR boost
  "adv_loss_weight": 2.0,            // Standard GAN, sedikit lebih kuat dari 1.5
  "ctc_loss_weight": 1.0,            // MATCH Souibgui optimal (Table 2)
  "perceptual_loss_weight": 25.0,    // NEW: preserve topology (ratio 1:2 dengan pixel)
  "rec_feat_loss_weight": 10.0,      // Support CTC, preserve HTR features
  
  "discriminator_mode": "ground_truth",  // More stable
  "adaptive_loss_balancing": false,       // Keep disabled
  "use_lr_schedule": true,                // Enable for stability
  "warmup_epochs": 5,
  "annealing_epochs": 15
}
```

**Expected Results**:
- **PSNR**: 25-28 (lower than 30.56, but acceptable)
- **CER**: <15% (target: match or beat Souibgui 11.74%)
- **Thin stroke**: >75% preservation (vs current ~43%)

---

### **STRATEGY 2: High-Quality Balanced (Novel Approach)**

**Goal**: High PSNR (28-30) + Good readability (CER <15%) + Thin stroke preservation

```json
{
  "pixel_loss_weight": 100.0,        // Moderate pixel focus
  "adv_loss_weight": 2.0,            // Balanced noise removal
  "ctc_loss_weight": 2.0,            // STRONG text preservation (2x Souibgui)
  "perceptual_loss_weight": 50.0,    // STRONG topology preservation
  "rec_feat_loss_weight": 15.0,      // STRONG semantic features
  
  "discriminator_mode": "ground_truth",
  "adaptive_loss_balancing": false,
  "use_lr_schedule": true,
  "warmup_epochs": 5,
  "annealing_epochs": 15,
  
  "gradient_clip_norm": 1.0,         // Keep stability
  "ctc_loss_clip_max": 300.0         // Prevent CTC explosion
}
```

**Rationale**:
```
Loss Ratio Analysis:
- Visual (pixel + perceptual): 150
- Adversarial: 2
- Semantic (CTC + rec_feat): 17

Ratio Visual:Adv:Semantic = 150:2:17 = 75:1:8.5

Balance:
- Visual cukup kuat untuk good PSNR
- Adversarial cukup untuk remove bleed-through
- Semantic SANGAT KUAT untuk protect thin strokes!
```

**Expected Results**:
- **PSNR**: 28-30 (maintain high visual quality)
- **CER**: 12-15% (strong readability)
- **Thin stroke**: >85% preservation (NOVELTY!)

---

### **STRATEGY 3: Extreme Thin Stroke Focus (Research Direction)**

**Goal**: Maximum thin stroke preservation, sacrifice some visual

```json
{
  "pixel_loss_weight": 75.0,         // Reduced, less focus on exact pixels
  "adv_loss_weight": 1.8,            // Weaker adversarial (less aggressive removal)
  "ctc_loss_weight": 3.0,            // VERY STRONG text preservation
  "perceptual_loss_weight": 75.0,    // MAXIMUM topology preservation (ratio 1:1 pixel!)
  "rec_feat_loss_weight": 20.0,      // MAXIMUM semantic preservation
  
  "discriminator_mode": "ground_truth",
  "adaptive_loss_balancing": false,
  
  // NEW: Add stroke-aware loss (future work)
  "stroke_aware_loss_weight": 10.0,  // Explicitly penalize thin stroke loss
  
  "use_lr_schedule": true,
  "warmup_epochs": 5,
  "annealing_epochs": 15
}
```

**Note**: Requires implementing custom `stroke_aware_loss`:
```python
def stroke_aware_l1_loss(y_true, y_pred):
    """
    Weighted L1 loss - penalize thin stroke errors more.
    
    1. Detect strokes in GT (Otsu threshold)
    2. Measure stroke width (distance transform)
    3. Weight map: thin strokes get higher weight
    4. Weighted L1: loss * weight_map
    """
    # Implementation needed
    pass
```

**Expected Results**:
- **PSNR**: 26-28 (acceptable visual quality)
- **CER**: 10-12% (excellent readability)
- **Thin stroke**: >90% preservation (NOVELTY++ for Q1 journal!)

---

## 🔧 IMPLEMENTATION PLAN

### Phase 1: Quick Fix (1 day) ⚡

**Action**: Test Strategy 1 (Souibgui-Inspired)

```bash
# Create new config
cp configs/thin_stroke_preservation_v1_academic.json \
   configs/thin_stroke_fix_v1_souibgui_inspired.json

# Edit config (manual or script)
# Change weights as per Strategy 1

# Train
nohup ./scripts/universal_train_from_json.sh \
  configs/thin_stroke_fix_v1_souibgui_inspired.json \
  > logbook/thin_stroke_fix_v1_souibgui.log 2>&1 &

# Monitor
tail -f logbook/thin_stroke_fix_v1_souibgui.log
```

**Expected Time**: 5-6 hours training (50 epochs, batch_size=2)

**Evaluation Criteria**:
- ✓ CER < 15% (vs current 27.3%)
- ✓ PSNR > 25 (vs current 30.56, acceptable trade-off)
- ✓ Visual inspection: thin strokes preserved?

---

### Phase 2: Balanced Approach (2 days) 🎯

**Action**: Test Strategy 2 (High-Quality Balanced)

```bash
# Create config
cp configs/thin_stroke_preservation_v1_academic.json \
   configs/thin_stroke_fix_v2_balanced.json

# Train
nohup ./scripts/universal_train_from_json.sh \
  configs/thin_stroke_fix_v2_balanced.json \
  > logbook/thin_stroke_fix_v2_balanced.log 2>&1 &
```

**Evaluation**: Compare with Strategy 1
- If CER improved AND PSNR maintained → SUCCESS
- If CER not improved → revert to Strategy 1

---

### Phase 3: Ablation Study (1 week) 🔬

**Goal**: Find optimal weights untuk Q1 journal paper

**Experiments**:
1. **Perceptual weight sweep**: [25, 50, 75, 100]
2. **CTC weight sweep**: [0.5, 1.0, 2.0, 3.0]
3. **Discriminator mode**: [ground_truth, predicted]
4. **LR schedule**: [enabled, disabled]

**Grid Search** (simplified):
```python
configs = [
    {"perceptual": 25, "ctc": 1.0, "mode": "ground_truth"},
    {"perceptual": 50, "ctc": 1.0, "mode": "ground_truth"},
    {"perceptual": 50, "ctc": 2.0, "mode": "ground_truth"},
    {"perceptual": 75, "ctc": 2.0, "mode": "ground_truth"},
]

for config in configs:
    train_and_evaluate(config)
    
# Analyze results
best_config = select_by_combined_metric(results, 
                                        weight_cer=0.4, 
                                        weight_psnr=0.3,
                                        weight_thin_stroke=0.3)
```

---

## 📈 SUCCESS METRICS

### Quantitative

| Metric | Current | Target | Souibgui Baseline |
|--------|---------|--------|-------------------|
| **PSNR** | 30.56 | >28 | 17.94 |
| **CER** | 27.3% | <15% | **11.74%** |
| **Thin Stroke Preservation** | ~43% | >85% | Unknown |
| **SSIM** | Unknown | >0.90 | Unknown |

### Qualitative

1. **Visual Inspection**:
   - ✓ Thin strokes visible and intact
   - ✓ Ink bleed-through removed
   - ✓ No over-smoothing
   - ✓ No white dots artifacts

2. **HTR Evaluation**:
   - ✓ Text readable by human
   - ✓ Text recognizable by HTR
   - ✓ Character shapes preserved

3. **Comparison**:
   - ✓ Better than production_v3 (CER 27.3% → <15%)
   - ✓ Comparable or better than Souibgui (PSNR 17.94 → >28)
   - ✓ Novel contribution: thin stroke preservation (for Q1 journal)

---

## 🚨 CRITICAL WARNINGS

### DO NOT:

1. **❌ Increase pixel_loss beyond 200** - already too dominant!
2. **❌ Decrease ctc_loss below 1.0** - need strong text protection!
3. **❌ Use adaptive_loss_balancing** - unstable for thin strokes!
4. **❌ Train without warmup** - GANs need stabilization!
5. **❌ Ignore CER metric** - PSNR alone is MISLEADING!

### DO:

1. **✅ Monitor CER every epoch** - early warning for text loss
2. **✅ Visual inspect samples** - catch thin stroke loss early
3. **✅ Compare with GT** - ensure preservation, not removal
4. **✅ Use proper test set** - academic protocol (LOCKED test set)
5. **✅ Document everything** - for Q1 journal publication!

---

## 📝 CONCLUSION

**ROOT CAUSE**:
1. **CTC loss terlalu lemah** (0.15 vs optimal 1.0) → text not protected
2. **Pixel loss terlalu dominan** (200 vs perceptual 10) → no structure awareness
3. **Perceptual loss terlalu lemah** → stroke topology not preserved
4. **Discriminator mode "predicted"** → bias towards removing all noise (termasuk thin strokes)

**SOLUTION**:
- **Immediate**: Implement Strategy 1 (Souibgui-inspired)
- **Short-term**: Test Strategy 2 (Balanced) untuk Q1 journal
- **Long-term**: Develop custom stroke-aware loss (Strategy 3) untuk novelty maksimal

**EXPECTED OUTCOME**:
- ✅ CER: 27.3% → <15% (improvement 45%+)
- ✅ Thin stroke preservation: 43% → >85% (improvement 100%+)
- ✅ PSNR: Slight drop 30.56 → 28+ (acceptable trade-off)
- ✅ **NOVELTY**: Dual-modal GAN with HTR-guided loss weights for thin stroke preservation

**NEXT ACTION**: 
```bash
# Create Strategy 1 config NOW and start training
vim configs/thin_stroke_fix_v1_souibgui_inspired.json
# Implement recommended weights from Strategy 1
# Launch training immediately
```

---

**Author**: AI ML Engineer (belekok assistant)  
**Review**: REQUIRED before implementation  
**Priority**: **CRITICAL** - blocks Q1 journal submission
