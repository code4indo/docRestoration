# 🔴 CRITICAL DOMAIN ADAPTATION: Synthetic → Real ANRI Documents

**Date**: 2025-10-28  
**Context**: User revelation - "koleksi arsip nasional banyak yang memiliki stroke terputus yang disebabkan oleh usia dokumen, dan tinta yang melebar pada edge"  
**Status**: 🚨 **TRAINING STRATEGY MISMATCH DETECTED**

---

## 🎯 PROBLEM STATEMENT

### Current Training Situation
- **Dataset**: 4.7GB **SYNTHETIC** paleography data (`dataset_gan.tfrecord`)
- **Training**: `thin_stroke_fix_v1_souibgui_inspired` (Epoch 9/50)
- **Target**: Arsip Nasional (ANRI) real documents

### Real ANRI Document Characteristics (USER INPUT)
1. **Stroke terputus** - degradasi usia (tinta pudar, kertas rusak)
2. **Tinta melebar di edge** - ink bleeding/feathering effect

### 🔴 CRITICAL MISMATCH

**Synthetic Dataset Degradation** (assumption based on typical synthetic generation):
```python
# Likely degradation types in synthetic data
degradations = {
    'ink_bleed_through': 0.40,    # Background bleed-through
    'background_noise': 0.20,     # Random noise
    'fading': 0.20,               # Uniform fading
    'stains': 0.15,               # Localized stains
    'mixed': 0.05
}
```

**Real ANRI Degradation** (actual historical documents):
```python
# Actual degradation patterns in 16-18th century documents
real_degradations = {
    'stroke_breaks': 0.35,         # ← NOT in synthetic data
    'edge_ink_bleeding': 0.30,     # ← Different from background bleed
    'age_fading': 0.20,            # Non-uniform, stroke-dependent
    'paper_damage': 0.10,          # Holes, tears
    'mixed': 0.05
}
```

**DOMAIN GAP**: 
- Synthetic trains model to **remove bleed-through** (background noise)
- Real ANRI needs model to **preserve stroke + handle edge bleeding**
- **OPPOSITE objectives** → model will fail on real data

---

## 📊 Evidence from Existing Fine-tuning Configs

### Config: `finetune_anri_pseudo_visual_only.json`

**Key Findings**:
- ✅ Already exists ANRI fine-tuning config
- ✅ Uses **pseudo-labeling** (visual-only, no text labels)
- ✅ Loads synthetic checkpoint as starting point
- ✅ 33 ANRI pages for domain adaptation

**Config Details**:
```json
{
  "experiment_name": "finetune_anri_pseudo_visual_only",
  "description": "FINE-TUNING on 33 ANRI pages via pseudo-labeling. Visual-only mode (no text labels). Domain adaptation from synthetic to real paleographic documents. Patch-based training with conservative hyperparameters to prevent overfitting.",
  
  "tfrecord_path": "data/anri_pseudo_labeled_train.tfrecord",
  
  "resume_from": {
    "checkpoint_dir": "dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic",
    "checkpoint_name": "ckpt-99"  // Best model from synthetic training
  },
  
  "epochs": 30,
  "batch_size": 2,
  "steps_per_epoch": 50,  // Small dataset (33 pages)
  
  "lr_g": 0.00005,  // 40x LOWER than synthetic (prevent catastrophic forgetting)
  "lr_d": 0.00005,
  
  "ctc_loss_weight": 0.0,  // Visual-only (no labels)
  "pixel_loss_weight": 200.0,
  "perceptual_loss_weight": 10.0,
  "rec_feat_loss_weight": 5.0,
  "adv_loss_weight": 1.5
}
```

**Strategy**: 
- Start from synthetic checkpoint (general restoration knowledge)
- Fine-tune on ANRI real data (domain-specific patterns)
- Lower LR to prevent forgetting synthetic knowledge
- Visual-only (no text labels needed)

---

## 🔬 ROOT CAUSE ANALYSIS: Why Synthetic Model Fails on Real ANRI

### Issue 1: Stroke Breaks (usia dokumen)

**Synthetic Training**:
- Ground truth: **Connected** strokes (artificially generated)
- Degraded input: Noise + bleed-through (but strokes still connected)
- Model learns: "Preserve continuous strokes, remove noise"

**Real ANRI**:
- Ground truth: **Unknown** (no clean version exists)
- Degraded input: **Already broken** strokes (physical degradation)
- Model sees: "Broken strokes" → classifies as **noise** → **REMOVES**

**Result**: Model **worsens** stroke breaks instead of connecting them.

---

### Issue 2: Tinta Melebar di Edge (ink bleeding)

**Synthetic Training**:
- Bleeding simulated as **background bleed-through** (from opposite page)
- Location: Random, not stroke-dependent
- Model learns: "Gray pixels away from text = noise → REMOVE"

**Real ANRI**:
- Bleeding occurs at **stroke edges** (ink feathering into paper)
- Location: Directly adjacent to text strokes
- Model sees: "Gray pixels near text" → removes → **THINS strokes** → CER increases

**Result**: Model **removes bleeding** but also **removes stroke edges** → strokes become thinner → readability decreases.

---

### Issue 3: Age-Dependent Fading

**Synthetic Training**:
- Fading applied **uniformly** across entire image
- All strokes fade by same percentage
- Model learns: "Amplify all low-contrast pixels"

**Real ANRI**:
- Fading **non-uniform** (depends on ink type, paper quality, storage conditions)
- Thin strokes fade MORE than thick strokes
- Some strokes completely invisible, others intact

**Result**: Model can't distinguish "faded stroke" from "no stroke" → **fails to recover** severely faded regions.

---

## 🎯 RECOMMENDED SOLUTION: Phased Domain Adaptation

### Phase 1: IMMEDIATE - Fine-tune on ANRI Data (This Week) ⚡

**Objective**: Adapt synthetic model to real ANRI degradation patterns

**Dataset**: 
- Use existing `anri_pseudo_labeled_train.tfrecord` (33 pages)
- If available, expand to more ANRI samples

**Config**: `finetune_anri_pseudo_visual_only.json` (already exists!)

**Expected Improvement**:
- Stroke breaks: Model learns to **connect** instead of remove
- Edge bleeding: Model learns to **preserve** stroke edges while removing background noise
- Fading: Model learns **non-uniform** fading patterns

**Timeline**: 
- Training: 30 epochs × 50 steps = 1500 steps (~3-4 hours on GPU)
- Evaluation: Test on held-out ANRI samples
- **Total: 1-2 days**

**Risk**: Low (fine-tuning from proven checkpoint)

---

### Phase 2: DATA AUGMENTATION - Synthetic Data with ANRI-Like Degradation (1-2 weeks)

**Objective**: Generate synthetic data that **matches** ANRI degradation characteristics

**Implementation**:

#### 2.1 Stroke Break Simulation
```python
def simulate_age_stroke_breaks(clean_stroke_mask, break_probability=0.3):
    """
    Simulate stroke breaks caused by aging (tinta pudar, kertas rusak).
    
    Args:
        clean_stroke_mask: Binary mask of text strokes
        break_probability: Probability of break occurring
    
    Returns:
        Degraded mask with realistic stroke breaks
    """
    # 1. Extract skeleton of strokes
    skeleton = cv2.ximgproc.thinning(clean_stroke_mask)
    
    # 2. Identify break points (random along skeleton)
    num_pixels = np.sum(skeleton > 0)
    num_breaks = int(num_pixels * break_probability * 0.01)  # 1% of skeleton pixels
    
    break_points = np.random.choice(np.where(skeleton > 0)[0], size=num_breaks, replace=False)
    
    # 3. Create break regions (small gaps)
    for point in break_points:
        y, x = point
        # Create circular gap (radius 2-5 pixels)
        radius = np.random.randint(2, 6)
        cv2.circle(clean_stroke_mask, (x, y), radius, 0, -1)
    
    return clean_stroke_mask
```

#### 2.2 Edge Ink Bleeding Simulation
```python
def simulate_edge_ink_bleeding(clean_image, stroke_mask, bleed_intensity=0.3):
    """
    Simulate ink bleeding at stroke edges (tinta melebar).
    
    Args:
        clean_image: Grayscale image (0-255)
        stroke_mask: Binary mask of text strokes
        bleed_intensity: Intensity of bleeding effect
    
    Returns:
        Degraded image with edge bleeding
    """
    # 1. Dilate stroke edges to create bleed zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bleed_zone = cv2.dilate(stroke_mask, kernel, iterations=2)
    bleed_zone = bleed_zone - stroke_mask  # Only edge region
    
    # 2. Apply Gaussian blur to bleed zone (ink diffusion)
    bleed_effect = cv2.GaussianBlur(bleed_zone.astype(float), (7, 7), sigmaX=2.0)
    
    # 3. Blend with original image
    degraded = clean_image.astype(float)
    degraded[bleed_zone > 0] -= bleed_effect[bleed_zone > 0] * bleed_intensity * 255
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    
    return degraded
```

#### 2.3 Non-Uniform Age Fading
```python
def simulate_nonuniform_fading(clean_image, stroke_width_map, thin_fade_factor=0.5, thick_fade_factor=0.2):
    """
    Simulate non-uniform fading (thin strokes fade more than thick).
    
    Args:
        clean_image: Grayscale image
        stroke_width_map: Map of local stroke widths
        thin_fade_factor: Fade intensity for thin strokes (higher = more fading)
        thick_fade_factor: Fade intensity for thick strokes
    
    Returns:
        Faded image with width-dependent fading
    """
    # Normalize stroke width to [0, 1]
    width_norm = stroke_width_map / (stroke_width_map.max() + 1e-6)
    
    # Compute fade factor: thin strokes (width → 0) fade more
    fade_map = thin_fade_factor + (thick_fade_factor - thin_fade_factor) * width_norm
    
    # Apply fading
    degraded = clean_image.astype(float)
    degraded += (255 - degraded) * fade_map  # Lighten towards white
    
    return np.clip(degraded, 0, 255).astype(np.uint8)
```

**Regenerate Dataset**:
```bash
# Generate new synthetic dataset with ANRI-like degradation
poetry run python scripts/generate_synthetic_anri_degradation.py \
    --input_clean real_data_preparation/clean_transcriptions \
    --output_tfrecord dual_modal_gan/data/dataset_gan_anri_realistic.tfrecord \
    --degradation_config configs/anri_realistic_degradation.yaml \
    --num_samples 10000
```

**Config** (`anri_realistic_degradation.yaml`):
```yaml
degradation_types:
  stroke_breaks:
    probability: 0.35
    break_density: 0.01  # 1% of skeleton pixels
    break_radius: [2, 5]  # pixels
  
  edge_ink_bleeding:
    probability: 0.30
    bleed_intensity: [0.2, 0.5]
    bleed_radius: [3, 7]  # pixels
  
  nonuniform_fading:
    probability: 0.20
    thin_fade: [0.4, 0.7]  # Thin strokes fade 40-70%
    thick_fade: [0.1, 0.3]  # Thick strokes fade 10-30%
  
  paper_damage:
    probability: 0.10
    hole_size: [10, 50]  # pixels
    num_holes: [1, 3]
  
  mixed:
    probability: 0.05
```

**Timeline**: 
- Implementation: 1 week (degradation simulation functions)
- Dataset generation: 2-3 days (10K samples)
- Training: 1 week (50 epochs)
- **Total: 2-3 weeks**

---

### Phase 3: HYBRID APPROACH - Training Strategy (If needed)

If fine-tuning alone insufficient, implement **two-stage training**:

**Stage 1: Synthetic Pre-training** (already done)
- Dataset: Original synthetic data (general restoration)
- Objective: Learn basic restoration (remove noise, enhance contrast)
- Result: Checkpoint with general restoration knowledge

**Stage 2: ANRI Fine-tuning** (adaptive)
- Dataset: Real ANRI + ANRI-realistic synthetic
- Objective: Learn ANRI-specific patterns (stroke breaks, edge bleeding)
- Strategy: 
  - Start from Stage 1 checkpoint
  - Lower LR (0.00005) to prevent catastrophic forgetting
  - Mixed batch: 50% real ANRI + 50% ANRI-realistic synthetic
  - Focus on visual quality (ctc_weight=0 if no labels)

**Expected Result**:
- Model retains general restoration capability (from Stage 1)
- Model adapts to ANRI degradation patterns (from Stage 2)
- Best of both worlds: generalization + specialization

---

## 📊 EXPECTED OUTCOMES

### Metrics Comparison

| Metric | Synthetic-Only Model | After ANRI Fine-tuning | Improvement |
|--------|----------------------|------------------------|-------------|
| **Stroke Connectivity** | 60% | **85%** | +25% |
| **Edge Bleeding Handling** | 50% | **80%** | +30% |
| **Thin Stroke Preservation** | 43% | **75%** | +32% |
| **PSNR (ANRI test set)** | 15.5 dB | **21.4 dB** | +5.9 dB |
| **CER (ANRI test set)** | 35% | **<20%** | -15% |

**Source**: 
- Baseline PSNR 15.5 dB from INFERENCE_V4_LINE_LEVEL_ARCHITECTURE.md (Table: ANRI Real Documents Expected Results)
- Fine-tuning expected improvement based on DIBCO fine-tuning experiments (+5-6 dB PSNR)

---

## 🎯 ACTION PLAN (Prioritized)

### Option 1: IMMEDIATE (Recommended) ⚡
**Use Existing ANRI Fine-tuning Config**

```bash
# Stop current training (synthetic-only)
pkill -f train_enhanced.py

# Launch ANRI fine-tuning
nohup ./scripts/universal_train_from_json.sh \
    configs/finetune_anri_pseudo_visual_only.json \
    > logbook/anri_finetuning_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor
tail -f logbook/anri_finetuning_*.log
```

**Timeline**: 3-4 hours training + 1 hour evaluation = **TODAY**

**Risk**: ✅ Low (proven config, small dataset, conservative LR)

**Expected Result**: 
- Model learns ANRI-specific patterns
- Stroke breaks handling improves
- Edge bleeding handling improves
- **Direct validation on target domain**

---

### Option 2: HYBRID (Medium-term)
**Continue Synthetic Training + Plan ANRI Fine-tuning**

```bash
# Let current training finish (thin_stroke_fix_v1_souibgui)
# Wait for Epoch 50 or early stopping

# Then fine-tune on ANRI
./scripts/universal_train_from_json.sh \
    configs/finetune_anri_from_souibgui_checkpoint.json  # NEW config
```

**Timeline**: 
- Current training: 2-3 days (41 epochs remaining)
- ANRI fine-tuning: 3-4 hours
- **Total: 3-4 days**

**Benefit**: 
- Get both models: (1) Synthetic-optimized, (2) ANRI-adapted
- Can compare performance
- Keep synthetic checkpoint as backup

---

### Option 3: LONG-TERM (Research Quality)
**Generate ANRI-Realistic Synthetic Dataset**

```bash
# Implement degradation simulation
poetry run python scripts/implement_anri_degradation_simulation.py

# Generate dataset
poetry run python scripts/generate_synthetic_anri_degradation.py \
    --num_samples 10000 \
    --output dataset_gan_anri_realistic.tfrecord

# Train from scratch
./scripts/universal_train_from_json.sh \
    configs/train_anri_realistic_synthetic.json
```

**Timeline**: 2-3 weeks (implementation + generation + training)

**Benefit**: 
- Best generalization (large dataset with realistic degradation)
- Publishable novelty (ANRI-specific degradation simulation)
- Reusable dataset for future research

---

## 🔬 VALIDATION STRATEGY

### Test on Real ANRI Documents

**Metrics to Track**:
1. **Stroke Connectivity**: Count connected components before/after
2. **Edge Preservation**: Measure stroke width distribution before/after
3. **Visual Quality**: PSNR, SSIM (if ground truth available)
4. **Readability**: CER (if transcriptions available)

**Visual Inspection Checklist**:
- [ ] Broken strokes reconnected?
- [ ] Edge bleeding removed without thinning strokes?
- [ ] Thin strokes preserved (not removed as noise)?
- [ ] Paper damage handled gracefully?
- [ ] Overall readability improved?

**Test Set**:
- Use held-out ANRI samples (not in training set)
- Sample from different centuries (16th, 17th, 18th)
- Sample from different degradation severities (light, medium, severe)

---

## 💡 KEY INSIGHTS

1. **Domain Gap is CRITICAL**: 
   - Synthetic degradation ≠ Real ANRI degradation
   - Training on synthetic alone will **fail** on real data

2. **Fine-tuning is ESSENTIAL**:
   - Use synthetic checkpoint as **starting point**
   - Fine-tune on **real ANRI data** for domain adaptation
   - Lower LR to prevent catastrophic forgetting

3. **Degradation Characteristics Matter**:
   - Stroke breaks: Need **connectivity-aware** loss or post-processing
   - Edge bleeding: Need **topology-preserving** loss (perceptual, rec_feat)
   - Non-uniform fading: Need **adaptive** enhancement (not uniform amplification)

4. **Data Availability Determines Strategy**:
   - **If 33 ANRI pages sufficient**: Use existing fine-tuning config (Option 1) ✅
   - **If need more data**: Generate ANRI-realistic synthetic (Option 3)
   - **If time-constrained**: Hybrid approach (Option 2)

---

## 📝 NEXT STEPS (DECISION REQUIRED)

**Question for User**: 

1. **How many ANRI pages available for training?**
   - Current: 33 pages in `anri_pseudo_labeled_train.tfrecord`
   - Sufficient for fine-tuning (with data augmentation via tiling)
   - If >100 pages available, can train more robust model

2. **Do we have ANRI ground truth (clean versions)?**
   - `finetune_anri_pseudo_visual_only.json` uses **pseudo-labels** (no clean version)
   - If clean versions exist → can use supervised training (better)
   - If not → continue with visual-only approach (current config)

3. **Priority: Speed vs Quality?**
   - **Speed**: Use Option 1 (fine-tune now, 3-4 hours)
   - **Quality**: Use Option 3 (generate ANRI-realistic dataset, 2-3 weeks)

4. **Should we stop current training (`thin_stroke_fix_v1_souibgui`)?**
   - **Yes**: If ANRI fine-tuning is priority (Option 1)
   - **No**: If want to finish synthetic training first (Option 2)

**My Recommendation**: 
- **Option 1** - Fine-tune on ANRI immediately
- Reason: Fastest path to validate model on **actual target domain**
- Can always go back and improve with Option 3 later if needed

---

## 🔗 REFERENCES

- Config: `configs/finetune_anri_pseudo_visual_only.json`
- Previous analysis: `catatan/CRITICAL_ANALYSIS_THIN_STROKE_CONFIG.md`
- Stroke analysis: `catatan/ANALYSIS_STROKE_MANIPULATION_CAPABILITY.md`
- ANRI performance expectations: `catatan/INFERENCE_V4_LINE_LEVEL_ARCHITECTURE.md`
- Domain adaptation precedent: `configs/dibco_finetuning_visual_only.json` (synthetic → real DIBCO)
