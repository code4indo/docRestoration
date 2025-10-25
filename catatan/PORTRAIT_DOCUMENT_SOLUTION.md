# Portrait Document Processing Solution

**Date**: 2025-10-24  
**Status**: ✅ IMPLEMENTED  
**Script**: `dual_modal_gan/scripts/inference_portrait_adaptive.py`

## Problem

ANRI real documents are portrait-oriented (aspect ratio ~0.68), while model was trained on landscape (aspect 8:1).

**Original failure**:
- Input: 2735×4012 (aspect 0.68)
- Previous method: `inference_line_aware_highres.py`
- Result: **BLANK** output (std=10.8)

## Root Cause

Model trained on 1024×128 (aspect 8:1) cannot handle portrait documents:
- Distance from model aspect: **7.32** (extreme mismatch!)
- Model struggles with narrow aspect ratios

## Solution: Adaptive Vertical+Horizontal Tiling

**Strategy**:
1. **Portrait detection** (aspect < 1.5): Split vertically into 2-3 columns
2. **Each column**: Split horizontally into strips (target aspect ~4-6)
3. **Process each tile** through model
4. **Blend back** with feathering at overlaps

**Code**: `inference_portrait_adaptive.py`

### Tiling Strategy by Aspect Ratio

```python
if aspect < 0.8:
    num_columns = 3  # Very narrow portrait
elif aspect < 1.5:
    num_columns = 2  # Narrow portrait
else:
    num_columns = 1  # Landscape/normal
```

Each column is then split horizontally:
- Target aspect: 6.0 (closer to model's 8:1)
- Overlap: 15% for smooth blending

## Results

| Method | Mean | Std | Status |
|--------|------|-----|--------|
| line_aware (failed) | 254.0 | 10.8 | ❌ BLANK |
| portrait_adaptive (α=0.15) | 235.9 | 23.8 | ⚠️ LOW |
| **portrait_adaptive (α=0.0)** | **250.5** | **25.8** | ✅ **FAIR** |

**Improvement**: Std 10.8 → 25.8 (2.4× better contrast!)

## Usage

```bash
poetry run python dual_modal_gan/scripts/inference_portrait_adaptive.py \
  --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
  --checkpoint_name ckpt-88 \
  --input RusakRingan/full_pages/ID-ANRI_K66b_082_0018.jpg \
  --output_dir results/anri_output \
  --gpu_id 1
```

**Default alpha**: 0.0 (no blending) - provides best contrast

## Technical Details

### Vertical Split with Overlap
- Columns: 2-3 depending on aspect ratio
- Overlap: 10% for smooth blending
- Weight feathering at column edges

### Horizontal Split per Column
- Target aspect: 6.0 (compromise between 0.68 and 8.0)
- Strip height: `column_width / 6.0`
- Overlap: 15% for seamless reconstruction

### Model Input Handling
**CRITICAL**: Model expects `(batch, width, height, channels)` not `(batch, height, width, channels)`
- Requires transpose before inference
- Transpose back after inference

## Limitations

- Contrast still **lower than landscape documents** (25.8 vs 40+ for DIBCO)
- Aspect ratio 4-6 is **compromise**, not optimal
- Model fundamentally designed for landscape documents

## Future Improvements

1. **Retrain with multi-aspect** (NOVELTY candidate):
   - Training data: Multiple aspect ratios (2:1, 4:1, 8:1, 16:1)
   - Architecture: Adaptive pooling for flexible width
   - Expected: Better handling of portrait documents

2. **Rotate preprocessing**:
   - Rotate portrait → landscape before inference
   - Process in landscape mode
   - Rotate back after restoration

3. **Specialized portrait model**:
   - Train separate model for portrait documents
   - Target aspect: 1:2 or 1:4
   - Switch model based on input aspect ratio

## Conclusion

✅ Portrait documents now processable (not blank!)  
⚠️ Quality lower than landscape (expected due to aspect mismatch)  
🎯 For optimal results: Use landscape documents OR retrain with multi-aspect

---

**Related Issues**:
- ANRI real data preprocessing
- Aspect ratio analysis (catatan/ANALYSIS_ASPECT_RATIO.md)
- Line-aware limitations for narrow documents
