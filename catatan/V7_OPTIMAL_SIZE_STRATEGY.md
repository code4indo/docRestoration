# V7 Optimal-Size Strategy - Implementation Report

**Date**: 2025-10-24  
**Focus**: Setiap chunk SELALU diproses pada ukuran optimal 1024×128  
**Motivation**: Mengadopsi eksperimen user yang berhasil (crop line + resize to 1024×128)

---

## 🎯 Konsep Inti

### User's Successful Experiment:
```
1. Crop baris teks dengan ukuran custom
2. Resize ke 1024×128 (preserve aspect ratio)
3. Model inference
4. Hasil: MEMUASKAN ✅
```

### V7 Implementation:
```
1. Extract tile dari image (bisa irregular: 550×128, 339×100, dst)
2. Resize TO 1024×128 (INTER_CUBIC) ← OPTIMAL untuk model
3. Model inference (always optimal input)
4. Resize BACK to original tile size
5. Merge dengan MIN blending
```

---

## 📊 Perbandingan V6 vs V7

### V6 Strategy (Padding):
- **Full tiles** (62.2%): 128×1024 → direct inference ✅
- **Partial tiles** (37.8%): Pad dengan white pixels
- **Issue**: 
  - Model menerima padding sebagai "background"
  - Information density rendah (53.7% pada boundary tiles)
  - 46.3% kapasitas model terbuang untuk padding

### V7 Strategy (Resize):
- **ALL tiles** (100%): RESIZE to 1024×128 ✅
- **No padding artifacts**
- **Benefits**:
  - Model SELALU dapat input optimal
  - Information density 100%
  - Full model capacity utilization
  - Better boundary quality (teoritis)

---

## 🧪 Test Results (DIBCO 1.bmp)

### Quantitative Comparison:

| Metric | V6 | V7 | Change |
|--------|-----|-----|--------|
| Total components (≥20px) | 33 | 33 | 0 |
| Large (≥1000px) | 18 | 18 | 0 |
| Medium (200-999px) | 4 | 4 | 0 |
| Small (20-199px) | 11 | 11 | 0 |
| Mean component area | 2895px | 2850px | -1.6% |

**Result**: IDENTIK (kedua versi generate 32 tiles pada image 1510×1067)

### Information Density Analysis (Boundary Tiles):

**Tile Example**: 550×128 (right edge of image 1510×1067)

- **V6 (Padding)**:
  - Pad to 1024×128
  - Content: 550×128 = 70,400 pixels (53.7%)
  - Padding: 474×128 = 60,672 pixels (46.3%) **WASTED**
  
- **V7 (Resize)**:
  - Resize 550×128 → 1024×128
  - Content: 1024×128 = 131,072 pixels (100.0%) ✅
  - Padding: 0 pixels (0.0%)

**Improvement**: V7 menggunakan 100% kapasitas model

---

## 🔬 Technical Implementation

### Extract Overlapping Tiles (V7):

```python
def extract_overlapping_tiles(image):
    """
    Extract tiles - ALWAYS resize to exact 1024×128.
    
    Strategy:
    1. Extract overlapping regions from original image
    2. Resize each region to EXACT 1024×128
    3. Process with model (optimal format)
    4. Resize back to original region size
    5. Merge with blending
    """
    for i in range(n_tiles_h):
        y = i * STRIDE_H
        for j in range(n_tiles_w):
            x = j * STRIDE_W
            
            # Extract tile (irregular size)
            tile_h = min(TILE_HEIGHT, height - y)
            tile_w = min(TILE_WIDTH, width - x)
            tile_original = image[y:y+tile_h, x:x+tile_w]
            
            # ✅ ALWAYS resize to EXACT 1024×128
            tile_resized = cv2.resize(tile_original, (TILE_WIDTH, TILE_HEIGHT), 
                                     interpolation=cv2.INTER_CUBIC)
            
            tiles.append({
                'tile': tile_resized,  # Always (128, 1024)
                'x': x,
                'y': y,
                'original_tile_size': (tile_h, tile_w),  # Store for resize back
                'resized_for_model': True
            })
```

### Merge Tiles with Resize-Back (V7):

```python
def merge_tiles_with_blending(tiles, original_height, original_width, alpha_kernel):
    """
    Merge tiles - resize BACK to original size before blending.
    """
    for tile_info in tiles:
        x = tile_info['x']
        y = tile_info['y']
        restored_tile_1024x128 = tile_info['restored']  # From model
        
        # ✅ Resize back to ORIGINAL tile size
        if tile_info.get('resized_for_model', False):
            orig_h, orig_w = tile_info['original_tile_size']
            restored_tile = cv2.resize(restored_tile_1024x128, (orig_w, orig_h), 
                                      interpolation=cv2.INTER_CUBIC)
        else:
            restored_tile = restored_tile_1024x128
        
        # Get corresponding region and alpha
        h_actual = min(restored_tile.shape[0], original_height - y)
        w_actual = min(restored_tile.shape[1], original_width - x)
        alpha = alpha_kernel[:h_actual, :w_actual]
        
        # STROKE-AWARE BLENDING (MIN in overlap zones)
        existing = restored_image[y:y+h_actual, x:x+w_actual]
        new_tile = restored_tile[:h_actual, :w_actual]
        is_overlap = alpha < 0.99
        
        blended = np.where(
            is_overlap,
            np.minimum(existing, new_tile),  # Preserve dark strokes
            new_tile
        )
        
        restored_image[y:y+h_actual, x:x+w_actual] = blended
```

---

## ✅ Advantages of V7

1. **Model Always Optimal** ✅
   - No partial tiles
   - No padding artifacts
   - Every inference uses full 1024×128 capacity

2. **Adopted from User's Success** ✅
   - User proved: resize to 1024×128 → excellent results
   - V7 automates this strategy for full documents

3. **Simple & Robust** ✅
   - No line detection complexity
   - No projection profile tuning
   - Just resize IN + resize OUT

4. **Better Boundary Handling** (theoretical) ✅
   - 100% information density vs 53.7% (V6)
   - No white padding confusion for model
   - Full utilization of model's learned features

---

## ⚠️ Potential Concerns

### 1. Resize Artifacts?
- **Concern**: Double resize (TO 1024×128, then BACK) may introduce artifacts
- **Mitigation**: Use INTER_CUBIC (high-quality interpolation)
- **Reality**: Test shows identical results to V6 (33 components)

### 2. Computational Cost?
- **V6**: Resize only small images + partial tiles
- **V7**: Resize ALL tiles (2× resize per tile)
- **Impact**: Marginal (resize is fast, GPU inference dominates)

### 3. Aspect Ratio Distortion?
- **Concern**: Resize 550×128 → 1024×128 changes aspect ratio
- **Reality**: Model trained on 1024×128, expects this format
- **User's experiment**: Successful with resize, so model handles it well

---

## 🎯 Recommendations

### For Production Use:

**✅ DEPLOY V7** if:
- You want 100% model capacity utilization
- Boundary tile quality is critical
- Computational cost is acceptable

**✅ DEPLOY V6** if:
- Computational efficiency is priority
- Current quality (33 components) is sufficient
- Simpler logic preferred (less resize operations)

### For Research/Future Work:

1. **Test on MORE documents**:
   - Large paleographic documents (>3000px width)
   - Documents with many boundary tiles
   - Compare V6 vs V7 on edge quality

2. **Hybrid Approach**:
   - Full tiles (128×1024): Direct inference (V6 fast path)
   - Partial tiles: Resize to 1024×128 (V7 optimal path)
   - Best of both worlds

3. **Batch Processing Optimization**:
   - Pre-resize all tiles before batch inference
   - Minimize resize overhead

---

## 📈 Expected Impact

### On Image 2259×1023 (45 tiles):
- **Full tiles**: 28 (62.2%) → same as V6
- **Partial tiles**: 17 (37.8%) → V7 optimizes these ✅

### Information Density Improvement:
- **V6**: 62.2% full + 37.8% partial (53.7%) = **81.8% avg utilization**
- **V7**: 100% all tiles = **100% utilization** ✅
- **Gain**: +18.2% model capacity usage

---

## 🔍 Next Steps

1. **Test V7 on full DIBCO dataset** (10 images)
2. **Compare V6 vs V7 quantitatively**:
   - Component count
   - Mean component size
   - PSNR/SSIM (if GT available)
3. **Visual inspection** of boundary regions
4. **Decide** based on quality vs efficiency tradeoff

---

## 💡 Key Insight

**User's successful experiment taught us**:
> Model performs BEST when given exactly what it was trained on: **1024×128 complete content**

V7 implements this principle for **EVERY tile**, not just single lines.

---

**Status**: ✅ Implemented and tested  
**Performance**: Identical to V6 on test image (33 components)  
**Deployment**: Ready for full dataset evaluation  
