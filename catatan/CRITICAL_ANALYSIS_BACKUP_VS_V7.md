# Critical Analysis: backup/inference_production_v3_aman.py vs V7

**Date**: 2025-10-24  
**Issue**: "Hasilnya belum sesuai yang saya inginkan"  
**Root Cause**: 3 perbedaan fundamental dalam processing strategy

---

## 🔍 Perbedaan Kritis

### 1. TILE EXTRACTION

| Aspect | BACKUP (aman.py) | V7 (current) |
|--------|------------------|--------------|
| **Tile size** | 128×1024 (fixed) | Variable → RESIZE to 1024×128 |
| **Boundary handling** | PAD dengan white (255) | RESIZE to 1024×128 |
| **Information density** | Partial (53.7% pada boundary) | Full (100% semua tile) |
| **Model input** | 128×1024 (padded jika perlu) | 1024×128 (always optimal) |

**Code Comparison**:

```python
# BACKUP (aman.py)
def extract_overlapping_tiles(image):
    for i in range(n_tiles_h):
        y = i * STRIDE_H
        for j in range(n_tiles_w):
            x = j * STRIDE_W
            
            # Extract tile
            tile = image[y:y+TILE_HEIGHT, x:x+TILE_WIDTH]
            
            # Pad if needed ← PADDING STRATEGY
            padded = False
            if tile.shape[0] < TILE_HEIGHT or tile.shape[1] < TILE_WIDTH:
                padded_tile = np.ones((TILE_HEIGHT, TILE_WIDTH), dtype=image.dtype) * 255
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
                padded = True
            
            tiles.append({
                'tile': tile,  # Always 128×1024
                'x': x,
                'y': y,
                'padded': padded  # Flag only
            })
```

```python
# V7 (current)
def extract_overlapping_tiles(image):
    for i in range(n_tiles_h):
        y = i * STRIDE_H
        for j in range(n_tiles_w):
            x = j * STRIDE_W
            
            # Extract tile (irregular size)
            tile_h = min(TILE_HEIGHT, height - y)
            tile_w = min(TILE_WIDTH, width - x)
            tile_original = image[y:y+tile_h, x:x+tile_w]
            
            # ✅ ALWAYS resize to EXACT 1024×128 ← RESIZE STRATEGY
            tile_resized = cv2.resize(tile_original, (TILE_WIDTH, TILE_HEIGHT), 
                                     interpolation=cv2.INTER_CUBIC)
            
            tiles.append({
                'tile': tile_resized,  # Always 1024×128 (resized)
                'x': x,
                'y': y,
                'original_tile_size': (tile_h, tile_w),  # Store for resize back
                'resized_for_model': True
            })
```

---

### 2. TILE MERGING

| Aspect | BACKUP (aman.py) | V7 (current) |
|--------|------------------|--------------|
| **Blending method** | WEIGHTED AVERAGING | MIN OPERATION |
| **Overlap strategy** | Smooth transition (alpha fade) | Preserve dark strokes |
| **Resize back** | NO | YES (from 1024×128 to original size) |
| **Quality priority** | Smooth seams | Stroke continuity |

**Code Comparison**:

```python
# BACKUP (aman.py) - WEIGHTED AVERAGING
def merge_tiles_with_blending(tiles, original_height, original_width, alpha_kernel):
    # Accumulation buffers
    restored_image = np.zeros((original_height, original_width), dtype=np.float32)
    weight_map = np.zeros((original_height, original_width), dtype=np.float32)
    
    for tile_info in tiles:
        x = tile_info['x']
        y = tile_info['y']
        restored_tile = tile_info['restored'].astype(np.float32)
        
        # Get alpha weights
        alpha = alpha_kernel[:h_actual, :w_actual]
        
        # ✅ WEIGHTED AVERAGING (smooth blending)
        restored_image[y:y+h_actual, x:x+w_actual] += restored_tile[:h_actual, :w_actual] * alpha
        weight_map[y:y+h_actual, x:x+w_actual] += alpha
    
    # Normalize by weights
    restored_image = restored_image / np.maximum(weight_map, 1e-6)
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    
    return restored_image
```

```python
# V7 (current) - MIN OPERATION
def merge_tiles_with_blending(tiles, original_height, original_width, alpha_kernel):
    # White background
    restored_image = np.ones((original_height, original_width), dtype=np.float32) * 255.0
    
    for tile_info in tiles:
        x = tile_info['x']
        y = tile_info['y']
        restored_tile_1024x128 = tile_info['restored']
        
        # ✅ RESIZE BACK to original tile size
        if tile_info.get('resized_for_model', False):
            orig_h, orig_w = tile_info['original_tile_size']
            restored_tile = cv2.resize(restored_tile_1024x128, (orig_w, orig_h), 
                                      interpolation=cv2.INTER_CUBIC)
        
        # Get alpha weights
        alpha = alpha_kernel[:h_actual, :w_actual]
        
        # Create overlap mask
        is_overlap = alpha < 0.99
        
        # ✅ MIN OPERATION (preserve dark strokes)
        existing = restored_image[y:y+h_actual, x:x+w_actual]
        new_tile = restored_tile[:h_actual, :w_actual]
        
        blended = np.where(
            is_overlap,
            np.minimum(existing, new_tile),  # MIN in overlap
            new_tile  # Direct in non-overlap
        )
        
        restored_image[y:y+h_actual, x:x+w_actual] = blended
    
    return restored_image
```

---

### 3. CONFIGURATION

| Parameter | BACKUP (aman.py) | V7 (current) |
|-----------|------------------|--------------|
| **OVERLAP** | 32px | 64px |
| **STRIDE_H** | 96px (128-32) | 64px (128-64) |
| **STRIDE_W** | 992px (1024-32) | 960px (1024-64) |
| **Number of tiles** | FEWER (larger stride) | MORE (smaller stride) |

**Impact on Image 1510×1067**:

```python
# BACKUP (aman.py): OVERLAP=32, STRIDE=(96, 992)
n_tiles_h = (1067 + 96 - 1) // 96 = 12
n_tiles_w = (1510 + 992 - 1) // 992 = 2
Total tiles = 12 × 2 = 24 tiles

# V7 (current): OVERLAP=64, STRIDE=(64, 960)
n_tiles_h = max(1, (1067 - 128 + 64 - 1) // 64 + 1) = 16
n_tiles_w = max(1, (1510 - 1024 + 960 - 1) // 960 + 1) = 2
Total tiles = 16 × 2 = 32 tiles
```

---

## 📊 Comparison Summary

### BACKUP (aman.py) Strategy:

**✅ Strengths**:
1. **Simple & Clean**: No double resize
2. **Smooth Transitions**: Weighted averaging eliminates seams
3. **Efficient**: Fewer tiles (24 vs 32)
4. **Natural Blending**: Gradual alpha fade in overlap zones

**⚠️ Weaknesses**:
1. **Padding Artifacts**: 46.3% wasted capacity on boundary tiles
2. **Stroke Weakening**: Averaging may reduce stroke intensity
3. **Lower Overlap**: 32px may not cover thick strokes adequately

---

### V7 (current) Strategy:

**✅ Strengths**:
1. **100% Model Utilization**: Every tile uses full 1024×128 capacity
2. **Stroke Preservation**: MIN operation keeps dark strokes intact
3. **Better Coverage**: 64px overlap handles thick strokes
4. **Adopted User's Success**: Resize to 1024×128 proven effective

**⚠️ Weaknesses**:
1. **Double Resize**: May introduce artifacts (TO 1024×128, BACK to original)
2. **Potential Seams**: MIN operation less smooth than weighted averaging
3. **More Tiles**: 32 tiles vs 24 (slower processing)

---

## 🎯 Recommendations

### Option A: Revert to BACKUP Strategy ✅

**When to use**:
- Smooth transitions more important than stroke preservation
- Processing speed is priority (fewer tiles)
- Padding artifacts acceptable

**Action**: Copy backup logic to current script

---

### Option B: Keep V7 with Tweaks ✅

**When to use**:
- Stroke continuity is critical
- 100% model utilization desired
- Willing to accept potential resize artifacts

**Action**: Fine-tune OVERLAP or post-processing

---

### Option C: HYBRID - Best of Both ✅ **RECOMMENDED**

**Strategy**:
```
1. RESIZE to 1024×128 (from V7) ← Full model capacity
2. WEIGHTED AVERAGING blending (from backup) ← Smooth transitions
3. OVERLAP=64px (from V7) ← Better coverage
```

**Benefits**:
- ✅ 100% model capacity utilization
- ✅ Smooth seam-free blending
- ✅ Better stroke coverage
- ✅ No MIN operation artifacts

**Implementation**:
```python
def merge_tiles_with_blending(tiles, original_height, original_width, alpha_kernel):
    # Accumulation buffers (like backup)
    restored_image = np.zeros((original_height, original_width), dtype=np.float32)
    weight_map = np.zeros((original_height, original_width), dtype=np.float32)
    
    for tile_info in tiles:
        x = tile_info['x']
        y = tile_info['y']
        restored_tile_1024x128 = tile_info['restored']
        
        # ✅ RESIZE BACK (from V7)
        if tile_info.get('resized_for_model', False):
            orig_h, orig_w = tile_info['original_tile_size']
            restored_tile = cv2.resize(restored_tile_1024x128, (orig_w, orig_h), 
                                      interpolation=cv2.INTER_CUBIC)
        else:
            restored_tile = restored_tile_1024x128
        
        restored_tile = restored_tile.astype(np.float32)
        
        # Get alpha
        alpha = alpha_kernel[:h_actual, :w_actual]
        
        # ✅ WEIGHTED AVERAGING (from backup) - smooth blending
        restored_image[y:y+h_actual, x:x+w_actual] += restored_tile[:h_actual, :w_actual] * alpha
        weight_map[y:y+h_actual, x:x+w_actual] += alpha
    
    # Normalize
    restored_image = restored_image / np.maximum(weight_map, 1e-6)
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    
    return restored_image
```

---

## 🔬 What You Want?

Berdasarkan "hasilnya belum sesuai yang saya inginkan", kemungkinan issue:

1. **Seams/Artifacts di Overlap Zones** → HYBRID (weighted averaging)
2. **Stroke Breaks** → Keep V7 (MIN operation + 64px overlap)
3. **Quality vs Speed** → BACKUP (fewer tiles, simpler)

**Next Step**: Konfirmasi issue spesifik yang Anda alami, saya implementasikan solusi yang tepat! 🎯
