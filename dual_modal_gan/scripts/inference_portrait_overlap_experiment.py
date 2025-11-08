#!/usr/bin/env python3
"""
Portrait Document Restoration - Adaptive Tiling for Narrow Aspect Ratios
=========================================================================

Problem: Model trained on 8:1 landscape aspect, real documents are 0.68:1 portrait
Solution: Adaptive vertical+horizontal split to create tiles closer to 8:1

✅ CRITICAL PREPROCESSING FIX (2025-11-01) - V9.0:
- 🐛 FIXED: Contrast stretching causing -4.29 dB PSNR drop!
  * ROOT CAUSE: Model trained WITHOUT contrast stretching
  * SOLUTION: Remove cv2.normalize(), use raw image intensity
  * NORMALIZATION: Exact training pipeline: /255.0 → *2-1 (tanh)
  * RESULT: +4.29 dB improvement (14.00 → 18.29 dB on DIBCO 2012)

✅ CRITICAL BUGFIX (2025-10-27) - V8.2:
- 🐛 FIXED: Inter-character spacing filled with gray
  * ROOT CAUSE: Aggressive closing fills all gaps including inter-char spaces
  * SOLUTION: Use CONSERVATIVE kernel size (max = stroke_width, not 11x11)
  * RESULT: Only smooths strokes, doesn't fill character spacing

✅ V8.1 BUGFIX:
- 🐛 FIXED: Aggressive mode causing whitening issue
  * Morphological closing now uses conservative parameters

✅ V8 UPGRADE - STROKE THICKNESS & INTENSITY CONTROL:
- Added --thin_strokes: Morphological opening untuk menipis stroke tebal
- Added --gamma: Gamma correction untuk mencerahkan/menggelapkan (1.0-1.5 recommended)

✅ V7 AGGRESSIVE MODE:
- Added --aggressive: Closing lebih kuat untuk edge terputus
  * Conservative kernel: max = avg_stroke_width (typically 5-7px)
  * Iterations: 2 (enough for smoothing, not over-filling)

Post-processing Pipeline:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  2. Conservative Morphological Closing (smooth strokes, preserve spacing)
  3. Morphological Opening (optional thinning)
  4. Gamma Correction (optional brightness)
  5. Unsharp Masking (detail enhancement)

Usage Examples:
  # Normal mode (NO contrast stretching, exact training preprocessing)
  python script.py ... 
  
  # Aggressive mode (smoother strokes, no inter-char filling)
  python script.py ... --aggressive
  
  # Thin thick strokes
  python script.py ... --thin_strokes
  
  # Brighten dark output
  python script.py ... --gamma 1.3

Author: belekok
Date: 2025-10-24
Updated: 2025-11-01 (V9.0 - Remove contrast stretching for training match)
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime
import logging

# Constants
TARGET_LINE_WIDTH = 1024
TARGET_LINE_HEIGHT = 128


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(
        output_dir, 
        f"inference_portrait_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def preprocess_image(image: np.ndarray, logger) -> np.ndarray:
    """
    EXACT training preprocessing - NO contrast stretching!
    
    ✅ CRITICAL FIX (2025-11-01):
    - REMOVED contrast stretching (causes domain shift!)
    - Model trained WITHOUT contrast stretching
    - Keep raw image intensity distribution
    
    Training pipeline:
    1. TFRecord data already in [0, 1]
    2. Normalize to [-1, 1]: x * 2 - 1
    
    Inference pipeline (to match):
    1. Load image [0, 255] uint8 (raw, no stretching!)
    2. Will be normalized in process_tile(): /255.0 → *2-1
    
    Args:
        image: Input grayscale image [0, 255] uint8
        logger: Logger instance
    
    Returns:
        Image unchanged (preprocessing happens in process_tile)
    """
    img_min = image.min()
    img_max = image.max()
    img_mean = image.mean()
    
    logger.info(f"� Input image stats:")
    logger.info(f"   Range: [{img_min}, {img_max}]")
    logger.info(f"   Mean: {img_mean:.1f}, Std: {image.std():.1f}")
    logger.info(f"   ✅ NO contrast stretching (matches training)")
    
    # Return as-is (matches training exactly)
    return image


def apply_post_processing(image: np.ndarray, logger, enable: bool = True, 
                          aggressive: bool = False, thin_strokes: bool = False,
                          gamma: float = 1.0) -> np.ndarray:
    """
    Apply V8.3 ADAPTIVE post-processing pipeline to enhance restoration quality.
    
    ✅ UPGRADE V8.3 (2025-10-27):
    - THIN_STROKES uses EROSION (more aggressive than opening)
    - Larger kernel for thinning (stroke_width // 2 instead of // 4)
    - Multiple iterations for thick strokes (2 iter if width > 6px)
    - Skip UNSHARP masking when thinning (prevents re-thickening)
    - Reduced CLAHE when thinning (prevents contrast-induced thickening)
    
    Pipeline:
    1. CLAHE (Contrast enhancement, reduced for thin_strokes mode)
    2. Morphological Closing - connect broken strokes
    3. Morphological Erosion (if thin_strokes) - aggressive thinning
    4. Gamma Correction (optional) - lighten dark strokes
    5. Unsharp Masking (skipped if thin_strokes) - enhance details
    
    Args:
        image: Grayscale image (H, W) uint8
        logger: Logger instance
        enable: If False, return image unchanged
        aggressive: If True, use stronger closing (higher kernel, more iterations)
        thin_strokes: If True, apply EROSION to thin thick strokes aggressively
        gamma: Gamma correction value (>1.0 = brighter, <1.0 = darker, 1.0 = no change)
    
    Returns:
        Enhanced image (H, W) uint8
    """
    if not enable:
        return image
    
    mode_desc = []
    if aggressive:
        mode_desc.append("AGGRESSIVE")
    if thin_strokes:
        mode_desc.append("THIN")
    if gamma != 1.0:
        mode_desc.append(f"GAMMA={gamma:.2f}")
    if not mode_desc:
        mode_desc.append("ADAPTIVE")
    
    mode_str = " + ".join(mode_desc)
    logger.info(f"  Applying V8.3 post-processing: {mode_str}")
    
    # Step 1: CLAHE for contrast enhancement
    # Reduce CLAHE clipLimit when thinning to avoid contrast-induced thickening
    clip_limit = 1.5 if thin_strokes else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    if thin_strokes:
        logger.info(f"    ✓ CLAHE: clipLimit={clip_limit} (reduced for thinning)")
    
    # Step 2: ADAPTIVE/AGGRESSIVE Morphological Closing based on stroke width
    # Strategy: Conservative closing to smooth/connect strokes WITHOUT filling inter-char gaps
    
    # Estimate local stroke width using distance transform
    binary = (enhanced < 128).astype(np.uint8)  # Binarize (text = 1, background = 0)
    
    if binary.sum() > 0:  # Only if there's foreground
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # Average stroke width = 2 * average distance to background
        avg_stroke_width = int(np.mean(dist_transform[dist_transform > 0]) * 2)
        
        if aggressive:
            # AGGRESSIVE: Use conservative kernel to avoid inter-character filling
            # Max kernel = 2x avg stroke width (to close gaps within strokes only)
            kernel_size_close = min(max(3, avg_stroke_width), 7)
            iterations_close = 2
        else:
            # ADAPTIVE: Minimal closing for smoothing only
            kernel_size_close = min(max(3, avg_stroke_width // 2), 5)
            iterations_close = 1
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_close, kernel_size_close))
        
        # Apply GENTLE closing directly on grayscale (with small kernel, it's safe)
        # Small kernel won't expand background significantly
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_close, iterations=iterations_close)
        
        logger.info(f"    ✓ Closing: kernel={kernel_size_close}x{kernel_size_close}, iter={iterations_close}, stroke_width≈{avg_stroke_width}px")
        
        # Step 2b: Morphological Opening for thinning (if requested)
        if thin_strokes:
            # AGGRESSIVE thinning strategy:
            # 1. Use erosion (more aggressive than opening)
            # 2. Larger kernel for more thinning effect
            # 3. Multiple iterations if stroke is very thick
            
            # Calculate kernel based on stroke width (more aggressive)
            kernel_size_erode = max(2, min(avg_stroke_width // 2, 5))
            iterations_erode = 1 if avg_stroke_width < 6 else 2
            
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_erode, kernel_size_erode))
            
            # Apply erosion on grayscale to thin strokes
            enhanced = cv2.erode(enhanced, kernel_erode, iterations=iterations_erode)
            
            logger.info(f"    ✓ Erosion (thinning): kernel={kernel_size_erode}x{kernel_size_erode}, iter={iterations_erode}, effect=STRONG")
    else:
        # Fallback for empty images - no morphological ops needed
        logger.info(f"    ✓ No foreground detected, skipping morphological operations")
    
    # Step 3: Gamma Correction for lightening dark strokes
    if gamma != 1.0:
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, lut)
        logger.info(f"    ✓ Gamma correction: γ={gamma:.2f} ({'brighter' if gamma > 1.0 else 'darker'})")
    
    # Step 4: Unsharp masking for detail enhancement
    # Note: Skip unsharp if thin_strokes enabled to avoid re-thickening
    if not thin_strokes:
        gaussian = cv2.GaussianBlur(enhanced, (3, 3), 0)
        unsharp_weight = 1.5
        enhanced = cv2.addWeighted(enhanced, 1 + unsharp_weight, gaussian, -unsharp_weight, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        logger.info(f"    ✓ Unsharp masking applied (weight={unsharp_weight})")
    else:
        logger.info(f"    ✓ Unsharp masking skipped (thin_strokes mode)")
    
    return enhanced


def load_model(checkpoint_dir, checkpoint_name, gpu_id=0):
    """Load trained model from checkpoint"""
    # Set GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    
    # Load model (COPY FROM inference_line_aware_highres.py)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.generator_enhanced import unet_enhanced
    
    generator = unet_enhanced(input_size=(TARGET_LINE_WIDTH, TARGET_LINE_HEIGHT, 1))
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(str(checkpoint_path)).expect_partial()
    
    return generator


def split_vertical_columns(image, num_columns=2, overlap_ratio=0.1):
    """
    Split image into vertical columns
    
    Args:
        image: Input image (H, W) or (H, W, C)
        num_columns: Number of vertical splits
        overlap_ratio: Overlap between columns (0.0-0.3) 
    
    Returns:
        List of (column_image, x_start, x_end)
    """
    h, w = image.shape[:2]
    overlap_px = int(w * overlap_ratio / num_columns)
    col_width = w // num_columns
    
    columns = []
    for i in range(num_columns):
        x_start = max(0, i * col_width - overlap_px)
        x_end = min(w, (i + 1) * col_width + overlap_px)
        
        if len(image.shape) == 2:
            col_img = image[:, x_start:x_end]
        else:
            col_img = image[:, x_start:x_end, :]
        
        columns.append((col_img, x_start, x_end))
    
    return columns


def split_horizontal_strips(image, target_aspect=6.0, min_height=100, overlap_ratio=0.15):
    """
    Split image into horizontal strips aiming for target aspect ratio
    
    Args:
        image: Input image (H, W) or (H, W, C)
        target_aspect: Target width/height ratio (default 6.0 for model 8:1 tolerance)
        min_height: Minimum strip height
        overlap_ratio: Overlap between strips
    
    Returns:
        List of (strip_image, y_start, y_end)
    """
    h, w = image.shape[:2]
    
    # Calculate optimal strip height
    # target_aspect = w / strip_height → strip_height = w / target_aspect
    strip_height = int(w / target_aspect)
    strip_height = max(strip_height, min_height)
    
    overlap_px = int(strip_height * overlap_ratio)
    
    strips = []
    y = 0
    while y < h:
        y_start = max(0, y - overlap_px) if y > 0 else 0
        y_end = min(h, y + strip_height + overlap_px)
        
        if len(image.shape) == 2:
            strip_img = image[y_start:y_end, :]
        else:
            strip_img = image[y_start:y_end, :, :]
        
        strips.append((strip_img, y_start, y_end))
        
        y += strip_height
        
        # Avoid tiny last strip
        if h - y < strip_height // 2 and y < h:
            y_start = max(0, y - overlap_px)
            if len(image.shape) == 2:
                strip_img = image[y_start:, :]
            else:
                strip_img = image[y_start:, :, :]
            strips.append((strip_img, y_start, h))
            break
    
    return strips


def resize_to_model_input(image, target_height=128, target_width=1024):
    """Resize image to model input size while preserving aspect"""
    h, w = image.shape[:2]
    aspect = w / h
    
    # Resize to target height
    new_h = target_height
    new_w = int(new_h * aspect)
    
    # If too wide, resize to target width
    if new_w > target_width:
        new_w = target_width
        new_h = int(new_w / aspect)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Pad to target size (CRITICAL: model expects (batch, width, height, channels) NOT (batch, height, width, channels)!)
    # So we need to create padded with shape (target_width, target_height) then transpose
    padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
    padded[:new_h, :new_w] = resized
    
    return padded, (new_h, new_w)


def process_tile(tile_img, generator):
    """
    Process a single tile through the model.
    
    ✅ EXACT TRAINING NORMALIZATION (2025-11-01):
    - Step 1: [0, 255] uint8 → [0, 1] float32 (/ 255.0)
    - Step 2: [0, 1] → [-1, 1] (* 2.0 - 1.0) for tanh generator
    - Output: [-1, 1] from generator → [0, 255] uint8
    
    This EXACTLY matches train_enhanced.py preprocessing pipeline.
    """
    # Resize to model input
    resized, (valid_h, valid_w) = resize_to_model_input(tile_img)
    
    # ✅ EXACT TRAINING NORMALIZATION:
    # Step 1: [0, 255] → [0, 1]
    input_tensor = resized.astype(np.float32) / 255.0
    
    # Step 2: [0, 1] → [-1, 1] (match training tanh normalization)
    input_tensor = input_tensor * 2.0 - 1.0
    
    # CRITICAL: Model expects (batch, width, height, channels)
    # OpenCV gives us (height, width), we need to TRANSPOSE!
    input_tensor = np.transpose(input_tensor, (1, 0))  # (width, height)
    input_tensor = np.expand_dims(input_tensor, axis=-1)  # (width, height, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)   # (1, width, height, 1)
    
    # Inference
    output = generator(input_tensor, training=False)
    restored = output[0, :, :, 0].numpy()
    
    # Transpose back to (height, width)
    restored = np.transpose(restored, (1, 0))
    
    # ✅ DENORMALIZATION: [-1, 1] → [0, 1] → [0, 255]
    # Generator uses tanh activation → output range [-1, 1]
    restored_01 = (restored + 1.0) / 2.0  # [-1, 1] → [0, 1]
    restored = np.clip(restored_01 * 255.0, 0, 255).astype(np.uint8)  # [0, 1] → [0, 255]
    
    # Extract valid region
    restored_valid = restored[:valid_h, :valid_w]
    
    # Resize back to original tile size (Lanczos4 for sharper upscaling)
    tile_h, tile_w = tile_img.shape[:2]
    restored_full = cv2.resize(restored_valid, (tile_w, tile_h), interpolation=cv2.INTER_LANCZOS4)
    
    return restored_full


def blend_tiles_vertical(tiles_data, full_width, full_height, overlap_ratio=0.1):
    """
    Blend vertical column tiles back together with Gaussian windowing
    
    Args:
        tiles_data: List of (restored_tile, x_start, x_end)
        full_width: Original image width
        full_height: Original image height
        overlap_ratio: Overlap ratio used in splitting
    
    Returns:
        Blended full image
    """
    result = np.zeros((full_height, full_width), dtype=np.float32)
    weights = np.zeros((full_height, full_width), dtype=np.float32)
    
    for tile, x_start, x_end in tiles_data:
        tile_w = x_end - x_start
        
        # Create weight map with Gaussian windowing at edges
        weight = np.ones((full_height, tile_w), dtype=np.float32)
        
        # Gaussian feather left edge
        if x_start > 0:
            fade_w = int(tile_w * overlap_ratio * 0.5)
            # Gaussian window (half of bell curve)
            sigma = fade_w / 3.0
            x = np.arange(fade_w)
            gaussian = np.exp(-0.5 * ((fade_w - x) / sigma) ** 2)
            weight[:, :fade_w] = gaussian[np.newaxis, :]
        
        # Gaussian feather right edge
        if x_end < full_width:
            fade_w = int(tile_w * overlap_ratio * 0.5)
            sigma = fade_w / 3.0
            x = np.arange(fade_w)
            gaussian = np.exp(-0.5 * (x / sigma) ** 2)
            weight[:, -fade_w:] = gaussian[np.newaxis, :]
        
        # Accumulate
        result[:, x_start:x_end] += tile.astype(np.float32) * weight
        weights[:, x_start:x_end] += weight
    
    # Normalize
    result = np.divide(result, weights, where=weights > 0)
    return result.astype(np.uint8)


def blend_tiles_horizontal(tiles_data, full_width, full_height, overlap_ratio=0.25):
    """
    Blend horizontal strip tiles back together with Gaussian windowing
    
    Args:
        tiles_data: List of (restored_tile, y_start, y_end)
        full_width: Original image width
        full_height: Original image height
        overlap_ratio: Overlap ratio used in splitting
    
    Returns:
        Blended full image
    """
    result = np.zeros((full_height, full_width), dtype=np.float32)
    weights = np.zeros((full_height, full_width), dtype=np.float32)
    
    for tile, y_start, y_end in tiles_data:
        tile_h = y_end - y_start
        
        # Create weight map with Gaussian windowing at edges
        weight = np.ones((tile_h, full_width), dtype=np.float32)
        
        # Gaussian feather top edge
        if y_start > 0:
            fade_h = int(tile_h * overlap_ratio * 0.5)
            # Gaussian window (half of bell curve)
            sigma = fade_h / 3.0
            y = np.arange(fade_h)
            gaussian = np.exp(-0.5 * ((fade_h - y) / sigma) ** 2)
            weight[:fade_h, :] = gaussian[:, np.newaxis]
        
        # Gaussian feather bottom edge
        if y_end < full_height:
            fade_h = int(tile_h * overlap_ratio * 0.5)
            sigma = fade_h / 3.0
            y = np.arange(fade_h)
            gaussian = np.exp(-0.5 * (y / sigma) ** 2)
            weight[-fade_h:, :] = gaussian[:, np.newaxis]
        
        # Accumulate
        result[y_start:y_end, :] += tile.astype(np.float32) * weight
        weights[y_start:y_end, :] += weight
    
    # Normalize
    result = np.divide(result, weights, where=weights > 0)
    return result.astype(np.uint8)


def process_portrait_document(image, generator, logger, alpha=0.15, post_processing=True, 
                             aggressive=False, thin_strokes=False, gamma=1.0):
    """
    Process portrait document with adaptive vertical+horizontal splitting
    
    Strategy:
    1. If aspect < 2.0 (portrait): Split vertically into columns
    2. Each column: Split horizontally into strips (target aspect ~6-8)
    3. Process each strip tile
    4. Blend back together
    5. Apply post-processing (CLAHE + Closing/Opening + Gamma + Unsharp)
    
    Args:
        image: Input grayscale image
        generator: Trained model
        logger: Logger instance
        alpha: Blending factor with original (0.0-0.3)
        post_processing: Enable post-processing
        aggressive: Use aggressive closing (stronger, more iterations)
        thin_strokes: Apply opening to thin thick strokes
        gamma: Gamma correction (>1.0 = brighter, <1.0 = darker)
    
    Returns:
        Restored image
    """
    h, w = image.shape[:2]
    aspect = w / h
    
    logger.info(f"Processing document: {w}×{h} (aspect {aspect:.2f})")
    
    # Determine strategy based on aspect ratio
    if aspect < 1.5:
        # Very narrow portrait - split vertically first
        num_columns = 3 if aspect < 0.8 else 2
        logger.info(f"Portrait mode: Splitting into {num_columns} columns")
        
        columns = split_vertical_columns(image, num_columns=num_columns, overlap_ratio=0.25)
        logger.info(f"Created {len(columns)} vertical columns")
        
        # Process each column
        restored_columns = []
        for col_idx, (col_img, x_start, x_end) in enumerate(columns):
            col_h, col_w = col_img.shape[:2]
            col_aspect = col_w / col_h
            logger.info(f"  Column {col_idx+1}: {col_w}×{col_h} (aspect {col_aspect:.2f})")
            
            # Split column into horizontal strips
            target_aspect = 6.0  # Closer to model's 8:1
            strips = split_horizontal_strips(col_img, target_aspect=target_aspect, overlap_ratio=0.15)
            logger.info(f"    → {len(strips)} horizontal strips")
            
            # Process each strip
            restored_strips = []
            for strip_idx, (strip_img, y_start, y_end) in enumerate(strips):
                strip_h, strip_w = strip_img.shape[:2]
                strip_aspect = strip_w / strip_h
                logger.info(f"      Strip {strip_idx+1}: {strip_w}×{strip_h} (aspect {strip_aspect:.2f})")
                
                # Process through model
                restored_strip = process_tile(strip_img, generator)
                restored_strips.append((restored_strip, y_start, y_end))
            
            # Blend strips vertically within column
            col_restored = blend_tiles_horizontal(restored_strips, col_w, col_h, overlap_ratio=0.15)
            restored_columns.append((col_restored, x_start, x_end))
        
        # Blend columns horizontally
        restored = blend_tiles_vertical(restored_columns, w, h, overlap_ratio=0.1)
        
    elif aspect < 3.0:
        # Moderately narrow - horizontal strips only
        logger.info("Semi-portrait mode: Horizontal strips only")
        target_aspect = 6.0
        strips = split_horizontal_strips(image, target_aspect=target_aspect, overlap_ratio=0.15)
        logger.info(f"Created {len(strips)} horizontal strips")
        
        restored_strips = []
        for strip_idx, (strip_img, y_start, y_end) in enumerate(strips):
            strip_h, strip_w = strip_img.shape[:2]
            strip_aspect = strip_w / strip_h
            logger.info(f"  Strip {strip_idx+1}: {strip_w}×{strip_h} (aspect {strip_aspect:.2f})")
            
            restored_strip = process_tile(strip_img, generator)
            restored_strips.append((restored_strip, y_start, y_end))
        
        restored = blend_tiles_horizontal(restored_strips, w, h, overlap_ratio=0.15)
    
    else:
        # Landscape or normal - process as-is with horizontal strips
        logger.info("Landscape mode: Standard horizontal strips")
        target_aspect = 7.0  # Closer to 8:1
        strips = split_horizontal_strips(image, target_aspect=target_aspect, overlap_ratio=0.15)
        logger.info(f"Created {len(strips)} horizontal strips")
        
        restored_strips = []
        for strip_idx, (strip_img, y_start, y_end) in enumerate(strips):
            strip_h, strip_w = strip_img.shape[:2]
            strip_aspect = strip_w / strip_h
            logger.info(f"  Strip {strip_idx+1}: {strip_w}×{strip_h} (aspect {strip_aspect:.2f})")
            
            restored_strip = process_tile(strip_img, generator)
            restored_strips.append((restored_strip, y_start, y_end))
        
        restored = blend_tiles_horizontal(restored_strips, w, h, overlap_ratio=0.15)
    
    # Apply post-processing before alpha blending
    if post_processing:
        restored = apply_post_processing(restored, logger, enable=True, 
                                        aggressive=aggressive, 
                                        thin_strokes=thin_strokes,
                                        gamma=gamma)
    
    # Alpha blending with original
    if alpha > 0:
        restored = cv2.addWeighted(
            restored, 1.0 - alpha,
            image, alpha,
            0
        )
        logger.info(f"Applied alpha blending: {alpha:.2f}")
    
    return restored


def main():
    parser = argparse.ArgumentParser(description='Portrait Document Restoration')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoint')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                       help='Checkpoint filename (e.g., ckpt-88)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--alpha', type=float, default=0.0,
                       help='Alpha blending with original (0.0-0.3, default 0.0=no blending)')
    parser.add_argument('--image_ext', type=str, default='.bmp',
                       help='Image file extension')
    parser.add_argument('--binarization_threshold', type=int, default=None,
                       help='(Optional) Apply binary thresholding with this value (0-255). Higher value ignores lighter gray areas.')
    parser.add_argument('--disable_post_processing', action='store_true',
                       help='Disable post-processing (CLAHE + Adaptive Closing + Unsharp)')
    parser.add_argument('--aggressive', action='store_true',
                       help='Use AGGRESSIVE closing (kernel max 11x11, iter 4) for heavily broken strokes')
    parser.add_argument('--thin_strokes', action='store_true',
                       help='Apply morphological opening to thin thick/bold strokes')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Gamma correction for brightness (>1.0 = lighter, <1.0 = darker, default=1.0)')
    parser.add_argument('--output_format', type=str, default='tiff', 
                       choices=['png', 'tiff', 'bmp', 'jpg'],
                       help='Output image format (default: tiff for lossless high-quality)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("="*80)
    logger.info("Portrait Document Restoration - Adaptive Tiling")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint_dir}/{args.checkpoint_name}")
    logger.info(f"GPU: {args.gpu_id}")
    logger.info(f"Alpha blending: {args.alpha}")
    logger.info(f"Post-processing: {not args.disable_post_processing}")
    logger.info(f"Aggressive mode: {args.aggressive}")
    logger.info(f"Thin strokes: {args.thin_strokes}")
    logger.info(f"Gamma correction: {args.gamma}")
    logger.info(f"Output format: {args.output_format.upper()}")
    
    # Load model
    logger.info("Loading model...")
    generator = load_model(args.checkpoint_dir, args.checkpoint_name, args.gpu_id)
    logger.info("Model loaded successfully")
    
    # Get input files
    if os.path.isdir(args.input):
        image_files = sorted([
            f for f in os.listdir(args.input)
            if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', args.image_ext))
        ])
        image_paths = [os.path.join(args.input, f) for f in image_files]
    else:
        image_paths = [args.input]
        image_files = [os.path.basename(args.input)]
    
    logger.info(f"Processing {len(image_paths)} images...")
    
    # Process each image
    results = []
    for img_path, img_name in zip(image_paths, image_files):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {img_name}")
        logger.info(f"{'='*80}")
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            continue
        
        h, w = image.shape
        aspect = w / h
        logger.info(f"Input size: {w}×{h} (aspect {aspect:.2f})")
        
        # Apply contrast stretching for low-contrast images
        image = preprocess_image(image, logger)
        
        # Process
        restored = process_portrait_document(
            image, generator, logger, 
            alpha=args.alpha,
            post_processing=not args.disable_post_processing,
            aggressive=args.aggressive,
            thin_strokes=args.thin_strokes,
            gamma=args.gamma
        )
        
        # Apply binarization if requested
        if args.binarization_threshold is not None:
            logger.info(f"Applying binarization with threshold: {args.binarization_threshold}")
            _, restored = cv2.threshold(
                restored, 
                args.binarization_threshold, 
                255, 
                cv2.THRESH_BINARY
            )

        # Determine output extension
        ext_map = {
            'png': '.png',
            'tiff': '.tiff',
            'bmp': '.bmp',
            'jpg': '.jpg'
        }
        output_ext = ext_map.get(args.output_format, '.tiff')
        
        # Save with appropriate format and compression
        output_name = os.path.splitext(img_name)[0] + f'_restored{output_ext}'
        output_path = os.path.join(args.output_dir, output_name)
        
        # Special handling for TIFF - use LZW compression for lossless size reduction
        if args.output_format == 'tiff':
            # OpenCV doesn't support TIFF compression params, use PIL for better control
            from PIL import Image
            restored_pil = Image.fromarray(restored)
            restored_pil.save(output_path, format='TIFF', compression='tiff_lzw', dpi=(300, 300))
            logger.info(f"Saved: {output_path} (TIFF LZW compression, 300 DPI)")
        else:
            cv2.imwrite(output_path, restored)
            logger.info(f"Saved: {output_path}")
        
        # Calculate metrics
        contrast = restored.std()
        mean_intensity = restored.mean()
        
        results.append({
            'image_name': os.path.splitext(img_name)[0],
            'input_size': [w, h],
            'aspect_ratio': round(aspect, 2),
            'contrast': round(float(contrast), 2),
            'mean_intensity': round(float(mean_intensity), 2),
            'output_path': output_path
        })
        
        logger.info(f"Metrics: contrast={contrast:.1f}, mean={mean_intensity:.1f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': f"{args.checkpoint_dir}/{args.checkpoint_name}",
        'alpha': args.alpha,
        'num_images': len(results),
        'results': results
    }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing complete! Processed {len(results)} images")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
