#!/usr/bin/env python3
"""
Portrait Document Restoration - Adaptive Tiling for Narrow Aspect Ratios
=========================================================================

Problem: Model trained on 8:1 landscape aspect, real documents are 0.68:1 portrait
Solution: Adaptive vertical+horizontal split to create tiles closer to 8:1

Author: belekok
Date: 2025-10-24
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
    Apply contrast stretching for low-contrast images.
    
    This fixes the Image #8 failure where compressed dynamic range [90-234]
    causes out-of-distribution input for the model.
    
    Args:
        image: Input grayscale image
        logger: Logger instance
    
    Returns:
        Preprocessed image with full [0, 255] range
    """
    img_std = image.std()
    img_min = image.min()
    img_max = image.max()
    img_range = img_max - img_min
    
    # Detect low contrast images
    if img_std < 30 or img_range < 200:
        # Apply linear contrast stretching
        stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        new_std = stretched.std()
        
        logger.info(f"🔧 Applied contrast stretching for low-contrast input:")
        logger.info(f"   Before: range=[{img_min}, {img_max}], std={img_std:.1f}")
        logger.info(f"   After:  range=[{stretched.min()}, {stretched.max()}], std={new_std:.1f}")
        logger.info(f"   Reason: Prevents model saturation for clean/compressed images")
        
        return stretched
    
    return image


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
    """Process a single tile through the model"""
    # Resize to model input
    resized, (valid_h, valid_w) = resize_to_model_input(tile_img)
    
    # CRITICAL FIX: Normalize to [-1, 1] to match TRAINING!
    # Training uses: degraded_images * 2.0 - 1.0 (see train_enhanced.py line 213)
    # Model expects tanh-normalized input range [-1, 1]
    input_tensor = resized.astype(np.float32) / 255.0  # [0, 255] → [0, 1]
    input_tensor = input_tensor * 2.0 - 1.0            # [0, 1] → [-1, 1] (MATCH TRAINING!)
    
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
    
    # CRITICAL FIX: Model output range is [-1, 1] not [0, 1]!
    # Generator uses tanh activation → output range [-1, 1]
    # Denormalize: [-1, 1] → [0, 255]
    restored = np.clip((restored + 1.0) * 127.5, 0, 255).astype(np.uint8)
    
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


def process_portrait_document(image, generator, logger, alpha=0.15):
    """
    Process portrait document with adaptive vertical+horizontal splitting
    
    Strategy:
    1. If aspect < 2.0 (portrait): Split vertically into columns
    2. Each column: Split horizontally into strips (target aspect ~6-8)
    3. Process each strip tile
    4. Blend back together
    
    Args:
        image: Input grayscale image
        generator: Trained model
        logger: Logger instance
        alpha: Blending factor with original (0.0-0.3)
    
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
        restored = process_portrait_document(image, generator, logger, alpha=args.alpha)
        
        # Save
        output_name = os.path.splitext(img_name)[0] + '_restored.png'
        output_path = os.path.join(args.output_dir, output_name)
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
