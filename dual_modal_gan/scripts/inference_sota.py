#!/usr/bin/env python3
"""
SOTA Document Restoration - Adaptive Quality-Based Processing
==============================================================

Target: Beat DE-GAN baseline (PSNR 24.9 dB, F-Measure 99.5%)

Adaptive Strategy:
- High-quality images (PSNR >18 dB): Minimal processing, preserve details
- Medium-quality images (15-18 dB): Moderate enhancement
- Low-quality images (PSNR <15 dB): Aggressive enhancement with bilateral filter

Key Improvements:
1. ✅ Adaptive overlap based on image quality
2. ✅ Adaptive alpha blending (0.0-0.3)
3. ✅ Conditional bilateral filtering
4. ✅ Quality-based parameter selection
5. ✅ Sharper upscaling (INTER_CUBIC instead of LANCZOS4)
6. ✅ Minimal smoothing for clean images

Author: belekok
Date: 2025-10-25
Version: SOTA Adaptive v2.0
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
        f"inference_sota_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def assess_image_quality(image: np.ndarray) -> dict:
    """
    Assess image quality to determine optimal processing parameters.
    
    Returns quality metrics for adaptive parameter selection.
    """
    # Calculate quality indicators
    std = float(image.std())
    mean = float(image.mean())
    img_range = float(image.max() - image.min())
    
    # Edge density (high edge density = more detail)
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Contrast score (higher = better quality)
    contrast_score = std / 255.0
    
    # Brightness score (closer to 128 = better)
    brightness_score = 1.0 - abs(mean - 128) / 128.0
    
    # Combined quality score (0-1, higher = better)
    quality_score = (contrast_score * 0.6) + (brightness_score * 0.2) + (edge_density * 0.2)
    
    # Categorize quality
    if quality_score > 0.35:
        category = "high"
    elif quality_score > 0.25:
        category = "medium"
    else:
        category = "low"
    
    return {
        'std': std,
        'mean': mean,
        'range': img_range,
        'edge_density': edge_density,
        'contrast_score': contrast_score,
        'quality_score': quality_score,
        'category': category
    }


def get_adaptive_parameters(quality: dict) -> dict:
    """
    Select optimal processing parameters based on image quality.
    
    Strategy:
    - High quality: Preserve details, minimal smoothing
    - Medium quality: Balanced enhancement
    - Low quality: Aggressive enhancement, seam smoothing
    """
    category = quality['category']
    
    if category == "high":
        # Preserve fine details for high-quality images
        params = {
            'overlap_vertical': 0.15,      # Lower overlap
            'overlap_horizontal': 0.20,    # Lower overlap
            'alpha': 0.0,                  # No blending with original
            'use_bilateral': False,        # No smoothing
            'bilateral_d': 0,
            'bilateral_sigma_color': 0,
            'bilateral_sigma_space': 0,
            'upscale_method': cv2.INTER_CUBIC,  # Sharper
            'sigma_ratio': 4.0,            # Sharper Gaussian weights
        }
    elif category == "medium":
        # Balanced enhancement
        params = {
            'overlap_vertical': 0.20,
            'overlap_horizontal': 0.25,
            'alpha': 0.15,                 # Light blending
            'use_bilateral': True,
            'bilateral_d': 7,              # Smaller kernel
            'bilateral_sigma_color': 50,   # Less aggressive
            'bilateral_sigma_space': 50,
            'upscale_method': cv2.INTER_LANCZOS4,
            'sigma_ratio': 3.5,
        }
    else:  # low quality
        # Aggressive enhancement for degraded images
        params = {
            'overlap_vertical': 0.25,      # Maximum overlap
            'overlap_horizontal': 0.30,    # Maximum overlap
            'alpha': 0.20,                 # Moderate blending
            'use_bilateral': True,
            'bilateral_d': 9,              # Larger kernel
            'bilateral_sigma_color': 75,   # More aggressive
            'bilateral_sigma_space': 75,
            'upscale_method': cv2.INTER_LANCZOS4,
            'sigma_ratio': 3.0,            # Smoother weights
        }
    
    return params


def preprocess_image(image: np.ndarray, logger) -> np.ndarray:
    """Apply contrast stretching for low-contrast images."""
    img_std = image.std()
    img_min = image.min()
    img_max = image.max()
    img_range = img_max - img_min
    
    if img_std < 30 or img_range < 200:
        stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        new_std = stretched.std()
        
        logger.info(f"🔧 Applied contrast stretching:")
        logger.info(f"   Before: range=[{img_min}, {img_max}], std={img_std:.1f}")
        logger.info(f"   After:  range=[{stretched.min()}, {stretched.max()}], std={new_std:.1f}")
        
        return stretched
    
    return image


def apply_adaptive_bilateral_filter(image: np.ndarray, params: dict, logger) -> np.ndarray:
    """Apply bilateral filter only if parameters indicate it's needed."""
    if not params['use_bilateral'] or params['bilateral_d'] == 0:
        return image
    
    filtered = cv2.bilateralFilter(
        image, 
        d=params['bilateral_d'],
        sigmaColor=params['bilateral_sigma_color'],
        sigmaSpace=params['bilateral_sigma_space']
    )
    
    logger.info(f"✨ Applied bilateral filter (d={params['bilateral_d']}, "
               f"σ_color={params['bilateral_sigma_color']}, σ_space={params['bilateral_sigma_space']})")
    
    return filtered


def load_model(checkpoint_dir, checkpoint_name, gpu_id=0):
    """Load trained model from checkpoint"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.generator_enhanced import unet_enhanced
    
    generator = unet_enhanced(input_size=(TARGET_LINE_WIDTH, TARGET_LINE_HEIGHT, 1))
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(str(checkpoint_path)).expect_partial()
    
    return generator


def split_vertical_columns(image, num_columns=2, overlap_ratio=0.20):
    """Split image into vertical columns with adaptive overlap"""
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


def split_horizontal_strips(image, target_aspect=6.0, min_height=100, overlap_ratio=0.25):
    """Split image into horizontal strips with adaptive overlap"""
    h, w = image.shape[:2]
    
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
    """Resize image to model input size"""
    h, w = image.shape[:2]
    aspect = w / h
    
    new_h = target_height
    new_w = int(new_h * aspect)
    
    if new_w > target_width:
        new_w = target_width
        new_h = int(new_w / aspect)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
    padded[:new_h, :new_w] = resized
    
    return padded, (new_h, new_w)


def process_tile(tile_img, generator, upscale_method=cv2.INTER_LANCZOS4):
    """Process a single tile with adaptive upscaling method"""
    resized, (valid_h, valid_w) = resize_to_model_input(tile_img)
    
    # Normalize to [0, 1]
    input_tensor = resized.astype(np.float32) / 255.0
    
    # CRITICAL FIX: Normalize to [-1, 1] to match TRAINING!
    # Training uses: degraded_images * 2.0 - 1.0 (see train_enhanced.py line 213)
    input_tensor = input_tensor * 2.0 - 1.0  # [0, 1] → [-1, 1] (MATCH TRAINING!)
    
    input_tensor = np.transpose(input_tensor, (1, 0))
    input_tensor = np.expand_dims(input_tensor, axis=-1)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    output = generator(input_tensor, training=False)
    restored = output[0, :, :, 0].numpy()
    
    restored = np.transpose(restored, (1, 0))
    
    # CRITICAL FIX: Denormalize from [-1, 1] to [0, 255] (tanh output range)
    restored = np.clip((restored + 1.0) * 127.5, 0, 255).astype(np.uint8)
    
    restored_valid = restored[:valid_h, :valid_w]
    
    tile_h, tile_w = tile_img.shape[:2]
    restored_full = cv2.resize(restored_valid, (tile_w, tile_h), interpolation=upscale_method)
    
    return restored_full


def create_smooth_gaussian_weight(size, sigma_ratio=3.5):
    """Create Gaussian weight for blending with adaptive smoothness"""
    sigma = size / sigma_ratio
    x = np.arange(size)
    gaussian = np.exp(-0.5 * ((size - x) / sigma) ** 2)
    return gaussian


def blend_tiles_vertical(tiles_data, full_width, full_height, overlap_ratio=0.20, sigma_ratio=3.5):
    """Blend vertical tiles with adaptive parameters"""
    result = np.zeros((full_height, full_width), dtype=np.float32)
    weights = np.zeros((full_height, full_width), dtype=np.float32)
    
    for tile, x_start, x_end in tiles_data:
        tile_w = x_end - x_start
        weight = np.ones((full_height, tile_w), dtype=np.float32)
        
        if x_start > 0:
            fade_w = int(tile_w * overlap_ratio * 0.6)
            gaussian = create_smooth_gaussian_weight(fade_w, sigma_ratio=sigma_ratio)
            weight[:, :fade_w] = gaussian[np.newaxis, :]
        
        if x_end < full_width:
            fade_w = int(tile_w * overlap_ratio * 0.6)
            gaussian = create_smooth_gaussian_weight(fade_w, sigma_ratio=sigma_ratio)
            weight[:, -fade_w:] = gaussian[::-1][np.newaxis, :]
        
        result[:, x_start:x_end] += tile.astype(np.float32) * weight
        weights[:, x_start:x_end] += weight
    
    result = np.divide(result, weights, where=weights > 0)
    return result.astype(np.uint8)


def blend_tiles_horizontal(tiles_data, full_width, full_height, overlap_ratio=0.25, sigma_ratio=3.5):
    """Blend horizontal tiles with adaptive parameters"""
    result = np.zeros((full_height, full_width), dtype=np.float32)
    weights = np.zeros((full_height, full_width), dtype=np.float32)
    
    for tile, y_start, y_end in tiles_data:
        tile_h = y_end - y_start
        weight = np.ones((tile_h, full_width), dtype=np.float32)
        
        if y_start > 0:
            fade_h = int(tile_h * overlap_ratio * 0.6)
            gaussian = create_smooth_gaussian_weight(fade_h, sigma_ratio=sigma_ratio)
            weight[:fade_h, :] = gaussian[:, np.newaxis]
        
        if y_end < full_height:
            fade_h = int(tile_h * overlap_ratio * 0.6)
            gaussian = create_smooth_gaussian_weight(fade_h, sigma_ratio=sigma_ratio)
            weight[-fade_h:, :] = gaussian[::-1][:, np.newaxis]
        
        result[y_start:y_end, :] += tile.astype(np.float32) * weight
        weights[y_start:y_end, :] += weight
    
    result = np.divide(result, weights, where=weights > 0)
    return result.astype(np.uint8)


def process_adaptive_document(image, generator, logger, params):
    """
    Process document with ADAPTIVE quality-based parameters
    
    Args:
        image: Input grayscale image
        generator: Trained model
        logger: Logger instance
        params: Adaptive parameters from get_adaptive_parameters()
    
    Returns:
        Restored image
    """
    h, w = image.shape[:2]
    aspect = w / h
    
    logger.info(f"Processing document: {w}×{h} (aspect {aspect:.2f})")
    logger.info(f"📊 Adaptive params: quality={params.get('quality_category', 'N/A')}, "
               f"overlap_h={params['overlap_horizontal']:.2f}, alpha={params['alpha']:.2f}, "
               f"bilateral={params['use_bilateral']}")
    
    # Determine strategy based on aspect ratio
    if aspect < 1.5:
        # Portrait mode
        num_columns = 3 if aspect < 0.8 else 2
        logger.info(f"Portrait mode: {num_columns} columns")
        
        columns = split_vertical_columns(image, num_columns=num_columns, 
                                        overlap_ratio=params['overlap_vertical'])
        logger.info(f"Created {len(columns)} vertical columns (overlap={params['overlap_vertical']*100:.0f}%)")
        
        restored_columns = []
        for col_idx, (col_img, x_start, x_end) in enumerate(columns):
            col_h, col_w = col_img.shape[:2]
            col_aspect = col_w / col_h
            logger.info(f"  Column {col_idx+1}: {col_w}×{col_h} (aspect {col_aspect:.2f})")
            
            strips = split_horizontal_strips(col_img, target_aspect=6.0, 
                                           overlap_ratio=params['overlap_horizontal'])
            logger.info(f"    → {len(strips)} horizontal strips (overlap={params['overlap_horizontal']*100:.0f}%)")
            
            restored_strips = []
            for strip_idx, (strip_img, y_start, y_end) in enumerate(strips):
                strip_h, strip_w = strip_img.shape[:2]
                strip_aspect = strip_w / strip_h
                logger.info(f"      Strip {strip_idx+1}: {strip_w}×{strip_h} (aspect {strip_aspect:.2f})")
                
                restored_strip = process_tile(strip_img, generator, upscale_method=params['upscale_method'])
                restored_strips.append((restored_strip, y_start, y_end))
            
            col_restored = blend_tiles_horizontal(restored_strips, col_w, col_h, 
                                                 overlap_ratio=params['overlap_horizontal'],
                                                 sigma_ratio=params['sigma_ratio'])
            restored_columns.append((col_restored, x_start, x_end))
        
        restored = blend_tiles_vertical(restored_columns, w, h, 
                                       overlap_ratio=params['overlap_vertical'],
                                       sigma_ratio=params['sigma_ratio'])
        
    elif aspect < 3.0:
        # Semi-portrait mode
        logger.info("Semi-portrait mode: Horizontal strips only")
        strips = split_horizontal_strips(image, target_aspect=6.0, 
                                        overlap_ratio=params['overlap_horizontal'])
        logger.info(f"Created {len(strips)} horizontal strips (overlap={params['overlap_horizontal']*100:.0f}%)")
        
        restored_strips = []
        for strip_idx, (strip_img, y_start, y_end) in enumerate(strips):
            strip_h, strip_w = strip_img.shape[:2]
            strip_aspect = strip_w / strip_h
            logger.info(f"  Strip {strip_idx+1}: {strip_w}×{strip_h} (aspect {strip_aspect:.2f})")
            
            restored_strip = process_tile(strip_img, generator, upscale_method=params['upscale_method'])
            restored_strips.append((restored_strip, y_start, y_end))
        
        restored = blend_tiles_horizontal(restored_strips, w, h, 
                                         overlap_ratio=params['overlap_horizontal'],
                                         sigma_ratio=params['sigma_ratio'])
    
    else:
        # Landscape mode
        logger.info("Landscape mode: Standard horizontal strips")
        strips = split_horizontal_strips(image, target_aspect=7.0, 
                                        overlap_ratio=params['overlap_horizontal'])
        logger.info(f"Created {len(strips)} horizontal strips (overlap={params['overlap_horizontal']*100:.0f}%)")
        
        restored_strips = []
        for strip_idx, (strip_img, y_start, y_end) in enumerate(strips):
            strip_h, strip_w = strip_img.shape[:2]
            strip_aspect = strip_w / strip_h
            logger.info(f"  Strip {strip_idx+1}: {strip_w}×{strip_h} (aspect {strip_aspect:.2f})")
            
            restored_strip = process_tile(strip_img, generator, upscale_method=params['upscale_method'])
            restored_strips.append((restored_strip, y_start, y_end))
        
        restored = blend_tiles_horizontal(restored_strips, w, h, 
                                         overlap_ratio=params['overlap_horizontal'],
                                         sigma_ratio=params['sigma_ratio'])
    
    # Post-processing: Adaptive bilateral filter
    restored = apply_adaptive_bilateral_filter(restored, params, logger)
    
    # Adaptive alpha blending
    if params['alpha'] > 0:
        restored = cv2.addWeighted(
            restored, 1.0 - params['alpha'],
            image, params['alpha'],
            0
        )
        logger.info(f"🎨 Applied alpha blending: {params['alpha']:.2f}")
    
    return restored


def main():
    parser = argparse.ArgumentParser(
        description='SOTA Document Restoration - Adaptive Quality-Based Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target: Beat DE-GAN baseline (PSNR 24.9 dB, F-Measure 99.5%)

Adaptive Strategy:
- High quality: Minimal processing, preserve details
- Medium quality: Balanced enhancement
- Low quality: Aggressive enhancement with bilateral filter

Examples:
  # Adaptive processing on DIBCO 2013
  python inference_sota.py --checkpoint_dir checkpoints/best_model --checkpoint_name ckpt-88 \\
    --input dibco_datasets/2013/imgs --output_dir results/dibco_2013_sota --gpu_id 1
        """
    )
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoint')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                       help='Checkpoint filename (e.g., ckpt-88)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--image_ext', type=str, default='.png',
                       help='Image file extension for batch processing (default: .png)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = setup_logging(args.output_dir)
    logger.info("="*80)
    logger.info("SOTA Document Restoration - Adaptive Quality-Based Processing")
    logger.info("="*80)
    logger.info(f"Target: Beat DE-GAN (PSNR 24.9 dB, F-Measure 99.5%)")
    logger.info(f"Checkpoint: {args.checkpoint_dir}/{args.checkpoint_name}")
    logger.info(f"GPU: {args.gpu_id}")
    
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
    
    logger.info(f"Processing {len(image_paths)} images with ADAPTIVE strategy...")
    
    results = []
    for img_path, img_name in zip(image_paths, image_files):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {img_name}")
        logger.info(f"{'='*80}")
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            continue
        
        h, w = image.shape
        aspect = w / h
        logger.info(f"Input size: {w}×{h} (aspect {aspect:.2f})")
        
        # Preprocess
        image = preprocess_image(image, logger)
        
        # Assess quality and get adaptive parameters
        quality = assess_image_quality(image)
        logger.info(f"🔍 Quality assessment:")
        logger.info(f"   Score: {quality['quality_score']:.3f}, Category: {quality['category'].upper()}")
        logger.info(f"   Contrast: {quality['contrast_score']:.3f}, Edge density: {quality['edge_density']:.4f}")
        
        params = get_adaptive_parameters(quality)
        params['quality_category'] = quality['category']
        
        # Process with adaptive parameters
        restored = process_adaptive_document(image, generator, logger, params)
        
        # Save
        output_name = os.path.splitext(img_name)[0] + '_restored.png'
        output_path = os.path.join(args.output_dir, output_name)
        cv2.imwrite(output_path, restored)
        logger.info(f"💾 Saved: {output_path}")
        
        # Metrics
        contrast = restored.std()
        mean_intensity = restored.mean()
        
        results.append({
            'image_name': os.path.splitext(img_name)[0],
            'input_size': [w, h],
            'aspect_ratio': round(aspect, 2),
            'quality_score': round(float(quality['quality_score']), 3),
            'quality_category': quality['category'],
            'adaptive_params': {
                'overlap_h': params['overlap_horizontal'],
                'overlap_v': params['overlap_vertical'],
                'alpha': params['alpha'],
                'bilateral': params['use_bilateral']
            },
            'contrast': round(float(contrast), 2),
            'mean_intensity': round(float(mean_intensity), 2),
            'output_path': output_path
        })
        
        logger.info(f"📊 Metrics: contrast={contrast:.1f}, mean={mean_intensity:.1f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'version': 'SOTA Adaptive v2.0',
        'target': 'Beat DE-GAN (PSNR 24.9 dB)',
        'checkpoint': f"{args.checkpoint_dir}/{args.checkpoint_name}",
        'strategy': 'Adaptive quality-based processing',
        'num_images': len(results),
        'results': results
    }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processing complete! Processed {len(results)} images")
    logger.info(f"📄 Summary saved to: {summary_path}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
