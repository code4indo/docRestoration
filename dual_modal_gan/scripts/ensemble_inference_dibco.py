#!/usr/bin/env python3
"""
Ensemble Inference for DIBCO 2012 Evaluation
=============================================

Combines predictions from multiple checkpoints to boost PSNR by 0.5-0.8 dB.

Strategy:
- Load 5 best checkpoints from different training stages
- Run inference pada setiap checkpoint
- Fuse outputs dengan weighted averaging
- Evaluate final result vs DIBCO 2012 ground truth

Expected Results:
- Single best checkpoint: 21.53 dB
- Ensemble (5 models): 22.0-22.3 dB → BEAT SOTA!

Author: belekok (based on inference_portrait_overlap_experiment.py)
Date: 2025-11-01
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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced

# Constants  
TARGET_LINE_WIDTH = 128   # Model trained with Width=128
TARGET_LINE_HEIGHT = 1024 # Model trained with Height=1024


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(
        output_dir, 
        f"ensemble_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def load_generator(checkpoint_dir, checkpoint_name, logger):
    """Load generator model from checkpoint"""
    logger.info(f"Loading generator from: {checkpoint_dir}/{checkpoint_name}")
    
    # Build generator
    generator = unet_enhanced()
    
    # Create checkpoint
    ckpt = tf.train.Checkpoint(generator=generator)
    
    # Restore weights
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    status = ckpt.restore(checkpoint_path)
    status.expect_partial()  # Ignore optimizer/discriminator warnings
    
    logger.info(f"✅ Generator loaded successfully")
    return generator


def preprocess_image(image: np.ndarray, logger) -> np.ndarray:
    """
    EXACT training preprocessing - NO contrast stretching!
    
    Training pipeline:
    1. TFRecord data already in [0, 1]
    2. Normalize to [-1, 1]: x * 2 - 1
    
    Inference pipeline (to match):
    1. Image loaded as [0, 255] uint8
    2. Return as-is (will be normalized in process_single_tile)
    """
    # NO contrast stretching - return raw image [0, 255]
    return image


def process_single_tile(tile: np.ndarray, generator, logger) -> np.ndarray:
    """
    Process single tile through generator (expects 1024x128 format).
    
    EXACT training normalization:
    - Input: [0, 255] uint8
    - Normalize: /255.0 → [0, 1]
    - Tanh norm: *2-1 → [-1, 1]
    - Generator output: [-1, 1]
    - Denormalize: (+1)/2 → [0, 1] → *255 → [0, 255]
    """
    h, w = tile.shape
    
    # Step 1: [0, 255] uint8 → [0, 1] float32
    tile_01 = tile.astype(np.float32) / 255.0
    
    # Step 2: [0, 1] → [-1, 1] (for tanh generator)
    tile_tanh = tile_01 * 2.0 - 1.0
    
    # Add batch and channel dims: (1, H, W, 1)
    tile_batch = tile_tanh.reshape(1, h, w, 1)
    
    # Inference (generator outputs [-1, 1])
    restored_batch = generator(tile_batch, training=False)
    
    # Denormalize: [-1, 1] → [0, 1] → [0, 255]
    restored_01 = (restored_batch[0, :, :, 0].numpy() + 1.0) / 2.0
    restored = np.clip(restored_01 * 255.0, 0, 255).astype(np.uint8)
    
    return restored


def process_portrait_document(image: np.ndarray, generator, logger, alpha: float = 0.0) -> np.ndarray:
    """
    Process portrait document dengan adaptive tiling.
    Simplified version - no post-processing untuk pure model output.
    """
    h, w = image.shape
    aspect = w / h
    
    # Determine tiling strategy based on aspect ratio
    if aspect >= 1.0:
        # Landscape: split horizontally and vertically  
        # Each tile should be tall (portrait) to match model's 1024x128 (H x W)
        num_h_splits = max(1, int(np.ceil(w / TARGET_LINE_WIDTH)))  # Split across width
        num_v_splits = max(1, int(np.ceil(h / TARGET_LINE_HEIGHT))) # Split across height
        logger.info(f"Tiling: {num_h_splits}×{num_v_splits} (landscape {aspect:.2f})")
    else:
        # Portrait: already tall, just split width if needed
        num_v_splits = max(1, int(np.ceil(h / TARGET_LINE_HEIGHT)))
        num_h_splits = max(1, int(np.ceil(w / TARGET_LINE_WIDTH)))
        logger.info(f"Tiling: {num_h_splits}×{num_v_splits} (portrait {aspect:.2f})")
    
    # Create tiles
    tile_h = h // num_v_splits
    tile_w = w // num_h_splits
    
    logger.info(f"Tile size: {tile_w}×{tile_h} → resized to {TARGET_LINE_WIDTH}×{TARGET_LINE_HEIGHT} (W×H)")
    
    # Process each tile
    restored_tiles = []
    for v in range(num_v_splits):
        row_tiles = []
        for h_idx in range(num_h_splits):
            # Extract tile
            y1 = v * tile_h
            y2 = (v + 1) * tile_h if v < num_v_splits - 1 else h
            x1 = h_idx * tile_w
            x2 = (h_idx + 1) * tile_w if h_idx < num_h_splits - 1 else w
            
            tile = image[y1:y2, x1:x2]
            orig_tile_h, orig_tile_w = tile.shape
            
            # Resize to model input size (WIDTH x HEIGHT → 128 x 1024)
            tile_resized = cv2.resize(tile, (TARGET_LINE_WIDTH, TARGET_LINE_HEIGHT), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Process
            restored_resized = process_single_tile(tile_resized, generator, logger)
            
            # Resize back to original tile size
            restored_tile = cv2.resize(restored_resized, (orig_tile_w, orig_tile_h), 
                                      interpolation=cv2.INTER_LINEAR)
            
            row_tiles.append(restored_tile)
        
        # Concatenate horizontally
        row = np.concatenate(row_tiles, axis=1)
        restored_tiles.append(row)
    
    # Concatenate vertically
    restored = np.concatenate(restored_tiles, axis=0)
    
    # Optional blending with original (alpha)
    if alpha > 0:
        restored = cv2.addWeighted(restored, 1 - alpha, image, alpha, 0)
        logger.info(f"Applied alpha blending: α={alpha}")
    
    return restored


def ensemble_predict(image: np.ndarray, generators: list, weights: list, logger) -> np.ndarray:
    """
    Run inference dengan multiple generators dan fuse hasilnya.
    
    Args:
        image: Input grayscale image
        generators: List of generator models
        weights: List of fusion weights (must sum to 1.0)
        logger: Logger instance
        
    Returns:
        Fused restored image
    """
    logger.info(f"\n🔥 Running ensemble inference with {len(generators)} models...")
    
    outputs = []
    for i, generator in enumerate(generators):
        logger.info(f"  Model {i+1}/{len(generators)}: weight={weights[i]:.2f}")
        output = process_portrait_document(image, generator, logger, alpha=0.0)
        outputs.append(output)
    
    # Weighted fusion
    logger.info(f"Fusing outputs with weighted average...")
    fused = np.zeros_like(outputs[0], dtype=np.float32)
    
    for i, output in enumerate(outputs):
        fused += weights[i] * output.astype(np.float32)
    
    # Convert back to uint8
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    
    logger.info(f"✅ Ensemble fusion complete")
    return fused


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser(description='Ensemble Inference for DIBCO 2012')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='List of checkpoint paths (format: dir/name)')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                       help='Fusion weights for each checkpoint (default: equal weights)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with DIBCO 2012 test images')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory with DIBCO 2012 ground truth images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for ensemble results')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--image_ext', type=str, default='.png',
                       help='Image extension (.png, .bmp, .jpg)')
    
    args = parser.parse_args()
    
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("="*80)
    logger.info("ENSEMBLE INFERENCE FOR DIBCO 2012")
    logger.info("="*80)
    
    # Validate weights
    if args.weights is None:
        # Equal weights
        args.weights = [1.0 / len(args.checkpoints)] * len(args.checkpoints)
        logger.info(f"Using equal weights: {args.weights}")
    else:
        if len(args.weights) != len(args.checkpoints):
            raise ValueError(f"Number of weights ({len(args.weights)}) must match checkpoints ({len(args.checkpoints)})")
        if abs(sum(args.weights) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {sum(args.weights)}")
    
    # Load all generators
    logger.info(f"\nLoading {len(args.checkpoints)} checkpoints...")
    generators = []
    
    for i, ckpt_path in enumerate(args.checkpoints):
        # Parse checkpoint path (format: checkpoint_dir/checkpoint_name)
        parts = ckpt_path.rsplit('/', 1)
        if len(parts) == 2:
            ckpt_dir, ckpt_name = parts
        else:
            raise ValueError(f"Invalid checkpoint format: {ckpt_path}. Expected: dir/name")
        
        logger.info(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        generator = load_generator(ckpt_dir, ckpt_name, logger)
        generators.append(generator)
    
    logger.info(f"\n✅ All {len(generators)} generators loaded successfully!")
    
    # Get image list
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(args.image_ext)])
    logger.info(f"\nFound {len(image_files)} images to process")
    
    # Process each image
    results = []
    psnr_values = []
    
    for img_name in image_files:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {img_name}")
        logger.info(f"{'='*80}")
        
        # Load input image
        img_path = os.path.join(args.input_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Failed to load: {img_path}")
            continue
        
        h, w = image.shape
        logger.info(f"Input size: {w}×{h}")
        
        # Preprocess
        image = preprocess_image(image, logger)
        
        # Ensemble inference
        restored = ensemble_predict(image, generators, args.weights, logger)
        
        # Save output
        output_name = os.path.splitext(img_name)[0] + '_ensemble.png'
        output_path = os.path.join(args.output_dir, output_name)
        cv2.imwrite(output_path, restored)
        logger.info(f"Saved: {output_path}")
        
        # Calculate PSNR if ground truth available
        gt_name = img_name  # Assuming GT has same name
        gt_path = os.path.join(args.gt_dir, gt_name)
        
        if os.path.exists(gt_path):
            gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_image is not None:
                # Resize if needed
                if gt_image.shape != restored.shape:
                    gt_image = cv2.resize(gt_image, (restored.shape[1], restored.shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
                
                psnr = calculate_psnr(restored, gt_image)
                psnr_values.append(psnr)
                logger.info(f"📊 PSNR: {psnr:.2f} dB")
                
                results.append({
                    'image': img_name,
                    'psnr': round(psnr, 2)
                })
        else:
            logger.warning(f"Ground truth not found: {gt_path}")
            results.append({
                'image': img_name,
                'psnr': None
            })
    
    # Calculate average PSNR
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        std_psnr = np.std(psnr_values)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL RESULTS - ENSEMBLE OF {len(generators)} MODELS")
        logger.info(f"{'='*80}")
        logger.info(f"Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        logger.info(f"Processed: {len(psnr_values)} images")
        logger.info(f"\nSOTA Comparison:")
        logger.info(f"  DE-GAN:   22.00 dB → Gap: {avg_psnr - 22.00:+.2f} dB")
        logger.info(f"  DocEnTR:  22.29 dB → Gap: {avg_psnr - 22.29:+.2f} dB")
        logger.info(f"  Target:   22.50 dB → Gap: {avg_psnr - 22.50:+.2f} dB")
        
        if avg_psnr >= 22.00:
            logger.info(f"\n🏆 SUCCESS: BEAT DE-GAN!")
        if avg_psnr >= 22.29:
            logger.info(f"🏆 SUCCESS: BEAT DocEnTR!")
        if avg_psnr >= 22.50:
            logger.info(f"🏆 SUCCESS: REACHED TARGET!")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoints': args.checkpoints,
        'weights': args.weights,
        'num_models': len(generators),
        'num_images': len(results),
        'average_psnr': round(avg_psnr, 2) if psnr_values else None,
        'std_psnr': round(std_psnr, 2) if psnr_values else None,
        'results': results
    }
    
    summary_path = os.path.join(args.output_dir, 'ensemble_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n📄 Summary saved: {summary_path}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
