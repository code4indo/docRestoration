#!/usr/bin/env python3
"""
Full-Size Document Restoration Inference Script for Production V3
================================================================

This script performs inference on full-size degraded document images using
the trained Enhanced Generator (production_v3) with overlapping tile strategy.

Key Features:
- Overlapping tile processing (128×1024, 32px overlap) to eliminate seams
- Alpha blending for smooth tile transitions
- Batch processing (batch_size=4) for GPU efficiency
- Comprehensive DIBCO metrics: PSNR, SSIM, F-Measure, NRM, MPM
- High-quality PNG output
- Side-by-side comparison visualization

Model Characteristics:
- Generator: Enhanced U-Net (ResBlocks + Attention Gates)
- Input format: (1024, 128, 1) - Width × Height × Channel
- Output activation: tanh (range [-1, 1])
- Best checkpoint: ckpt-88 (PSNR: 30.91 dB)

Usage:
    python inference_production_v3.py --checkpoint_dir <path> --input_dir <path> --output_dir <path>
    
Example:
    python inference_production_v3.py \\
        --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \\
        --input_dir dibco_datasets/DIPCO2016_dataset \\
        --gt_dir dibco_datasets/DIPCO2016_Dataset_GT \\
        --output_dir results/inference_production_v3_dibco2016

Author: AI Assistant
Date: 2025-10-22
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# CRITICAL FIX: Disable XLA JIT compilation to avoid libdevice.10.bc error
# Must be set BEFORE importing TensorFlow
# ============================================================================
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit'
# Force disable XLA completely
os.environ['TF_DISABLE_XLA'] = '1'
# Alternative: point to actual libdevice location
os.environ['XLA_NVVM_LIBDEVICE_PATH'] = '/usr/lib/nvidia-cuda-toolkit/libdevice'

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Triple-check: Disable XLA JIT compilation at TF level
tf.config.optimizer.set_jit(False)
# Disable XLA globally
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced


# ============================================================================
# Configuration
# ============================================================================

TILE_HEIGHT = 128
TILE_WIDTH = 1024
OVERLAP = 32
BATCH_SIZE = 4

# Stride for overlapping tiles
STRIDE_H = TILE_HEIGHT - OVERLAP  # 96
STRIDE_W = TILE_WIDTH - OVERLAP   # 992


# ============================================================================
# GPU Configuration
# ============================================================================

def configure_gpu(gpu_id=1):
    """
    Configure GPU with memory growth and device selection.
    
    Args:
        gpu_id (int): GPU device ID to use. None for CPU.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TF from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            if gpu_id is not None and gpu_id < len(gpus):
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                logging.info(f"✓ Using GPU {gpu_id}: {gpus[gpu_id].name}")
            else:
                logging.warning(f"⚠️  GPU {gpu_id} not available, using default GPU")
        except RuntimeError as e:
            logging.error(f"❌ GPU configuration error: {e}")
    else:
        logging.warning("⚠️  No GPU found, using CPU")


# ============================================================================
# Alpha Blending Utilities
# ============================================================================

def create_alpha_blending_kernel(height, width, fade_size):
    """
    Create 2D alpha blending map for smooth tile transitions.
    
    The kernel has value 1.0 in the center and fades to 0.0 at the edges
    over the fade_size region. This allows overlapping tiles to blend
    smoothly without visible seams.
    
    Args:
        height (int): Tile height
        width (int): Tile width
        fade_size (int): Number of pixels to fade at edges
        
    Returns:
        np.ndarray: Alpha blending kernel of shape (height, width)
    """
    alpha_h = np.ones(height, dtype=np.float32)
    alpha_w = np.ones(width, dtype=np.float32)
    
    # Create fade-in at start and fade-out at end
    if fade_size > 0:
        alpha_h[:fade_size] = np.linspace(0, 1, fade_size)
        alpha_h[-fade_size:] = np.linspace(1, 0, fade_size)
        alpha_w[:fade_size] = np.linspace(0, 1, fade_size)
        alpha_w[-fade_size:] = np.linspace(1, 0, fade_size)
    
    # Create 2D map using outer product
    alpha_map = np.outer(alpha_h, alpha_w)
    return alpha_map


# ============================================================================
# Preprocessing and Postprocessing
# ============================================================================

def preprocess_tile(tile):
    """
    Preprocess tile for generator input.
    
    Critical transformations:
    1. Convert to grayscale if needed
    2. Normalize to [0, 1]
    3. Transpose (H, W) -> (W, H) for model compatibility
    4. Add channel dimension
    5. Normalize to [-1, 1] for tanh generator
    
    Args:
        tile (np.ndarray): Grayscale tile of shape (H, W), range [0, 255]
    
    Returns:
        np.ndarray: Preprocessed tile of shape (1024, 128, 1), range [-1, 1]
    """
    # Convert to grayscale if needed
    if len(tile.shape) == 3:
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    
    # Ensure correct size
    if tile.shape != (TILE_HEIGHT, TILE_WIDTH):
        tile = cv2.resize(tile, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    tile = tile.astype(np.float32) / 255.0
    
    # CRITICAL: Transpose (H, W) -> (W, H)
    tile = tile.T  # Now (1024, 128)
    
    # Add channel dimension
    tile = tile[..., np.newaxis]  # (1024, 128, 1)
    
    # Normalize to [-1, 1] for tanh generator
    tile = tile * 2.0 - 1.0
    
    return tile


def postprocess_tile(tile):
    """
    Postprocess generator output to image format.
    
    Inverse transformations of preprocess_tile:
    1. Transpose (W, H) -> (H, W)
    2. Denormalize from [-1, 1] to [0, 255]
    3. Clip to valid range
    
    Args:
        tile (np.ndarray): Generator output of shape (1024, 128, 1), range [-1, 1]
    
    Returns:
        np.ndarray: Image tile of shape (128, 1024), range [0, 255], dtype uint8
    """
    # Remove channel dimension if present
    if len(tile.shape) == 3:
        tile = tile[:, :, 0]  # (1024, 128)
    
    # CRITICAL: Transpose (W, H) -> (H, W)
    tile = tile.T  # (128, 1024)
    
    # Denormalize from [-1, 1] to [0, 255]
    tile = (tile + 1.0) * 127.5
    tile = np.clip(tile, 0, 255).astype(np.uint8)
    
    return tile


# ============================================================================
# Tile Extraction
# ============================================================================

def extract_overlapping_tiles(image):
    """
    Extract overlapping tiles from full-size image.
    
    Args:
        image (np.ndarray): Full-size grayscale image (H, W)
    
    Returns:
        list: List of dicts with keys:
            - 'tile': tile image (128, 1024)
            - 'x': x coordinate in original image
            - 'y': y coordinate in original image
            - 'padded': whether tile was padded
    """
    height, width = image.shape[:2]
    tiles = []
    
    # Calculate number of tiles needed
    n_tiles_h = (height + STRIDE_H - 1) // STRIDE_H
    n_tiles_w = (width + STRIDE_W - 1) // STRIDE_W
    
    for i in range(n_tiles_h):
        y = i * STRIDE_H
        for j in range(n_tiles_w):
            x = j * STRIDE_W
            
            # Extract tile
            tile = image[y:y+TILE_HEIGHT, x:x+TILE_WIDTH]
            
            # Pad if needed (at image boundaries)
            padded = False
            if tile.shape[0] < TILE_HEIGHT or tile.shape[1] < TILE_WIDTH:
                padded_tile = np.ones((TILE_HEIGHT, TILE_WIDTH), dtype=image.dtype) * 255
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
                padded = True
            
            tiles.append({
                'tile': tile,
                'x': x,
                'y': y,
                'padded': padded
            })
    
    logging.info(f"  Extracted {len(tiles)} tiles ({n_tiles_h}×{n_tiles_w})")
    return tiles


# ============================================================================
# Batch Inference
# ============================================================================

def process_tiles_batch(generator, tiles, alpha_kernel):
    """
    Process tiles in batches with generator.
    
    Args:
        generator (tf.keras.Model): Trained generator model
        tiles (list): List of tile dictionaries
        alpha_kernel (np.ndarray): Alpha blending kernel
    
    Returns:
        list: List of tile dictionaries with added 'restored' key
    """
    num_tiles = len(tiles)
    logging.info(f"  Processing {num_tiles} tiles in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, num_tiles, BATCH_SIZE), desc="  Batches"):
        batch_tiles = tiles[i:i+BATCH_SIZE]
        
        # Preprocess batch
        batch_data = np.stack([preprocess_tile(t['tile']) for t in batch_tiles], axis=0)
        
        # Inference
        restored_batch = generator(batch_data, training=False).numpy()
        
        # Postprocess batch
        for j, tile_info in enumerate(batch_tiles):
            restored = postprocess_tile(restored_batch[j])
            tile_info['restored'] = restored
    
    return tiles


# ============================================================================
# Tile Merging
# ============================================================================

def merge_tiles_with_blending(tiles, original_height, original_width, alpha_kernel):
    """
    Merge overlapping tiles with alpha blending to eliminate seams.
    
    Args:
        tiles (list): List of tile dictionaries with 'restored' key
        original_height (int): Original image height
        original_width (int): Original image width
        alpha_kernel (np.ndarray): Alpha blending kernel
    
    Returns:
        np.ndarray: Restored full-size image (H, W), dtype uint8
    """
    # Allocate output buffers
    restored_image = np.zeros((original_height, original_width), dtype=np.float32)
    weight_map = np.zeros((original_height, original_width), dtype=np.float32)
    
    logging.info(f"  Merging tiles with alpha blending...")
    
    for tile_info in tiles:
        x = tile_info['x']
        y = tile_info['y']
        restored_tile = tile_info['restored'].astype(np.float32)
        
        # Calculate actual region to paste (handle boundary tiles)
        h_actual = min(TILE_HEIGHT, original_height - y)
        w_actual = min(TILE_WIDTH, original_width - x)
        
        # Get corresponding alpha weights
        alpha = alpha_kernel[:h_actual, :w_actual]
        
        # Accumulate weighted tiles
        restored_image[y:y+h_actual, x:x+w_actual] += restored_tile[:h_actual, :w_actual] * alpha
        weight_map[y:y+h_actual, x:x+w_actual] += alpha
    
    # Normalize by accumulated weights
    restored_image = restored_image / np.maximum(weight_map, 1e-6)
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    
    return restored_image


# ============================================================================
# Post-Processing Pipeline (V5 Enhancement)
# ============================================================================

def apply_post_processing(image: np.ndarray, enable: bool = True) -> np.ndarray:
    """
    Apply V5 post-processing pipeline to enhance restoration quality.
    
    Pipeline:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) - recover contrast
    2. Morphological Closing - connect broken strokes
    3. Unsharp Masking - enhance details and edges
    
    Args:
        image: Grayscale image (H, W) uint8
        enable: If False, return image unchanged
    
    Returns:
        Enhanced image (H, W) uint8
    """
    if not enable:
        return image
    
    logging.info(f"  Applying post-processing pipeline (CLAHE + Closing + Unsharp)...")
    
    # Step 1: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Step 2: Morphological closing to connect broken strokes
    # Use ellipse kernel to preserve stroke directionality
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Step 3: Unsharp masking for detail enhancement
    gaussian = cv2.GaussianBlur(enhanced, (3, 3), 0)
    unsharp_weight = 1.5
    enhanced = cv2.addWeighted(enhanced, 1 + unsharp_weight, gaussian, -unsharp_weight, 0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


# ============================================================================
# DIBCO Metrics Calculation
# ============================================================================

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index.
    Using scikit-image implementation for accuracy.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=255)
    except ImportError:
        logging.warning("⚠️  scikit-image not available, SSIM approximation used")
        # Simple SSIM approximation
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        mu1 = cv2.GaussianBlur(img1.astype(np.float32), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2.astype(np.float32), (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1.astype(np.float32) ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2.astype(np.float32) ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1.astype(np.float32) * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return np.mean(ssim_map)


def calculate_f_measure(img_pred, img_gt):
    """
    Calculate F-Measure (Pseudo F-Measure for binary images).
    """
    # Binarize images (threshold at 128)
    pred_bin = (img_pred > 128).astype(np.float32)
    gt_bin = (img_gt > 128).astype(np.float32)
    
    # Calculate TP, FP, FN
    tp = np.sum(pred_bin * gt_bin)
    fp = np.sum(pred_bin * (1 - gt_bin))
    fn = np.sum((1 - pred_bin) * gt_bin)
    
    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Calculate F-measure
    f_measure = 2 * precision * recall / (precision + recall + 1e-10)
    return f_measure


def calculate_nrm(img_pred, img_gt):
    """
    Calculate Negative Rate Metric (NRM).
    Lower is better.
    """
    pred_bin = (img_pred > 128).astype(np.float32)
    gt_bin = (img_gt > 128).astype(np.float32)
    
    fn = np.sum((1 - pred_bin) * gt_bin)
    fp = np.sum(pred_bin * (1 - gt_bin))
    
    nrm = (fn + fp) / np.sum(gt_bin + 1e-10)
    return nrm


def calculate_mpm(img_pred, img_gt):
    """
    Calculate Misclassification Penalty Metric (MPM).
    Lower is better.
    """
    pred_bin = (img_pred > 128).astype(np.float32)
    gt_bin = (img_gt > 128).astype(np.float32)
    
    # Misclassified pixels
    misclass = np.abs(pred_bin - gt_bin)
    
    # Distance transform for penalty
    dist_fg = cv2.distanceTransform((1 - gt_bin).astype(np.uint8), cv2.DIST_L2, 5)
    dist_bg = cv2.distanceTransform(gt_bin.astype(np.uint8), cv2.DIST_L2, 5)
    
    # Calculate penalties
    penalty = misclass * (dist_fg + dist_bg)
    mpm = np.sum(penalty) / (np.sum(gt_bin) + 1e-10)
    
    return mpm


def calculate_all_metrics(restored, gt):
    """
    Calculate all DIBCO metrics.
    
    Args:
        restored (np.ndarray): Restored image
        gt (np.ndarray): Ground truth image
    
    Returns:
        dict: Dictionary with all metrics (converted to Python float)
    """
    # Ensure same size
    if restored.shape != gt.shape:
        gt = cv2.resize(gt, (restored.shape[1], restored.shape[0]))
    
    metrics = {
        'psnr': float(calculate_psnr(restored, gt)),
        'ssim': float(calculate_ssim(restored, gt)),
        'f_measure': float(calculate_f_measure(restored, gt)),
        'nrm': float(calculate_nrm(restored, gt)),
        'mpm': float(calculate_mpm(restored, gt))
    }
    
    return metrics


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_image(degraded, restored, gt, metrics):
    """
    Create side-by-side comparison image with metrics overlay.
    
    Args:
        degraded (np.ndarray): Degraded input image
        restored (np.ndarray): Restored output image
        gt (np.ndarray): Ground truth image
        metrics (dict): Calculated metrics
    
    Returns:
        np.ndarray: Comparison image
    """
    # Ensure all images are same size
    h, w = restored.shape[:2]
    degraded = cv2.resize(degraded, (w, h))
    gt = cv2.resize(gt, (w, h))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(degraded, cmap='gray')
    axes[0].set_title('Degraded Input', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(restored, cmap='gray')
    axes[1].set_title('Restored Output', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(gt, cmap='gray')
    axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add metrics text
    metrics_text = (
        f"PSNR: {metrics['psnr']:.2f} dB\n"
        f"SSIM: {metrics['ssim']:.4f}\n"
        f"F-Measure: {metrics['f_measure']:.4f}\n"
        f"NRM: {metrics['nrm']:.4f}\n"
        f"MPM: {metrics['mpm']:.4f}"
    )
    
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Convert to numpy array
    fig.canvas.draw()
    comparison = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    comparison = comparison.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return comparison


# ============================================================================
# Main Inference Pipeline
# ============================================================================

def load_generator(checkpoint_path):
    """
    Load trained generator from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file (without extension)
    
    Returns:
        tf.keras.Model: Loaded generator model
    """
    logging.info(f"Loading generator from: {checkpoint_path}")
    
    # Create model
    generator = unet_enhanced(input_size=(TILE_WIDTH, TILE_HEIGHT, 1))
    
    # Create checkpoint
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Restore weights
    status = checkpoint.restore(checkpoint_path).expect_partial()
    
    logging.info(f"✓ Generator loaded successfully")
    logging.info(f"  Model: Enhanced U-Net (ResBlocks + Attention)")
    logging.info(f"  Parameters: ~21.8M")
    logging.info(f"  Input: ({TILE_WIDTH}, {TILE_HEIGHT}, 1)")
    logging.info(f"  Output: tanh activation [-1, 1]")
    
    return generator


def process_document(generator, image_path, gt_path, output_dir, alpha_kernel, enable_postprocess=True):
    """
    Process a single full-size document image.
    
    Args:
        generator (tf.keras.Model): Trained generator model
        image_path (str): Path to degraded image
        gt_path (str): Path to ground truth image (optional)
        output_dir (Path): Output directory
        alpha_kernel (np.ndarray): Alpha blending kernel
        enable_postprocess (bool): Enable V5 post-processing pipeline
    
    Returns:
        dict: Processing results and metrics
    """
    image_name = Path(image_path).stem
    logging.info(f"\n{'='*70}")
    logging.info(f"Processing: {image_name}")
    logging.info(f"{'='*70}")
    
    # Load degraded image
    degraded = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if degraded is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    height, width = degraded.shape
    logging.info(f"  Image size: {width}×{height} pixels")
    
    # Extract overlapping tiles
    tiles = extract_overlapping_tiles(degraded)
    
    # Process tiles in batches
    tiles = process_tiles_batch(generator, tiles, alpha_kernel)
    
    # Merge tiles with blending
    restored = merge_tiles_with_blending(tiles, height, width, alpha_kernel)
    
    # Apply V5 post-processing pipeline
    restored = apply_post_processing(restored, enable=enable_postprocess)
    
    # Save restored image (high-quality PNG)
    output_path = output_dir / f"{image_name}_restored.png"
    cv2.imwrite(str(output_path), restored, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    logging.info(f"  ✓ Saved restored image: {output_path.name}")
    
    # Calculate metrics if GT available
    metrics = None
    if gt_path and Path(gt_path).exists():
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            metrics = calculate_all_metrics(restored, gt)
            logging.info(f"  Metrics:")
            logging.info(f"    PSNR: {metrics['psnr']:.2f} dB")
            logging.info(f"    SSIM: {metrics['ssim']:.4f}")
            logging.info(f"    F-Measure: {metrics['f_measure']:.4f}")
            logging.info(f"    NRM: {metrics['nrm']:.4f}")
            logging.info(f"    MPM: {metrics['mpm']:.4f}")
            
            # Create comparison image
            comparison = create_comparison_image(degraded, restored, gt, metrics)
            comparison_path = output_dir / f"{image_name}_comparison.png"
            cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            logging.info(f"  ✓ Saved comparison: {comparison_path.name}")
    
    return {
        'image_name': image_name,
        'size': (width, height),
        'num_tiles': len(tiles),
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description='Full-Size Document Restoration Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint files')
    parser.add_argument('--checkpoint_name', type=str, default='ckpt-88',
                       help='Checkpoint name (default: ckpt-88 = best model)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to single image file or directory containing images')
    parser.add_argument('--gt_dir', type=str, default=None,
                       help='Directory containing ground truth images (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--gpu_id', type=int, default=1,
                       help='GPU device ID (default: 1, use -1 for CPU)')
    parser.add_argument('--image_ext', type=str, default='.bmp',
                       help='Image file extension (default: .bmp)')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Disable V5 post-processing (CLAHE + Closing + Unsharp)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("Full-Size Document Restoration Inference - Production V3")
    logging.info("="*70)
    logging.info(f"Checkpoint: {args.checkpoint_dir}/{args.checkpoint_name}")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"GT directory: {args.gt_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"GPU ID: {args.gpu_id}")
    logging.info("")
    
    # Configure GPU
    configure_gpu(args.gpu_id if args.gpu_id >= 0 else None)
    
    # Load generator
    checkpoint_path = str(Path(args.checkpoint_dir) / args.checkpoint_name)
    generator = load_generator(checkpoint_path)
    
    # Create alpha blending kernel
    alpha_kernel = create_alpha_blending_kernel(TILE_HEIGHT, TILE_WIDTH, OVERLAP)
    logging.info(f"\nAlpha blending kernel created: {TILE_HEIGHT}×{TILE_WIDTH}, fade={OVERLAP}px")
    
    # Find all input images (support both single file and directory)
    input_path = Path(args.input_dir)
    
    if input_path.is_file():
        # Single file provided
        image_files = [input_path]
        logging.info(f"\nProcessing single image: {input_path.name}")
    elif input_path.is_dir():
        # Directory provided
        image_files = sorted(input_path.glob(f'*{args.image_ext}'))
        if not image_files:
            logging.error(f"❌ No images found in {input_path} with extension {args.image_ext}")
            return
        logging.info(f"\nFound {len(image_files)} images to process")
    else:
        logging.error(f"❌ Input path does not exist: {input_path}")
        return
    
    # Process all images
    all_results = []
    all_metrics = []
    
    for image_path in image_files:
        # Find corresponding GT image
        gt_path = None
        if args.gt_dir:
            gt_name = image_path.stem + '_gt' + args.image_ext
            gt_path = Path(args.gt_dir) / gt_name
        
        try:
            result = process_document(generator, image_path, gt_path, output_dir, alpha_kernel, 
                                     enable_postprocess=not args.no_postprocess)
            all_results.append(result)
            
            if result['metrics']:
                all_metrics.append(result['metrics'])
        
        except Exception as e:
            logging.error(f"❌ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate and save average metrics
    if all_metrics:
        avg_metrics = {
            'psnr': np.mean([m['psnr'] for m in all_metrics]),
            'ssim': np.mean([m['ssim'] for m in all_metrics]),
            'f_measure': np.mean([m['f_measure'] for m in all_metrics]),
            'nrm': np.mean([m['nrm'] for m in all_metrics]),
            'mpm': np.mean([m['mpm'] for m in all_metrics])
        }
        
        logging.info(f"\n{'='*70}")
        logging.info("AVERAGE METRICS")
        logging.info(f"{'='*70}")
        logging.info(f"PSNR: {avg_metrics['psnr']:.2f} dB")
        logging.info(f"SSIM: {avg_metrics['ssim']:.4f}")
        logging.info(f"F-Measure: {avg_metrics['f_measure']:.4f}")
        logging.info(f"NRM: {avg_metrics['nrm']:.4f}")
        logging.info(f"MPM: {avg_metrics['mpm']:.4f}")
        
        # Save metrics to CSV
        import csv
        csv_path = output_dir / 'metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'PSNR', 'SSIM', 'F-Measure', 'NRM', 'MPM'])
            
            for result in all_results:
                if result['metrics']:
                    m = result['metrics']
                    writer.writerow([
                        result['image_name'],
                        f"{m['psnr']:.2f}",
                        f"{m['ssim']:.4f}",
                        f"{m['f_measure']:.4f}",
                        f"{m['nrm']:.4f}",
                        f"{m['mpm']:.4f}"
                    ])
            
            writer.writerow([])
            writer.writerow(['AVERAGE',
                           f"{avg_metrics['psnr']:.2f}",
                           f"{avg_metrics['ssim']:.4f}",
                           f"{avg_metrics['f_measure']:.4f}",
                           f"{avg_metrics['nrm']:.4f}",
                           f"{avg_metrics['mpm']:.4f}"])
        
        logging.info(f"\n✓ Metrics saved to: {csv_path}")
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'num_images': len(image_files),
        'processed': len(all_results),
        'average_metrics': avg_metrics if all_metrics else None,
        'results': all_results
    }
    
    json_path = output_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"✓ Summary saved to: {json_path}")
    logging.info(f"\n{'='*70}")
    logging.info("✓ INFERENCE COMPLETED SUCCESSFULLY")
    logging.info(f"{'='*70}")


if __name__ == '__main__':
    main()
