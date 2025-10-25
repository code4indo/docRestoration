#!/usr/bin/env python3
"""
DIBCO 2013 Evaluation dengan BINARISASI OUTPUT
Mengevaluasi performa dengan format binary yang sesuai dengan ground truth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    """
    Binarisasi dengan Otsu's method
    
    Args:
        image: Grayscale image (0-255)
        
    Returns:
        Binary image (0 or 255)
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_sauvola(image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """
    Binarisasi dengan Sauvola's method (better for document images)
    
    Args:
        image: Grayscale image (0-255)
        window_size: Local window size
        k: Sauvola parameter
        
    Returns:
        Binary image (0 or 255)
    """
    # Convert to float
    img_float = image.astype(np.float64)
    
    # Calculate local mean and std
    mean = cv2.boxFilter(img_float, -1, (window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(img_float**2, -1, (window_size, window_size), normalize=True)
    std = np.sqrt(mean_sq - mean**2)
    
    # Sauvola threshold
    threshold = mean * (1 + k * ((std / 128.0) - 1))
    
    # Binarize
    binary = np.where(img_float > threshold, 255, 0).astype(np.uint8)
    
    return binary


def calculate_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """Calculate PSNR"""
    mse = np.mean((gt.astype(float) - pred.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def calculate_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """Calculate SSIM"""
    return ssim(gt, pred, data_range=255)


def calculate_f_measure(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate F-Measure (DIBCO metric)
    
    F-Measure = 2 * Precision * Recall / (Precision + Recall)
    """
    # Ensure binary
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)
    
    # Calculate TP, FP, FN
    tp = np.sum(np.logical_and(gt_bin == 1, pred_bin == 1))
    fp = np.sum(np.logical_and(gt_bin == 0, pred_bin == 1))
    fn = np.sum(np.logical_and(gt_bin == 1, pred_bin == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure


def calculate_nrm(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate Negative Rate Metric (DIBCO metric)
    
    NRM = (FN + FP) / (2 * TP + FP + FN)
    """
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)
    
    tp = np.sum(np.logical_and(gt_bin == 1, pred_bin == 1))
    fp = np.sum(np.logical_and(gt_bin == 0, pred_bin == 1))
    fn = np.sum(np.logical_and(gt_bin == 1, pred_bin == 0))
    
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 1.0
    
    nrm = (fn + fp) / denominator
    return nrm


def calculate_mpm(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Calculate Misclassification Penalty Metric (DIBCO metric)
    
    MPM = (FN² + FP²) / (TP + FN + FP)
    """
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)
    
    tp = np.sum(np.logical_and(gt_bin == 1, pred_bin == 1))
    fp = np.sum(np.logical_and(gt_bin == 0, pred_bin == 1))
    fn = np.sum(np.logical_and(gt_bin == 1, pred_bin == 0))
    
    denominator = tp + fn + fp
    if denominator == 0:
        return 1.0
    
    mpm = (fn**2 + fp**2) / denominator
    return mpm


def evaluate_image(
    gt_path: Path, 
    restored_path: Path,
    binarization_method: str = 'sauvola'
) -> Dict[str, float]:
    """
    Evaluate single image with binarization
    
    Args:
        gt_path: Ground truth image path
        restored_path: Restored image path
        binarization_method: 'otsu' or 'sauvola'
        
    Returns:
        Dictionary of metrics
    """
    # Load images
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    restored = cv2.imread(str(restored_path), cv2.IMREAD_GRAYSCALE)
    
    if gt is None:
        raise ValueError(f"Cannot load ground truth: {gt_path}")
    if restored is None:
        raise ValueError(f"Cannot load restored: {restored_path}")
    
    # Binarize restored image
    if binarization_method == 'otsu':
        restored_bin = binarize_otsu(restored)
    elif binarization_method == 'sauvola':
        restored_bin = binarize_sauvola(restored)
    else:
        raise ValueError(f"Unknown binarization method: {binarization_method}")
    
    # Calculate metrics
    metrics = {
        'psnr': calculate_psnr(gt, restored_bin),
        'ssim': calculate_ssim(gt, restored_bin),
        'f_measure': calculate_f_measure(gt, restored_bin),
        'nrm': calculate_nrm(gt, restored_bin),
        'mpm': calculate_mpm(gt, restored_bin),
    }
    
    return metrics, restored_bin


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DIBCO 2013 results with BINARIZATION'
    )
    parser.add_argument(
        '--gt_dir',
        type=str,
        required=True,
        help='Ground truth directory'
    )
    parser.add_argument(
        '--restored_dir',
        type=str,
        required=True,
        help='Restored images directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for metrics and binary images'
    )
    parser.add_argument(
        '--binarization',
        type=str,
        default='sauvola',
        choices=['otsu', 'sauvola'],
        help='Binarization method (default: sauvola)'
    )
    
    args = parser.parse_args()
    
    gt_dir = Path(args.gt_dir)
    restored_dir = Path(args.restored_dir)
    output_dir = Path(args.output_dir)
    
    # Create binary output directory
    binary_dir = output_dir / 'binary'
    binary_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("DIBCO 2013 Evaluation - WITH BINARIZATION")
    logger.info("=" * 80)
    logger.info(f"Ground Truth: {gt_dir}")
    logger.info(f"Restored: {restored_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Binarization: {args.binarization.upper()}")
    logger.info("")
    
    # Find ground truth images
    gt_images = sorted(gt_dir.glob('*.png'))
    logger.info(f"Found {len(gt_images)} ground truth images")
    
    # Evaluate each image
    all_metrics = []
    
    for gt_path in tqdm(gt_images, desc="Evaluating"):
        # Get corresponding restored image
        img_name = gt_path.stem
        
        # Try different naming patterns
        possible_names = [
            f"{img_name}_restored.png",
            f"{img_name}.png",
        ]
        
        restored_path = None
        for name in possible_names:
            candidate = restored_dir / name
            if candidate.exists():
                restored_path = candidate
                break
        
        if restored_path is None:
            logger.warning(f"⚠️  Skipped {img_name}: No restored image found")
            continue
        
        # Evaluate
        try:
            metrics, binary_img = evaluate_image(gt_path, restored_path, args.binarization)
            
            # Save binary image
            binary_path = binary_dir / f"{img_name}_binary.png"
            cv2.imwrite(str(binary_path), binary_img)
            
            # Add to results
            metrics['image'] = img_name
            all_metrics.append(metrics)
            
            logger.info(
                f"✅ {img_name:>3}: "
                f"PSNR={metrics['psnr']:6.2f} dB, "
                f"SSIM={metrics['ssim']:.4f}, "
                f"F-measure={metrics['f_measure']*100:.2f}%"
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing {img_name}: {e}")
            continue
    
    if not all_metrics:
        logger.error("No images were successfully evaluated!")
        return
    
    # Calculate aggregate statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("Aggregate Statistics")
    logger.info("=" * 80)
    
    metrics_keys = ['psnr', 'ssim', 'f_measure', 'nrm', 'mpm']
    stats = {}
    
    for key in metrics_keys:
        values = [m[key] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        stats[key] = {
            'mean': float(mean),
            'std': float(std),
            'min': float(min_val),
            'max': float(max_val)
        }
        
        # Format output
        if key == 'psnr':
            logger.info(f"PSNR        : {mean:6.2f} ± {std:5.2f} dB [{min_val:6.2f}, {max_val:6.2f}]")
        elif key in ['ssim', 'f_measure']:
            logger.info(f"{key.upper():12s}: {mean:.4f} ± {std:.4f} [{min_val:.4f}, {max_val:.4f}]")
        else:
            logger.info(f"{key.upper():12s}: {mean:.4f} ± {std:.4f} [{min_val:.4f}, {max_val:.4f}] (lower is better)")
    
    # Save results
    results = {
        'binarization_method': args.binarization,
        'num_images': len(all_metrics),
        'aggregate': stats,
        'per_image': all_metrics
    }
    
    # Save JSON
    json_path = output_dir / f'evaluation_metrics_binary_{args.binarization}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✅ JSON saved: {json_path}")
    
    # Save CSV
    import csv
    csv_path = output_dir / f'evaluation_metrics_binary_{args.binarization}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image'] + metrics_keys)
        writer.writeheader()
        writer.writerows(all_metrics)
    logger.info(f"✅ CSV saved: {csv_path}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
