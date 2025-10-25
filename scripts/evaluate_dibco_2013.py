#!/usr/bin/env python3
"""
DIBCO 2013 Evaluation Script - Academic Metrics
==============================================

Evaluates restored DIBCO 2013 images against ground truth using standard metrics:
- PSNR (Peak Signal-to-Noise Ratio) - Higher is better
- SSIM (Structural Similarity Index) - Higher is better
- F-Measure (Pseudo F-Measure) - Higher is better
- NRM (Negative Rate Metric) - Lower is better
- MPM (Misclassification Penalty Metric) - Lower is better

Input:
- Ground Truth: dibco_datasets/2013/gt_imgs/*.png
- Restored Results: results/dibco_2013/*_restored.png (from inference_portrait_overlap_experiment.py)

Output:
- JSON: results/dibco_2013/evaluation_metrics.json
- CSV: results/dibco_2013/evaluation_metrics.csv

Author: belekok
Date: 2025-10-25
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import cv2
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Functions (from inference_production_v3.py)
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
        logger.warning("⚠️  scikit-image not available, SSIM approximation used")
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
    Higher is better (0-1 range, typically reported as percentage).
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
    
    nrm = (fn + fp) / (np.sum(gt_bin) + 1e-10)
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
        restored: Restored image (grayscale, 0-255)
        gt: Ground truth image (grayscale, 0-255)
    
    Returns:
        dict: All metrics
    """
    metrics = {
        'psnr': calculate_psnr(restored, gt),
        'ssim': calculate_ssim(restored, gt),
        'f_measure': calculate_f_measure(restored, gt),
        'nrm': calculate_nrm(restored, gt),
        'mpm': calculate_mpm(restored, gt)
    }
    return metrics


# ============================================================================
# Evaluation Pipeline
# ============================================================================

def load_image_grayscale(path):
    """Load image and convert to grayscale if needed."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def evaluate_dibco_2013(gt_dir, restored_dir, output_dir):
    """
    Evaluate all DIBCO 2013 images.
    
    Args:
        gt_dir: Path to ground truth images
        restored_dir: Path to restored images
        output_dir: Path to save evaluation results
    """
    logger.info("="*80)
    logger.info("DIBCO 2013 Evaluation - Academic Metrics")
    logger.info("="*80)
    logger.info(f"Ground Truth: {gt_dir}")
    logger.info(f"Restored: {restored_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    # Get all GT images
    gt_dir = Path(gt_dir)
    restored_dir = Path(restored_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gt_files = sorted(gt_dir.glob("*.png"))
    if not gt_files:
        logger.error(f"No GT images found in {gt_dir}")
        return
    
    logger.info(f"Found {len(gt_files)} ground truth images")
    
    # Evaluate each image
    results = []
    
    for gt_path in tqdm(gt_files, desc="Evaluating"):
        image_name = gt_path.stem  # e.g., "1", "2", ..., "16"
        restored_path = restored_dir / f"{image_name}_restored.png"
        
        if not restored_path.exists():
            logger.warning(f"⚠️  Restored image not found: {restored_path}")
            continue
        
        # Load images
        try:
            gt_img = load_image_grayscale(gt_path)
            restored_img = load_image_grayscale(restored_path)
            
            # Ensure same size
            if gt_img.shape != restored_img.shape:
                logger.warning(f"⚠️  Size mismatch for {image_name}: GT {gt_img.shape} vs Restored {restored_img.shape}")
                # Resize restored to match GT
                restored_img = cv2.resize(restored_img, (gt_img.shape[1], gt_img.shape[0]), 
                                         interpolation=cv2.INTER_LANCZOS4)
            
            # Calculate metrics
            metrics = calculate_all_metrics(restored_img, gt_img)
            
            # Store results
            result = {
                'image_name': image_name,
                'gt_size': list(gt_img.shape),
                'restored_size': list(restored_img.shape),
                **metrics
            }
            results.append(result)
            
            # Log individual result
            logger.info(f"✅ {image_name:>3s}: PSNR={metrics['psnr']:6.2f} dB, "
                       f"SSIM={metrics['ssim']:.4f}, F-measure={metrics['f_measure']*100:5.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error processing {image_name}: {e}")
            continue
    
    # Calculate aggregate statistics
    if not results:
        logger.error("No results to aggregate!")
        return
    
    logger.info("")
    logger.info("="*80)
    logger.info("Aggregate Statistics")
    logger.info("="*80)
    
    aggregate = {}
    for metric in ['psnr', 'ssim', 'f_measure', 'nrm', 'mpm']:
        values = [r[metric] for r in results]
        aggregate[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
        
        # Log aggregate
        if metric in ['psnr']:
            logger.info(f"{metric.upper():12s}: {aggregate[metric]['mean']:6.2f} ± {aggregate[metric]['std']:5.2f} dB "
                       f"[{aggregate[metric]['min']:6.2f}, {aggregate[metric]['max']:6.2f}]")
        elif metric in ['ssim', 'f_measure']:
            logger.info(f"{metric.upper():12s}: {aggregate[metric]['mean']:.4f} ± {aggregate[metric]['std']:.4f} "
                       f"[{aggregate[metric]['min']:.4f}, {aggregate[metric]['max']:.4f}]")
        else:
            logger.info(f"{metric.upper():12s}: {aggregate[metric]['mean']:.4f} ± {aggregate[metric]['std']:.4f} "
                       f"[{aggregate[metric]['min']:.4f}, {aggregate[metric]['max']:.4f}] (lower is better)")
    
    # Prepare final output
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'DIBCO 2013',
        'num_images': len(results),
        'checkpoint': 'production_v3_academic_split_70_15_15/best_model/ckpt-88',
        'inference_method': 'inference_portrait_overlap_experiment.py',
        'aggregate_metrics': aggregate,
        'per_image_results': results
    }
    
    # Save JSON
    json_path = output_dir / 'evaluation_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n✅ JSON saved: {json_path}")
    
    # Save CSV
    csv_path = output_dir / 'evaluation_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'psnr', 'ssim', 'f_measure', 'nrm', 'mpm'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'image_name': r['image_name'],
                'psnr': f"{r['psnr']:.2f}",
                'ssim': f"{r['ssim']:.4f}",
                'f_measure': f"{r['f_measure']:.4f}",
                'nrm': f"{r['nrm']:.4f}",
                'mpm': f"{r['mpm']:.4f}"
            })
    logger.info(f"✅ CSV saved: {csv_path}")
    
    logger.info("")
    logger.info("="*80)
    logger.info("Evaluation Complete!")
    logger.info("="*80)
    
    return output


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate DIBCO 2013 results with academic metrics')
    parser.add_argument('--gt_dir', type=str, 
                       default='dibco_datasets/2013/gt_imgs',
                       help='Ground truth directory')
    parser.add_argument('--restored_dir', type=str,
                       default='results/dibco_2013',
                       help='Restored images directory')
    parser.add_argument('--output_dir', type=str,
                       default='results/dibco_2013',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    evaluate_dibco_2013(args.gt_dir, args.restored_dir, args.output_dir)


if __name__ == '__main__':
    main()
