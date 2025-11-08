#!/usr/bin/env python3
"""
PSNR Verification Script - Independent PSNR Calculation Audit

Purpose:
    Verify PSNR calculations from training by recalculating from saved sample images.
    Compare training-reported PSNR vs independently calculated PSNR to detect bugs.

Process:
    1. Load comparison images (degraded|clean|generated)
    2. Extract clean (GT) and generated (restored) regions
    3. Calculate PSNR using multiple methods:
       - OpenCV cv2.PSNR
       - scikit-image metrics.peak_signal_noise_ratio
       - Manual MSE formula
       - TensorFlow tf.image.psnr
    4. Compare against training log reported PSNR
    5. Identify any calculation discrepancies

Expected Outcome:
    If PSNR calculation is correct: All methods should agree (~same value)
    If PSNR calculation is buggy: Discrepancy > 0.5 dB detected
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
import argparse
from pathlib import Path
import re

def calculate_psnr_opencv(img1, img2):
    """Calculate PSNR using OpenCV (expects values in [0, 255])"""
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)
    return cv2.PSNR(img1, img2)

def calculate_psnr_manual(img1, img2, max_val=255.0):
    """Calculate PSNR manually using MSE formula"""
    # Convert to float if needed
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # PSNR = 10 * log10(MAX^2 / MSE)
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

def calculate_psnr_skimage(img1, img2):
    """Calculate PSNR using scikit-image"""
    # Normalize to [0, 1] if needed
    if img1.max() > 1.0:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
    
    return skimage_psnr(img1, img2, data_range=1.0)

def calculate_psnr_tensorflow(img1, img2):
    """Calculate PSNR using TensorFlow (matching training script)"""
    # Normalize to [0, 1] if needed
    if img1.max() > 1.0:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
    
    # Convert to tensors (add batch + channel dims)
    img1_tensor = tf.convert_to_tensor(img1[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    img2_tensor = tf.convert_to_tensor(img2[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    
    # Calculate PSNR (matching training: max_val=1.0)
    psnr_tensor = tf.image.psnr(img1_tensor, img2_tensor, max_val=1.0)
    
    return psnr_tensor.numpy()[0]

def extract_regions_from_comparison(comparison_img):
    """
    Extract degraded, clean, and generated regions from vertical concatenation.
    
    Comparison format (VERTICAL):
        [Degraded Image]
        [Clean/GT Image]  
        [Generated/Restored Image]
    
    Returns:
        degraded, clean, generated (all as grayscale numpy arrays)
    """
    height, width = comparison_img.shape[:2]
    
    # Each region is 1/3 of total height
    region_height = height // 3
    
    # Extract regions
    degraded = comparison_img[0:region_height, :]
    clean = comparison_img[region_height:2*region_height, :]
    generated = comparison_img[2*region_height:3*region_height, :]
    
    # Convert to grayscale if needed
    if len(degraded.shape) == 3:
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    if len(clean.shape) == 3:
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    if len(generated.shape) == 3:
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
    
    return degraded, clean, generated

def parse_training_log_psnr(log_path, epoch):
    """
    Parse training log to extract reported PSNR for given epoch.
    
    Returns:
        psnr_mean, psnr_std from validation
    """
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Find validation stats for this epoch
    # Pattern: Epoch XX/50 ... Validation PSNR: XX.XX ± XX.XX
    pattern = rf'Epoch\s+{epoch}/\d+.*?PSNR:\s+([\d.]+)\s+±\s+([\d.]+)'
    match = re.search(pattern, log_content, re.DOTALL)
    
    if match:
        psnr_mean = float(match.group(1))
        psnr_std = float(match.group(2))
        return psnr_mean, psnr_std
    else:
        return None, None

def verify_sample_psnr(sample_dir, log_path, epoch):
    """
    Verify PSNR calculation for all samples in a given epoch.
    
    Args:
        sample_dir: Directory containing comparison images
        log_path: Path to training log file
        epoch: Epoch number to verify
    
    Returns:
        dict with verification results
    """
    # Find all comparison images for this epoch
    comparison_pattern = f'comparison_epoch_{epoch:04d}_sample_*.png'
    comparison_files = sorted(Path(sample_dir).glob(comparison_pattern))
    
    if not comparison_files:
        print(f"❌ No comparison images found for epoch {epoch} in {sample_dir}")
        return None
    
    print(f"\n{'='*80}")
    print(f"📊 PSNR VERIFICATION - EPOCH {epoch}")
    print(f"{'='*80}")
    print(f"Found {len(comparison_files)} comparison images")
    
    # Get training log PSNR
    log_psnr_mean, log_psnr_std = parse_training_log_psnr(log_path, epoch)
    if log_psnr_mean is not None:
        print(f"\n📝 Training Log Reported PSNR: {log_psnr_mean:.4f} ± {log_psnr_std:.4f} dB")
    else:
        print(f"\n⚠️  Could not parse PSNR from training log for epoch {epoch}")
    
    # Calculate PSNR for each sample using multiple methods
    all_psnr_opencv = []
    all_psnr_manual = []
    all_psnr_skimage = []
    all_psnr_tensorflow = []
    
    for i, comp_file in enumerate(comparison_files):
        # Load comparison image
        comparison_img = cv2.imread(str(comp_file), cv2.IMREAD_GRAYSCALE)
        
        # Extract regions
        degraded, clean, generated = extract_regions_from_comparison(comparison_img)
        
        # Calculate PSNR using multiple methods (clean vs generated)
        psnr_opencv = calculate_psnr_opencv(clean, generated)
        psnr_manual = calculate_psnr_manual(clean, generated, max_val=255.0)
        psnr_skimage = calculate_psnr_skimage(clean, generated)
        psnr_tensorflow = calculate_psnr_tensorflow(clean, generated)
        
        all_psnr_opencv.append(psnr_opencv)
        all_psnr_manual.append(psnr_manual)
        all_psnr_skimage.append(psnr_skimage)
        all_psnr_tensorflow.append(psnr_tensorflow)
        
        # Print first sample details
        if i == 0:
            print(f"\n🔍 Sample 0 Detailed Verification:")
            print(f"   OpenCV PSNR:      {psnr_opencv:.4f} dB")
            print(f"   Manual PSNR:      {psnr_manual:.4f} dB")
            print(f"   Skimage PSNR:     {psnr_skimage:.4f} dB")
            print(f"   TensorFlow PSNR:  {psnr_tensorflow:.4f} dB")
            print(f"   Image shapes: Clean={clean.shape}, Generated={generated.shape}")
            print(f"   Value ranges: Clean=[{clean.min()}, {clean.max()}], Generated=[{generated.min()}, {generated.max()}]")
    
    # Calculate statistics
    opencv_mean = np.mean(all_psnr_opencv)
    opencv_std = np.std(all_psnr_opencv, ddof=1)
    
    manual_mean = np.mean(all_psnr_manual)
    manual_std = np.std(all_psnr_manual, ddof=1)
    
    skimage_mean = np.mean(all_psnr_skimage)
    skimage_std = np.std(all_psnr_skimage, ddof=1)
    
    tensorflow_mean = np.mean(all_psnr_tensorflow)
    tensorflow_std = np.std(all_psnr_tensorflow, ddof=1)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📈 PSNR CALCULATION COMPARISON (Mean ± Std across {len(comparison_files)} samples)")
    print(f"{'='*80}")
    print(f"OpenCV cv2.PSNR:              {opencv_mean:.4f} ± {opencv_std:.4f} dB")
    print(f"Manual MSE Formula:           {manual_mean:.4f} ± {manual_std:.4f} dB")
    print(f"Skimage PSNR:                 {skimage_mean:.4f} ± {skimage_std:.4f} dB")
    print(f"TensorFlow tf.image.psnr:     {tensorflow_mean:.4f} ± {tensorflow_std:.4f} dB")
    
    if log_psnr_mean is not None:
        print(f"\n{'='*80}")
        print(f"📊 TRAINING LOG vs INDEPENDENT VERIFICATION")
        print(f"{'='*80}")
        print(f"Training Log PSNR:            {log_psnr_mean:.4f} ± {log_psnr_std:.4f} dB")
        print(f"TensorFlow Recalculated:      {tensorflow_mean:.4f} ± {tensorflow_std:.4f} dB")
        
        # Check for discrepancy
        psnr_diff = abs(log_psnr_mean - tensorflow_mean)
        print(f"\n🎯 Discrepancy: {psnr_diff:.4f} dB")
        
        if psnr_diff < 0.1:
            print(f"✅ PSNR calculation is CORRECT (difference < 0.1 dB)")
        elif psnr_diff < 0.5:
            print(f"⚠️  Minor discrepancy detected (0.1-0.5 dB)")
        else:
            print(f"❌ CRITICAL: Significant PSNR calculation error (> 0.5 dB)")
            print(f"   This suggests a BUG in training script PSNR calculation!")
    
    # Check consistency across methods
    print(f"\n{'='*80}")
    print(f"🔬 METHOD CONSISTENCY CHECK")
    print(f"{'='*80}")
    
    # Calculate max deviation from TensorFlow (reference)
    max_deviation = max(
        abs(opencv_mean - tensorflow_mean),
        abs(manual_mean - tensorflow_mean),
        abs(skimage_mean - tensorflow_mean)
    )
    
    print(f"Max deviation from TensorFlow: {max_deviation:.4f} dB")
    
    if max_deviation < 0.1:
        print(f"✅ All methods agree (deviation < 0.1 dB)")
        print(f"   PSNR calculation formula is CONSISTENT")
    else:
        print(f"⚠️  Some methods show deviation > 0.1 dB")
        print(f"   OpenCV diff:  {abs(opencv_mean - tensorflow_mean):.4f} dB")
        print(f"   Manual diff:  {abs(manual_mean - tensorflow_mean):.4f} dB")
        print(f"   Skimage diff: {abs(skimage_mean - tensorflow_mean):.4f} dB")
    
    return {
        'epoch': epoch,
        'num_samples': len(comparison_files),
        'log_psnr_mean': log_psnr_mean,
        'log_psnr_std': log_psnr_std,
        'opencv_mean': opencv_mean,
        'opencv_std': opencv_std,
        'manual_mean': manual_mean,
        'manual_std': manual_std,
        'skimage_mean': skimage_mean,
        'skimage_std': skimage_std,
        'tensorflow_mean': tensorflow_mean,
        'tensorflow_std': tensorflow_std,
        'discrepancy': psnr_diff if log_psnr_mean else None
    }

def main():
    parser = argparse.ArgumentParser(description='Verify PSNR calculation from training')
    parser.add_argument('--sample_dir', type=str, required=True,
                        help='Directory containing comparison images')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Training log file path')
    parser.add_argument('--epochs', type=int, nargs='+', default=[50],
                        help='Epoch numbers to verify (default: 50)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🔍 PSNR CALCULATION VERIFICATION TOOL")
    print(f"{'='*80}")
    print(f"Sample Directory: {args.sample_dir}")
    print(f"Training Log:     {args.log_file}")
    print(f"Epochs to verify: {args.epochs}")
    
    results = []
    for epoch in args.epochs:
        result = verify_sample_psnr(args.sample_dir, args.log_file, epoch)
        if result:
            results.append(result)
    
    # Final summary
    if results:
        print(f"\n{'='*80}")
        print(f"📋 FINAL SUMMARY")
        print(f"{'='*80}")
        
        for result in results:
            if result['discrepancy'] is not None:
                status = "✅" if result['discrepancy'] < 0.5 else "❌"
                print(f"{status} Epoch {result['epoch']}: "
                      f"Log={result['log_psnr_mean']:.2f} dB, "
                      f"TF={result['tensorflow_mean']:.2f} dB, "
                      f"Diff={result['discrepancy']:.4f} dB")

if __name__ == '__main__':
    main()
