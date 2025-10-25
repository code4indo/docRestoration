#!/usr/bin/env python3
"""
Quick comparison between baseline and enhanced version
"""
import cv2
import numpy as np
from pathlib import Path

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim_simple(img1, img2):
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1, img2, data_range=255)

# Paths
gt = cv2.imread('dibco_datasets/2013/gt_imgs/14.png', cv2.IMREAD_GRAYSCALE)
baseline = cv2.imread('results/dibco_2013/14_restored.png', cv2.IMREAD_GRAYSCALE)
enhanced = cv2.imread('results/dibco_2013_prod_test/14_restored.png', cv2.IMREAD_GRAYSCALE)

print("="*60)
print("Image 14 (Worst Performer) - Comparison")
print("="*60)
print("\nBaseline (inference_portrait_overlap_experiment.py):")
print(f"  PSNR: {calculate_psnr(baseline, gt):.2f} dB")
print(f"  SSIM: {calculate_ssim_simple(baseline, gt):.4f}")
print(f"  Contrast: {baseline.std():.1f}")

print("\nEnhanced (inference_prod.py):")
print(f"  PSNR: {calculate_psnr(enhanced, gt):.2f} dB")
print(f"  SSIM: {calculate_ssim_simple(enhanced, gt):.4f}")
print(f"  Contrast: {enhanced.std():.1f}")

print("\nImprovements:")
psnr_gain = calculate_psnr(enhanced, gt) - calculate_psnr(baseline, gt)
ssim_gain = calculate_ssim_simple(enhanced, gt) - calculate_ssim_simple(baseline, gt)
print(f"  PSNR: {'+' if psnr_gain > 0 else ''}{psnr_gain:.2f} dB")
print(f"  SSIM: {'+' if ssim_gain > 0 else ''}{ssim_gain:.4f}")
print("="*60)
