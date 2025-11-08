#!/usr/bin/env python3
"""
Quick PSNR Calculator for DIBCO 2012 Results
Calculates PSNR between restored images and ground truth
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json

def calculate_metrics(restored_path, gt_path):
    """Calculate PSNR and SSIM between restored and GT images"""
    # Load images as grayscale
    restored = cv2.imread(str(restored_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    
    if restored is None:
        raise ValueError(f"Failed to load restored image: {restored_path}")
    if gt is None:
        raise ValueError(f"Failed to load GT image: {gt_path}")
    
    # Resize restored to match GT if sizes differ
    if restored.shape != gt.shape:
        print(f"  ⚠️  Resizing {restored_path.name}: {restored.shape} → {gt.shape}")
        restored = cv2.resize(restored, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] for PSNR/SSIM calculation
    restored_norm = restored.astype(np.float32) / 255.0
    gt_norm = gt.astype(np.float32) / 255.0
    
    # Calculate metrics
    psnr_value = psnr(gt_norm, restored_norm, data_range=1.0)
    ssim_value = ssim(gt_norm, restored_norm, data_range=1.0)
    
    return psnr_value, ssim_value

def main():
    # Paths
    restored_dir = Path("results/dibco2012_ckpt99")
    gt_dir = Path("dibco_datasets/2012/gt_imgs")
    
    print("="*80)
    print("📊 DIBCO 2012 PSNR Calculation")
    print("="*80)
    print(f"Restored images: {restored_dir}")
    print(f"Ground truth:    {gt_dir}")
    print()
    
    # Get all restored images
    restored_files = sorted(restored_dir.glob("*_restored.png"))
    
    if not restored_files:
        print(f"❌ No restored images found in {restored_dir}")
        return
    
    results = []
    
    print(f"Found {len(restored_files)} restored images\n")
    print("-"*80)
    
    for restored_path in restored_files:
        # Extract image ID (e.g., "1_restored.png" → "1")
        image_id = restored_path.stem.replace("_restored", "")
        
        # Find corresponding GT image
        gt_path = gt_dir / f"{image_id}.png"
        
        if not gt_path.exists():
            print(f"⚠️  No GT found for {image_id}, skipping...")
            continue
        
        # Calculate metrics
        try:
            psnr_value, ssim_value = calculate_metrics(restored_path, gt_path)
            
            result = {
                'image_id': image_id,
                'psnr': float(psnr_value),
                'ssim': float(ssim_value)
            }
            results.append(result)
            
            print(f"✅ {image_id:>3s}: PSNR = {psnr_value:6.2f} dB, SSIM = {ssim_value:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {image_id}: {e}")
    
    # Calculate statistics
    if results:
        print()
        print("="*80)
        print("📈 AGGREGATE STATISTICS")
        print("="*80)
        
        psnr_values = [r['psnr'] for r in results]
        ssim_values = [r['ssim'] for r in results]
        
        psnr_mean = np.mean(psnr_values)
        psnr_std = np.std(psnr_values, ddof=1)
        psnr_min = np.min(psnr_values)
        psnr_max = np.max(psnr_values)
        psnr_median = np.median(psnr_values)
        
        ssim_mean = np.mean(ssim_values)
        ssim_std = np.std(ssim_values, ddof=1)
        ssim_min = np.min(ssim_values)
        ssim_max = np.max(ssim_values)
        ssim_median = np.median(ssim_values)
        
        print(f"\nNumber of images: {len(results)}")
        print()
        print("PSNR Statistics:")
        print(f"  Mean:   {psnr_mean:6.2f} dB")
        print(f"  Std:    {psnr_std:6.2f} dB")
        print(f"  Min:    {psnr_min:6.2f} dB")
        print(f"  Max:    {psnr_max:6.2f} dB")
        print(f"  Median: {psnr_median:6.2f} dB")
        print()
        print("SSIM Statistics:")
        print(f"  Mean:   {ssim_mean:.4f}")
        print(f"  Std:    {ssim_std:.4f}")
        print(f"  Min:    {ssim_min:.4f}")
        print(f"  Max:    {ssim_max:.4f}")
        print(f"  Median: {ssim_median:.4f}")
        
        # Assessment
        print()
        print("="*80)
        print("🎯 ASSESSMENT")
        print("="*80)
        
        if psnr_mean >= 30:
            print(f"✅ EXCELLENT PSNR: {psnr_mean:.2f} dB (Target ≥30 dB ACHIEVED!)")
        elif psnr_mean >= 25:
            print(f"✅ VERY GOOD PSNR: {psnr_mean:.2f} dB (Above 25 dB)")
        elif psnr_mean >= 22:
            print(f"✅ GOOD PSNR: {psnr_mean:.2f} dB (SOTA range for DIBCO)")
        else:
            print(f"⚠️  MODERATE PSNR: {psnr_mean:.2f} dB (Below 22 dB)")
        
        if ssim_mean >= 0.95:
            print(f"✅ EXCELLENT SSIM: {ssim_mean:.4f} (Target ≥0.95 ACHIEVED!)")
        elif ssim_mean >= 0.90:
            print(f"✅ GOOD SSIM: {ssim_mean:.4f} (Above 0.90)")
        else:
            print(f"⚠️  MODERATE SSIM: {ssim_mean:.4f}")
        
        # Save results to JSON
        output_file = restored_dir / "psnr_ssim_results.json"
        summary = {
            'checkpoint': 'thin_stroke_preservation_v1_academic/best_model/ckpt-99',
            'dataset': 'DIBCO 2012',
            'num_images': len(results),
            'statistics': {
                'psnr': {
                    'mean': float(psnr_mean),
                    'std': float(psnr_std),
                    'min': float(psnr_min),
                    'max': float(psnr_max),
                    'median': float(psnr_median)
                },
                'ssim': {
                    'mean': float(ssim_mean),
                    'std': float(ssim_std),
                    'min': float(ssim_min),
                    'max': float(ssim_max),
                    'median': float(ssim_median)
                }
            },
            'per_image_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print()
        print(f"📄 Results saved to: {output_file}")
        print("="*80)
    else:
        print("\n❌ No results to report")

if __name__ == '__main__':
    main()
