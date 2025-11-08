#!/usr/bin/env python3
"""
Analyze per-image PSNR patterns to understand what makes some images easier/harder
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
import json

def analyze_image_characteristics(img_path, gt_path):
    """Analyze image characteristics that might affect PSNR"""
    
    degraded = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    
    # Degraded image characteristics
    deg_mean = degraded.mean()
    deg_std = degraded.std()
    deg_min = degraded.min()
    deg_max = degraded.max()
    deg_range = deg_max - deg_min
    
    # GT characteristics (binary)
    gt_text_ratio = (gt == 0).sum() / gt.size  # Ratio of text pixels
    
    # Calculate baseline PSNR (degraded + Otsu vs GT)
    _, deg_binary = cv2.threshold(degraded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if deg_binary.shape != gt.shape:
        deg_binary = cv2.resize(deg_binary, (gt.shape[1], gt.shape[0]))
    
    baseline_psnr = psnr(gt.astype(float)/255, deg_binary.astype(float)/255, data_range=1.0)
    
    # Degradation level estimate
    # Low contrast = low std, narrow range
    is_low_contrast = deg_std < 30 or deg_range < 200
    
    # High mean = very bright (less degradation?)
    is_very_bright = deg_mean > 230
    
    return {
        'degraded_mean': float(deg_mean),
        'degraded_std': float(deg_std),
        'degraded_range': int(deg_range),
        'degraded_min': int(deg_min),
        'degraded_max': int(deg_max),
        'gt_text_ratio': float(gt_text_ratio),
        'baseline_psnr': float(baseline_psnr),
        'is_low_contrast': bool(is_low_contrast),
        'is_very_bright': bool(is_very_bright),
        'image_size': degraded.shape
    }

def main():
    input_dir = Path("dibco_datasets/2012/imgs")
    gt_dir = Path("dibco_datasets/2012/gt_imgs")
    
    print("="*80)
    print("🔍 Per-Image Characteristics Analysis")
    print("="*80)
    print()
    
    results = []
    
    for img_path in sorted(input_dir.glob("*.png")):
        img_id = img_path.stem
        gt_path = gt_dir / f"{img_id}.png"
        
        if not gt_path.exists():
            continue
        
        chars = analyze_image_characteristics(img_path, gt_path)
        chars['image_id'] = img_id
        results.append(chars)
        
        status = "✅" if chars['baseline_psnr'] > 18 else "⚠️"
        
        print(f"{status} Image {img_id:>3s}:")
        print(f"   Baseline PSNR:  {chars['baseline_psnr']:6.2f} dB")
        print(f"   Degraded:       mean={chars['degraded_mean']:5.1f}, std={chars['degraded_std']:5.1f}, range=[{chars['degraded_min']}, {chars['degraded_max']}]")
        print(f"   Text ratio:     {chars['gt_text_ratio']*100:5.2f}%")
        print(f"   Low contrast:   {chars['is_low_contrast']}")
        print(f"   Very bright:    {chars['is_very_bright']}")
        print()
    
    # Sort by baseline PSNR
    results_sorted = sorted(results, key=lambda x: x['baseline_psnr'], reverse=True)
    
    print("="*80)
    print("📊 SUMMARY")
    print("="*80)
    print()
    print("Top 3 easiest images (highest baseline PSNR):")
    for i, r in enumerate(results_sorted[:3], 1):
        print(f"  {i}. Image {r['image_id']:>3s}: {r['baseline_psnr']:6.2f} dB "
              f"(mean={r['degraded_mean']:.1f}, std={r['degraded_std']:.1f})")
    
    print()
    print("Bottom 3 hardest images (lowest baseline PSNR):")
    for i, r in enumerate(results_sorted[-3:], 1):
        print(f"  {i}. Image {r['image_id']:>3s}: {r['baseline_psnr']:6.2f} dB "
              f"(mean={r['degraded_mean']:.1f}, std={r['degraded_std']:.1f})")
    
    print()
    print("Correlation analysis:")
    
    # Calculate correlations
    psnrs = [r['baseline_psnr'] for r in results]
    means = [r['degraded_mean'] for r in results]
    stds = [r['degraded_std'] for r in results]
    ranges = [r['degraded_range'] for r in results]
    text_ratios = [r['gt_text_ratio'] for r in results]
    
    corr_mean = np.corrcoef(psnrs, means)[0, 1]
    corr_std = np.corrcoef(psnrs, stds)[0, 1]
    corr_range = np.corrcoef(psnrs, ranges)[0, 1]
    corr_text = np.corrcoef(psnrs, text_ratios)[0, 1]
    
    print(f"  PSNR vs degraded_mean:   {corr_mean:+.3f}")
    print(f"  PSNR vs degraded_std:    {corr_std:+.3f}")
    print(f"  PSNR vs degraded_range:  {corr_range:+.3f}")
    print(f"  PSNR vs text_ratio:      {corr_text:+.3f}")
    
    print()
    print("="*80)
    print("💡 INSIGHTS")
    print("="*80)
    
    # Find patterns
    low_contrast_imgs = [r for r in results if r['is_low_contrast']]
    low_contrast_psnr_avg = np.mean([r['baseline_psnr'] for r in low_contrast_imgs])
    high_contrast_imgs = [r for r in results if not r['is_low_contrast']]
    high_contrast_psnr_avg = np.mean([r['baseline_psnr'] for r in high_contrast_imgs]) if high_contrast_imgs else 0
    
    print(f"Low contrast images ({len(low_contrast_imgs)}): avg PSNR = {low_contrast_psnr_avg:.2f} dB")
    print(f"High contrast images ({len(high_contrast_imgs)}): avg PSNR = {high_contrast_psnr_avg:.2f} dB")
    
    if low_contrast_psnr_avg < high_contrast_psnr_avg - 2:
        print("\n⚠️  Low-contrast images perform significantly worse!")
        print("   → Pre-processing should focus on contrast enhancement")
    
    # Save results
    output_file = Path("results/dibco2012_image_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'per_image': results_sorted,
            'correlations': {
                'psnr_vs_mean': float(corr_mean),
                'psnr_vs_std': float(corr_std),
                'psnr_vs_range': float(corr_range),
                'psnr_vs_text_ratio': float(corr_text)
            },
            'insights': {
                'low_contrast_count': len(low_contrast_imgs),
                'low_contrast_avg_psnr': float(low_contrast_psnr_avg),
                'high_contrast_count': len(high_contrast_imgs),
                'high_contrast_avg_psnr': float(high_contrast_psnr_avg)
            }
        }, f, indent=2)
    
    print(f"\n📄 Analysis saved to: {output_file}")

if __name__ == '__main__':
    main()
