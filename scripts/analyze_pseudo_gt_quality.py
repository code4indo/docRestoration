#!/usr/bin/env python3
"""
Analyze quality of pseudo-GT pairs

Computes metrics for ALL degraded-pseudoGT pairs to enable filtering
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm


def compute_pair_metrics(degraded_path: Path, pseudo_gt_path: Path) -> Dict:
    """Compute quality metrics for a single pair"""
    # Load images
    degraded = cv2.imread(str(degraded_path), cv2.IMREAD_GRAYSCALE)
    pseudo_gt = cv2.imread(str(pseudo_gt_path), cv2.IMREAD_GRAYSCALE)
    
    if degraded is None or pseudo_gt is None:
        return None
    
    # Basic statistics
    deg_mean = float(degraded.mean())
    pgt_mean = float(pseudo_gt.mean())
    deg_std = float(degraded.std())
    pgt_std = float(pseudo_gt.std())
    
    # Text pixels (< 200 threshold)
    deg_text_ratio = float((degraded < 200).sum() / degraded.size)
    pgt_text_ratio = float((pseudo_gt < 200).sum() / pseudo_gt.size)
    
    # Text preservation
    if deg_text_ratio > 0.001:
        preservation = pgt_text_ratio / deg_text_ratio
    else:
        preservation = 0.0
    
    # Contrast
    deg_contrast = int(degraded.max() - degraded.min())
    pgt_contrast = int(pseudo_gt.max() - pseudo_gt.min())
    
    # SSIM (manual implementation, simple version)
    # For speed, use correlation-based similarity
    deg_norm = (degraded - deg_mean) / (deg_std + 1e-8)
    pgt_norm = (pseudo_gt - pgt_mean) / (pgt_std + 1e-8)
    correlation = float(np.mean(deg_norm * pgt_norm))
    ssim_approx = (correlation + 1) / 2  # Map [-1,1] to [0,1]
    
    metrics = {
        'degraded_path': str(degraded_path.name),
        'pseudo_gt_path': str(pseudo_gt_path.name),
        'degraded_mean': deg_mean,
        'pseudo_gt_mean': pgt_mean,
        'degraded_std': deg_std,
        'pseudo_gt_std': pgt_std,
        'degraded_text_ratio': deg_text_ratio,
        'pseudo_gt_text_ratio': pgt_text_ratio,
        'text_preservation': preservation,
        'degraded_contrast': deg_contrast,
        'pseudo_gt_contrast': pgt_contrast,
        'ssim_approx': ssim_approx,
        'intensity_change': pgt_mean - deg_mean,
        'contrast_improvement': pgt_contrast - deg_contrast
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze pseudo-GT quality')
    parser.add_argument('--degraded_dir', type=str, required=True,
                        help='Directory with degraded images')
    parser.add_argument('--pseudo_gt_dir', type=str, required=True,
                        help='Directory with pseudo-GT images')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Output JSON file with metrics')
    
    args = parser.parse_args()
    
    degraded_dir = Path(args.degraded_dir)
    pseudo_gt_dir = Path(args.pseudo_gt_dir)
    output_json = Path(args.output_json)
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all degraded images
    degraded_files = sorted(list(degraded_dir.glob('*.jpg')) + list(degraded_dir.glob('*.png')))
    
    print("="*80)
    print("PSEUDO-GT QUALITY ANALYSIS")
    print("="*80)
    print()
    print(f"Degraded dir: {degraded_dir}")
    print(f"Pseudo-GT dir: {pseudo_gt_dir}")
    print(f"Found {len(degraded_files)} degraded images")
    print()
    
    # Compute metrics for all pairs
    all_metrics = []
    skipped = 0
    
    for degraded_path in tqdm(degraded_files, desc="Analyzing pairs"):
        # Find corresponding pseudo-GT
        pseudo_gt_name = degraded_path.stem + '_restored.png'
        pseudo_gt_path = pseudo_gt_dir / pseudo_gt_name
        
        if not pseudo_gt_path.exists():
            skipped += 1
            continue
        
        metrics = compute_pair_metrics(degraded_path, pseudo_gt_path)
        
        if metrics is not None:
            all_metrics.append(metrics)
    
    print()
    print(f"Analyzed: {len(all_metrics)} pairs")
    print(f"Skipped: {skipped} pairs (missing pseudo-GT)")
    print()
    
    # Compute aggregate statistics
    if all_metrics:
        preservations = [m['text_preservation'] for m in all_metrics]
        pgt_means = [m['pseudo_gt_mean'] for m in all_metrics]
        ssims = [m['ssim_approx'] for m in all_metrics]
        
        stats = {
            'total_pairs': len(all_metrics),
            'text_preservation': {
                'mean': float(np.mean(preservations)),
                'median': float(np.median(preservations)),
                'std': float(np.std(preservations)),
                'min': float(np.min(preservations)),
                'max': float(np.max(preservations)),
                'percentile_10': float(np.percentile(preservations, 10)),
                'percentile_90': float(np.percentile(preservations, 90))
            },
            'pseudo_gt_mean_intensity': {
                'mean': float(np.mean(pgt_means)),
                'median': float(np.median(pgt_means)),
                'std': float(np.std(pgt_means)),
                'min': float(np.min(pgt_means)),
                'max': float(np.max(pgt_means))
            },
            'ssim_approx': {
                'mean': float(np.mean(ssims)),
                'median': float(np.median(ssims)),
                'std': float(np.std(ssims)),
                'min': float(np.min(ssims)),
                'max': float(np.max(ssims))
            }
        }
        
        print("="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        print()
        print(f"Text Preservation:")
        print(f"  Mean: {stats['text_preservation']['mean']:.3f} ({stats['text_preservation']['mean']*100:.1f}%)")
        print(f"  Median: {stats['text_preservation']['median']:.3f}")
        print(f"  Range: {stats['text_preservation']['min']:.3f} to {stats['text_preservation']['max']:.3f}")
        print(f"  10th-90th percentile: {stats['text_preservation']['percentile_10']:.3f} to {stats['text_preservation']['percentile_90']:.3f}")
        print()
        print(f"Pseudo-GT Mean Intensity:")
        print(f"  Mean: {stats['pseudo_gt_mean_intensity']['mean']:.1f}")
        print(f"  Median: {stats['pseudo_gt_mean_intensity']['median']:.1f}")
        print(f"  Range: {stats['pseudo_gt_mean_intensity']['min']:.1f} to {stats['pseudo_gt_mean_intensity']['max']:.1f}")
        print()
        print(f"SSIM (approx):")
        print(f"  Mean: {stats['ssim_approx']['mean']:.3f}")
        print(f"  Median: {stats['ssim_approx']['median']:.3f}")
        print()
        
        # Quality assessment
        avg_preservation = stats['text_preservation']['mean']
        avg_pgt_mean = stats['pseudo_gt_mean_intensity']['mean']
        
        print("="*80)
        print("QUALITY VERDICT")
        print("="*80)
        print()
        
        if avg_preservation > 0.7 and 180 <= avg_pgt_mean <= 230:
            print("✅ GOOD: High text preservation, appropriate brightness")
        elif avg_preservation > 0.4 and 170 <= avg_pgt_mean <= 240:
            print("⚠️  MIXED: Acceptable but needs filtering")
            print(f"   Recommendation: Filter to keep preservation 0.4-0.85, mean 180-230")
        else:
            print("❌ POOR: Too aggressive cleaning")
            print(f"   Text preservation: {avg_preservation:.1%} (target >40%)")
            print(f"   Mean intensity: {avg_pgt_mean:.1f} (target 180-230)")
        print()
    
    # Save to JSON
    output_data = {
        'metadata': {
            'degraded_dir': str(degraded_dir),
            'pseudo_gt_dir': str(pseudo_gt_dir),
            'total_pairs': len(all_metrics),
            'skipped_pairs': skipped
        },
        'aggregate_stats': stats if all_metrics else {},
        'per_sample_metrics': all_metrics
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Metrics saved to: {output_json}")
    print()


if __name__ == '__main__':
    main()
