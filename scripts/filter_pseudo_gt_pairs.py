#!/usr/bin/env python3
"""
Filter pseudo-GT pairs based on quality metrics

Keeps only high-quality pairs for fine-tuning
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def apply_quality_filters(
    metrics: Dict,
    min_preservation: float,
    max_preservation: float,
    min_pseudo_gt_mean: float,
    max_pseudo_gt_mean: float,
    min_ssim: float
) -> bool:
    """Check if sample passes quality filters"""
    
    # Text preservation filter
    if not (min_preservation <= metrics['text_preservation'] <= max_preservation):
        return False
    
    # Pseudo-GT mean intensity filter
    if not (min_pseudo_gt_mean <= metrics['pseudo_gt_mean'] <= max_pseudo_gt_mean):
        return False
    
    # SSIM filter
    if metrics['ssim_approx'] < min_ssim:
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Filter pseudo-GT pairs by quality')
    parser.add_argument('--quality_json', type=str, required=True,
                        help='JSON file with quality metrics from analyze script')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for filtered pairs')
    parser.add_argument('--min_preservation', type=float, default=0.40,
                        help='Minimum text preservation ratio (default: 0.40)')
    parser.add_argument('--max_preservation', type=float, default=0.85,
                        help='Maximum text preservation ratio (default: 0.85)')
    parser.add_argument('--min_pseudo_gt_mean', type=float, default=180,
                        help='Minimum pseudo-GT mean intensity (default: 180)')
    parser.add_argument('--max_pseudo_gt_mean', type=float, default=230,
                        help='Maximum pseudo-GT mean intensity (default: 230)')
    parser.add_argument('--min_ssim', type=float, default=0.60,
                        help='Minimum SSIM (default: 0.60)')
    
    args = parser.parse_args()
    
    quality_json = Path(args.quality_json)
    output_dir = Path(args.output_dir)
    
    # Load quality metrics
    with open(quality_json, 'r') as f:
        data = json.load(f)
    
    metrics_list = data['per_sample_metrics']
    degraded_dir = Path(data['metadata']['degraded_dir'])
    pseudo_gt_dir = Path(data['metadata']['pseudo_gt_dir'])
    
    print("="*80)
    print("PSEUDO-GT QUALITY FILTERING")
    print("="*80)
    print()
    print(f"Input metrics: {quality_json}")
    print(f"Total pairs: {len(metrics_list)}")
    print()
    print("Filter criteria:")
    print(f"  Text preservation: {args.min_preservation:.2f} - {args.max_preservation:.2f}")
    print(f"  Pseudo-GT mean: {args.min_pseudo_gt_mean:.1f} - {args.max_pseudo_gt_mean:.1f}")
    print(f"  SSIM (approx): >= {args.min_ssim:.2f}")
    print()
    
    # Create output directories
    filtered_degraded_dir = output_dir / 'train' / 'degraded'
    filtered_pseudo_gt_dir = output_dir / 'train' / 'pseudo_gt'
    
    filtered_degraded_dir.mkdir(parents=True, exist_ok=True)
    filtered_pseudo_gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter pairs
    kept_pairs = []
    filter_reasons = {
        'text_preservation_low': 0,
        'text_preservation_high': 0,
        'pseudo_gt_mean_low': 0,
        'pseudo_gt_mean_high': 0,
        'ssim_low': 0
    }
    
    for metrics in tqdm(metrics_list, desc="Filtering pairs"):
        passed = apply_quality_filters(
            metrics,
            args.min_preservation,
            args.max_preservation,
            args.min_pseudo_gt_mean,
            args.max_pseudo_gt_mean,
            args.min_ssim
        )
        
        if passed:
            kept_pairs.append(metrics)
        else:
            # Track rejection reasons
            if metrics['text_preservation'] < args.min_preservation:
                filter_reasons['text_preservation_low'] += 1
            elif metrics['text_preservation'] > args.max_preservation:
                filter_reasons['text_preservation_high'] += 1
            
            if metrics['pseudo_gt_mean'] < args.min_pseudo_gt_mean:
                filter_reasons['pseudo_gt_mean_low'] += 1
            elif metrics['pseudo_gt_mean'] > args.max_pseudo_gt_mean:
                filter_reasons['pseudo_gt_mean_high'] += 1
            
            if metrics['ssim_approx'] < args.min_ssim:
                filter_reasons['ssim_low'] += 1
    
    print()
    print("="*80)
    print("FILTERING RESULTS")
    print("="*80)
    print()
    print(f"Input pairs: {len(metrics_list)}")
    print(f"Kept pairs: {len(kept_pairs)} ({len(kept_pairs)/len(metrics_list)*100:.1f}%)")
    print(f"Rejected pairs: {len(metrics_list) - len(kept_pairs)}")
    print()
    print("Rejection reasons (samples may fail multiple filters):")
    print(f"  Text preservation too low (<{args.min_preservation}): {filter_reasons['text_preservation_low']}")
    print(f"  Text preservation too high (>{args.max_preservation}): {filter_reasons['text_preservation_high']}")
    print(f"  Pseudo-GT too dark (<{args.min_pseudo_gt_mean}): {filter_reasons['pseudo_gt_mean_low']}")
    print(f"  Pseudo-GT too bright (>{args.max_pseudo_gt_mean}): {filter_reasons['pseudo_gt_mean_high']}")
    print(f"  SSIM too low (<{args.min_ssim}): {filter_reasons['ssim_low']}")
    print()
    
    # Copy filtered pairs
    print("Copying filtered pairs...")
    for metrics in tqdm(kept_pairs, desc="Copying files"):
        degraded_src = degraded_dir / metrics['degraded_path']
        pseudo_gt_src = pseudo_gt_dir / metrics['pseudo_gt_path']
        
        degraded_dst = filtered_degraded_dir / metrics['degraded_path']
        pseudo_gt_dst = filtered_pseudo_gt_dir / metrics['pseudo_gt_path']
        
        if degraded_src.exists():
            shutil.copy2(degraded_src, degraded_dst)
        if pseudo_gt_src.exists():
            shutil.copy2(pseudo_gt_src, pseudo_gt_dst)
    
    # Save filtered metadata
    filtered_metadata = {
        'filter_params': vars(args),
        'statistics': {
            'input_pairs': len(metrics_list),
            'kept_pairs': len(kept_pairs),
            'rejection_rate': (len(metrics_list) - len(kept_pairs)) / len(metrics_list),
            'rejection_reasons': filter_reasons
        },
        'filtered_pairs': kept_pairs
    }
    
    metadata_path = output_dir / 'filtered_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(filtered_metadata, f, indent=2)
    
    print()
    print("="*80)
    print("✅ FILTERING COMPLETE")
    print("="*80)
    print()
    print(f"Filtered degraded: {filtered_degraded_dir}")
    print(f"Filtered pseudo-GT: {filtered_pseudo_gt_dir}")
    print(f"Metadata: {metadata_path}")
    print()
    print(f"Ready for TFRecord creation with {len(kept_pairs)} high-quality pairs")
    print()


if __name__ == '__main__':
    main()
