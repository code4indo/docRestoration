#!/usr/bin/env python3
"""
Filter pseudo-labeled patches based on quality metrics

Strategy:
- Automatic filtering using confidence metrics
- Generate visual inspection samples
- Create filtered dataset for training

Quality criteria:
- SSIM in reasonable range (0.6-0.95)
- Text preservation > threshold (0.7)
- Not over-cleaned (std > threshold)
- Contrast improvement reasonable

Author: Pseudo-Labeling Pipeline  
Date: 2025-10-28
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_metrics(metrics_path: Path) -> Dict:
    """Load pseudo-label generation metrics"""
    with open(metrics_path) as f:
        return json.load(f)


def filter_patches(
    metrics: List[Dict],
    ssim_min: float = 0.6,
    ssim_max: float = 0.95,
    text_preservation_min: float = 0.7,
    text_preservation_max: float = 1.3,
    restored_std_min: float = 20.0,
    top_percentile: float = 0.7,
) -> List[Dict]:
    """
    Filter patches based on quality criteria
    
    Args:
        metrics: List of patch metrics
        ssim_min: Minimum SSIM (too low = failed restoration)
        ssim_max: Maximum SSIM (too high = no improvement)
        text_preservation_min: Minimum text preservation ratio (avoid over-cleaning)
        text_preservation_max: Maximum text preservation ratio (avoid adding noise)
        restored_std_min: Minimum std in restored (avoid over-cleaning)
        top_percentile: Keep top X% by SSIM score
    
    Returns:
        List of filtered patch metrics
    """
    filtered = []
    
    for m in metrics:
        # Basic quality checks
        if m['ssim'] < ssim_min or m['ssim'] > ssim_max:
            continue
        
        if m['text_preservation_ratio'] < text_preservation_min:
            continue  # Over-cleaned
        
        if m['text_preservation_ratio'] > text_preservation_max:
            continue  # Added too much
        
        if m['restored_std'] < restored_std_min:
            continue  # Over-cleaned (low contrast)
        
        filtered.append(m)
    
    # Sort by SSIM (quality indicator)
    filtered.sort(key=lambda x: x['ssim'], reverse=True)
    
    # Keep top percentile
    n_keep = int(len(filtered) * top_percentile)
    filtered = filtered[:n_keep]
    
    return filtered


def create_visual_inspection_samples(
    input_dir: Path,
    output_dir: Path,
    patch_names: List[str],
    n_samples: int = 20,
    split: str = 'train'
):
    """
    Create side-by-side comparison images for manual inspection
    
    Args:
        input_dir: Base directory with degraded and pseudo_gt subdirs
        output_dir: Output directory for inspection samples
        patch_names: List of patch filenames to sample from
        n_samples: Number of samples to create
        split: 'train' or 'val'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Random sample
    import random
    random.seed(42)
    sampled = random.sample(patch_names, min(n_samples, len(patch_names)))
    
    degraded_dir = input_dir / split / 'degraded'
    pseudo_gt_dir = input_dir / split / 'pseudo_gt'
    
    for i, patch_name in enumerate(sampled, 1):
        degraded_path = degraded_dir / patch_name
        pseudo_gt_path = pseudo_gt_dir / patch_name
        
        if not degraded_path.exists() or not pseudo_gt_path.exists():
            continue
        
        # Load images
        degraded = Image.open(degraded_path).convert('L')
        pseudo_gt = Image.open(pseudo_gt_path).convert('L')
        
        # Create side-by-side comparison
        width, height = degraded.size
        comparison = Image.new('L', (width * 2, height))
        comparison.paste(degraded, (0, 0))
        comparison.paste(pseudo_gt, (width, 0))
        
        # Save
        output_path = output_dir / f"sample_{i:03d}_{patch_name}"
        comparison.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Filter pseudo-labeled patches by quality'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='DokumenRusak/anri_patches',
        help='Directory containing patches and metrics'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='DokumenRusak/anri_patches_filtered',
        help='Output directory for filtered patches'
    )
    parser.add_argument(
        '--ssim_min',
        type=float,
        default=0.6,
        help='Minimum SSIM threshold (default: 0.6)'
    )
    parser.add_argument(
        '--ssim_max',
        type=float,
        default=0.95,
        help='Maximum SSIM threshold (default: 0.95)'
    )
    parser.add_argument(
        '--text_preservation_min',
        type=float,
        default=0.7,
        help='Minimum text preservation ratio (default: 0.7)'
    )
    parser.add_argument(
        '--restored_std_min',
        type=float,
        default=20.0,
        help='Minimum std in restored image (default: 20.0)'
    )
    parser.add_argument(
        '--top_percentile',
        type=float,
        default=0.7,
        help='Keep top X%% by quality (default: 0.7 = 70%%)'
    )
    parser.add_argument(
        '--inspection_samples',
        type=int,
        default=20,
        help='Number of visual inspection samples (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("PSEUDO-PATCH QUALITY FILTERING")
    print("="*80)
    print()
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("Filtering criteria:")
    print(f"  SSIM range: [{args.ssim_min}, {args.ssim_max}]")
    print(f"  Text preservation min: {args.text_preservation_min}")
    print(f"  Restored std min: {args.restored_std_min}")
    print(f"  Keep top: {args.top_percentile*100:.0f}%")
    print()
    
    # Load metrics
    metrics_path = input_dir / 'pseudo_label_metrics.json'
    if not metrics_path.exists():
        print(f"❌ Metrics file not found: {metrics_path}")
        print("Run generate_pseudo_labels_patches.py first!")
        return
    
    metrics_data = load_metrics(metrics_path)
    train_metrics = metrics_data['train_metrics']
    val_metrics = metrics_data['val_metrics']
    
    print(f"Loaded metrics:")
    print(f"  Training patches: {len(train_metrics)}")
    print(f"  Validation patches: {len(val_metrics)}")
    print()
    
    # Filter training patches
    print("Filtering training patches...")
    train_filtered = filter_patches(
        train_metrics,
        ssim_min=args.ssim_min,
        ssim_max=args.ssim_max,
        text_preservation_min=args.text_preservation_min,
        restored_std_min=args.restored_std_min,
        top_percentile=args.top_percentile,
    )
    
    # Filter validation patches
    print("Filtering validation patches...")
    val_filtered = filter_patches(
        val_metrics,
        ssim_min=args.ssim_min,
        ssim_max=args.ssim_max,
        text_preservation_min=args.text_preservation_min,
        restored_std_min=args.restored_std_min,
        top_percentile=args.top_percentile,
    )
    
    print()
    print("="*80)
    print("FILTERING RESULTS")
    print("="*80)
    print()
    print(f"Training patches:")
    print(f"  Before: {len(train_metrics)}")
    print(f"  After: {len(train_filtered)}")
    print(f"  Kept: {len(train_filtered)/len(train_metrics)*100:.1f}%")
    print()
    print(f"Validation patches:")
    print(f"  Before: {len(val_metrics)}")
    print(f"  After: {len(val_filtered)}")
    print(f"  Kept: {len(val_filtered)/len(val_metrics)*100:.1f}%")
    print()
    
    # Copy filtered patches to output directory
    train_degraded_out = output_dir / 'train' / 'degraded'
    train_clean_out = output_dir / 'train' / 'clean'
    val_degraded_out = output_dir / 'val' / 'degraded'
    val_clean_out = output_dir / 'val' / 'clean'
    
    train_degraded_out.mkdir(parents=True, exist_ok=True)
    train_clean_out.mkdir(parents=True, exist_ok=True)
    val_degraded_out.mkdir(parents=True, exist_ok=True)
    val_clean_out.mkdir(parents=True, exist_ok=True)
    
    print("Copying filtered training patches...")
    for m in tqdm(train_filtered, desc="Train"):
        patch_name = m['patch_filename']
        
        # Copy degraded
        src = input_dir / 'train' / 'degraded' / patch_name
        dst = train_degraded_out / patch_name
        shutil.copy2(src, dst)
        
        # Copy pseudo-GT
        src = input_dir / 'train' / 'pseudo_gt' / patch_name
        dst = train_clean_out / patch_name
        shutil.copy2(src, dst)
    
    print("Copying filtered validation patches...")
    for m in tqdm(val_filtered, desc="Val"):
        patch_name = m['patch_filename']
        
        # Copy degraded
        src = input_dir / 'val' / 'degraded' / patch_name
        dst = val_degraded_out / patch_name
        shutil.copy2(src, dst)
        
        # Copy pseudo-GT
        src = input_dir / 'val' / 'pseudo_gt' / patch_name
        dst = val_clean_out / patch_name
        shutil.copy2(src, dst)
    
    print()
    
    # Create visual inspection samples
    print(f"Creating {args.inspection_samples} visual inspection samples...")
    inspection_dir = output_dir / 'visual_inspection'
    
    create_visual_inspection_samples(
        input_dir,
        inspection_dir / 'train',
        [m['patch_filename'] for m in train_filtered],
        n_samples=args.inspection_samples,
        split='train'
    )
    
    create_visual_inspection_samples(
        input_dir,
        inspection_dir / 'val',
        [m['patch_filename'] for m in val_filtered],
        n_samples=min(10, len(val_filtered)),
        split='val'
    )
    
    print(f"Inspection samples saved to: {inspection_dir}")
    print()
    
    # Save filtered metrics
    filtered_metrics = {
        'filtering_config': {
            'ssim_min': args.ssim_min,
            'ssim_max': args.ssim_max,
            'text_preservation_min': args.text_preservation_min,
            'restored_std_min': args.restored_std_min,
            'top_percentile': args.top_percentile,
        },
        'train_filtered': train_filtered,
        'val_filtered': val_filtered,
        'statistics': {
            'train_before': len(train_metrics),
            'train_after': len(train_filtered),
            'train_kept_ratio': len(train_filtered) / len(train_metrics) if train_metrics else 0,
            'val_before': len(val_metrics),
            'val_after': len(val_filtered),
            'val_kept_ratio': len(val_filtered) / len(val_metrics) if val_metrics else 0,
        }
    }
    
    # Calculate filtered statistics
    if train_filtered:
        train_ssim = [m['ssim'] for m in train_filtered]
        train_text_pres = [m['text_preservation_ratio'] for m in train_filtered]
        
        filtered_metrics['train_quality'] = {
            'ssim_mean': float(np.mean(train_ssim)),
            'ssim_std': float(np.std(train_ssim)),
            'ssim_min': float(np.min(train_ssim)),
            'ssim_max': float(np.max(train_ssim)),
            'text_preservation_mean': float(np.mean(train_text_pres)),
            'text_preservation_std': float(np.std(train_text_pres)),
        }
        
        print("Filtered training quality:")
        print(f"  SSIM: {np.mean(train_ssim):.3f} ± {np.std(train_ssim):.3f} (range: {np.min(train_ssim):.3f}-{np.max(train_ssim):.3f})")
        print(f"  Text preservation: {np.mean(train_text_pres):.3f} ± {np.std(train_text_pres):.3f}")
        print()
    
    metrics_output_path = output_dir / 'filtered_metrics.json'
    with open(metrics_output_path, 'w') as f:
        json.dump(filtered_metrics, f, indent=2)
    
    print(f"Filtered metrics saved to: {metrics_output_path}")
    print()
    
    print("="*80)
    print("✅ QUALITY FILTERING COMPLETE")
    print("="*80)
    print()
    print("Manual inspection:")
    print(f"  Check samples in: {inspection_dir}")
    print("  Verify degraded → pseudo-GT looks reasonable")
    print()
    print("Next step:")
    print("  python scripts/create_pseudo_tfrecord.py \\")
    print(f"    --input_dir {output_dir}")
    print()


if __name__ == '__main__':
    main()
