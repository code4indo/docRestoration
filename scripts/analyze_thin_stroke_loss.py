#!/usr/bin/env python3
"""
Analyze Thin Stroke Loss in Document Restoration

Quantifies how many thin strokes are lost during GAN restoration.
This diagnostic helps identify the thin stroke preservation problem.

Usage:
    poetry run python scripts/analyze_thin_stroke_loss.py \
        --degraded dibco_datasets/2012/imgs \
        --restored results/dibco_2012_fully_fixed \
        --gt dibco_datasets/2012/gt_imgs \
        --visualize
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_stroke_width_map(binary_image):
    """
    Compute per-pixel stroke width using distance transform.
    
    Args:
        binary_image: Binary image (0=background, 255=foreground)
    
    Returns:
        stroke_width_map: Per-pixel stroke width (diameter)
    """
    # Distance transform: distance to nearest background pixel
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    
    # Stroke width = 2 * distance (diameter)
    stroke_width = dist_transform * 2.0
    
    return stroke_width


def classify_strokes_by_width(stroke_width_map, thin_threshold=2.5, thick_threshold=5.0):
    """
    Classify strokes into thin/medium/thick categories.
    
    Args:
        stroke_width_map: Per-pixel stroke width
        thin_threshold: Width below this = thin
        thick_threshold: Width above this = thick
    
    Returns:
        dict with masks for each category
    """
    foreground = stroke_width_map > 0
    
    thin_mask = (stroke_width_map > 0) & (stroke_width_map < thin_threshold)
    medium_mask = (stroke_width_map >= thin_threshold) & (stroke_width_map < thick_threshold)
    thick_mask = stroke_width_map >= thick_threshold
    
    return {
        'thin': thin_mask,
        'medium': medium_mask,
        'thick': thick_mask,
        'foreground': foreground
    }


def analyze_stroke_preservation(gt_path, restored_path, degraded_path=None):
    """
    Analyze which strokes are preserved/lost in restoration.
    
    Returns:
        dict with detailed statistics
    """
    # Load images
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    restored = cv2.imread(str(restored_path), cv2.IMREAD_GRAYSCALE)
    
    if gt is None or restored is None:
        return None
    
    # Binarize (Otsu threshold)
    _, gt_bin = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, restored_bin = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Compute stroke widths
    gt_stroke_width = compute_stroke_width_map(gt_bin)
    restored_stroke_width = compute_stroke_width_map(restored_bin)
    
    # Classify strokes by width
    gt_classes = classify_strokes_by_width(gt_stroke_width)
    restored_classes = classify_strokes_by_width(restored_stroke_width)
    
    # Compute preservation metrics
    stats = {}
    
    for category in ['thin', 'medium', 'thick']:
        gt_pixels = np.sum(gt_classes[category])
        
        if gt_pixels > 0:
            # Pixels preserved (present in both GT and restored)
            preserved_pixels = np.sum(gt_classes[category] & restored_classes['foreground'])
            preservation_rate = preserved_pixels / gt_pixels * 100
        else:
            preservation_rate = 0.0
        
        stats[category] = {
            'gt_pixels': int(gt_pixels),
            'preserved_pixels': int(preserved_pixels) if gt_pixels > 0 else 0,
            'preservation_rate': preservation_rate
        }
    
    # Overall statistics
    total_gt = np.sum(gt_classes['foreground'])
    total_restored = np.sum(restored_classes['foreground'])
    
    stats['overall'] = {
        'gt_pixels': int(total_gt),
        'restored_pixels': int(total_restored),
        'pixel_difference': int(total_restored - total_gt),
        'pixel_difference_pct': (total_restored - total_gt) / total_gt * 100 if total_gt > 0 else 0
    }
    
    # Stroke width statistics
    gt_widths = gt_stroke_width[gt_stroke_width > 0]
    restored_widths = restored_stroke_width[restored_stroke_width > 0]
    
    stats['stroke_width'] = {
        'gt_mean': float(np.mean(gt_widths)) if len(gt_widths) > 0 else 0,
        'gt_median': float(np.median(gt_widths)) if len(gt_widths) > 0 else 0,
        'restored_mean': float(np.mean(restored_widths)) if len(restored_widths) > 0 else 0,
        'restored_median': float(np.median(restored_widths)) if len(restored_widths) > 0 else 0,
    }
    
    return stats


def visualize_stroke_analysis(gt_path, restored_path, output_path):
    """Create visualization showing stroke preservation/loss."""
    # Load images
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    restored = cv2.imread(str(restored_path), cv2.IMREAD_GRAYSCALE)
    
    # Binarize
    _, gt_bin = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, restored_bin = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Compute stroke widths
    gt_stroke_width = compute_stroke_width_map(gt_bin)
    
    # Classify strokes
    gt_classes = classify_strokes_by_width(gt_stroke_width)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Ground truth analysis
    axes[0, 0].imshow(gt, cmap='gray')
    axes[0, 0].set_title('Ground Truth (Original)')
    axes[0, 0].axis('off')
    
    # Show stroke width heatmap
    stroke_heatmap = np.ma.masked_where(gt_stroke_width == 0, gt_stroke_width)
    axes[0, 1].imshow(stroke_heatmap, cmap='jet', vmin=0, vmax=10)
    axes[0, 1].set_title('GT Stroke Width (pixels)')
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
    
    # Show stroke categories
    category_map = np.zeros_like(gt, dtype=np.uint8)
    category_map[gt_classes['thin']] = 85   # Dark gray
    category_map[gt_classes['medium']] = 170  # Light gray
    category_map[gt_classes['thick']] = 255   # White
    
    axes[0, 2].imshow(category_map, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('GT Stroke Categories\n(Dark=Thin, Gray=Medium, Bright=Thick)')
    axes[0, 2].axis('off')
    
    # Row 2: Restoration analysis
    axes[1, 0].imshow(restored, cmap='gray')
    axes[1, 0].set_title('Restored Image')
    axes[1, 0].axis('off')
    
    # Show what was preserved/lost
    preserved_thin = gt_classes['thin'] & (restored_bin > 0)
    lost_thin = gt_classes['thin'] & (restored_bin == 0)
    
    preservation_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
    preservation_map[preserved_thin] = [0, 255, 0]  # Green = preserved thin
    preservation_map[lost_thin] = [255, 0, 0]       # Red = lost thin
    
    axes[1, 1].imshow(preservation_map)
    axes[1, 1].set_title('Thin Stroke Preservation\n(Green=Preserved, Red=Lost)')
    axes[1, 1].axis('off')
    
    # Show difference map
    diff = gt_bin.astype(int) - restored_bin.astype(int)
    diff_colored = np.zeros((*gt.shape, 3), dtype=np.uint8)
    diff_colored[diff > 0] = [255, 0, 0]   # Red = removed (in GT, not in restored)
    diff_colored[diff < 0] = [0, 0, 255]   # Blue = added (not in GT, in restored)
    
    axes[1, 2].imshow(diff_colored)
    axes[1, 2].set_title('Pixel Difference\n(Red=Removed, Blue=Added)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze thin stroke preservation')
    parser.add_argument('--gt', '--gt_dir', dest='gt_dir', required=True, help='Ground truth directory')
    parser.add_argument('--restored', '--restored_dir', dest='restored_dir', required=True, help='Restored images directory')
    parser.add_argument('--degraded', '--degraded_dir', dest='degraded_dir', help='Degraded images directory (optional)')
    parser.add_argument('--output', '--output_dir', dest='output_dir', default='results/stroke_analysis', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--thin_threshold', type=float, default=2.5, help='Thin stroke threshold (pixels)')
    parser.add_argument('--thick_threshold', type=float, default=5.0, help='Thick stroke threshold (pixels)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    gt_dir = Path(args.gt_dir)
    restored_dir = Path(args.restored_dir)
    
    gt_images = sorted(gt_dir.glob('*.png')) + sorted(gt_dir.glob('*.bmp')) + sorted(gt_dir.glob('*.jpg'))
    
    if len(gt_images) == 0:
        print(f"❌ No images found in {gt_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"🔬 THIN STROKE PRESERVATION ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Ground Truth:  {gt_dir}")
    print(f"Restored:      {restored_dir}")
    print(f"Images:        {len(gt_images)}")
    print(f"Output:        {output_dir}")
    print(f"\nThresholds:")
    print(f"  Thin:   < {args.thin_threshold} px")
    print(f"  Medium: {args.thin_threshold}-{args.thick_threshold} px")
    print(f"  Thick:  > {args.thick_threshold} px\n")
    
    # Analyze each image
    all_stats = []
    
    for gt_path in tqdm(gt_images, desc="Analyzing images"):
        # Find corresponding restored image
        # Try multiple naming conventions
        base_name = gt_path.stem.replace('_gt', '')
        
        # Try: {name}_restored.png (inference_portrait_overlap_experiment.py format)
        restored_path = restored_dir / f"{base_name}_restored.png"
        
        if not restored_path.exists():
            # Try: {name}.png (direct match)
            restored_path = restored_dir / gt_path.name
        
        if not restored_path.exists():
            # Try: {name}.bmp
            restored_path = restored_dir / f"{base_name}.bmp"
            
        if not restored_path.exists():
            print(f"⚠️  Skipping {gt_path.name}: no corresponding restored image")
            continue
        
        # Analyze
        stats = analyze_stroke_preservation(gt_path, restored_path)
        
        if stats is not None:
            stats['filename'] = gt_path.name
            all_stats.append(stats)
            
            # Visualize if requested
            if args.visualize:
                vis_path = output_dir / f"{gt_path.stem}_analysis.png"
                visualize_stroke_analysis(gt_path, restored_path, vis_path)
    
    # Compute aggregate statistics
    if len(all_stats) == 0:
        print("❌ No images analyzed successfully")
        return
    
    # Average preservation rates
    avg_thin = np.mean([s['thin']['preservation_rate'] for s in all_stats])
    avg_medium = np.mean([s['medium']['preservation_rate'] for s in all_stats])
    avg_thick = np.mean([s['thick']['preservation_rate'] for s in all_stats])
    
    total_gt_thin = sum([s['thin']['gt_pixels'] for s in all_stats])
    total_preserved_thin = sum([s['thin']['preserved_pixels'] for s in all_stats])
    overall_thin_rate = total_preserved_thin / total_gt_thin * 100 if total_gt_thin > 0 else 0
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY RESULTS ({len(all_stats)} images)")
    print(f"{'='*80}\n")
    
    print(f"Stroke Preservation Rates:")
    print(f"  Thin strokes:    {avg_thin:.1f}% (overall: {overall_thin_rate:.1f}%)")
    print(f"  Medium strokes:  {avg_medium:.1f}%")
    print(f"  Thick strokes:   {avg_thick:.1f}%")
    
    print(f"\nStroke Width Statistics:")
    avg_gt_width = np.mean([s['stroke_width']['gt_mean'] for s in all_stats])
    avg_restored_width = np.mean([s['stroke_width']['restored_mean'] for s in all_stats])
    print(f"  GT mean width:       {avg_gt_width:.2f} px")
    print(f"  Restored mean width: {avg_restored_width:.2f} px")
    print(f"  Change:              {avg_restored_width - avg_gt_width:+.2f} px ({(avg_restored_width/avg_gt_width - 1)*100:+.1f}%)")
    
    # Diagnosis
    print(f"\n{'='*80}")
    print(f"🔍 DIAGNOSIS")
    print(f"{'='*80}\n")
    
    if overall_thin_rate < 70:
        print(f"⚠️  CRITICAL: Thin stroke preservation is LOW ({overall_thin_rate:.1f}%)")
        print(f"    {100 - overall_thin_rate:.1f}% of thin strokes are LOST during restoration!")
        print(f"\n    Recommended actions:")
        print(f"    1. Implement post-processing stroke preservation (quick fix)")
        print(f"    2. Add stroke-aware loss function (training enhancement)")
        print(f"    3. Rebalance synthetic data (long-term fix)")
    elif overall_thin_rate < 85:
        print(f"⚠️  Thin stroke preservation is MODERATE ({overall_thin_rate:.1f}%)")
        print(f"    Some thin strokes are lost. Consider improvements.")
    else:
        print(f"✅ Thin stroke preservation is GOOD ({overall_thin_rate:.1f}%)")
        print(f"    Most thin strokes are preserved.")
    
    # Save detailed results
    results_file = output_dir / 'stroke_analysis_summary.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'thin_preservation_rate': overall_thin_rate,
                'medium_preservation_rate': avg_medium,
                'thick_preservation_rate': avg_thick,
                'avg_gt_stroke_width': avg_gt_width,
                'avg_restored_stroke_width': avg_restored_width
            },
            'per_image': all_stats
        }, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {results_file}")
    
    if args.visualize:
        print(f"✅ Visualizations saved to: {output_dir}/")


if __name__ == '__main__':
    main()
