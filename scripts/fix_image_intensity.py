#!/usr/bin/env python3
"""
Script untuk memperbaiki intensitas pixel gambar hasil training
Masalah: Denormalisasi yang salah menyebabkan text abu-abu (127-150) bukan hitam (0-50)
Solusi: Remap pixel values dari [127, 255] ke [0, 255]
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def fix_image_intensity(img, method='linear_remap'):
    """
    Fix image intensity untuk menghitamkan text
    
    Args:
        img: Input image (grayscale)
        method: 'linear_remap', 'histogram_equalization', atau 'adaptive'
    
    Returns:
        Fixed image with blacker text
    """
    if method == 'linear_remap':
        # Simple linear remapping: stretch [min, max] to [0, 255]
        min_val = img.min()
        max_val = img.max()
        
        if max_val > min_val:
            # Remap range
            img_fixed = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            img_fixed = img.copy()
    
    elif method == 'histogram_equalization':
        # Histogram equalization untuk meningkatkan contrast
        img_fixed = cv2.equalizeHist(img.astype(np.uint8))
    
    elif method == 'adaptive':
        # Adaptive thresholding dengan perbaikan
        # First, stretch the histogram
        min_val = img.min()
        max_val = img.max()
        img_stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # Then apply adaptive contrast
        img_fixed = cv2.convertScaleAbs(img_stretched, alpha=1.2, beta=-30)
    
    elif method == 'contrast_stretch':
        # Contrast stretching dengan clipping pada percentile
        p2, p98 = np.percentile(img, (2, 98))
        img_fixed = np.clip((img - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return img_fixed

def fix_comparison_image(input_path, output_path, method='linear_remap', dry_run=False):
    """
    Fix comparison image (3 sections: degraded, GT, restored)
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Gagal membaca: {input_path}")
        return False
    
    # Split menjadi 3 bagian vertikal
    height = img.shape[0]
    section_height = height // 3
    
    degraded = img[0:section_height, :]
    ground_truth = img[section_height:2*section_height, :]
    restored = img[2*section_height:, :]
    
    # Fix each section
    degraded_fixed = fix_image_intensity(degraded, method)
    ground_truth_fixed = fix_image_intensity(ground_truth, method)
    restored_fixed = fix_image_intensity(restored, method)
    
    # Concatenate back
    img_fixed = np.vstack([degraded_fixed, ground_truth_fixed, restored_fixed])
    
    if not dry_run:
        # Save fixed image
        cv2.imwrite(output_path, img_fixed)
    
    # Report statistics
    def get_stats(section, name):
        text_region = section[section < 100]
        if len(text_region) > 0:
            text_mean = np.mean(text_region)
            text_percent = len(text_region) / section.size * 100
            return f"{name}: text_mean={text_mean:.1f}, text%={text_percent:.1f}%"
        else:
            return f"{name}: no dark pixels"
    
    stats_before = [
        get_stats(degraded, "Deg"),
        get_stats(ground_truth, "GT"),
        get_stats(restored, "Res")
    ]
    
    stats_after = [
        get_stats(degraded_fixed, "Deg"),
        get_stats(ground_truth_fixed, "GT"),
        get_stats(restored_fixed, "Res")
    ]
    
    return {
        'input': input_path,
        'output': output_path,
        'stats_before': stats_before,
        'stats_after': stats_after
    }

def process_directory(input_dir, output_dir=None, pattern="*.png", method='linear_remap', dry_run=False):
    """Process all images in directory"""
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir / "fixed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(input_dir.glob(pattern))
    
    if not image_files:
        print(f"❌ No files found matching: {pattern}")
        return
    
    print(f"✅ Found {len(image_files)} files")
    print(f"📂 Output directory: {output_dir}")
    print(f"🔧 Method: {method}")
    
    if dry_run:
        print("🧪 DRY RUN MODE - No files will be saved")
    
    results = []
    
    for img_file in tqdm(image_files, desc="Processing"):
        output_path = output_dir / img_file.name
        
        if 'comparison' in img_file.name:
            result = fix_comparison_image(str(img_file), str(output_path), method, dry_run)
        else:
            # Single image
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_fixed = fix_image_intensity(img, method)
                if not dry_run:
                    cv2.imwrite(str(output_path), img_fixed)
                result = {'input': str(img_file), 'output': str(output_path)}
            else:
                result = None
        
        if result:
            results.append(result)
    
    # Print sample results
    if results and 'stats_before' in results[0]:
        print(f"\n{'='*80}")
        print("📊 SAMPLE RESULTS (first 3 images):")
        print(f"{'='*80}")
        
        for i, result in enumerate(results[:3]):
            print(f"\n{Path(result['input']).name}:")
            print("  Before:")
            for stat in result['stats_before']:
                print(f"    {stat}")
            print("  After:")
            for stat in result['stats_after']:
                print(f"    {stat}")
    
    print(f"\n✅ Processed {len(results)} images")
    print(f"📂 Saved to: {output_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Fix image intensity to make text blacker")
    parser.add_argument('--input', '-i', required=True, help='Input directory')
    parser.add_argument('--output', '-o', help='Output directory (default: input/fixed)')
    parser.add_argument('--pattern', '-p', default='*.png', help='File pattern (default: *.png)')
    parser.add_argument('--method', '-m', 
                       choices=['linear_remap', 'histogram_equalization', 'adaptive', 'contrast_stretch'],
                       default='linear_remap',
                       help='Fixing method (default: linear_remap)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no files saved)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("🔧 IMAGE INTENSITY FIXER")
    print(f"{'='*80}\n")
    
    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        method=args.method,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
