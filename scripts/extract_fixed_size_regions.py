#!/usr/bin/env python3
"""
Extract fixed-size 1024×128 regions from ANRI full pages

Strategy:
- Sliding window across entire page (horizontal and vertical)
- Extract ALL 1024×128 regions that contain text
- No line detection needed - model handles it
- Maximum data utilization

Author: Simplified Extraction Pipeline
Date: 2025-10-28
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from tqdm import tqdm


def has_sufficient_text(region: np.ndarray, min_text_ratio: float = 0.05) -> bool:
    """
    Check if region contains sufficient text content
    
    Args:
        region: Image region (grayscale)
        min_text_ratio: Minimum ratio of dark pixels (text)
    
    Returns:
        True if region has enough text
    """
    # Count pixels below threshold (text pixels)
    text_pixels = (region < 200).sum()
    text_ratio = text_pixels / region.size
    
    return text_ratio >= min_text_ratio


def extract_regions_from_page(
    page_path: Path,
    output_dir: Path,
    page_id: str,
    target_width: int = 1024,
    target_height: int = 128,
    stride_x: int = 512,  # 50% overlap horizontal
    stride_y: int = 64,   # 50% overlap vertical
    min_text_ratio: float = 0.05
) -> List[Dict]:
    """
    Extract fixed-size regions from page using sliding window
    
    Returns:
        List of metadata for each extracted region
    """
    # Load image
    image = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image: {page_path}")
    
    img_height, img_width = image.shape
    
    # Extract regions with sliding window
    region_metadata = []
    region_count = 0
    
    # Slide vertically
    for y in range(0, img_height - target_height + 1, stride_y):
        # Slide horizontally
        for x in range(0, img_width - target_width + 1, stride_x):
            # Extract region
            region = image[y:y+target_height, x:x+target_width]
            
            # Check if region has sufficient text
            if not has_sufficient_text(region, min_text_ratio):
                continue
            
            # Generate filename
            region_filename = f"{page_id}_region_{region_count:04d}.jpg"
            region_path = output_dir / region_filename
            
            # Save
            cv2.imwrite(str(region_path), region, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Store metadata
            metadata = {
                'region_filename': region_filename,
                'page_filename': page_path.name,
                'page_id': page_id,
                'region_index': region_count,
                'position': (x, y),
                'size': (target_width, target_height)
            }
            region_metadata.append(metadata)
            region_count += 1
    
    return region_metadata


def main():
    parser = argparse.ArgumentParser(description='Extract fixed 1024×128 regions from ANRI pages')
    parser.add_argument('--input_dir', type=str, default='DokumenRusak/full_pages_ANRI',
                        help='Directory with ANRI full pages')
    parser.add_argument('--output_dir', type=str, default='outputs/anri_regions_1024x128',
                        help='Output directory')
    parser.add_argument('--target_width', type=int, default=1024)
    parser.add_argument('--target_height', type=int, default=128)
    parser.add_argument('--stride_x', type=int, default=512,
                        help='Horizontal stride (overlap)')
    parser.add_argument('--stride_y', type=int, default=64,
                        help='Vertical stride (overlap)')
    parser.add_argument('--min_text_ratio', type=float, default=0.05,
                        help='Minimum text content ratio (0.05 = 5%)')
    parser.add_argument('--train_ratio', type=float, default=0.85)
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    train_degraded_dir = output_dir / 'train' / 'degraded'
    val_degraded_dir = output_dir / 'val' / 'degraded'
    
    train_degraded_dir.mkdir(parents=True, exist_ok=True)
    val_degraded_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all pages
    page_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg')))
    
    if not page_files:
        raise ValueError(f"No images found in {input_dir}")
    
    print("="*80)
    print("FIXED-SIZE REGION EXTRACTION (GRID-BASED)")
    print("="*80)
    print()
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Region size: {args.target_width}×{args.target_height}")
    print(f"Stride: {args.stride_x}×{args.stride_y} (overlap)")
    print(f"Min text ratio: {args.min_text_ratio*100:.1f}%")
    print()
    print(f"Found {len(page_files)} pages")
    print()
    
    # Split train/val
    n_train = int(len(page_files) * args.train_ratio)
    train_pages = page_files[:n_train]
    val_pages = page_files[n_train:]
    
    print(f"Split: {len(train_pages)} train / {len(val_pages)} val pages")
    print()
    
    # Extract training regions
    print("Extracting training regions...")
    all_train_metadata = []
    
    for page_path in tqdm(train_pages, desc="Train"):
        page_id = page_path.stem
        
        try:
            region_metadata = extract_regions_from_page(
                page_path,
                train_degraded_dir,
                page_id,
                args.target_width,
                args.target_height,
                args.stride_x,
                args.stride_y,
                args.min_text_ratio
            )
            all_train_metadata.extend(region_metadata)
        except Exception as e:
            print(f"\nError on {page_path.name}: {e}")
            continue
    
    print()
    
    # Extract validation regions
    print("Extracting validation regions...")
    all_val_metadata = []
    
    for page_path in tqdm(val_pages, desc="Val"):
        page_id = page_path.stem
        
        try:
            region_metadata = extract_regions_from_page(
                page_path,
                val_degraded_dir,
                page_id,
                args.target_width,
                args.target_height,
                args.stride_x,
                args.stride_y,
                args.min_text_ratio
            )
            all_val_metadata.extend(region_metadata)
        except Exception as e:
            print(f"\nError on {page_path.name}: {e}")
            continue
    
    print()
    
    # Save metadata
    metadata = {
        'extraction_params': vars(args),
        'statistics': {
            'n_train_pages': len(train_pages),
            'n_val_pages': len(val_pages),
            'n_train_regions': len(all_train_metadata),
            'n_val_regions': len(all_val_metadata),
            'total_regions': len(all_train_metadata) + len(all_val_metadata),
            'avg_regions_per_train_page': len(all_train_metadata) / len(train_pages) if train_pages else 0,
            'avg_regions_per_val_page': len(all_val_metadata) / len(val_pages) if val_pages else 0
        },
        'train_regions': all_train_metadata,
        'val_regions': all_val_metadata
    }
    
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print()
    print(f"Training: {len(train_pages)} pages → {len(all_train_metadata)} regions")
    if train_pages:
        print(f"  Avg: {len(all_train_metadata) / len(train_pages):.1f} regions/page")
    print()
    print(f"Validation: {len(val_pages)} pages → {len(all_val_metadata)} regions")
    if val_pages:
        print(f"  Avg: {len(all_val_metadata) / len(val_pages):.1f} regions/page")
    print()
    print(f"Total: {len(all_train_metadata) + len(all_val_metadata)} regions")
    print()
    print(f"Metadata: {metadata_path}")
    print()
    print("="*80)
    print("✅ EXTRACTION COMPLETE")
    print("="*80)
    print()
    print("Next: Use extracted regions directly as training data")
    print("      OR generate pseudo-GT if needed")
    print()


if __name__ == '__main__':
    main()
