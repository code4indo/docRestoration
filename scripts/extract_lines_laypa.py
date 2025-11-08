#!/usr/bin/env python3
"""
Extract text lines from ANRI pages using LAYPA baseline detection

Strategy:
- Use LAYPA for robust baseline detection (handles degraded docs)
- Extract lines based on baseline coordinates  
- Resize to 1024×128 to match training distribution

Author: Line-Level Pseudo-Labeling Pipeline
Date: 2025-10-28
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add dual_modal_gan to path for LAYPA imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "dual_modal_gan" / "scripts"))

from line_detection_laypa import LaypaLineDetector


def resize_to_target(
    line: np.ndarray,
    target_width: int = 1024,
    target_height: int = 128
) -> np.ndarray:
    """
    Resize line to target size while maintaining aspect ratio
    
    Strategy:
    - Resize to fit within target dimensions
    - Pad with white background to reach exact target size
    """
    if line is None or line.size == 0:
        return None
    
    line_h, line_w = line.shape
    
    # Calculate scaling
    scale_w = target_width / line_w
    scale_h = target_height / line_h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale beyond original
    
    # Resize
    new_w = int(line_w * scale)
    new_h = int(line_h * scale)
    
    if new_w > 0 and new_h > 0:
        resized = cv2.resize(line, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        return None
    
    # Create canvas with white background
    canvas = np.full((target_height, target_width), 255, dtype=np.uint8)
    
    # Center line on canvas
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Place resized line
    y_end = min(y_offset + new_h, target_height)
    x_end = min(x_offset + new_w, target_width)
    resized_h = y_end - y_offset
    resized_w = x_end - x_offset
    
    canvas[y_offset:y_end, x_offset:x_end] = resized[:resized_h, :resized_w]
    
    return canvas


def extract_lines_from_page_laypa(
    page_path: Path,
    output_dir: Path,
    page_id: str,
    laypa_detector: LaypaLineDetector,
    target_width: int = 1024,
    target_height: int = 128,
    min_line_width: int = 100,
    padding: int = 10
) -> List[Dict]:
    """
    Extract text lines from page using LAYPA baseline detection
    
    Returns:
        List of metadata for each extracted line
    """
    # Load image
    image = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image: {page_path}")
    
    # Detect baselines with LAYPA
    temp_output = output_dir / "temp_laypa"
    temp_output.mkdir(exist_ok=True)
    
    try:
        boxes = laypa_detector.detect_lines(str(page_path), str(temp_output))
    except Exception as e:
        print(f"Warning: LAYPA detection failed for {page_path.name}: {e}")
        return []
    
    # Extract and save each line
    line_metadata = []
    
    for line_idx, box in enumerate(boxes):
        # Unpack bounding box
        x1, y1, x2, y2 = box
        
        # Filter short lines (likely noise)
        line_width = x2 - x1
        line_height = y2 - y1
        
        if line_width < min_line_width:
            continue
        
        # Extract line region from bounding box
        line_image = image[y1:y2, x1:x2]
        
        if line_image is None or line_image.size == 0:
            continue
        
        # Resize to target
        resized_line = resize_to_target(line_image, target_width, target_height)
        
        if resized_line is None:
            continue
        
        # Generate filename
        line_filename = f"{page_id}_line_{line_idx:03d}.jpg"
        line_path = output_dir / line_filename
        
        # Save
        cv2.imwrite(str(line_path), resized_line, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Store metadata
        metadata = {
            'line_filename': line_filename,
            'page_filename': page_path.name,
            'page_id': page_id,
            'line_index': line_idx,
            'bounding_box': (x1, y1, x2, y2),
            'original_width': line_width,
            'original_height': line_height,
            'resized_width': target_width,
            'resized_height': target_height
        }
        line_metadata.append(metadata)
    
    return line_metadata


def main():
    parser = argparse.ArgumentParser(description='Extract lines from ANRI pages using LAYPA')
    parser.add_argument('--input_dir', type=str, default='DokumenRusak/full_pages_ANRI',
                        help='Directory with ANRI full pages')
    parser.add_argument('--output_dir', type=str, default='outputs/anri_lines_laypa',
                        help='Output directory')
    parser.add_argument('--target_width', type=int, default=1024)
    parser.add_argument('--target_height', type=int, default=128)
    parser.add_argument('--min_line_width', type=int, default=100,
                        help='Minimum line width to keep')
    parser.add_argument('--padding', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=0.85)
    parser.add_argument('--gpu_id', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    train_degraded_dir = output_dir / 'train' / 'degraded'
    val_degraded_dir = output_dir / 'val' / 'degraded'
    
    train_degraded_dir.mkdir(parents=True, exist_ok=True)
    val_degraded_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LAYPA detector
    print("Initializing LAYPA baseline detector...")
    laypa = LaypaLineDetector(gpu_id=args.gpu_id)
    print()
    
    # Find all pages
    page_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg')))
    
    if not page_files:
        raise ValueError(f"No images found in {input_dir}")
    
    print("="*80)
    print("LINE EXTRACTION FROM ANRI PAGES (LAYPA-BASED)")
    print("="*80)
    print()
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {args.target_width}×{args.target_height}")
    print(f"Min line width: {args.min_line_width} px")
    print()
    print(f"Found {len(page_files)} pages")
    print()
    
    # Split train/val
    n_train = int(len(page_files) * args.train_ratio)
    train_pages = page_files[:n_train]
    val_pages = page_files[n_train:]
    
    print(f"Split: {len(train_pages)} train / {len(val_pages)} val")
    print()
    
    # Extract training lines
    print("Extracting training lines...")
    all_train_metadata = []
    
    for page_path in tqdm(train_pages, desc="Train"):
        page_id = page_path.stem
        
        try:
            line_metadata = extract_lines_from_page_laypa(
                page_path,
                train_degraded_dir,
                page_id,
                laypa,
                args.target_width,
                args.target_height,
                args.min_line_width,
                args.padding
            )
            all_train_metadata.extend(line_metadata)
        except Exception as e:
            print(f"\nError on {page_path.name}: {e}")
            continue
    
    print()
    
    # Extract validation lines
    print("Extracting validation lines...")
    all_val_metadata = []
    
    for page_path in tqdm(val_pages, desc="Val"):
        page_id = page_path.stem
        
        try:
            line_metadata = extract_lines_from_page_laypa(
                page_path,
                val_degraded_dir,
                page_id,
                laypa,
                args.target_width,
                args.target_height,
                args.min_line_width,
                args.padding
            )
            all_val_metadata.extend(line_metadata)
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
            'n_train_lines': len(all_train_metadata),
            'n_val_lines': len(all_val_metadata),
            'total_lines': len(all_train_metadata) + len(all_val_metadata),
            'avg_lines_per_train_page': len(all_train_metadata) / len(train_pages) if train_pages else 0,
            'avg_lines_per_val_page': len(all_val_metadata) / len(val_pages) if val_pages else 0
        },
        'train_lines': all_train_metadata,
        'val_lines': all_val_metadata
    }
    
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print()
    print(f"Training: {len(train_pages)} pages → {len(all_train_metadata)} lines")
    if train_pages:
        print(f"  Avg: {len(all_train_metadata) / len(train_pages):.1f} lines/page")
    print()
    print(f"Validation: {len(val_pages)} pages → {len(all_val_metadata)} lines")
    if val_pages:
        print(f"  Avg: {len(all_val_metadata) / len(val_pages):.1f} lines/page")
    print()
    print(f"Total: {len(all_train_metadata) + len(all_val_metadata)} lines")
    print()
    print(f"Metadata: {metadata_path}")
    print()
    print("="*80)
    print("✅ EXTRACTION COMPLETE")
    print("="*80)
    print()
    print("Next: Generate pseudo-GT via inference")
    print()


if __name__ == '__main__':
    main()
