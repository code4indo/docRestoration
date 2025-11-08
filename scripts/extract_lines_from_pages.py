#!/usr/bin/env python3
"""
Extract horizontal text lines from ANRI full-page documents

Strategy:
- Horizontal projection profile analysis
- Adaptive line spacing detection
- Extract lines with padding
- Resize to 1024×128 to match training distribution

Author: Line-Level Pseudo-Labeling Pipeline
Date: 2025-10-28
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_projection_profile(image: np.ndarray) -> np.ndarray:
    """
    Calculate horizontal projection profile (sum of pixels per row)
    
    Args:
        image: Grayscale image (H, W)
    
    Returns:
        profile: Array of length H with sum of pixels per row
    """
    # Invert image so text is white on black background for easier analysis
    inverted = 255 - image
    
    # Sum pixels horizontally (across columns) for each row
    profile = np.sum(inverted, axis=1)
    
    return profile


def smooth_profile(profile: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing to reduce noise"""
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(profile, kernel, mode='same')
    return smoothed


def detect_line_boundaries(
    profile: np.ndarray, 
    min_line_height: int = 30,
    min_gap_height: int = 15,
    threshold_ratio: float = 0.15
) -> List[Tuple[int, int]]:
    """
    Detect text line boundaries from projection profile
    
    Args:
        profile: Horizontal projection profile
        min_line_height: Minimum height of a text line in pixels
        min_gap_height: Minimum gap between lines
        threshold_ratio: Ratio of max profile to consider as text
    
    Returns:
        List of (start_row, end_row) tuples
    """
    # Determine threshold for text vs background
    max_val = np.max(profile)
    threshold = max_val * threshold_ratio
    
    # Binary mask: 1 where text exists, 0 where background
    text_mask = (profile > threshold).astype(int)
    
    # Find transitions (0→1 = line start, 1→0 = line end)
    diff = np.diff(text_mask)
    line_starts = np.where(diff == 1)[0] + 1  # +1 because diff shifts index
    line_ends = np.where(diff == -1)[0] + 1
    
    # Handle edge cases
    if text_mask[0] == 1:  # Image starts with text
        line_starts = np.insert(line_starts, 0, 0)
    if text_mask[-1] == 1:  # Image ends with text
        line_ends = np.append(line_ends, len(text_mask))
    
    # Pair starts and ends
    lines = []
    for start, end in zip(line_starts, line_ends):
        line_height = end - start
        
        # Filter by minimum line height
        if line_height >= min_line_height:
            lines.append((int(start), int(end)))
    
    # Merge lines that are too close (likely same line split by noise)
    merged_lines = []
    if lines:
        current_start, current_end = lines[0]
        
        for start, end in lines[1:]:
            gap = start - current_end
            
            if gap < min_gap_height:
                # Merge with current line
                current_end = end
            else:
                # Save current line and start new one
                merged_lines.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Don't forget the last line
        merged_lines.append((current_start, current_end))
    
    return merged_lines


def extract_line_with_padding(
    image: np.ndarray, 
    start_row: int, 
    end_row: int, 
    padding: int = 10
) -> np.ndarray:
    """
    Extract a line segment with vertical padding
    
    Args:
        image: Full page image
        start_row: Starting row of text line
        end_row: Ending row of text line
        padding: Pixels to add above and below
    
    Returns:
        Line image with padding
    """
    h, w = image.shape
    
    # Add padding
    padded_start = max(0, start_row - padding)
    padded_end = min(h, end_row + padding)
    
    # Extract line
    line = image[padded_start:padded_end, :]
    
    return line


def resize_line_to_target(line: np.ndarray, target_width: int = 1024, target_height: int = 128) -> np.ndarray:
    """
    Resize line to target dimensions (1024×128) while maintaining aspect ratio
    
    Strategy:
    - If line is wider than 1024, resize to fit width
    - If line is narrower, pad with white background
    - Height is always adjusted to 128
    """
    line_h, line_w = line.shape
    
    # Calculate scaling to fit target dimensions
    scale_w = target_width / line_w
    scale_h = target_height / line_h
    
    # Use minimum scale to fit within target (maintaining aspect ratio)
    scale = min(scale_w, scale_h)
    
    # Resize
    new_w = int(line_w * scale)
    new_h = int(line_h * scale)
    
    if new_w > 0 and new_h > 0:
        resized = cv2.resize(line, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Edge case: line too small
        resized = line
        new_h, new_w = resized.shape
    
    # Create target canvas (white background)
    canvas = np.full((target_height, target_width), 255, dtype=np.uint8)
    
    # Center the resized line on canvas
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Ensure we don't exceed canvas bounds
    y_end = min(y_offset + new_h, target_height)
    x_end = min(x_offset + new_w, target_width)
    resized_h_actual = y_end - y_offset
    resized_w_actual = x_end - x_offset
    
    # Place resized line on canvas
    canvas[y_offset:y_end, x_offset:x_end] = resized[:resized_h_actual, :resized_w_actual]
    
    return canvas


def extract_lines_from_page(
    page_path: Path,
    output_dir: Path,
    page_id: str,
    target_width: int = 1024,
    target_height: int = 128,
    min_line_height: int = 30,
    padding: int = 10
) -> List[Dict]:
    """
    Extract all text lines from a single page
    
    Returns:
        List of metadata dicts for each extracted line
    """
    # Load image
    image = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image: {page_path}")
    
    # Calculate projection profile
    profile = calculate_projection_profile(image)
    
    # Smooth to reduce noise
    smoothed_profile = smooth_profile(profile, window_size=7)
    
    # Detect line boundaries
    line_boundaries = detect_line_boundaries(
        smoothed_profile,
        min_line_height=min_line_height,
        min_gap_height=15,
        threshold_ratio=0.15
    )
    
    # Extract and save each line
    line_metadata = []
    
    for line_idx, (start_row, end_row) in enumerate(line_boundaries):
        # Extract line with padding
        line_image = extract_line_with_padding(image, start_row, end_row, padding=padding)
        
        # Resize to target dimensions
        resized_line = resize_line_to_target(line_image, target_width, target_height)
        
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
            'original_start_row': int(start_row),
            'original_end_row': int(end_row),
            'original_height': int(end_row - start_row),
            'padded_start_row': max(0, start_row - padding),
            'padded_end_row': min(image.shape[0], end_row + padding),
            'resized_width': target_width,
            'resized_height': target_height
        }
        line_metadata.append(metadata)
    
    return line_metadata


def main():
    parser = argparse.ArgumentParser(description='Extract horizontal text lines from ANRI pages')
    parser.add_argument('--input_dir', type=str, default='DokumenRusak/full_pages_ANRI',
                        help='Directory containing full-page ANRI documents')
    parser.add_argument('--output_dir', type=str, default='outputs/anri_lines',
                        help='Output directory for extracted lines')
    parser.add_argument('--target_width', type=int, default=1024,
                        help='Target line width (match training)')
    parser.add_argument('--target_height', type=int, default=128,
                        help='Target line height (match training)')
    parser.add_argument('--min_line_height', type=int, default=30,
                        help='Minimum line height in pixels')
    parser.add_argument('--padding', type=int, default=10,
                        help='Vertical padding around each line')
    parser.add_argument('--train_ratio', type=float, default=0.85,
                        help='Ratio of pages for training (rest for validation)')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    train_degraded_dir = output_dir / 'train' / 'degraded'
    val_degraded_dir = output_dir / 'val' / 'degraded'
    
    train_degraded_dir.mkdir(parents=True, exist_ok=True)
    val_degraded_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all ANRI pages
    page_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg')))
    
    if not page_files:
        raise ValueError(f"No images found in {input_dir}")
    
    print("="*80)
    print("LINE EXTRACTION FROM ANRI FULL-PAGE DOCUMENTS")
    print("="*80)
    print()
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target line size: {args.target_width}×{args.target_height}")
    print(f"Min line height: {args.min_line_height} px")
    print(f"Padding: {args.padding} px")
    print()
    print(f"Found {len(page_files)} full-page documents")
    print()
    
    # Split into train and validation sets
    n_train = int(len(page_files) * args.train_ratio)
    train_pages = page_files[:n_train]
    val_pages = page_files[n_train:]
    
    print(f"Split: {len(train_pages)} train / {len(val_pages)} validation pages")
    print()
    
    # Extract lines from training pages
    print("Extracting training lines...")
    all_train_metadata = []
    
    for page_path in tqdm(train_pages, desc="Train"):
        page_id = page_path.stem
        
        try:
            line_metadata = extract_lines_from_page(
                page_path,
                train_degraded_dir,
                page_id,
                args.target_width,
                args.target_height,
                args.min_line_height,
                args.padding
            )
            all_train_metadata.extend(line_metadata)
        except Exception as e:
            print(f"\nError processing {page_path.name}: {e}")
            continue
    
    print()
    
    # Extract lines from validation pages
    print("Extracting validation lines...")
    all_val_metadata = []
    
    for page_path in tqdm(val_pages, desc="Val"):
        page_id = page_path.stem
        
        try:
            line_metadata = extract_lines_from_page(
                page_path,
                val_degraded_dir,
                page_id,
                args.target_width,
                args.target_height,
                args.min_line_height,
                args.padding
            )
            all_val_metadata.extend(line_metadata)
        except Exception as e:
            print(f"\nError processing {page_path.name}: {e}")
            continue
    
    print()
    
    # Save metadata
    metadata = {
        'extraction_params': {
            'target_width': args.target_width,
            'target_height': args.target_height,
            'min_line_height': args.min_line_height,
            'padding': args.padding,
            'train_ratio': args.train_ratio
        },
        'statistics': {
            'n_train_pages': len(train_pages),
            'n_val_pages': len(val_pages),
            'n_train_lines': len(all_train_metadata),
            'n_val_lines': len(all_val_metadata),
            'total_lines': len(all_train_metadata) + len(all_val_metadata)
        },
        'train_lines': all_train_metadata,
        'val_lines': all_val_metadata
    }
    
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print()
    print(f"Training set:")
    print(f"  Pages: {len(train_pages)}")
    print(f"  Lines extracted: {len(all_train_metadata)}")
    if train_pages:
        print(f"  Avg lines/page: {len(all_train_metadata) / len(train_pages):.1f}")
    print()
    print(f"Validation set:")
    print(f"  Pages: {len(val_pages)}")
    print(f"  Lines extracted: {len(all_val_metadata)}")
    if val_pages:
        print(f"  Avg lines/page: {len(all_val_metadata) / len(val_pages):.1f}")
    print()
    print(f"Total lines: {len(all_train_metadata) + len(all_val_metadata)}")
    print()
    print(f"Metadata saved to: {metadata_path}")
    print()
    print("="*80)
    print("✅ LINE EXTRACTION COMPLETE")
    print("="*80)
    print()
    print("Next step:")
    print(f"  Verify quality by inspecting: {train_degraded_dir}")
    print(f"  Then generate pseudo-GT with inference script")
    print()


if __name__ == '__main__':
    main()
