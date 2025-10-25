#!/usr/bin/env python3
"""
Line Detection using Kraken
===========================

This module provides line detection functionality using Kraken's 
state-of-the-art layout analysis for historical documents.

Installation:
    pip install kraken

Usage:
    from line_detection_kraken import detect_lines_kraken
    
    lines = detect_lines_kraken('document.jpg')
    for line in lines:
        print(f"Line at y={line['bbox'][1]}, height={line['bbox'][3]-line['bbox'][1]}")

Author: AI Assistant + Belekok
Date: 2025-10-22
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2

try:
    from kraken import binarization, pageseg
    from kraken.lib import models
    from PIL import Image
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False
    logging.warning("⚠️  Kraken not installed. Install with: pip install kraken")


def detect_lines_kraken(
    image_path: str,
    model: str = 'default',
    min_line_height: int = 40,
    max_line_height: int = 200
) -> List[Dict]:
    """
    Detect text lines using Kraken layout analysis.
    
    Args:
        image_path: Path to input document image
        model: Kraken model name (default: 'default')
        min_line_height: Minimum line height to keep
        max_line_height: Maximum line height to keep
    
    Returns:
        List of line dictionaries with keys:
            - bbox: (x, y, x2, y2) bounding box
            - polygon: List of (x, y) points for line polygon
            - baseline: List of (x, y) points for baseline
    """
    if not KRAKEN_AVAILABLE:
        raise ImportError("Kraken is not installed. Run: pip install kraken")
    
    logging.info(f"  Using Kraken for line detection...")
    
    # Load image
    im = Image.open(image_path)
    
    # Binarize if needed (Kraken works better on binarized images)
    im_bin = binarization.nlbin(im)
    
    # Segment lines
    try:
        # Use default segmentation model
        regions = pageseg.segment(im_bin, model=None)
        
        lines = []
        for region in regions['boxes']:
            # Extract bounding box
            x1, y1, x2, y2 = region[0], region[1], region[2], region[3]
            height = y2 - y1
            
            # Filter by height
            if min_line_height <= height <= max_line_height:
                lines.append({
                    'bbox': (x1, y1, x2, y2),
                    'height': height,
                    'width': x2 - x1
                })
        
        logging.info(f"    Kraken detected {len(lines)} valid lines")
        return lines
    
    except Exception as e:
        logging.error(f"    ❌ Kraken segmentation failed: {e}")
        return []


def extract_line_images_kraken(
    image: np.ndarray,
    lines: List[Dict],
    margin_top: int = 5,
    margin_bottom: int = 5
) -> List[np.ndarray]:
    """
    Extract line images from detected bounding boxes.
    
    Args:
        image: Full document image (grayscale)
        lines: List of line dictionaries from detect_lines_kraken
        margin_top: Top margin in pixels
        margin_bottom: Bottom margin in pixels
    
    Returns:
        List of extracted line images
    """
    height, width = image.shape[:2]
    extracted_lines = []
    
    for line_info in lines:
        x1, y1, x2, y2 = line_info['bbox']
        
        # Apply margins
        y1_margin = max(0, y1 - margin_top)
        y2_margin = min(height, y2 + margin_bottom)
        x1_margin = max(0, x1)
        x2_margin = min(width, x2)
        
        # Extract line
        line_image = image[y1_margin:y2_margin, x1_margin:x2_margin]
        extracted_lines.append(line_image)
    
    return extracted_lines


def visualize_kraken_lines(
    image: np.ndarray,
    lines: List[Dict],
    output_path: Path
):
    """
    Create visualization of Kraken-detected lines.
    
    Args:
        image: Original document image
        lines: List of line dictionaries
        output_path: Path to save visualization
    """
    # Create RGB version
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = image.copy()
    
    # Draw bounding boxes
    for i, line_info in enumerate(lines):
        x1, y1, x2, y2 = line_info['bbox']
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw line number
        cv2.putText(vis_image, f"L{i+1}", (x1 + 10, y1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Add title
    title = f"Kraken Line Detection: {len(lines)} lines"
    cv2.putText(vis_image, title, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imwrite(str(output_path), vis_image)
    logging.info(f"    ✓ Saved Kraken visualization: {output_path.name}")


# ============================================================================
# Test Script
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Kraken line detection')
    parser.add_argument('--input', type=str, required=True,
                       help='Input document image')
    parser.add_argument('--output', type=str, default='kraken_lines_detected.png',
                       help='Output visualization')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("="*70)
    print("Kraken Line Detection Test")
    print("="*70)
    
    # Detect lines
    lines = detect_lines_kraken(args.input)
    
    print(f"\nDetected {len(lines)} lines:")
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line['bbox']
        print(f"  Line {i+1}: bbox=({x1}, {y1}, {x2}, {y2}), "
              f"height={line['height']}, width={line['width']}")
    
    # Load image and visualize
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    visualize_kraken_lines(image, lines, Path(args.output))
    
    print(f"\n✓ Visualization saved to: {args.output}")
    print("="*70)
