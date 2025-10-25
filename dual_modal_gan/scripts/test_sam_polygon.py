#!/usr/bin/env python3
"""
Test SAM Polygon Segmentation for Cursive Text
==============================================

Test script to validate that SAM extracts text lines as POLYGONS
that follow the actual text contour (including ascenders/descenders)
instead of simple rectangular bounding boxes.

Expected Output:
- Polygon masks that adapt to cursive text shapes
- Visualization showing polygon vs rectangle difference
- Extracted lines with transparent/white background outside text region
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

# Add scripts dir to path
scripts_dir = Path(__file__).parent
sys.path.append(str(scripts_dir))

from line_detection_sam_polygon import SAMLineSegmenter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_lines_simple(img: np.ndarray) -> list:
    """Deteksi line bounding boxes sederhana."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Projection profile
    h_projection = np.sum(binary, axis=1)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(h_projection, height=np.mean(h_projection), distance=10)
    
    if len(peaks) == 0:
        logger.warning("No text lines detected")
        return []
    
    # Group peaks into lines
    lines = []
    current_start = peaks[0]
    current_end = peaks[0]
    
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] > 20:
            y1 = max(0, current_start - 10)
            y2 = min(img.shape[0], current_end + 10)
            lines.append([0, y1, img.shape[1], y2])
            current_start = peaks[i]
            current_end = peaks[i]
        else:
            current_end = peaks[i]
    
    # Last line
    y1 = max(0, current_start - 10)
    y2 = min(img.shape[0], current_end + 10)
    lines.append([0, y1, img.shape[1], y2])
    
    logger.info(f"Detected {len(lines)} text lines")
    return lines


def main():
    parser = argparse.ArgumentParser(description='Test SAM Polygon Segmentation')
    parser.add_argument('--input_image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--sam_checkpoint', type=str,
                       default='models/sam/sam_vit_b_01ec64.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--output_dir', type=str, default='/tmp/sam_polygon_test',
                       help='Output directory')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check CUDA
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load image
    logger.info(f"Loading: {args.input_image}")
    img = cv2.imread(args.input_image)
    if img is None:
        logger.error(f"Failed to load image")
        return
    
    logger.info(f"Image size: {img.shape}")
    
    # Detect lines
    logger.info("Detecting text lines...")
    rough_boxes = detect_lines_simple(img)
    
    if len(rough_boxes) == 0:
        logger.error("No lines detected")
        return
    
    # Initialize SAM segmenter
    logger.info("Initializing SAM polygon segmenter...")
    segmenter = SAMLineSegmenter(
        checkpoint_path=args.sam_checkpoint,
        model_type='vit_b',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        margin=5  # Small margin for polygon expansion
    )
    
    # Segment lines as polygons
    logger.info(f"Segmenting {len(rough_boxes)} lines with SAM...")
    masks, polygons = segmenter.segment_lines(img, rough_boxes, verbose=True)
    
    # Statistics
    stats = segmenter.get_statistics()
    logger.info("\n" + "="*60)
    logger.info("SAM Polygon Segmentation Statistics:")
    logger.info(f"  Total lines: {stats['total_boxes']}")
    logger.info(f"  Segmented: {stats['segmented_lines']}")
    logger.info(f"  Fallback (rectangles): {stats['fallback_boxes']}")
    logger.info(f"  Segmentation rate: {stats['segmentation_rate']*100:.1f}%")
    logger.info(f"  Avg mask coverage: {stats['avg_mask_coverage']:.1f}%")
    logger.info("="*60)
    
    # Visualize polygon vs rectangle
    vis_path = os.path.join(args.output_dir, 'polygon_vs_rectangle.png')
    segmenter.visualize_segmentation(img, rough_boxes, masks, polygons, vis_path)
    
    # Extract lines with polygon masks
    logger.info("Extracting line regions with polygon masks...")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    for i, (mask, polygon) in enumerate(zip(masks, polygons)):
        # Extract with mask (preserves irregular shape)
        line_masked = segmenter.extract_line_with_mask(
            gray_img, mask, polygon,
            background_color=255,  # White background
            output_size=None  # Keep original size
        )
        
        # Save masked line
        masked_path = os.path.join(args.output_dir, f'line_{i+1:02d}_polygon_masked.png')
        cv2.imwrite(masked_path, line_masked)
        
        # Also save polygon coordinates as text
        poly_path = os.path.join(args.output_dir, f'line_{i+1:02d}_polygon_coords.txt')
        np.savetxt(poly_path, polygon, fmt='%d', header=f'Polygon points: {len(polygon)}')
    
    logger.info(f"\n✓ Test completed successfully!")
    logger.info(f"Results: {args.output_dir}")
    logger.info(f"  - Visualization: polygon_vs_rectangle.png")
    logger.info(f"  - Masked lines: {len(masks)} files (*_polygon_masked.png)")
    logger.info(f"  - Polygon coords: {len(polygons)} files (*_polygon_coords.txt)")
    
    # Compare polygon vs rectangle coverage
    logger.info("\n" + "="*60)
    logger.info("Polygon Shape Analysis (first 5 lines):")
    for i, (polygon, bbox) in enumerate(zip(polygons[:5], rough_boxes[:5])):
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Calculate polygon area
        poly_area = cv2.contourArea(polygon.astype(np.int32))
        coverage = (poly_area / bbox_area) * 100 if bbox_area > 0 else 0
        
        # Polygon complexity (number of vertices)
        n_points = len(polygon)
        
        logger.info(
            f"  Line {i+1}: {n_points} vertices, "
            f"area={poly_area:.0f}px², "
            f"bbox coverage={coverage:.1f}%"
        )
    logger.info("="*60)


if __name__ == "__main__":
    main()
