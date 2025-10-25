#!/usr/bin/env python3
"""
SAM Line Refinement Test Script
================================
Test SAM refinement dengan CUDA untuk mendeteksi text boundaries yang lebih presisi.

Usage:
    python test_sam_refinement.py --input_image <path> --sam_checkpoint <path>
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

# Add project root and scripts dir to path
project_root = Path(__file__).parent.parent.parent
scripts_dir = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(scripts_dir))

from line_detection_sam import SAMLineRefiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_lines_simple(img: np.ndarray) -> list:
    """Deteksi line bounding boxes sederhana menggunakan projection profile."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Projection profile
    h_projection = np.sum(binary, axis=1)
    
    # Find peaks (text regions)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(h_projection, height=np.mean(h_projection), distance=10)
    
    if len(peaks) == 0:
        logger.warning("No text lines detected")
        return []
    
    # Group consecutive peaks into lines
    lines = []
    current_line_start = peaks[0]
    current_line_end = peaks[0]
    
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] > 20:  # Gap threshold
            # End current line
            y1 = max(0, current_line_start - 10)
            y2 = min(img.shape[0], current_line_end + 10)
            lines.append([0, y1, img.shape[1], y2])
            
            # Start new line
            current_line_start = peaks[i]
            current_line_end = peaks[i]
        else:
            current_line_end = peaks[i]
    
    # Add last line
    y1 = max(0, current_line_start - 10)
    y2 = min(img.shape[0], current_line_end + 10)
    lines.append([0, y1, img.shape[1], y2])
    
    logger.info(f"Detected {len(lines)} text lines using simple projection profile")
    return lines


def visualize_boxes(img: np.ndarray, original_boxes: list, refined_boxes: list, output_path: str):
    """Visualisasi perbandingan original vs refined boxes."""
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    # Draw original boxes (red)
    for box in original_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, "Original", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw refined boxes (green)
    for box in refined_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "SAM Refined", (x1+5, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, vis)
    logger.info(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test SAM Line Refinement with CUDA')
    parser.add_argument('--input_image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--sam_checkpoint', type=str, 
                       default='models/sam/sam_vit_b_01ec64.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    parser.add_argument('--output_dir', type=str, default='/tmp/sam_test',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check CUDA
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    # Load image
    logger.info(f"Loading image: {args.input_image}")
    img = cv2.imread(args.input_image)
    if img is None:
        logger.error(f"Failed to load image: {args.input_image}")
        return
    
    logger.info(f"Image shape: {img.shape}")
    
    # Detect lines dengan simple method
    logger.info("Detecting text lines with projection profile...")
    original_boxes = detect_lines_simple(img)
    
    if len(original_boxes) == 0:
        logger.error("No text lines detected, aborting")
        return
    
    # Initialize SAM refiner
    logger.info(f"Initializing SAM refiner: {args.sam_model_type}")
    logger.info(f"Checkpoint: {args.sam_checkpoint}")
    
    try:
        sam_refiner = SAMLineRefiner(
            checkpoint_path=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("✓ SAM refiner initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SAM: {e}")
        return
    
    # Refine boxes dengan SAM
    logger.info(f"Refining {len(original_boxes)} boxes with SAM...")
    refined_boxes = sam_refiner.refine_boxes(img, original_boxes)
    
    # Get statistics
    stats = sam_refiner.get_statistics()
    logger.info("\n" + "="*60)
    logger.info("SAM Refinement Statistics:")
    logger.info(f"  Total boxes: {stats['total_boxes']}")
    logger.info(f"  Refined boxes: {stats['refined_boxes']}")
    logger.info(f"  Fallback boxes: {stats['fallback_boxes']}")
    logger.info(f"  Refinement rate: {stats['refinement_rate']*100:.1f}%")
    logger.info(f"  Fallback rate: {stats['fallback_rate']*100:.1f}%")
    logger.info(f"  Avg height reduction: {stats['avg_height_reduction']:.1f}px")
    logger.info("="*60)
    
    # Visualize comparison
    output_vis = os.path.join(args.output_dir, 'sam_refinement_comparison.png')
    visualize_boxes(img, original_boxes, refined_boxes, output_vis)
    
    # Save individual refined line crops
    logger.info("Extracting refined line crops...")
    for i, box in enumerate(refined_boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        line_crop = img[y1:y2, x1:x2]
        
        crop_path = os.path.join(args.output_dir, f'line_{i+1:02d}_refined.png')
        cv2.imwrite(crop_path, line_crop)
    
    logger.info(f"\n✓ Test completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  - Visualization: {output_vis}")
    logger.info(f"  - Line crops: {len(refined_boxes)} files")


if __name__ == "__main__":
    main()
