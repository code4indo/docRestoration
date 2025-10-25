#!/usr/bin/env python3
"""
DocTR-based Line Detection for Document Restoration
Uses lightweight DB_ResNet50 or LinkNet for text line detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import sys

try:
    # Use TensorFlow backend to avoid PyTorch conflicts
    import os
    os.environ['USE_TF'] = '1'
    os.environ['USE_TORCH'] = '0'
    
    from doctr.models import detection_predictor
    from doctr.io import DocumentFile
except ImportError as e:
    print(f"ERROR: DocTR import failed: {e}")
    print("Please install: pip install python-doctr[tf]")
    sys.exit(1)


class DocTRLineDetector:
    """
    DocTR-based line detection for document images
    Uses DB_ResNet50 or LinkNet architecture
    """
    
    def __init__(self, arch='db_resnet50', min_confidence=0.5, device='cpu'):
        """
        Initialize DocTR detector
        
        Args:
            arch: Model architecture ('db_resnet50' or 'linknet_resnet18')
            min_confidence: Minimum confidence threshold for detection
            device: Device for inference ('cpu' or 'cuda')
        """
        self.arch = arch
        self.min_confidence = min_confidence
        self.device = device
        
        print(f"Loading DocTR model: {arch}...")
        self.model = detection_predictor(arch=arch, pretrained=True)
        print("DocTR model loaded successfully!")
    
    def detect_text_regions(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using DocTR
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2) in pixel coordinates
        """
        # Load document
        doc = DocumentFile.from_images(image_path)
        
        # Run detection
        result = self.model(doc)
        
        # Get image dimensions
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Extract bounding boxes
        boxes = []
        for page in result.pages:
            for block in page.blocks:
                # Get normalized coordinates (0-1 range)
                x1_norm, y1_norm = block.geometry[0]
                x2_norm, y2_norm = block.geometry[1]
                
                # Convert to pixel coordinates
                x1 = int(x1_norm * width)
                y1 = int(y1_norm * height)
                x2 = int(x2_norm * width)
                y2 = int(y2_norm * height)
                
                boxes.append((x1, y1, x2, y2))
        
        print(f"DocTR detected {len(boxes)} text region(s)")
        return boxes
    
    def regions_to_lines(self, 
                        boxes: List[Tuple[int, int, int, int]], 
                        image_shape: Tuple[int, int],
                        min_line_height: int = 30,
                        max_line_height: int = 200) -> List[Tuple[int, int, int, int]]:
        """
        Convert detected regions to line-level boxes
        
        Args:
            boxes: List of detected bounding boxes
            image_shape: (height, width) of image
            min_line_height: Minimum line height in pixels
            max_line_height: Maximum line height in pixels
            
        Returns:
            List of line bounding boxes
        """
        height, width = image_shape
        lines = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            box_height = y2 - y1
            box_width = x2 - x1
            
            # If box is already line-sized, keep it
            if min_line_height <= box_height <= max_line_height:
                lines.append(box)
            
            # If box is too tall (paragraph/block), split it
            elif box_height > max_line_height:
                # Estimate number of lines (assume ~80px per line)
                estimated_line_height = 80
                num_lines = max(1, int(box_height / estimated_line_height))
                line_height = box_height // num_lines
                
                for i in range(num_lines):
                    line_y1 = y1 + i * line_height
                    line_y2 = line_y1 + line_height if i < num_lines - 1 else y2
                    
                    # Only add if reasonable size
                    if line_y2 - line_y1 >= min_line_height:
                        lines.append((x1, line_y1, x2, line_y2))
            
            # If box is too small, skip it
            else:
                print(f"  Skipping small box: height={box_height}px")
        
        # Sort lines by vertical position
        lines.sort(key=lambda b: b[1])
        
        print(f"Converted to {len(lines)} line(s)")
        return lines
    
    def detect_lines(self, 
                    image_path: str,
                    min_line_height: int = 30,
                    max_line_height: int = 200) -> List[Tuple[int, int, int, int]]:
        """
        Complete pipeline: detect regions and convert to lines
        
        Args:
            image_path: Path to input image
            min_line_height: Minimum line height
            max_line_height: Maximum line height
            
        Returns:
            List of line bounding boxes (x1, y1, x2, y2)
        """
        # Detect text regions
        boxes = self.detect_text_regions(image_path)
        
        if not boxes:
            print("WARNING: No text regions detected!")
            return []
        
        # Load image to get dimensions
        img = cv2.imread(image_path)
        image_shape = img.shape[:2]
        
        # Convert to lines
        lines = self.regions_to_lines(boxes, image_shape, min_line_height, max_line_height)
        
        return lines
    
    def visualize_detections(self, 
                           image_path: str,
                           boxes: List[Tuple[int, int, int, int]],
                           output_path: Optional[str] = None,
                           title: str = "DocTR Line Detection"):
        """
        Visualize detected lines on image
        
        Args:
            image_path: Path to input image
            boxes: List of bounding boxes to visualize
            output_path: Optional path to save visualization
            title: Title for the plot
        """
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(img_rgb)
        
        # Draw boxes
        for idx, (x1, y1, x2, y2) in enumerate(boxes, 1):
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='orange', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add line number
            ax.text(x1 + 5, y1 + 20, f'L{idx}', 
                   color='orange', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_title(f"{title} ({len(boxes)} lines)", fontsize=14, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='DocTR Line Detection for Document Restoration')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output_dir', type=str, default='results/line_detection_doctr',
                       help='Output directory for visualizations')
    parser.add_argument('--arch', type=str, default='db_resnet50',
                       choices=['db_resnet50', 'linknet_resnet18'],
                       help='DocTR model architecture')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                       help='Minimum confidence threshold')
    parser.add_argument('--min_line_height', type=int, default=30,
                       help='Minimum line height in pixels')
    parser.add_argument('--max_line_height', type=int, default=200,
                       help='Maximum line height in pixels')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print(f"\n{'='*60}")
    print(f"DocTR Line Detection")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Architecture: {args.arch}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    detector = DocTRLineDetector(
        arch=args.arch,
        min_confidence=args.min_confidence,
        device=args.device
    )
    
    # Detect lines
    print("\nDetecting text lines...")
    lines = detector.detect_lines(
        args.input,
        min_line_height=args.min_line_height,
        max_line_height=args.max_line_height
    )
    
    if not lines:
        print("\nERROR: No lines detected!")
        return
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Detection Results")
    print(f"{'='*60}")
    print(f"Total lines detected: {len(lines)}")
    print(f"\nLine details:")
    for idx, (x1, y1, x2, y2) in enumerate(lines, 1):
        width = x2 - x1
        height = y2 - y1
        print(f"  Line {idx:2d}: ({x1:4d}, {y1:4d}) -> ({x2:4d}, {y2:4d})  "
              f"[{width:4d}×{height:3d}px]")
    
    # Visualize
    input_path = Path(args.input)
    output_path = output_dir / f"{input_path.stem}_doctr_lines.png"
    
    print(f"\nGenerating visualization...")
    detector.visualize_detections(
        args.input,
        lines,
        str(output_path),
        title=f"DocTR {args.arch} Line Detection"
    )
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
