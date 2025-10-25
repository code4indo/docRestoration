#!/usr/bin/env python3
"""
Watershed-based Line Segmentation for Document Images
Uses morphological operations + watershed algorithm for precise line detection
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, rectangle, remove_small_objects
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from typing import List, Tuple, Optional


class WatershedLineDetector:
    """
    Watershed-based line detection using marker-controlled watershed algorithm
    """
    
    def __init__(self, 
                 min_line_height: int = 30,
                 max_line_height: int = 200,
                 min_line_width: int = 100):
        """
        Initialize watershed detector
        
        Args:
            min_line_height: Minimum acceptable line height
            max_line_height: Maximum acceptable line height
            min_line_width: Minimum acceptable line width
        """
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_line_width = min_line_width
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for watershed segmentation
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary image ready for watershed
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 10
        )
        
        # Denoise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary
    
    def detect_lines_watershed(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text lines using watershed segmentation
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of line bounding boxes (x1, y1, x2, y2)
        """
        height, width = image.shape[:2]
        
        # Preprocess
        binary = self.preprocess_image(image)
        
        # Horizontal projection for initial line separation
        h_proj = np.sum(binary, axis=1)
        
        # Find valleys (gaps between lines) using morphological operations
        # Close gaps horizontally to merge words into lines
        h_kernel_width = max(50, width // 40)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
        h_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel)
        
        # Find sure background (large gaps)
        sure_bg = cv2.dilate(h_closed, disk(3), iterations=3)
        
        # Find sure foreground (text core)
        dist_transform = cv2.distanceTransform(h_closed, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the unknown region with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        image_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        markers = watershed(-dist_transform, markers, mask=h_closed)
        
        # Extract line regions from watershed markers
        lines = []
        for label in np.unique(markers):
            if label <= 1:  # Skip background
                continue
            
            # Create mask for this label
            mask = np.zeros_like(binary, dtype=np.uint8)
            mask[markers == label] = 255
            
            # Find bounding box
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if (self.min_line_height <= h <= self.max_line_height and 
                    w >= self.min_line_width):
                    lines.append((x, y, x + w, y + h))
        
        # Sort by vertical position
        lines.sort(key=lambda b: b[1])
        
        # Merge overlapping lines
        lines = self._merge_overlapping_lines(lines)
        
        print(f"Watershed detected {len(lines)} line(s)")
        return lines
    
    def detect_lines_projection_watershed(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Hybrid: Use projection profile to identify line regions, then refine with watershed
        
        Args:
            image: Input image
            
        Returns:
            List of line bounding boxes
        """
        height, width = image.shape[:2]
        
        # Preprocess
        binary = self.preprocess_image(image)
        
        # Horizontal projection
        h_proj = np.sum(binary, axis=1)
        
        # Smooth projection
        from scipy.ndimage import gaussian_filter1d
        h_proj_smooth = gaussian_filter1d(h_proj, sigma=2)
        
        # Find peaks (text lines) and valleys (gaps)
        mean_val = np.mean(h_proj_smooth[h_proj_smooth > 0])
        threshold = mean_val * 0.3
        
        # Detect line regions
        in_line = False
        line_start = 0
        rough_lines = []
        
        for i, val in enumerate(h_proj_smooth):
            if not in_line and val > threshold:
                line_start = i
                in_line = True
            elif in_line and val <= threshold:
                rough_lines.append((0, line_start, width, i))
                in_line = False
        
        if in_line:
            rough_lines.append((0, line_start, width, height))
        
        print(f"Projection detected {len(rough_lines)} rough line region(s)")
        
        # Refine each rough line using watershed
        refined_lines = []
        for x1, y1, x2, y2 in rough_lines:
            # Extract line region
            line_roi = binary[y1:y2, x1:x2]
            
            if line_roi.size == 0:
                continue
            
            # Apply watershed to refine boundaries
            h_proj_roi = np.sum(line_roi, axis=1)
            
            # Find actual text boundaries (remove empty margins)
            nonzero = np.where(h_proj_roi > 0)[0]
            if len(nonzero) > 0:
                actual_y1 = y1 + nonzero[0]
                actual_y2 = y1 + nonzero[-1]
                
                # Find horizontal boundaries
                v_proj_roi = np.sum(line_roi, axis=0)
                nonzero_x = np.where(v_proj_roi > 0)[0]
                if len(nonzero_x) > 0:
                    actual_x1 = x1 + nonzero_x[0]
                    actual_x2 = x1 + nonzero_x[-1]
                    
                    # Validate size
                    h = actual_y2 - actual_y1
                    w = actual_x2 - actual_x1
                    if (self.min_line_height <= h <= self.max_line_height and 
                        w >= self.min_line_width):
                        refined_lines.append((actual_x1, actual_y1, actual_x2, actual_y2))
        
        print(f"Watershed refinement resulted in {len(refined_lines)} line(s)")
        return refined_lines
    
    def _merge_overlapping_lines(self, 
                                 lines: List[Tuple[int, int, int, int]],
                                 overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Merge vertically overlapping line boxes
        
        Args:
            lines: List of bounding boxes
            overlap_threshold: Minimum overlap ratio to merge
            
        Returns:
            Merged line boxes
        """
        if not lines:
            return []
        
        merged = []
        current = list(lines[0])
        
        for next_box in lines[1:]:
            x1_c, y1_c, x2_c, y2_c = current
            x1_n, y1_n, x2_n, y2_n = next_box
            
            # Check vertical overlap
            overlap_start = max(y1_c, y1_n)
            overlap_end = min(y2_c, y2_n)
            overlap = max(0, overlap_end - overlap_start)
            
            h_c = y2_c - y1_c
            h_n = y2_n - y1_n
            
            overlap_ratio = overlap / min(h_c, h_n) if min(h_c, h_n) > 0 else 0
            
            if overlap_ratio > overlap_threshold:
                # Merge
                current = [
                    min(x1_c, x1_n),
                    min(y1_c, y1_n),
                    max(x2_c, x2_n),
                    max(y2_c, y2_n)
                ]
            else:
                merged.append(tuple(current))
                current = list(next_box)
        
        merged.append(tuple(current))
        return merged
    
    def visualize_detections(self,
                           image_path: str,
                           boxes: List[Tuple[int, int, int, int]],
                           output_path: Optional[str] = None,
                           title: str = "Watershed Line Detection"):
        """
        Visualize detected lines
        
        Args:
            image_path: Path to input image
            boxes: List of bounding boxes
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
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add line number
            ax.text(x1 + 5, y1 + 20, f'L{idx}',
                   color='lime', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
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
    parser = argparse.ArgumentParser(description='Watershed-based Line Detection')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output_dir', type=str, default='results/line_detection_watershed',
                       help='Output directory')
    parser.add_argument('--method', type=str, default='hybrid',
                       choices=['watershed', 'hybrid'],
                       help='Detection method: watershed or hybrid (projection+watershed)')
    parser.add_argument('--min_line_height', type=int, default=30,
                       help='Minimum line height')
    parser.add_argument('--max_line_height', type=int, default=200,
                       help='Maximum line height')
    parser.add_argument('--min_line_width', type=int, default=100,
                       help='Minimum line width')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print(f"\n{'='*60}")
    print(f"Watershed Line Detection")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Method: {args.method}")
    print(f"{'='*60}\n")
    
    detector = WatershedLineDetector(
        min_line_height=args.min_line_height,
        max_line_height=args.max_line_height,
        min_line_width=args.min_line_width
    )
    
    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"ERROR: Could not load image: {args.input}")
        return
    
    # Detect lines
    print("Detecting lines...")
    if args.method == 'watershed':
        lines = detector.detect_lines_watershed(image)
    else:  # hybrid
        lines = detector.detect_lines_projection_watershed(image)
    
    if not lines:
        print("\nERROR: No lines detected!")
        return
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Detection Results")
    print(f"{'='*60}")
    print(f"Total lines: {len(lines)}")
    print(f"\nLine details:")
    for idx, (x1, y1, x2, y2) in enumerate(lines, 1):
        width = x2 - x1
        height = y2 - y1
        print(f"  Line {idx:2d}: ({x1:4d}, {y1:4d}) -> ({x2:4d}, {y2:4d})  "
              f"[{width:4d}×{height:3d}px]")
    
    # Visualize
    input_path = Path(args.input)
    output_path = output_dir / f"{input_path.stem}_watershed_lines.png"
    
    print(f"\nGenerating visualization...")
    detector.visualize_detections(
        args.input,
        lines,
        str(output_path),
        title=f"Watershed Line Detection ({args.method})"
    )
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
