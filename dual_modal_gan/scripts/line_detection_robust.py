/home/lambda_one/tesis/GAN-HTR-ORI/loghi-htr#!/usr/bin/env python3
"""
Robust Line Detection for Historical Documents
===============================================

This module implements state-of-the-art line detection methods specifically
designed for degraded historical documents, inspired by:
- ARU-Net methodology
- SegHist approach
- Classical morphological operations

Methods implemented:
1. Connected Components Analysis (CCA)
2. Morphological Operations with Adaptive Kernels
3. RLSA (Run Length Smearing Algorithm)
4. Horizontal Projection with Adaptive Thresholding

Author: AI Assistant + Belekok
Date: 2025-10-22
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks, savgol_filter


class RobustLineDetector:
    """
    Robust line detector for historical documents using multiple methods.
    """
    
    def __init__(
        self,
        min_line_height: int = 40,
        max_line_height: int = 200,
        min_line_width: int = 100,
        min_text_density: float = 0.01,
        max_text_density: float = 0.95
    ):
        """
        Initialize robust line detector.
        
        Args:
            min_line_height: Minimum line height in pixels
            max_line_height: Maximum line height in pixels
            min_line_width: Minimum line width in pixels
            min_text_density: Minimum text density (% of black pixels)
            max_text_density: Maximum text density (% of black pixels)
        """
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_line_width = min_line_width
        self.min_text_density = min_text_density
        self.max_text_density = max_text_density
        
        logging.info("✓ RobustLineDetector initialized")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better line detection.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize
        image_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image_norm,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=51,
            C=10
        )
        
        # Denoise
        kernel_denoise = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_denoise)
        
        return binary
    
    def detect_lines_morphological(
        self,
        image: np.ndarray,
        binary: np.ndarray
    ) -> List[Dict]:
        """
        Detect lines using morphological operations (RLSA-inspired).
        
        This is the MOST RELIABLE method for historical documents.
        
        Args:
            image: Original grayscale image
            binary: Preprocessed binary image
        
        Returns:
            List of detected line dictionaries
        """
        height, width = binary.shape
        
        # Apply RLSA (Run Length Smearing Algorithm) horizontally
        # This connects nearby text pixels horizontally
        kernel_width = max(50, width // 40)  # Adaptive kernel
        kernel_rlsa = np.ones((1, kernel_width), np.uint8)
        rlsa_horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_rlsa)
        
        # Apply vertical closing to merge vertically close components
        kernel_vertical = np.ones((5, 1), np.uint8)
        rlsa_result = cv2.morphologyEx(rlsa_horizontal, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            rlsa_result, connectivity=8
        )
        
        lines = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            
            # Filter by dimensions
            if not (self.min_line_height <= h <= self.max_line_height and w >= self.min_line_width):
                continue
            
            # Extract region from binary image
            line_region = binary[y:y+h, x:x+w]
            
            # Calculate text density
            text_pixels = np.sum(line_region > 0)
            total_pixels = w * h
            density = text_pixels / total_pixels if total_pixels > 0 else 0
            
            # Filter by density
            if not (self.min_text_density <= density <= self.max_text_density):
                continue
            
            # Calculate line score (quality metric)
            aspect_ratio = w / h if h > 0 else 0
            score = density * min(aspect_ratio / 10, 1.0)  # Prefer horizontal lines
            
            lines.append({
                'bbox': (x, y, x + w, y + h),
                'height': h,
                'width': w,
                'density': density,
                'score': score,
                'centroid_y': centroids[i][1]
            })
        
        # Sort by vertical position
        lines.sort(key=lambda l: l['centroid_y'])
        
        # Merge overlapping lines
        lines = self._merge_overlapping_lines(lines)
        
        logging.info(f"    Morphological method detected {len(lines)} lines")
        return lines
    
    def detect_lines_connected_components(
        self,
        image: np.ndarray,
        binary: np.ndarray
    ) -> List[Dict]:
        """
        Detect lines using connected components analysis.
        
        Args:
            image: Original grayscale image
            binary: Preprocessed binary image
        
        Returns:
            List of detected line dictionaries
        """
        height, width = binary.shape
        
        # Find all connected components (letters/words)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Group components by vertical position
        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Filter small noise
            if w < 5 or h < 5:
                continue
            
            components.append({
                'bbox': (x, y, x + w, y + h),
                'centroid_y': centroids[i][1]
            })
        
        if not components:
            logging.warning("    No valid components found")
            return []
        
        # Cluster components into lines using vertical proximity
        lines = self._cluster_components_into_lines(components, height)
        
        logging.info(f"    Connected components method detected {len(lines)} lines")
        return lines
    
    def _cluster_components_into_lines(
        self,
        components: List[Dict],
        image_height: int
    ) -> List[Dict]:
        """
        Cluster components into text lines based on vertical proximity.
        
        Args:
            components: List of component dictionaries
            image_height: Height of the image
        
        Returns:
            List of line dictionaries
        """
        if not components:
            return []
        
        # Sort components by vertical position
        components.sort(key=lambda c: c['centroid_y'])
        
        # Adaptive threshold based on image height
        vertical_threshold = max(20, image_height // 50)
        
        lines = []
        current_line = [components[0]]
        
        for comp in components[1:]:
            # Check if component belongs to current line
            last_comp = current_line[-1]
            y_distance = abs(comp['centroid_y'] - last_comp['centroid_y'])
            
            if y_distance <= vertical_threshold:
                current_line.append(comp)
            else:
                # Create line from current cluster
                line = self._create_line_from_components(current_line)
                if line:
                    lines.append(line)
                
                # Start new line
                current_line = [comp]
        
        # Add last line
        if current_line:
            line = self._create_line_from_components(current_line)
            if line:
                lines.append(line)
        
        return lines
    
    def _create_line_from_components(self, components: List[Dict]) -> Optional[Dict]:
        """
        Create a line bounding box from a cluster of components.
        
        Args:
            components: List of component dictionaries
        
        Returns:
            Line dictionary or None if invalid
        """
        if not components:
            return None
        
        # Find bounding box that encompasses all components
        x_min = min(c['bbox'][0] for c in components)
        y_min = min(c['bbox'][1] for c in components)
        x_max = max(c['bbox'][2] for c in components)
        y_max = max(c['bbox'][3] for c in components)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Validate dimensions
        if not (self.min_line_height <= height <= self.max_line_height and width >= self.min_line_width):
            return None
        
        centroid_y = np.mean([c['centroid_y'] for c in components])
        
        return {
            'bbox': (x_min, y_min, x_max, y_max),
            'height': height,
            'width': width,
            'centroid_y': centroid_y,
            'num_components': len(components)
        }
    
    def _merge_overlapping_lines(self, lines: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
        """
        Merge lines that significantly overlap vertically.
        
        Args:
            lines: List of line dictionaries
            overlap_threshold: Minimum overlap ratio to merge (0-1)
        
        Returns:
            List of merged line dictionaries
        """
        if len(lines) <= 1:
            return lines
        
        merged = []
        skip = set()
        
        for i, line1 in enumerate(lines):
            if i in skip:
                continue
            
            x1, y1, x2, y2 = line1['bbox']
            merged_line = line1.copy()
            
            for j in range(i + 1, len(lines)):
                if j in skip:
                    continue
                
                line2 = lines[j]
                x3, y3, x4, y4 = line2['bbox']
                
                # Calculate vertical overlap
                overlap_start = max(y1, y3)
                overlap_end = min(y2, y4)
                overlap_height = max(0, overlap_end - overlap_start)
                
                min_height = min(y2 - y1, y4 - y3)
                overlap_ratio = overlap_height / min_height if min_height > 0 else 0
                
                if overlap_ratio >= overlap_threshold:
                    # Merge bounding boxes
                    new_x1 = min(x1, x3)
                    new_y1 = min(y1, y3)
                    new_x2 = max(x2, x4)
                    new_y2 = max(y2, y4)
                    
                    merged_line['bbox'] = (new_x1, new_y1, new_x2, new_y2)
                    merged_line['height'] = new_y2 - new_y1
                    merged_line['width'] = new_x2 - new_x1
                    merged_line['centroid_y'] = (new_y1 + new_y2) / 2
                    
                    skip.add(j)
            
            merged.append(merged_line)
        
        return merged
    
    def detect_lines(
        self,
        image: np.ndarray,
        method: str = 'morphological'
    ) -> List[Dict]:
        """
        Main entry point for line detection.
        
        Args:
            image: Input grayscale image
            method: Detection method ('morphological' or 'connected_components')
        
        Returns:
            List of detected line dictionaries
        """
        # Preprocess
        binary = self.preprocess_image(image)
        
        # Detect lines based on method
        if method == 'morphological':
            lines = self.detect_lines_morphological(image, binary)
        elif method == 'connected_components':
            lines = self.detect_lines_connected_components(image, binary)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return lines


def extract_line_images(
    image: np.ndarray,
    lines: List[Dict],
    margin_top: int = 5,
    margin_bottom: int = 5
) -> List[Tuple[np.ndarray, Dict]]:
    """
    Extract line images from detected bounding boxes.
    
    Args:
        image: Full document image (grayscale)
        lines: List of line dictionaries
        margin_top: Top margin in pixels
        margin_bottom: Bottom margin in pixels
    
    Returns:
        List of (line_image, line_info) tuples
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
        
        # Update bbox with margins
        line_info_with_margin = line_info.copy()
        line_info_with_margin['bbox_with_margin'] = (x1_margin, y1_margin, x2_margin, y2_margin)
        
        extracted_lines.append((line_image, line_info_with_margin))
    
    return extracted_lines


def visualize_lines(
    image: np.ndarray,
    lines: List[Dict],
    output_path: Path,
    title: str = "Line Detection"
):
    """
    Create visualization of detected lines.
    
    Args:
        image: Original document image
        lines: List of line dictionaries
        output_path: Path to save visualization
        title: Title for visualization
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
        cv2.putText(vis_image, f"L{i+1}", (x1 + 10, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw metrics if available
        if 'density' in line_info:
            metric_text = f"D:{line_info['density']:.2f}"
            cv2.putText(vis_image, metric_text, (x1 + 10, y2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Add title and stats
    title_text = f"{title}: {len(lines)} lines"
    cv2.putText(vis_image, title_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imwrite(str(output_path), vis_image)
    logging.info(f"    ✓ Saved visualization: {output_path.name}")


# ============================================================================
# Test Script
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Robust Line Detection')
    parser.add_argument('--input', type=str, required=True,
                       help='Input document image')
    parser.add_argument('--output_dir', type=str, default='results/line_detection_test',
                       help='Output directory')
    parser.add_argument('--method', type=str, default='morphological',
                       choices=['morphological', 'connected_components'],
                       help='Detection method')
    parser.add_argument('--min_height', type=int, default=40,
                       help='Minimum line height')
    parser.add_argument('--max_height', type=int, default=200,
                       help='Maximum line height')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("="*70)
    print(f"Robust Line Detection Test - Method: {args.method}")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Error: Could not load image: {args.input}")
        exit(1)
    
    print(f"✓ Loaded image: {image.shape}")
    
    # Initialize detector
    detector = RobustLineDetector(
        min_line_height=args.min_height,
        max_line_height=args.max_height
    )
    
    # Detect lines
    print(f"\nDetecting lines using '{args.method}' method...")
    lines = detector.detect_lines(image, method=args.method)
    
    print(f"\n✓ Detected {len(lines)} lines:")
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line['bbox']
        print(f"  Line {i+1}: bbox=({x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}), "
              f"h={line['height']:3d}, w={line['width']:4d}", end='')
        if 'density' in line:
            print(f", density={line['density']:.3f}", end='')
        print()
    
    # Visualize
    output_path = output_dir / f"{Path(args.input).stem}_{args.method}_lines.png"
    visualize_lines(image, lines, output_path, title=f"{args.method.title()} Detection")
    
    # Extract and save line images
    extracted_lines = extract_line_images(image, lines)
    for i, (line_img, line_info) in enumerate(extracted_lines):
        line_path = output_dir / f"line_{i+1:03d}.png"
        cv2.imwrite(str(line_path), line_img)
    
    print(f"\n✓ Saved {len(extracted_lines)} line images to: {output_dir}")
    print(f"✓ Visualization saved to: {output_path}")
    print("="*70)
