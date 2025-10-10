#!/usr/bin/env python3
"""
Add Visual Reference Grid to Document

Untuk debugging/validasi alignment:
- Menambahkan garis horizontal di setiap baseline
- Menambahkan garis vertikal sebagai reference
- Warna berbeda: original (merah), restored (hijau)

Usage:
    poetry run python dual_modal_gan/scripts/add_reference_grid.py \
        --input image.png \
        --baselines baselines.xml \
        --output image_with_grid.png
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple


def parse_pagexml_baselines_simple(pagexml_path: str) -> List[List[Tuple[int, int]]]:
    """Parse PageXML untuk extract baseline coordinates."""
    tree = ET.parse(pagexml_path)
    root = tree.getroot()
    
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    
    baselines = []
    for textline in root.findall('.//pc:TextLine', ns):
        baseline_elem = textline.find('pc:Baseline', ns)
        if baseline_elem is None:
            continue
        
        baseline_points_str = baseline_elem.get('points')
        if not baseline_points_str:
            continue
        
        baseline_points = []
        for point_str in baseline_points_str.split():
            x, y = map(int, point_str.split(','))
            baseline_points.append((x, y))
        
        baselines.append(baseline_points)
    
    return baselines


def add_baseline_grid(image: np.ndarray, baselines: List[List[Tuple[int, int]]], 
                     color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Add horizontal baseline grid lines pada image.
    
    Args:
        image: Input image
        baselines: List of baseline polylines
        color: Line color (B, G, R)
        thickness: Line thickness
    
    Returns:
        Image dengan grid lines
    """
    img_with_grid = image.copy()
    h, w = img_with_grid.shape[:2]
    
    # Draw each baseline sebagai polyline
    for baseline_points in baselines:
        if len(baseline_points) < 2:
            continue
        
        # Draw polyline
        pts = np.array(baseline_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_with_grid, [pts], isClosed=False, 
                     color=color, thickness=thickness)
        
        # Calculate average y position untuk horizontal reference line
        avg_y = int(np.mean([p[1] for p in baseline_points]))
        
        # Draw horizontal reference line (straight)
        cv2.line(img_with_grid, (0, avg_y), (w, avg_y), 
                color=color, thickness=1, lineType=cv2.LINE_AA)
        
        # Add tick marks at start and end
        start_x = baseline_points[0][0]
        end_x = baseline_points[-1][0]
        
        # Start tick (vertical)
        cv2.line(img_with_grid, (start_x, avg_y - 20), (start_x, avg_y + 20),
                color=color, thickness=2)
        
        # End tick (vertical)
        cv2.line(img_with_grid, (end_x, avg_y - 20), (end_x, avg_y + 20),
                color=color, thickness=2)
    
    return img_with_grid


def add_vertical_grid(image: np.ndarray, spacing: int = 200, 
                     color=(128, 128, 128), thickness=1) -> np.ndarray:
    """
    Add vertical reference grid lines.
    
    Args:
        image: Input image
        spacing: Spacing between vertical lines (pixels)
        color: Line color
        thickness: Line thickness
    
    Returns:
        Image dengan vertical grid
    """
    img_with_grid = image.copy()
    h, w = img_with_grid.shape[:2]
    
    # Draw vertical lines
    for x in range(0, w, spacing):
        cv2.line(img_with_grid, (x, 0), (x, h), 
                color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
        # Add coordinate label at top
        label = f"{x}"
        cv2.putText(img_with_grid, label, (x + 5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_with_grid


def add_horizontal_grid(image: np.ndarray, spacing: int = 200,
                       color=(128, 128, 128), thickness=1) -> np.ndarray:
    """
    Add horizontal reference grid lines (untuk validasi spacing).
    
    Args:
        image: Input image
        spacing: Spacing between horizontal lines
        color: Line color
        thickness: Line thickness
    
    Returns:
        Image dengan horizontal grid
    """
    img_with_grid = image.copy()
    h, w = img_with_grid.shape[:2]
    
    # Draw horizontal lines
    for y in range(0, h, spacing):
        cv2.line(img_with_grid, (0, y), (w, y),
                color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
        # Add coordinate label at left
        label = f"{y}"
        cv2.putText(img_with_grid, label, (5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_with_grid


def add_legend(image: np.ndarray) -> np.ndarray:
    """Add legend explaining grid colors."""
    img_with_legend = image.copy()
    h, w = img_with_legend.shape[:2]
    
    # Legend background (semi-transparent white box)
    legend_h = 120
    legend_w = 300
    legend_x = w - legend_w - 20
    legend_y = 20
    
    # Draw white background
    cv2.rectangle(img_with_legend, 
                 (legend_x, legend_y), 
                 (legend_x + legend_w, legend_y + legend_h),
                 (255, 255, 255), -1)
    
    # Draw border
    cv2.rectangle(img_with_legend,
                 (legend_x, legend_y),
                 (legend_x + legend_w, legend_y + legend_h),
                 (0, 0, 0), 2)
    
    # Add title
    cv2.putText(img_with_legend, "REFERENCE GRID",
               (legend_x + 10, legend_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add legend items
    y_offset = legend_y + 50
    
    # Green: Baseline
    cv2.line(img_with_legend, 
            (legend_x + 10, y_offset), 
            (legend_x + 40, y_offset),
            (0, 255, 0), 3)
    cv2.putText(img_with_legend, "Baseline (actual)",
               (legend_x + 50, y_offset + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Green thin: Horizontal reference
    y_offset += 25
    cv2.line(img_with_legend,
            (legend_x + 10, y_offset),
            (legend_x + 40, y_offset),
            (0, 255, 0), 1)
    cv2.putText(img_with_legend, "Horizontal ref (straight)",
               (legend_x + 50, y_offset + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Gray: Grid
    y_offset += 25
    cv2.line(img_with_legend,
            (legend_x + 10, y_offset),
            (legend_x + 40, y_offset),
            (128, 128, 128), 1)
    cv2.putText(img_with_legend, "Coordinate grid",
               (legend_x + 50, y_offset + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img_with_legend


def main():
    parser = argparse.ArgumentParser(
        description="Add visual reference grid untuk validasi alignment"
    )
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--pagexml', required=True, help='PageXML file dengan baselines')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--grid-spacing', type=int, default=200, 
                       help='Spacing untuk coordinate grid (default: 200)')
    parser.add_argument('--baseline-color', default='green',
                       choices=['green', 'red', 'blue', 'yellow'],
                       help='Baseline color')
    parser.add_argument('--no-coord-grid', action='store_true',
                       help='Disable coordinate grid (only baselines)')
    
    args = parser.parse_args()
    
    # Color mapping
    color_map = {
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255)
    }
    baseline_color = color_map[args.baseline_color]
    
    print("=" * 80)
    print("🎨 ADDING REFERENCE GRID")
    print("=" * 80)
    print(f"Input:   {args.input}")
    print(f"PageXML: {args.pagexml}")
    print(f"Output:  {args.output}")
    print("")
    
    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ Cannot load image: {args.input}")
        return
    
    print(f"📐 Image size: {image.shape[1]}×{image.shape[0]}")
    
    # Parse baselines
    print(f"📖 Parsing baselines from PageXML...")
    baselines = parse_pagexml_baselines_simple(args.pagexml)
    print(f"✅ Found {len(baselines)} baselines")
    
    # Add coordinate grid (optional)
    if not args.no_coord_grid:
        print(f"📊 Adding coordinate grid (spacing={args.grid_spacing}px)...")
        image = add_vertical_grid(image, spacing=args.grid_spacing)
        image = add_horizontal_grid(image, spacing=args.grid_spacing)
    
    # Add baseline grid
    print(f"📏 Adding baseline grid (color={args.baseline_color})...")
    image = add_baseline_grid(image, baselines, color=baseline_color, thickness=2)
    
    # Add legend
    print(f"🏷️  Adding legend...")
    image = add_legend(image)
    
    # Save
    cv2.imwrite(args.output, image)
    print(f"💾 Saved: {args.output}")
    print("")
    print("=" * 80)
    print("✅ REFERENCE GRID COMPLETE")
    print("=" * 80)
    print("")
    print("🔍 Visual inspection:")
    print(f"   - Green thick lines: Actual baseline polylines")
    print(f"   - Green thin lines:  Straight horizontal references")
    print(f"   - Green ticks:       Baseline start/end markers")
    print(f"   - Gray grid:         Coordinate reference system")
    print("")
    print("💡 Validation:")
    print(f"   - Check if baselines align dengan horizontal references")
    print(f"   - Measure deviation menggunakan coordinate grid")
    print(f"   - Compare before/after alignment enforcement")
    print("")


if __name__ == "__main__":
    main()
