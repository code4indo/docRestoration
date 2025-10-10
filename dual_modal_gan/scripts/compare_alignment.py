#!/usr/bin/env python3
"""
Compare Baseline Alignment: Before vs After Enforcement

Visualize baseline straightness dan measure quantitative improvement.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def draw_baselines_on_image(image, baseline_y_positions, color=(0, 255, 0), thickness=2):
    """Draw horizontal lines di setiap baseline position."""
    img_with_lines = image.copy()
    
    for y in baseline_y_positions:
        cv2.line(img_with_lines, (0, int(y)), (img_with_lines.shape[1], int(y)), 
                color, thickness)
    
    return img_with_lines


def measure_baseline_variance(image):
    """
    Measure variance dalam baseline positions (proxy untuk zigzag).
    
    Strategy:
    - Detect horizontal lines using Hough transform
    - Measure variance in y-positions
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Hough line detection (horizontal bias)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=200, maxLineGap=50)
    
    if lines is None:
        return None, None
    
    # Extract y-positions dari horizontal lines
    y_positions = []
    horizontal_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Check if line is approximately horizontal
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        if angle < np.pi / 36:  # Less than 5 degrees
            y_avg = (y1 + y2) / 2
            y_positions.append(y_avg)
            horizontal_lines.append((x1, y1, x2, y2))
    
    if len(y_positions) < 2:
        return None, None
    
    # Calculate variance
    variance = np.var(np.diff(y_positions))  # Variance of spacing
    
    return variance, y_positions


def create_comparison_plot(img_before, img_after, 
                          variance_before, variance_after,
                          output_path):
    """Create side-by-side comparison dengan metrics."""
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Before
    axes[0].imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Before Alignment\nBaseline Variance: {variance_before:.2f}px²', 
                     fontsize=14, fontweight='bold', color='red')
    axes[0].axis('off')
    
    # After
    axes[1].imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'After Alignment\nBaseline Variance: {variance_after:.2f}px²', 
                     fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # Overall assessment
    improvement = ((variance_before - variance_after) / variance_before) * 100
    
    if improvement > 50:
        assessment = f"✅ SIGNIFICANT IMPROVEMENT (-{improvement:.1f}%)"
        color = 'green'
    elif improvement > 20:
        assessment = f"✅ MODERATE IMPROVEMENT (-{improvement:.1f}%)"
        color = 'orange'
    elif improvement > 0:
        assessment = f"✅ SLIGHT IMPROVEMENT (-{improvement:.1f}%)"
        color = 'blue'
    else:
        assessment = f"⚠️ NO IMPROVEMENT ({improvement:.1f}%)"
        color = 'red'
    
    fig.text(0.5, 0.02, assessment,
            ha='center', fontsize=16, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved comparison: {output_path}")
    print(f"  📊 Improvement: {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline alignment")
    parser.add_argument('--before', required=True, help='Image before alignment')
    parser.add_argument('--after', required=True, help='Image after alignment')
    parser.add_argument('--output', default='alignment_comparison.png', 
                       help='Output comparison image')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 BASELINE ALIGNMENT COMPARISON")
    print("=" * 80)
    print(f"Before: {args.before}")
    print(f"After:  {args.after}")
    print("")
    
    # Load images
    img_before = cv2.imread(args.before)
    img_after = cv2.imread(args.after)
    
    if img_before is None or img_after is None:
        print("❌ Error loading images!")
        return
    
    # Measure baseline variance
    print("📏 Measuring baseline variance...")
    variance_before, y_pos_before = measure_baseline_variance(img_before)
    variance_after, y_pos_after = measure_baseline_variance(img_after)
    
    if variance_before is None or variance_after is None:
        print("⚠️  Could not detect baselines automatically")
        print("    Manual visual inspection required")
        return
    
    print(f"  Before: {variance_before:.2f}px² ({len(y_pos_before)} lines detected)")
    print(f"  After:  {variance_after:.2f}px² ({len(y_pos_after)} lines detected)")
    print("")
    
    # Create comparison
    print("🎨 Creating comparison visualization...")
    create_comparison_plot(img_before, img_after, 
                          variance_before, variance_after,
                          args.output)
    
    print("")
    print("=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
