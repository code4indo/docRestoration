#!/usr/bin/env python3
"""
Compare Different Line Detection Methods on DIBCO Dataset
Tests: Robust Morphological, Watershed, and Projection Profile
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import json
import time
from typing import List, Tuple, Dict
import sys

# Import our detection methods
sys.path.append(str(Path(__file__).parent))
from line_detection_robust import RobustLineDetector
from line_detection_watershed import WatershedLineDetector


def detect_line_boundaries_projection(image: np.ndarray, 
                                     min_line_height: int = 30) -> List[Tuple[int, int, int, int]]:
    """
    Projection profile baseline method
    
    Args:
        image: Input grayscale image
        min_line_height: Minimum line height
        
    Returns:
        List of line bounding boxes (x1, y1, x2, y2)
    """
    height, width = image.shape[:2]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal projection
    h_proj = np.sum(binary, axis=1)
    
    # Smooth
    from scipy.ndimage import gaussian_filter1d
    h_proj_smooth = gaussian_filter1d(h_proj, sigma=2)
    
    # Find lines
    mean_val = np.mean(h_proj_smooth[h_proj_smooth > 0])
    threshold = mean_val * 0.3
    
    in_line = False
    line_start = 0
    lines = []
    
    for i, val in enumerate(h_proj_smooth):
        if not in_line and val > threshold:
            line_start = i
            in_line = True
        elif in_line and val <= threshold:
            if i - line_start >= min_line_height:
                lines.append((0, line_start, width, i))
            in_line = False
    
    if in_line:
        lines.append((0, line_start, width, height))
    
    return lines


def evaluate_on_image(image_path: str, methods: Dict) -> Dict:
    """
    Run all methods on a single image and collect results
    
    Args:
        image_path: Path to image
        methods: Dict of method_name -> detector_instance
        
    Returns:
        Dict with results for each method
    """
    print(f"\nProcessing: {Path(image_path).name}")
    print("-" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not load image")
        return {}
    
    results = {}
    
    for method_name, detector in methods.items():
        try:
            start_time = time.time()
            
            if method_name == "projection":
                lines = detect_line_boundaries_projection(image)
            elif method_name == "robust_morphological":
                lines_dict = detector.detect_lines(image, method='morphological')
                # Convert from Dict format to tuple format (x1, y1, x2, y2)
                lines = [l['bbox'] for l in lines_dict]
            elif method_name == "watershed":
                lines = detector.detect_lines_watershed(image)
            else:
                lines = []
            
            elapsed = time.time() - start_time
            
            results[method_name] = {
                'lines': lines,
                'num_lines': len(lines),
                'time_sec': elapsed,
                'success': len(lines) > 0
            }
            
            print(f"  {method_name:25s}: {len(lines):2d} lines in {elapsed:.3f}s")
            
        except Exception as e:
            print(f"  {method_name:25s}: FAILED - {e}")
            results[method_name] = {
                'lines': [],
                'num_lines': 0,
                'time_sec': 0,
                'success': False,
                'error': str(e)
            }
    
    return results


def visualize_comparison(image_path: str, 
                        results: Dict,
                        output_path: str):
    """
    Create side-by-side comparison visualization
    
    Args:
        image_path: Path to input image
        results: Results from evaluate_on_image
        output_path: Where to save visualization
    """
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Count successful methods
    methods = list(results.keys())
    n_methods = len(methods)
    
    # Create figure
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 10))
    if n_methods == 1:
        axes = [axes]
    
    colors = {
        'projection': 'red',
        'robust_morphological': 'orange',
        'watershed': 'lime'
    }
    
    for idx, method_name in enumerate(methods):
        ax = axes[idx]
        ax.imshow(img_rgb)
        
        result = results[method_name]
        lines = result.get('lines', [])
        color = colors.get(method_name, 'white')
        
        # Draw boxes
        for line_idx, (x1, y1, x2, y2) in enumerate(lines, 1):
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add line number
            ax.text(x1 + 5, y1 + 20, f'{line_idx}',
                   color=color, fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Title with stats
        title = f"{method_name.replace('_', ' ').title()}\n"
        title += f"{result['num_lines']} lines, {result['time_sec']:.3f}s"
        ax.set_title(title, fontsize=12, weight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_full_comparison(input_dir: str, 
                       output_dir: str,
                       max_images: int = None):
    """
    Run comparison on entire DIBCO dataset
    
    Args:
        input_dir: Directory with DIBCO images
        output_dir: Where to save results
        max_images: Optional limit on number of images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = sorted(list(input_path.glob("*.bmp")) + 
                        list(input_path.glob("*.png")) +
                        list(input_path.glob("*.jpg")))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n{'='*60}")
    print(f"Line Detection Method Comparison")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {len(image_files)}")
    print(f"{'='*60}")
    
    # Initialize detectors
    print("\nInitializing detectors...")
    methods = {
        'projection': None,  # Function-based, no instance needed
        'robust_morphological': RobustLineDetector(),
        'watershed': WatershedLineDetector(min_line_height=20, max_line_height=200)
    }
    print("Done!")
    
    # Process all images
    all_results = {}
    
    for img_path in image_files:
        img_name = img_path.stem
        
        # Run evaluation
        results = evaluate_on_image(str(img_path), methods)
        all_results[img_name] = results
        
        # Create visualization
        vis_output = output_path / f"{img_name}_comparison.png"
        visualize_comparison(str(img_path), results, str(vis_output))
        print(f"  Saved: {vis_output}")
    
    # Generate summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    
    summary = {}
    for method_name in methods.keys():
        num_lines_list = []
        time_list = []
        success_count = 0
        
        for img_results in all_results.values():
            if method_name in img_results:
                result = img_results[method_name]
                if result['success']:
                    num_lines_list.append(result['num_lines'])
                    time_list.append(result['time_sec'])
                    success_count += 1
        
        if num_lines_list:
            summary[method_name] = {
                'success_rate': f"{success_count}/{len(all_results)} ({100*success_count/len(all_results):.1f}%)",
                'avg_lines': np.mean(num_lines_list),
                'std_lines': np.std(num_lines_list),
                'avg_time': np.mean(time_list),
                'total_lines': sum(num_lines_list)
            }
        else:
            summary[method_name] = {
                'success_rate': "0/0 (0%)",
                'avg_lines': 0,
                'std_lines': 0,
                'avg_time': 0,
                'total_lines': 0
            }
    
    # Print summary table
    print(f"\n{'Method':<25} {'Success Rate':<15} {'Avg Lines':<12} {'Avg Time':<12} {'Total Lines'}")
    print("-" * 80)
    
    for method_name, stats in summary.items():
        print(f"{method_name:<25} {stats['success_rate']:<15} "
              f"{stats['avg_lines']:>5.1f} ± {stats['std_lines']:<4.1f} "
              f"{stats['avg_time']:>8.3f}s    {stats['total_lines']:>5d}")
    
    # Save summary to JSON
    summary_file = output_path / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': {k: {m: {key: val for key, val in v.items() if key != 'lines'} 
                                    for m, v in img_results.items()}
                               for k, img_results in all_results.items()}
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    print(f"\n{'='*60}")
    print("Comparison Complete!")
    print(f"{'='*60}\n")
    
    return summary, all_results


def main():
    parser = argparse.ArgumentParser(description='Compare Line Detection Methods')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with images')
    parser.add_argument('--output_dir', type=str, 
                       default='results/line_detection_comparison',
                       help='Output directory')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    run_full_comparison(args.input_dir, args.output_dir, args.max_images)


if __name__ == "__main__":
    main()
