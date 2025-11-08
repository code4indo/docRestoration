"""
Detailed visual quality comparison between two inference scripts.
Compare sharpness, contrast, and detail preservation.
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage

def calculate_sharpness(image):
    """Calculate image sharpness using Laplacian variance."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

def calculate_contrast(image):
    """Calculate RMS contrast."""
    return image.std()

def calculate_edge_strength(image):
    """Calculate average edge strength using Sobel."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return magnitude.mean()

def calculate_detail_preservation(image):
    """Measure high-frequency content (detail)."""
    # High-pass filter to extract details
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    high_freq = cv2.absdiff(image, blurred)
    return high_freq.mean(), high_freq.std()

def calculate_stroke_continuity(image):
    """Measure stroke continuity using morphological analysis."""
    # Binarize (assuming text is dark on light background after restoration)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (text should be white on black)
    if binary.mean() > 127:
        binary = 255 - binary
    
    # Calculate connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter out background and very small noise
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
    valid_components = areas[areas > 10]  # Ignore tiny noise
    
    return {
        'num_strokes': len(valid_components),
        'avg_stroke_area': valid_components.mean() if len(valid_components) > 0 else 0,
        'stroke_area_std': valid_components.std() if len(valid_components) > 0 else 0
    }

def analyze_image_quality(image_path):
    """Comprehensive quality analysis."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    metrics = {
        'mean_intensity': img.mean(),
        'std_intensity': img.std(),
        'sharpness': calculate_sharpness(img),
        'contrast': calculate_contrast(img),
        'edge_strength': calculate_edge_strength(img),
        'min_value': img.min(),
        'max_value': img.max(),
        'dynamic_range': img.max() - img.min(),
    }
    
    # Detail preservation
    detail_mean, detail_std = calculate_detail_preservation(img)
    metrics['detail_mean'] = detail_mean
    metrics['detail_std'] = detail_std
    
    # Stroke continuity
    stroke_metrics = calculate_stroke_continuity(img)
    metrics.update(stroke_metrics)
    
    return metrics

def main():
    # Paths
    results_dir1 = Path("results/restored_data_latih/restored")  # inference_data_latih.py (NO post-process)
    results_dir2 = Path("RusakRingan/data_latih")  # inference_portrait_overlap_experiment.py (WITH post-process)
    
    print("="*80)
    print("VISUAL QUALITY COMPARISON")
    print("="*80)
    print(f"Script 1 (NO post-processing): {results_dir1}")
    print(f"Script 2 (WITH post-processing): {results_dir2}")
    print()
    
    # Find common files
    files1 = {f.stem.replace('_restored', ''): f for f in results_dir1.glob("*_restored.png")}
    files2 = {f.stem.replace('_restored', ''): f for f in results_dir2.glob("*_restored.png")}
    
    common_files = set(files1.keys()) & set(files2.keys())
    
    if not common_files:
        print("❌ No common files found!")
        return
    
    print(f"Analyzing {len(common_files)} image pairs\n")
    
    all_results = []
    
    for filename in sorted(common_files):
        img1_path = files1[filename]
        img2_path = files2[filename]
        
        print(f"\n{'='*80}")
        print(f"Image: {filename}")
        print(f"{'='*80}")
        
        metrics1 = analyze_image_quality(img1_path)
        metrics2 = analyze_image_quality(img2_path)
        
        if metrics1 is None or metrics2 is None:
            print("❌ Failed to load images")
            continue
        
        # Compare metrics
        comparison = {}
        for key in metrics1.keys():
            val1 = metrics1[key]
            val2 = metrics2[key]
            diff = val2 - val1
            percent_change = (diff / val1 * 100) if val1 != 0 else 0
            
            comparison[key] = {
                'script1': val1,
                'script2': val2,
                'diff': diff,
                'percent_change': percent_change
            }
        
        all_results.append({
            'filename': filename,
            'metrics1': metrics1,
            'metrics2': metrics2,
            'comparison': comparison
        })
        
        # Print detailed comparison
        print(f"\n{'Metric':<25} {'Script1':<15} {'Script2':<15} {'Change':<15} {'Winner'}")
        print("-" * 80)
        
        # Key metrics for document restoration
        key_metrics = [
            ('sharpness', 'higher_better'),
            ('contrast', 'higher_better'),
            ('edge_strength', 'higher_better'),
            ('detail_mean', 'higher_better'),
            ('detail_std', 'higher_better'),
            ('num_strokes', 'lower_better'),  # Fewer broken strokes = better continuity
            ('dynamic_range', 'higher_better'),
        ]
        
        for metric, direction in key_metrics:
            comp = comparison[metric]
            val1 = comp['script1']
            val2 = comp['script2']
            change = comp['percent_change']
            
            if direction == 'higher_better':
                winner = "Script2 ✓" if val2 > val1 else "Script1 ✓" if val1 > val2 else "Tie"
            else:  # lower_better
                winner = "Script1 ✓" if val1 < val2 else "Script2 ✓" if val2 < val1 else "Tie"
            
            print(f"{metric:<25} {val1:<15.2f} {val2:<15.2f} {change:+.2f}% {winner}")
    
    # Overall summary
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    if not all_results:
        print("No results to summarize")
        return
    
    # Count wins per script
    wins_script1 = 0
    wins_script2 = 0
    
    key_metric_names = ['sharpness', 'contrast', 'edge_strength', 'detail_mean', 'detail_std', 'num_strokes', 'dynamic_range']
    key_metric_directions = ['higher_better', 'higher_better', 'higher_better', 'higher_better', 'higher_better', 'lower_better', 'higher_better']
    
    for result in all_results:
        for metric, direction in zip(key_metric_names, key_metric_directions):
            comp = result['comparison'][metric]
            val1 = comp['script1']
            val2 = comp['script2']
            
            if direction == 'higher_better':
                if val2 > val1:
                    wins_script2 += 1
                elif val1 > val2:
                    wins_script1 += 1
            else:  # lower_better
                if val1 < val2:
                    wins_script1 += 1
                elif val2 < val1:
                    wins_script2 += 1
    
    total_comparisons = len(all_results) * len(key_metric_names)
    
    print(f"Total comparisons: {total_comparisons}")
    print(f"Script 1 (NO post-processing) wins: {wins_script1} ({wins_script1/total_comparisons*100:.1f}%)")
    print(f"Script 2 (WITH post-processing) wins: {wins_script2} ({wins_script2/total_comparisons*100:.1f}%)")
    print(f"Ties: {total_comparisons - wins_script1 - wins_script2}")
    
    print("\n" + "="*80)
    if wins_script1 > wins_script2:
        print("🏆 WINNER: Script 1 (inference_data_latih.py - NO post-processing)")
        print("   → Model mentah tanpa post-processing menghasilkan kualitas lebih baik")
    elif wins_script2 > wins_script1:
        print("🏆 WINNER: Script 2 (inference_portrait_overlap_experiment.py - WITH post-processing)")
        print("   → Post-processing (closing + unsharp mask) meningkatkan kualitas")
    else:
        print("🤝 TIE: Kedua script menghasilkan kualitas yang setara")
    print("="*80)
    
    # Detailed metric averages
    print(f"\n{'='*80}")
    print("AVERAGE METRICS ACROSS ALL IMAGES")
    print(f"{'='*80}\n")
    
    avg_metrics1 = {}
    avg_metrics2 = {}
    
    for key in all_results[0]['metrics1'].keys():
        vals1 = [r['metrics1'][key] for r in all_results]
        vals2 = [r['metrics2'][key] for r in all_results]
        avg_metrics1[key] = np.mean(vals1)
        avg_metrics2[key] = np.mean(vals2)
    
    print(f"{'Metric':<25} {'Script1 (Avg)':<20} {'Script2 (Avg)':<20}")
    print("-" * 80)
    for key in key_metric_names:
        val1 = avg_metrics1[key]
        val2 = avg_metrics2[key]
        print(f"{key:<25} {val1:<20.2f} {val2:<20.2f}")

if __name__ == "__main__":
    main()
