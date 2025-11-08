"""
Compare outputs from two different inference scripts to check if they produce identical results.
"""

import cv2
import numpy as np
from pathlib import Path

def compare_images(img1_path, img2_path):
    """Compare two images pixel by pixel."""
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return None, None, None, "Failed to load images"
    
    if img1.shape != img2.shape:
        return None, None, None, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    
    # Calculate differences
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_different = np.sum(diff > 0)
    percent_different = (num_different / diff.size) * 100
    
    # Calculate PSNR between the two restored versions
    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate statistics
    stats = {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'num_different_pixels': num_different,
        'percent_different': percent_different,
        'psnr_between_restored': psnr,
        'img1_mean': np.mean(img1),
        'img2_mean': np.mean(img2),
        'identical': max_diff < 1e-5
    }
    
    return stats, img1, img2, None

def main():
    # Paths
    results_dir1 = Path("results/restored_data_latih/restored")  # inference_data_latih.py
    results_dir2 = Path("RusakRingan/data_latih")  # inference_portrait_overlap_experiment.py
    
    print("="*80)
    print("Comparing Inference Script Outputs")
    print("="*80)
    print(f"Script 1 (inference_data_latih.py): {results_dir1}")
    print(f"Script 2 (inference_portrait_overlap_experiment.py): {results_dir2}")
    print()
    
    # Find common files
    files1 = {f.stem.replace('_restored', ''): f for f in results_dir1.glob("*_restored.png")}
    files2 = {f.stem.replace('_restored', ''): f for f in results_dir2.glob("*_restored.png")}
    
    common_files = set(files1.keys()) & set(files2.keys())
    
    if not common_files:
        print("❌ No common files found!")
        print(f"Files in dir1: {list(files1.keys())}")
        print(f"Files in dir2: {list(files2.keys())}")
        return
    
    print(f"Found {len(common_files)} common files to compare\n")
    
    all_stats = []
    
    for filename in sorted(common_files):
        img1_path = files1[filename]
        img2_path = files2[filename]
        
        print(f"Comparing: {filename}")
        print(f"  File 1: {img1_path.name}")
        print(f"  File 2: {img2_path.name}")
        
        stats, img1, img2, error = compare_images(img1_path, img2_path)
        
        if error:
            print(f"  ❌ Error: {error}\n")
            continue
        
        all_stats.append(stats)
        
        print(f"  Shape: {img1.shape}")
        print(f"  Identical: {'✅ YES' if stats['identical'] else '❌ NO'}")
        print(f"  Max difference: {stats['max_diff']:.2f}")
        print(f"  Mean difference: {stats['mean_diff']:.4f}")
        print(f"  Different pixels: {stats['num_different_pixels']:,} ({stats['percent_different']:.2f}%)")
        print(f"  PSNR between restored: {stats['psnr_between_restored']:.2f} dB")
        print(f"  Mean intensity:")
        print(f"    Script 1: {stats['img1_mean']:.2f}")
        print(f"    Script 2: {stats['img2_mean']:.2f}")
        print()
    
    # Summary
    if all_stats:
        print("="*80)
        print("SUMMARY")
        print("="*80)
        
        num_identical = sum(1 for s in all_stats if s['identical'])
        avg_max_diff = np.mean([s['max_diff'] for s in all_stats])
        avg_mean_diff = np.mean([s['mean_diff'] for s in all_stats])
        avg_psnr = np.mean([s['psnr_between_restored'] for s in all_stats if np.isfinite(s['psnr_between_restored'])])
        avg_percent_diff = np.mean([s['percent_different'] for s in all_stats])
        
        print(f"Total files compared: {len(all_stats)}")
        print(f"Identical outputs: {num_identical}/{len(all_stats)}")
        print(f"Average max difference: {avg_max_diff:.2f}")
        print(f"Average mean difference: {avg_mean_diff:.4f}")
        print(f"Average PSNR between restored: {avg_psnr:.2f} dB")
        print(f"Average percent different: {avg_percent_diff:.2f}%")
        print()
        
        if num_identical == len(all_stats):
            print("✅ HASIL IDENTIK: Kedua script menghasilkan output yang sama persis!")
        elif avg_psnr > 40:
            print("✅ HASIL SANGAT MIRIP: Perbedaan minimal (PSNR > 40 dB)")
        elif avg_psnr > 30:
            print("⚠️ HASIL MIRIP: Ada perbedaan kecil (PSNR 30-40 dB)")
        else:
            print("❌ HASIL BERBEDA: Ada perbedaan signifikan (PSNR < 30 dB)")

if __name__ == "__main__":
    main()
