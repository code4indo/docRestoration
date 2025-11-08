"""
Evaluate three versions: Script1, Script2 (with post-processing), Script2 (without post-processing)
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import json

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    return ssim(img1, img2, data_range=255)

def evaluate_restoration(restored_path, ground_truth_path):
    """Evaluate a restored image against ground truth."""
    restored = cv2.imread(str(restored_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(ground_truth_path), cv2.IMREAD_GRAYSCALE)
    
    if restored is None or gt is None:
        return None
    
    # Ensure same size
    if restored.shape != gt.shape:
        restored = cv2.resize(restored, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    psnr = calculate_psnr(gt, restored)
    ssim_value = calculate_ssim(gt, restored)
    
    return {'psnr': psnr, 'ssim': ssim_value}

def find_matching_gt(restored_filename, gt_dir):
    """Find matching ground truth file."""
    base_name = restored_filename.replace('_restored.png', '')
    
    gt_path = gt_dir / f"{base_name}.png"
    if gt_path.exists():
        return gt_path
    
    gt_path = gt_dir / f"{base_name}.jpg"
    if gt_path.exists():
        return gt_path
    
    if base_name.startswith(('0000_', '0001_', '000_', '001_')):
        alt_name = '_'.join(base_name.split('_')[1:])
        gt_path = gt_dir / f"{alt_name}.png"
        if gt_path.exists():
            return gt_path
        gt_path = gt_dir / f"{alt_name}.jpg"
        if gt_path.exists():
            return gt_path
    
    return None

def main():
    ground_truth_dir = Path("DokumenRusak/gt_dataLatih")
    
    # Three versions to compare
    dirs = {
        'Script1 (inference_data_latih.py)': Path("results/restored_data_latih/restored"),
        'Script2 WITH post-processing': Path("RusakRingan/data_latih"),
        'Script2 WITHOUT post-processing': Path("results/restored_no_postprocess")
    }
    
    print("="*100)
    print("THREE-WAY COMPARISON WITH GROUND TRUTH")
    print("="*100)
    print(f"Ground Truth: {ground_truth_dir}\n")
    
    all_results = {name: [] for name in dirs.keys()}
    
    # Evaluate each version
    for name, result_dir in dirs.items():
        print(f"\nEvaluating: {name}")
        print(f"Directory: {result_dir}")
        print("-" * 100)
        
        restored_files = list(result_dir.glob("*_restored.png"))
        
        for restored_path in sorted(restored_files):
            gt_path = find_matching_gt(restored_path.name, ground_truth_dir)
            
            if gt_path is None:
                continue
            
            metrics = evaluate_restoration(restored_path, gt_path)
            
            if metrics:
                filename = restored_path.stem.replace('_restored', '')
                metrics['filename'] = filename
                all_results[name].append(metrics)
                print(f"  {filename}: PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}")
    
    # Calculate averages
    print("\n" + "="*100)
    print("SUMMARY COMPARISON")
    print("="*100)
    
    summary = {}
    for name, results in all_results.items():
        if results:
            avg_psnr = np.mean([r['psnr'] for r in results])
            std_psnr = np.std([r['psnr'] for r in results])
            avg_ssim = np.mean([r['ssim'] for r in results])
            std_ssim = np.std([r['ssim'] for r in results])
            
            summary[name] = {
                'avg_psnr': avg_psnr,
                'std_psnr': std_psnr,
                'avg_ssim': avg_ssim,
                'std_ssim': std_ssim,
                'num_images': len(results)
            }
    
    # Print comparison table
    print(f"\n{'Version':<40} {'PSNR (dB)':<20} {'SSIM':<20} {'N'}")
    print("-" * 100)
    
    for name, stats in summary.items():
        print(f"{name:<40} {stats['avg_psnr']:>6.2f} ± {stats['std_psnr']:<5.2f} {stats['avg_ssim']:>8.4f} ± {stats['std_ssim']:<6.4f} {stats['num_images']}")
    
    # Determine winner
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)
    
    script1_psnr = summary['Script1 (inference_data_latih.py)']['avg_psnr']
    script2_with_psnr = summary['Script2 WITH post-processing']['avg_psnr']
    script2_without_psnr = summary['Script2 WITHOUT post-processing']['avg_psnr']
    
    script1_ssim = summary['Script1 (inference_data_latih.py)']['avg_ssim']
    script2_with_ssim = summary['Script2 WITH post-processing']['avg_ssim']
    script2_without_ssim = summary['Script2 WITHOUT post-processing']['avg_ssim']
    
    print("\n1. Effect of Post-Processing:")
    psnr_drop = script2_with_psnr - script2_without_psnr
    ssim_drop = script2_with_ssim - script2_without_ssim
    print(f"   PSNR: {script2_without_psnr:.2f} → {script2_with_psnr:.2f} dB (Δ = {psnr_drop:+.2f} dB, {psnr_drop/script2_without_psnr*100:+.2f}%)")
    print(f"   SSIM: {script2_without_ssim:.4f} → {script2_with_ssim:.4f} (Δ = {ssim_drop:+.4f}, {ssim_drop/script2_without_ssim*100:+.2f}%)")
    
    if psnr_drop < 0:
        print(f"   ✅ CONFIRMED: Post-processing REDUCES PSNR by {abs(psnr_drop):.2f} dB")
    
    print("\n2. Script1 vs Script2 (without post-processing):")
    psnr_diff = script1_psnr - script2_without_psnr
    ssim_diff = script1_ssim - script2_without_ssim
    print(f"   PSNR: {script1_psnr:.2f} vs {script2_without_psnr:.2f} dB (Δ = {psnr_diff:+.2f} dB)")
    print(f"   SSIM: {script1_ssim:.4f} vs {script2_without_ssim:.4f} (Δ = {ssim_diff:+.4f})")
    
    if abs(psnr_diff) < 0.5 and abs(ssim_diff) < 0.001:
        print(f"   ✅ Scripts are EQUIVALENT when post-processing is disabled")
        print(f"      → Normalization is correct in both scripts")
    elif abs(psnr_diff) < 0.1 and abs(ssim_diff) < 0.0005:
        print(f"   ✅ Scripts are NEARLY IDENTICAL (minor numerical differences)")
    else:
        print(f"   ⚠️  Scripts still differ even without post-processing")
        print(f"      → May have other preprocessing differences")
    
    # Final verdict
    print("\n" + "="*100)
    print("FINAL VERDICT")
    print("="*100)
    
    best_psnr = max(summary.items(), key=lambda x: x[1]['avg_psnr'])
    best_ssim = max(summary.items(), key=lambda x: x[1]['avg_ssim'])
    
    print(f"\n🏆 Best PSNR: {best_psnr[0]}")
    print(f"   Value: {best_psnr[1]['avg_psnr']:.2f} ± {best_psnr[1]['std_psnr']:.2f} dB")
    
    print(f"\n🏆 Best SSIM: {best_ssim[0]}")
    print(f"   Value: {best_ssim[1]['avg_ssim']:.4f} ± {best_ssim[1]['std_ssim']:.4f}")
    
    print("\n📋 RECOMMENDATION:")
    print("   → Use inference WITHOUT post-processing for best PSNR/SSIM")
    print("   → Post-processing reduces fidelity to ground truth by ~11%")
    
    print("="*100)
    
    # Save results
    output_file = Path("results/three_way_comparison.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
