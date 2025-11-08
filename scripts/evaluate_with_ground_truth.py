"""
Objective evaluation with ground truth: Compare PSNR and SSIM between two inference scripts.
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
        print(f"  ⚠️  Shape mismatch: restored {restored.shape} vs GT {gt.shape}")
        # Resize restored to match GT
        restored = cv2.resize(restored, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    psnr = calculate_psnr(gt, restored)
    ssim_value = calculate_ssim(gt, restored)
    
    return {
        'psnr': psnr,
        'ssim': ssim_value,
        'shape': restored.shape
    }

def find_matching_gt(restored_filename, gt_dir):
    """Find matching ground truth file for a restored image."""
    # Remove _restored suffix
    base_name = restored_filename.replace('_restored.png', '')
    
    # Try exact match first
    gt_path = gt_dir / f"{base_name}.png"
    if gt_path.exists():
        return gt_path
    
    # Try with .jpg extension
    gt_path = gt_dir / f"{base_name}.jpg"
    if gt_path.exists():
        return gt_path
    
    # Try without prefix numbers (0000_, 0001_, etc.)
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
    # Paths
    ground_truth_dir = Path("DokumenRusak/gt_dataLatih")
    results_dir1 = Path("results/restored_data_latih/restored")  # NO post-processing
    results_dir2 = Path("RusakRingan/data_latih")  # WITH post-processing
    
    print("="*80)
    print("OBJECTIVE EVALUATION WITH GROUND TRUTH")
    print("="*80)
    print(f"Ground Truth: {ground_truth_dir}")
    print(f"Script 1 (NO post-processing): {results_dir1}")
    print(f"Script 2 (WITH post-processing): {results_dir2}")
    print()
    
    if not ground_truth_dir.exists():
        print(f"❌ Ground truth directory not found: {ground_truth_dir}")
        return
    
    # Get all restored files from both scripts
    restored_files1 = list(results_dir1.glob("*_restored.png"))
    restored_files2 = list(results_dir2.glob("*_restored.png"))
    
    print(f"Found {len(restored_files1)} files in Script 1")
    print(f"Found {len(restored_files2)} files in Script 2")
    print()
    
    results_script1 = []
    results_script2 = []
    
    # Process Script 1 results
    print("Evaluating Script 1 (NO post-processing)...")
    for restored_path in sorted(restored_files1):
        gt_path = find_matching_gt(restored_path.name, ground_truth_dir)
        
        if gt_path is None:
            print(f"  ⚠️  No GT found for: {restored_path.name}")
            continue
        
        print(f"  Evaluating: {restored_path.name}")
        print(f"    GT: {gt_path.name}")
        
        metrics = evaluate_restoration(restored_path, gt_path)
        
        if metrics:
            metrics['filename'] = restored_path.stem.replace('_restored', '')
            metrics['gt_file'] = gt_path.name
            results_script1.append(metrics)
            print(f"    PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    
    print()
    
    # Process Script 2 results
    print("Evaluating Script 2 (WITH post-processing)...")
    for restored_path in sorted(restored_files2):
        gt_path = find_matching_gt(restored_path.name, ground_truth_dir)
        
        if gt_path is None:
            print(f"  ⚠️  No GT found for: {restored_path.name}")
            continue
        
        print(f"  Evaluating: {restored_path.name}")
        print(f"    GT: {gt_path.name}")
        
        metrics = evaluate_restoration(restored_path, gt_path)
        
        if metrics:
            metrics['filename'] = restored_path.stem.replace('_restored', '')
            metrics['gt_file'] = gt_path.name
            results_script2.append(metrics)
            print(f"    PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    
    print()
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    if not results_script1 or not results_script2:
        print("❌ No valid results to compare")
        return
    
    # Calculate averages
    avg_psnr1 = np.mean([r['psnr'] for r in results_script1])
    avg_ssim1 = np.mean([r['ssim'] for r in results_script1])
    std_psnr1 = np.std([r['psnr'] for r in results_script1])
    std_ssim1 = np.std([r['ssim'] for r in results_script1])
    
    avg_psnr2 = np.mean([r['psnr'] for r in results_script2])
    avg_ssim2 = np.mean([r['ssim'] for r in results_script2])
    std_psnr2 = np.std([r['psnr'] for r in results_script2])
    std_ssim2 = np.std([r['ssim'] for r in results_script2])
    
    print()
    print(f"{'Metric':<30} {'Script 1 (NO post)':<25} {'Script 2 (WITH post)':<25} {'Winner'}")
    print("-"*80)
    
    # PSNR comparison
    psnr_winner = "Script 1 ✓" if avg_psnr1 > avg_psnr2 else "Script 2 ✓" if avg_psnr2 > avg_psnr1 else "Tie"
    print(f"{'PSNR (dB)':<30} {avg_psnr1:>8.2f} ± {std_psnr1:<5.2f} {avg_psnr2:>13.2f} ± {std_psnr2:<5.2f} {psnr_winner:>10}")
    
    # SSIM comparison
    ssim_winner = "Script 1 ✓" if avg_ssim1 > avg_ssim2 else "Script 2 ✓" if avg_ssim2 > avg_ssim1 else "Tie"
    print(f"{'SSIM':<30} {avg_ssim1:>8.4f} ± {std_ssim1:<5.4f} {avg_ssim2:>13.4f} ± {std_ssim2:<5.4f} {ssim_winner:>10}")
    
    print()
    print("="*80)
    
    # Detailed per-image comparison
    print("\nDETAILED PER-IMAGE COMPARISON")
    print("="*80)
    
    # Match files by filename
    for r1 in results_script1:
        r2 = next((r for r in results_script2 if r['filename'] == r1['filename']), None)
        if r2:
            psnr_diff = r2['psnr'] - r1['psnr']
            ssim_diff = r2['ssim'] - r1['ssim']
            
            print(f"\n{r1['filename']}")
            print(f"  PSNR: {r1['psnr']:6.2f} dB (Script1) vs {r2['psnr']:6.2f} dB (Script2) | Δ = {psnr_diff:+.2f} dB")
            print(f"  SSIM: {r1['ssim']:6.4f} (Script1) vs {r2['ssim']:6.4f} (Script2) | Δ = {ssim_diff:+.4f}")
            
            # Determine winner for this image
            psnr_img_winner = "Script2" if psnr_diff > 0 else "Script1" if psnr_diff < 0 else "Tie"
            ssim_img_winner = "Script2" if ssim_diff > 0 else "Script1" if ssim_diff < 0 else "Tie"
            print(f"  Winner: PSNR={psnr_img_winner}, SSIM={ssim_img_winner}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    psnr_improvement = avg_psnr2 - avg_psnr1
    ssim_improvement = avg_ssim2 - avg_ssim1
    
    print(f"\nScript 2 (WITH post-processing) vs Script 1 (NO post-processing):")
    print(f"  PSNR: {psnr_improvement:+.2f} dB ({psnr_improvement/avg_psnr1*100:+.2f}%)")
    print(f"  SSIM: {ssim_improvement:+.4f} ({ssim_improvement/avg_ssim1*100:+.2f}%)")
    
    print()
    
    if psnr_improvement > 0 and ssim_improvement > 0:
        print("🏆 WINNER: Script 2 (inference_portrait_overlap_experiment.py)")
        print("   → Post-processing MENINGKATKAN kualitas objektif (PSNR & SSIM lebih tinggi)")
    elif psnr_improvement < 0 and ssim_improvement < 0:
        print("🏆 WINNER: Script 1 (inference_data_latih.py)")
        print("   → Tanpa post-processing menghasilkan kualitas objektif lebih baik")
    else:
        print("🤝 MIXED RESULTS")
        if psnr_improvement > 0:
            print("   → Script 2 lebih baik di PSNR")
        else:
            print("   → Script 1 lebih baik di PSNR")
        if ssim_improvement > 0:
            print("   → Script 2 lebih baik di SSIM")
        else:
            print("   → Script 1 lebih baik di SSIM")
    
    print("="*80)
    
    # Save results
    output_file = Path("results/evaluation_with_ground_truth.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'ground_truth_dir': str(ground_truth_dir),
        'script1': {
            'name': 'inference_data_latih.py (NO post-processing)',
            'path': str(results_dir1),
            'num_images': len(results_script1),
            'avg_psnr': float(avg_psnr1),
            'std_psnr': float(std_psnr1),
            'avg_ssim': float(avg_ssim1),
            'std_ssim': float(std_ssim1),
            'results': results_script1
        },
        'script2': {
            'name': 'inference_portrait_overlap_experiment.py (WITH post-processing)',
            'path': str(results_dir2),
            'num_images': len(results_script2),
            'avg_psnr': float(avg_psnr2),
            'std_psnr': float(std_psnr2),
            'avg_ssim': float(avg_ssim2),
            'std_ssim': float(std_ssim2),
            'results': results_script2
        },
        'comparison': {
            'psnr_improvement': float(psnr_improvement),
            'ssim_improvement': float(ssim_improvement),
            'psnr_improvement_percent': float(psnr_improvement/avg_psnr1*100),
            'ssim_improvement_percent': float(ssim_improvement/avg_ssim1*100)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
