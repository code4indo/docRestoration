#!/usr/bin/env python3
"""
DIBCO 2012 Binarization Strategies Test
Tests different post-processing methods to maximize PSNR on binary GT
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
from tqdm import tqdm

def otsu_binarization(image):
    """Simple Otsu thresholding"""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def adaptive_mean_binarization(image, block_size=15):
    """Adaptive mean thresholding"""
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, block_size, 2
    )
    return binary

def adaptive_gaussian_binarization(image, block_size=15):
    """Adaptive Gaussian thresholding"""
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, 2
    )
    return binary

def sauvola_binarization(image, window_size=15, k=0.2):
    """
    Sauvola local thresholding - best for historical documents
    T(x,y) = m(x,y) * [1 + k * (s(x,y)/R - 1)]
    where m is local mean, s is local std, R is dynamic range
    """
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Calculate local mean and std
    mean = cv2.boxFilter(img_float, -1, (window_size, window_size))
    mean_sq = cv2.boxFilter(img_float**2, -1, (window_size, window_size))
    std = np.sqrt(mean_sq - mean**2)
    
    # Sauvola threshold
    R = 128.0  # Dynamic range for grayscale
    threshold = mean * (1 + k * (std / R - 1))
    
    # Apply threshold
    binary = np.where(img_float > threshold, 255, 0).astype(np.uint8)
    
    return binary

def niblack_binarization(image, window_size=15, k=-0.2):
    """
    Niblack local thresholding
    T(x,y) = m(x,y) + k * s(x,y)
    """
    img_float = image.astype(np.float32)
    
    mean = cv2.boxFilter(img_float, -1, (window_size, window_size))
    mean_sq = cv2.boxFilter(img_float**2, -1, (window_size, window_size))
    std = np.sqrt(mean_sq - mean**2)
    
    threshold = mean + k * std
    binary = np.where(img_float > threshold, 255, 0).astype(np.uint8)
    
    return binary

def denoise_then_otsu(image):
    """Denoise first, then Otsu"""
    # Bilateral filter to preserve edges while removing noise
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def morphological_cleanup(binary, kernel_size=3):
    """Morphological opening to remove small noise"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned

def evaluate_strategy(restored_dir, gt_dir, strategy_name, binarization_func, 
                     apply_morph=False, save_outputs=False):
    """
    Evaluate a binarization strategy
    
    Args:
        strategy_name: Name of the strategy
        binarization_func: Function that takes grayscale image and returns binary
        apply_morph: Whether to apply morphological cleanup
        save_outputs: Whether to save binarized images
    """
    results = []
    
    restored_files = sorted(restored_dir.glob("*_restored.png"))
    
    if save_outputs:
        output_dir = restored_dir.parent / f"dibco2012_{strategy_name}"
        output_dir.mkdir(exist_ok=True)
    
    for restored_path in restored_files:
        image_id = restored_path.stem.replace("_restored", "")
        gt_path = gt_dir / f"{image_id}.png"
        
        if not gt_path.exists():
            continue
        
        # Load images
        restored = cv2.imread(str(restored_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if restored.shape != gt.shape:
            restored = cv2.resize(restored, (gt.shape[1], gt.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
        
        # Apply binarization
        binary = binarization_func(restored)
        
        # Optional morphological cleanup
        if apply_morph:
            binary = morphological_cleanup(binary, kernel_size=2)
        
        # Save if requested
        if save_outputs:
            cv2.imwrite(str(output_dir / f"{image_id}_binary.png"), binary)
        
        # Calculate metrics
        binary_norm = binary.astype(np.float32) / 255.0
        gt_norm = gt.astype(np.float32) / 255.0
        
        psnr_value = psnr(gt_norm, binary_norm, data_range=1.0)
        ssim_value = ssim(gt_norm, binary_norm, data_range=1.0)
        
        results.append({
            'image_id': image_id,
            'psnr': float(psnr_value),
            'ssim': float(ssim_value)
        })
    
    # Calculate statistics
    if results:
        psnr_values = [r['psnr'] for r in results]
        ssim_values = [r['ssim'] for r in results]
        
        stats = {
            'strategy': strategy_name,
            'morphological_cleanup': apply_morph,
            'num_images': len(results),
            'psnr': {
                'mean': float(np.mean(psnr_values)),
                'std': float(np.std(psnr_values, ddof=1)),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values)),
                'median': float(np.median(psnr_values))
            },
            'ssim': {
                'mean': float(np.mean(ssim_values)),
                'std': float(np.std(ssim_values, ddof=1)),
                'min': float(np.min(ssim_values)),
                'max': float(np.max(ssim_values)),
                'median': float(np.median(ssim_values))
            },
            'per_image': results
        }
        
        return stats
    
    return None

def main():
    # Paths
    restored_dir = Path("results/dibco2012_ckpt99")
    gt_dir = Path("dibco_datasets/2012/gt_imgs")
    
    print("="*80)
    print("🧪 DIBCO 2012 Binarization Strategy Evaluation")
    print("="*80)
    print(f"Testing different post-processing methods to maximize PSNR")
    print(f"Target: PSNR > 23 dB")
    print()
    
    # Define strategies to test
    strategies = [
        ("otsu", otsu_binarization, False),
        ("otsu_morph", otsu_binarization, True),
        ("adaptive_mean_11", lambda img: adaptive_mean_binarization(img, 11), False),
        ("adaptive_mean_15", lambda img: adaptive_mean_binarization(img, 15), False),
        ("adaptive_gaussian_11", lambda img: adaptive_gaussian_binarization(img, 11), False),
        ("adaptive_gaussian_15", lambda img: adaptive_gaussian_binarization(img, 15), False),
        ("sauvola_15_k02", lambda img: sauvola_binarization(img, 15, 0.2), False),
        ("sauvola_15_k03", lambda img: sauvola_binarization(img, 15, 0.3), False),
        ("sauvola_25_k02", lambda img: sauvola_binarization(img, 25, 0.2), False),
        ("sauvola_35_k02", lambda img: sauvola_binarization(img, 35, 0.2), False),
        ("niblack_15", lambda img: niblack_binarization(img, 15, -0.2), False),
        ("denoise_otsu", denoise_then_otsu, False),
        ("denoise_otsu_morph", denoise_then_otsu, True),
    ]
    
    all_results = []
    
    print("Testing strategies...")
    print("-"*80)
    
    for strategy_name, binarization_func, apply_morph in tqdm(strategies, desc="Strategies"):
        stats = evaluate_strategy(
            restored_dir, gt_dir, strategy_name, 
            binarization_func, apply_morph,
            save_outputs=False  # Change to True to save outputs
        )
        
        if stats:
            all_results.append(stats)
            
            psnr_mean = stats['psnr']['mean']
            ssim_mean = stats['ssim']['mean']
            
            # Status indicator
            status = "✅" if psnr_mean > 23 else "⚠️"
            
            print(f"{status} {strategy_name:25s}: PSNR = {psnr_mean:6.2f} dB, SSIM = {ssim_mean:.4f}")
    
    # Sort by PSNR descending
    all_results.sort(key=lambda x: x['psnr']['mean'], reverse=True)
    
    print()
    print("="*80)
    print("🏆 BEST STRATEGIES (Top 5)")
    print("="*80)
    
    for i, stats in enumerate(all_results[:5], 1):
        psnr_mean = stats['psnr']['mean']
        ssim_mean = stats['ssim']['mean']
        strategy = stats['strategy']
        morph = " + morph" if stats['morphological_cleanup'] else ""
        
        print(f"{i}. {strategy}{morph}")
        print(f"   PSNR: {psnr_mean:6.2f} ± {stats['psnr']['std']:5.2f} dB "
              f"(min: {stats['psnr']['min']:.2f}, max: {stats['psnr']['max']:.2f})")
        print(f"   SSIM: {ssim_mean:.4f} ± {stats['ssim']['std']:.4f}")
        print()
    
    # Check if target achieved
    best = all_results[0]
    if best['psnr']['mean'] > 23:
        print("="*80)
        print(f"🎉 TARGET ACHIEVED! Best PSNR: {best['psnr']['mean']:.2f} dB (>23 dB)")
        print("="*80)
    else:
        print("="*80)
        print(f"⚠️  Target not reached. Best PSNR: {best['psnr']['mean']:.2f} dB")
        print("   Consider: Fine-tuning model on DIBCO dataset")
        print("="*80)
    
    # Save results
    output_file = restored_dir / "binarization_strategies_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': 'thin_stroke_preservation_v1_academic/best_model/ckpt-99',
            'dataset': 'DIBCO 2012',
            'target_psnr': 23.0,
            'all_strategies': all_results
        }, f, indent=2)
    
    print(f"\n📄 Full results saved to: {output_file}")
    
    # Ask if user wants to save best strategy outputs
    print()
    print("="*80)
    print("💡 NEXT STEPS")
    print("="*80)
    print(f"To save binarized images with best strategy ({best['strategy']}):")
    print(f"  Modify save_outputs=True for that strategy in this script")
    print()

if __name__ == '__main__':
    main()
