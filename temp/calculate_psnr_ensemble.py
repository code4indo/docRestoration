#!/usr/bin/env python3
"""
Quick PSNR Calculation for Ensemble Results
============================================
Calculate PSNR between ensemble outputs and ground truth.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # Ensure same data type
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def main():
    output_dir = "dual_modal_gan/outputs/ensemble_dibco_2012"
    gt_dir = "dibco_datasets/2012/gt_imgs"
    
    results = []
    
    # Get all ensemble output files
    output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('_ensemble.png')])
    
    print("=" * 80)
    print("ENSEMBLE PSNR CALCULATION - DIBCO 2012")
    print("=" * 80)
    print(f"\nOutput dir: {output_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"Found {len(output_files)} output files\n")
    
    for output_file in output_files:
        # Extract image name (1.png, 2.png, etc.)
        img_name = output_file.replace('_ensemble.png', '.png')
        
        # Load ensemble output
        output_path = os.path.join(output_dir, output_file)
        restored = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        
        # Load ground truth
        gt_path = os.path.join(gt_dir, img_name)
        
        if not os.path.exists(gt_path):
            print(f"⚠️  GT not found for {img_name}")
            continue
        
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if gt_image is None:
            print(f"⚠️  Failed to load GT: {img_name}")
            continue
        
        # Resize if needed
        if gt_image.shape != restored.shape:
            gt_image = cv2.resize(gt_image, (restored.shape[1], restored.shape[0]), 
                                 interpolation=cv2.INTER_AREA)
        
        # Calculate PSNR
        psnr = calculate_psnr(restored, gt_image)
        
        results.append({
            'image': img_name,
            'psnr': psnr
        })
        
        print(f"  {img_name:15s} → PSNR: {psnr:6.2f} dB")
    
    # Calculate statistics
    if results:
        psnrs = [r['psnr'] for r in results]
        avg_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        min_psnr = np.min(psnrs)
        max_psnr = np.max(psnrs)
        
        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS")
        print("=" * 80)
        print(f"Average PSNR:  {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"Min PSNR:      {min_psnr:.2f} dB")
        print(f"Max PSNR:      {max_psnr:.2f} dB")
        print(f"Images:        {len(results)}/14")
        
        print("\n" + "=" * 80)
        print("🎯 COMPARISON WITH SOTA")
        print("=" * 80)
        
        baselines = {
            "DE-GAN (2020)": 22.00,
            "DocEnTR (2022)": 22.29,
            "TARGET": 22.50,
            "AMBITIOUS": 24.00
        }
        
        for name, baseline in baselines.items():
            gap = avg_psnr - baseline
            status = "✅ BEAT" if gap > 0 else "❌ BELOW"
            print(f"{name:20s} {baseline:.2f} dB → Gap: {gap:+.2f} dB  {status}")
        
        # Save updated summary
        summary = {
            "timestamp": str(Path(output_dir).stat().st_mtime),
            "checkpoints": [
                "dibco_transfer_from_production_v3_fixed/best_model/ckpt-214",
                "dibco_transfer_from_production_v3_fixed/ckpt-220",
                "dibco_transfer_from_production_v3_fixed/ckpt-218",
                "production_v3_academic_split_70_15_15/best_model/ckpt-88"
            ],
            "weights": [0.35, 0.25, 0.20, 0.20],
            "num_models": 4,
            "num_images": len(results),
            "average_psnr": float(avg_psnr),
            "std_psnr": float(std_psnr),
            "min_psnr": float(min_psnr),
            "max_psnr": float(max_psnr),
            "results": results
        }
        
        summary_file = os.path.join(output_dir, "ensemble_summary_complete.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Complete summary saved: {summary_file}")
        
        # Final verdict
        print("\n" + "=" * 80)
        if avg_psnr >= 22.00:
            print("🎉 SUCCESS! Ensemble achieved target performance!")
            if avg_psnr >= 22.29:
                print("🏆 BEAT DocEnTR (SOTA)!")
        else:
            print("⚠️  Below target. Consider:")
            print("   - Add more checkpoints (5-7 models)")
            print("   - Tune ensemble weights")
            print("   - Add multi-scale inference")
        print("=" * 80)
        
    else:
        print("\n❌ No PSNR calculated - check GT files!")

if __name__ == "__main__":
    main()
