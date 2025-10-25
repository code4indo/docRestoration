#!/usr/bin/env python3
"""
Comprehensive Comparison: Baseline vs Enhanced Version
"""
import json
import pandas as pd

# Load evaluation results
with open('results/dibco_2013/evaluation_metrics.json') as f:
    baseline = json.load(f)

with open('results/dibco_2013_prod/evaluation_metrics.json') as f:
    enhanced = json.load(f)

print("="*80)
print("DIBCO 2013 Evaluation - Baseline vs Enhanced Comparison")
print("="*80)
print()

# Aggregate comparison
print("AGGREGATE METRICS COMPARISON:")
print("-" * 80)
print(f"{'Metric':<15} {'Baseline':<20} {'Enhanced':<20} {'Improvement':<20}")
print("-" * 80)

metrics = ['psnr', 'ssim', 'f_measure', 'nrm', 'mpm']
for metric in metrics:
    baseline_val = baseline['aggregate_metrics'][metric]['mean']
    enhanced_val = enhanced['aggregate_metrics'][metric]['mean']
    diff = enhanced_val - baseline_val
    
    if metric == 'psnr':
        print(f"{metric.upper():<15} {baseline_val:>6.2f} dB         {enhanced_val:>6.2f} dB         {diff:>+6.2f} dB")
    elif metric in ['nrm', 'mpm']:
        pct_change = (diff / baseline_val) * 100 if baseline_val != 0 else 0
        print(f"{metric.upper():<15} {baseline_val:>8.4f}           {enhanced_val:>8.4f}           {diff:>+7.4f} ({pct_change:>+5.1f}%)")
    else:
        pct_change = (diff / baseline_val) * 100 if baseline_val != 0 else 0
        print(f"{metric.upper():<15} {baseline_val:>8.4f}           {enhanced_val:>8.4f}           {diff:>+7.4f} ({pct_change:>+5.1f}%)")

print()
print("="*80)
print("DETAILED ANALYSIS:")
print("="*80)
print()

# Per-image comparison
print("Per-Image PSNR Comparison:")
print("-" * 60)
print(f"{'Image':<10} {'Baseline':>12} {'Enhanced':>12} {'Δ':>12} {'Status':<15}")
print("-" * 60)

improvements = 0
degradations = 0
for baseline_img in baseline['per_image_results']:
    img_name = baseline_img['image_name']
    enhanced_img = next((x for x in enhanced['per_image_results'] if x['image_name'] == img_name), None)
    
    if enhanced_img:
        baseline_psnr = baseline_img['psnr']
        enhanced_psnr = enhanced_img['psnr']
        diff = enhanced_psnr - baseline_psnr
        
        if diff > 0:
            status = "✅ Improved"
            improvements += 1
        elif diff < -0.1:
            status = "⚠️  Degraded"
            degradations += 1
        else:
            status = "≈ Same"
        
        print(f"{img_name:<10} {baseline_psnr:>10.2f} dB {enhanced_psnr:>10.2f} dB {diff:>+10.2f} dB  {status:<15}")

print("-" * 60)
print(f"Summary: {improvements} improved, {degradations} degraded, {16-improvements-degradations} similar")
print()

# Best and worst performers
print("="*80)
print("KEY INSIGHTS:")
print("="*80)
print()

print("✅ IMPROVEMENTS:")
print("  • Better blending: Increased overlap (20% vertical, 25% horizontal)")
print("  • Smoother transitions: Enhanced Gaussian weights (σ_ratio 3.5 vs 3.0)")  
print("  • Post-processing: Bilateral filter for seam smoothing")
print("  • Better alpha: 0.25 blending with original")
print()

print("📊 RESULTS:")
baseline_psnr = baseline['aggregate_metrics']['psnr']['mean']
enhanced_psnr = enhanced['aggregate_metrics']['psnr']['mean']
baseline_ssim = baseline['aggregate_metrics']['ssim']['mean']
enhanced_ssim = enhanced['aggregate_metrics']['ssim']['mean']

print(f"  • PSNR: {baseline_psnr:.2f} → {enhanced_psnr:.2f} dB ({enhanced_psnr-baseline_psnr:+.2f} dB)")
print(f"  • SSIM: {baseline_ssim:.4f} → {enhanced_ssim:.4f} ({enhanced_ssim-baseline_ssim:+.4f})")
print(f"  • F-Measure: {baseline['aggregate_metrics']['f_measure']['mean']:.4f} → {enhanced['aggregate_metrics']['f_measure']['mean']:.4f}")
print()

if enhanced_psnr > baseline_psnr and enhanced_ssim > baseline_ssim:
    print("✅ CONCLUSION: Enhanced version shows improvement in both PSNR and SSIM")
elif enhanced_psnr > baseline_psnr:
    print("⚠️  CONCLUSION: Enhanced version shows PSNR improvement but SSIM trade-off")
else:
    print("⚠️  CONCLUSION: Enhanced version shows mixed results, further tuning needed")

print()
print("🎯 NEXT STEPS:")
print("  1. Fine-tune bilateral filter parameters (d, σ_color, σ_space)")
print("  2. Experiment with alpha blending values (0.2, 0.25, 0.3)")
print("  3. Consider adaptive overlap based on image content")
print("  4. Add morphological operations for stroke preservation")
print()
print("="*80)
