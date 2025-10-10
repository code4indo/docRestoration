#!/usr/bin/env python3
"""
Comprehensive Baseline Analysis Script
Analyze all 3 baseline methods dan compare dengan GAN-HTR main model

Usage:
    poetry run python dual_modal_gan/scripts/analyze_baselines.py
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def find_latest_baseline_dir(pattern):
    """Find latest baseline output directory."""
    dirs = glob.glob(f"dual_modal_gan/outputs/{pattern}_*")
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def load_training_history(dir_path):
    """Load training history from JSON."""
    history_path = os.path.join(dir_path, "metrics", "training_history.json")
    if not os.path.exists(history_path):
        return None
    
    with open(history_path, 'r') as f:
        return json.load(f)


def analyze_baseline(name, dir_path):
    """Analyze single baseline training."""
    if not dir_path or not os.path.exists(dir_path):
        return None
    
    history = load_training_history(dir_path)
    if not history:
        return None
    
    # Get final metrics
    final_epoch = history['epoch'][-1] if history['epoch'] else 0
    final_psnr = history['psnr'][-1] if history['psnr'] else 0
    final_ssim = history['ssim'][-1] if history['ssim'] else 0
    final_g_loss = history['g_loss'][-1] if history['g_loss'] else 0
    
    # Calculate statistics
    max_psnr = max(history['psnr']) if history['psnr'] else 0
    max_ssim = max(history['ssim']) if history['ssim'] else 0
    min_g_loss = min(history['g_loss']) if history['g_loss'] else 0
    
    # Convergence analysis
    if len(history['psnr']) > 10:
        last_10_psnr = history['psnr'][-10:]
        psnr_std = np.std(last_10_psnr)
        converged = psnr_std < 0.5  # Consider converged if std < 0.5 dB
    else:
        psnr_std = 0
        converged = False
    
    return {
        'name': name,
        'dir': dir_path,
        'final_epoch': final_epoch,
        'final_psnr': final_psnr,
        'final_ssim': final_ssim,
        'final_g_loss': final_g_loss,
        'max_psnr': max_psnr,
        'max_ssim': max_ssim,
        'min_g_loss': min_g_loss,
        'psnr_std_last10': psnr_std,
        'converged': converged,
        'history': history
    }


def create_comparison_table(baselines, main_model):
    """Create comparison table."""
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Method':<25} {'Epochs':<10} {'PSNR (dB)':<12} {'SSIM':<10} {'G_Loss':<10} {'Converged':<10}")
    print(f"{'-'*80}")
    
    # Main model
    print(f"{'GAN-HTR (Main Model)':<25} {main_model['epochs']:<10} "
          f"{main_model['psnr']:<12.2f} {main_model['ssim']:<10.4f} "
          f"{'N/A':<10} {'✅':<10}")
    print(f"{'-'*80}")
    
    # Baselines
    for baseline in baselines:
        if baseline:
            converged_mark = '✅' if baseline['converged'] else '⏳'
            print(f"{baseline['name']:<25} {baseline['final_epoch']:<10} "
                  f"{baseline['final_psnr']:<12.2f} {baseline['final_ssim']:<10.4f} "
                  f"{baseline['final_g_loss']:<10.4f} {converged_mark:<10}")
        else:
            print(f"{'[Not Available]':<25} {'-':<10} {'-':<12} {'-':<10} {'-':<10} {'-':<10}")
    
    print(f"{'-'*80}\n")


def create_performance_gaps(baselines, main_model):
    """Calculate performance gaps vs main model."""
    print(f"\n{'='*80}")
    print("PERFORMANCE GAPS (vs GAN-HTR Main Model)")
    print(f"{'='*80}\n")
    
    print(f"{'Method':<25} {'PSNR Gap':<15} {'SSIM Gap':<15} {'% Improvement Needed':<25}")
    print(f"{'-'*80}")
    
    for baseline in baselines:
        if baseline:
            psnr_gap = main_model['psnr'] - baseline['final_psnr']
            ssim_gap = main_model['ssim'] - baseline['final_ssim']
            psnr_improvement_needed = (psnr_gap / baseline['final_psnr']) * 100 if baseline['final_psnr'] > 0 else 0
            
            print(f"{baseline['name']:<25} "
                  f"{psnr_gap:>+7.2f} dB{'':>5} "
                  f"{ssim_gap:>+7.4f}{'':>5} "
                  f"{psnr_improvement_needed:>7.1f}%")
    
    print(f"{'-'*80}\n")


def create_training_curves(baselines, main_model, output_dir):
    """Create training curves comparison plot."""
    print("\n📊 Creating training curves comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Baseline Training Comparison', fontsize=16, weight='bold')
    
    # PSNR curves
    ax1 = axes[0, 0]
    for baseline in baselines:
        if baseline and baseline['history']:
            ax1.plot(baseline['history']['epoch'], baseline['history']['psnr'], 
                    marker='o', markersize=3, label=baseline['name'], linewidth=2)
    ax1.axhline(main_model['psnr'], color='red', linestyle='--', linewidth=2, 
                label=f'GAN-HTR Target ({main_model["psnr"]:.2f} dB)')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('PSNR (dB)', fontsize=11)
    ax1.set_title('Peak Signal-to-Noise Ratio', fontsize=13, weight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # SSIM curves
    ax2 = axes[0, 1]
    for baseline in baselines:
        if baseline and baseline['history']:
            ax2.plot(baseline['history']['epoch'], baseline['history']['ssim'], 
                    marker='o', markersize=3, label=baseline['name'], linewidth=2)
    ax2.axhline(main_model['ssim'], color='red', linestyle='--', linewidth=2,
                label=f'GAN-HTR Target ({main_model["ssim"]:.4f})')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('SSIM', fontsize=11)
    ax2.set_title('Structural Similarity Index', fontsize=13, weight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Generator Loss curves
    ax3 = axes[1, 0]
    for baseline in baselines:
        if baseline and baseline['history']:
            ax3.plot(baseline['history']['epoch'], baseline['history']['g_loss'], 
                    marker='o', markersize=3, label=baseline['name'], linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Generator Loss', fontsize=11)
    ax3.set_title('Generator Loss Convergence', fontsize=13, weight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_yscale('log')  # Log scale untuk better visualization
    
    # Bar chart comparison (final metrics)
    ax4 = axes[1, 1]
    methods = []
    psnr_values = []
    ssim_values = []
    
    for baseline in baselines:
        if baseline:
            methods.append(baseline['name'].replace('Baseline ', '').replace(': ', '\n'))
            psnr_values.append(baseline['final_psnr'])
            ssim_values.append(baseline['final_ssim'] * 100)  # Scale to 0-100
    
    methods.append('GAN-HTR\n(Target)')
    psnr_values.append(main_model['psnr'])
    ssim_values.append(main_model['ssim'] * 100)
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, psnr_values, width, label='PSNR (dB)', alpha=0.8)
    bars2 = ax4.bar(x + width/2, ssim_values, width, label='SSIM (×100)', alpha=0.8)
    
    ax4.set_xlabel('Method', fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Final Performance Comparison', fontsize=13, weight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'baseline_comparison_curves.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  ✅ Saved to {output_path}")


def create_convergence_analysis(baselines, output_dir):
    """Analyze convergence behavior."""
    print("\n📈 Analyzing convergence behavior...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Convergence Analysis - Last 10 Epochs', fontsize=16, weight='bold')
    
    for idx, baseline in enumerate(baselines):
        if baseline and baseline['history'] and len(baseline['history']['psnr']) >= 10:
            ax = axes[idx]
            
            last_10_epochs = baseline['history']['epoch'][-10:]
            last_10_psnr = baseline['history']['psnr'][-10:]
            last_10_ssim = baseline['history']['ssim'][-10:]
            
            # PSNR
            ax.plot(last_10_epochs, last_10_psnr, 'o-', label='PSNR', linewidth=2, markersize=6)
            ax.axhline(np.mean(last_10_psnr), color='blue', linestyle='--', alpha=0.5, 
                      label=f'Mean: {np.mean(last_10_psnr):.2f}')
            
            # SSIM (scaled)
            ax_twin = ax.twinx()
            ax_twin.plot(last_10_epochs, last_10_ssim, 's-', color='orange', 
                        label='SSIM', linewidth=2, markersize=6)
            ax_twin.axhline(np.mean(last_10_ssim), color='orange', linestyle='--', alpha=0.5,
                          label=f'Mean: {np.mean(last_10_ssim):.4f}')
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('PSNR (dB)', fontsize=11, color='blue')
            ax_twin.set_ylabel('SSIM', fontsize=11, color='orange')
            ax.set_title(baseline['name'], fontsize=12, weight='bold')
            ax.grid(alpha=0.3)
            
            # Legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
            
            # Add std text
            psnr_std = np.std(last_10_psnr)
            converged_text = '✅ Converged' if psnr_std < 0.5 else '⏳ Still improving'
            ax.text(0.02, 0.98, f'Std: {psnr_std:.3f} dB\n{converged_text}',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'baseline_convergence_analysis.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  ✅ Saved to {output_path}")


def create_improvement_potential(baselines, main_model, output_dir):
    """Visualize improvement potential."""
    print("\n📊 Creating improvement potential analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Improvement Potential Analysis', fontsize=16, weight='bold')
    
    # PSNR improvement needed
    methods = []
    psnr_gaps = []
    ssim_gaps = []
    
    for baseline in baselines:
        if baseline:
            methods.append(baseline['name'].replace('Baseline ', '').replace(': ', '\n'))
            psnr_gaps.append(main_model['psnr'] - baseline['final_psnr'])
            ssim_gaps.append((main_model['ssim'] - baseline['final_ssim']) * 100)  # Percentage points
    
    # PSNR Gap
    bars1 = ax1.barh(methods, psnr_gaps, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax1.set_xlabel('PSNR Gap (dB)', fontsize=12)
    ax1.set_title('PSNR Gap vs GAN-HTR Target', fontsize=13, weight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (bar, gap) in enumerate(zip(bars1, psnr_gaps)):
        ax1.text(gap + 0.2, i, f'{gap:.2f} dB', va='center', fontsize=10, weight='bold')
    
    # SSIM Gap
    bars2 = ax2.barh(methods, ssim_gaps, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax2.set_xlabel('SSIM Gap (percentage points)', fontsize=12)
    ax2.set_title('SSIM Gap vs GAN-HTR Target', fontsize=13, weight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (bar, gap) in enumerate(zip(bars2, ssim_gaps)):
        ax2.text(gap + 0.2, i, f'{gap:.2f}pp', va='center', fontsize=10, weight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'baseline_improvement_potential.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  ✅ Saved to {output_path}")


def save_analysis_summary(baselines, main_model, output_dir):
    """Save analysis summary to JSON."""
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'main_model': main_model,
        'baselines': []
    }
    
    for baseline in baselines:
        if baseline:
            summary['baselines'].append({
                'name': baseline['name'],
                'directory': baseline['dir'],
                'final_epoch': int(baseline['final_epoch']),
                'final_psnr': float(baseline['final_psnr']),
                'final_ssim': float(baseline['final_ssim']),
                'max_psnr': float(baseline['max_psnr']),
                'max_ssim': float(baseline['max_ssim']),
                'psnr_gap_vs_main': float(main_model['psnr'] - baseline['final_psnr']),
                'ssim_gap_vs_main': float(main_model['ssim'] - baseline['final_ssim']),
                'converged': bool(baseline['converged']),
                'psnr_std_last10': float(baseline['psnr_std_last10'])
            })
    
    output_path = os.path.join(output_dir, 'baseline_analysis_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Analysis summary saved to {output_path}")


def main():
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BASELINE ANALYSIS")
    print(f"{'='*80}")
    
    # Main model metrics (from comprehensive evaluation)
    main_model = {
        'name': 'GAN-HTR (Main Model)',
        'epochs': 100,
        'psnr': 22.90,
        'ssim': 0.9538
    }
    
    # Find latest baseline directories
    print("\n🔍 Finding baseline training directories...")
    unet_dir = find_latest_baseline_dir("baseline_unet")
    gan_dir = find_latest_baseline_dir("baseline_standard_gan")
    ctc_dir = find_latest_baseline_dir("baseline_ctc_only")
    
    print(f"  U-Net: {unet_dir or 'Not found'}")
    print(f"  Standard GAN: {gan_dir or 'Not found'}")
    print(f"  CTC-Only: {ctc_dir or 'Not found'}")
    
    # Analyze each baseline
    print("\n📊 Analyzing baselines...")
    baselines = [
        analyze_baseline("Baseline 1: Plain U-Net", unet_dir),
        analyze_baseline("Baseline 2: Standard GAN", gan_dir),
        analyze_baseline("Baseline 3: CTC-Only", ctc_dir)
    ]
    
    # Filter out None values
    valid_baselines = [b for b in baselines if b is not None]
    
    if not valid_baselines:
        print("\n❌ No baseline training results found!")
        return
    
    # Create output directory
    output_dir = "dual_modal_gan/outputs/baseline_analysis"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Generate analyses
    create_comparison_table(baselines, main_model)
    create_performance_gaps(baselines, main_model)
    create_training_curves(baselines, main_model, output_dir)
    create_convergence_analysis(valid_baselines, output_dir)
    create_improvement_potential(valid_baselines, main_model, output_dir)
    save_analysis_summary(baselines, main_model, output_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  1. baseline_comparison_curves.png - Training curves comparison")
    print(f"  2. baseline_convergence_analysis.png - Convergence behavior")
    print(f"  3. baseline_improvement_potential.png - Gap analysis")
    print(f"  4. baseline_analysis_summary.json - Complete summary")
    print(f"\n📁 All files in: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
