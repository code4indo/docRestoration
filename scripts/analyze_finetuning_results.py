"""
Comprehensive analysis of DIBCO fine-tuning results.
Analyzes training metrics, visual quality, and model performance.
"""

import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def parse_training_log(log_path):
    """Extract all metrics from training log"""
    epochs_data = []
    
    with open(log_path, 'r') as f:
        current_epoch = {}
        
        for line in f:
            # Epoch number
            if match := re.search(r'Epoch (\d+)/(\d+)', line):
                if current_epoch:
                    epochs_data.append(current_epoch)
                current_epoch = {
                    'epoch': int(match.group(1)),
                    'total_epochs': int(match.group(2))
                }
            
            # PSNR
            if match := re.search(r'PSNR:\s+([\d.]+)\s+±\s+([\d.]+)\s+dB', line):
                current_epoch['psnr_mean'] = float(match.group(1))
                current_epoch['psnr_std'] = float(match.group(2))
            
            # SSIM
            if match := re.search(r'SSIM:\s+([\d.]+)\s+±\s+([\d.]+)', line):
                current_epoch['ssim_mean'] = float(match.group(1))
                current_epoch['ssim_std'] = float(match.group(2))
            
            # CER
            if match := re.search(r'CER:\s+([\d.]+)\s+±\s+([\d.]+)', line):
                current_epoch['cer_mean'] = float(match.group(1))
                current_epoch['cer_std'] = float(match.group(2))
            
            # Noise metrics
            if match := re.search(r'Noise Var:\s+([\d.e-]+)', line):
                current_epoch['noise_var'] = float(match.group(1))
            if match := re.search(r'Isolated White:\s+([\d.e-]+)', line):
                current_epoch['isolated_white'] = float(match.group(1))
            
            # Best model flag
            if 'Best model preserved' in line:
                current_epoch['is_best'] = True
            
            # Epoch completion time
            if match := re.search(r'Epoch \d+ completed in ([\d.]+)s', line):
                current_epoch['time_seconds'] = float(match.group(1))
        
        # Add last epoch
        if current_epoch and 'epoch' in current_epoch:
            epochs_data.append(current_epoch)
    
    return epochs_data

def analyze_performance(epochs_data):
    """Analyze training performance metrics"""
    print("\n" + "=" * 80)
    print("TRAINING PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Extract PSNR values
    psnr_values = [e['psnr_mean'] for e in epochs_data if 'psnr_mean' in e]
    ssim_values = [e['ssim_mean'] for e in epochs_data if 'ssim_mean' in e]
    
    print(f"\n📊 PSNR Analysis ({len(psnr_values)} epochs):")
    print(f"   Initial (Epoch 1):     {psnr_values[0]:.2f} dB")
    print(f"   Final (Epoch {len(psnr_values)}):      {psnr_values[-1]:.2f} dB")
    print(f"   Best:                  {max(psnr_values):.2f} dB (Epoch {psnr_values.index(max(psnr_values)) + 1})")
    print(f"   Worst:                 {min(psnr_values):.2f} dB (Epoch {psnr_values.index(min(psnr_values)) + 1})")
    print(f"   Mean ± Std:            {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
    print(f"   Overall Change:        {psnr_values[-1] - psnr_values[0]:+.2f} dB ({(psnr_values[-1] - psnr_values[0])/psnr_values[0]*100:+.1f}%)")
    
    print(f"\n📊 SSIM Analysis ({len(ssim_values)} epochs):")
    print(f"   Initial (Epoch 1):     {ssim_values[0]:.4f}")
    print(f"   Final (Epoch {len(ssim_values)}):      {ssim_values[-1]:.4f}")
    print(f"   Best:                  {max(ssim_values):.4f} (Epoch {ssim_values.index(max(ssim_values)) + 1})")
    print(f"   Mean ± Std:            {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print(f"   Overall Change:        {ssim_values[-1] - ssim_values[0]:+.4f} ({(ssim_values[-1] - ssim_values[0])/ssim_values[0]*100:+.1f}%)")
    
    # Training stability
    print(f"\n📈 Training Stability:")
    psnr_variance = np.var(psnr_values)
    psnr_range = max(psnr_values) - min(psnr_values)
    print(f"   PSNR Variance:         {psnr_variance:.4f}")
    print(f"   PSNR Range:            {psnr_range:.2f} dB")
    
    # Identify best models
    best_epochs = [e for e in epochs_data if e.get('is_best', False)]
    print(f"\n🏆 Best Model Updates: {len(best_epochs)} times")
    if len(best_epochs) > 0:
        print(f"   First best: Epoch {best_epochs[0]['epoch']}")
        print(f"   Last best:  Epoch {best_epochs[-1]['epoch']}")
    
    # Training efficiency
    times = [e['time_seconds'] for e in epochs_data if 'time_seconds' in e]
    if times:
        print(f"\n⏱️  Training Time:")
        print(f"   Average per epoch:     {np.mean(times):.1f} seconds")
        print(f"   Total time:            {sum(times)/60:.1f} minutes ({sum(times)/3600:.2f} hours)")
        print(f"   Fastest epoch:         {min(times):.1f} seconds")
        print(f"   Slowest epoch:         {max(times):.1f} seconds")
    
    return {
        'psnr_initial': psnr_values[0],
        'psnr_final': psnr_values[-1],
        'psnr_best': max(psnr_values),
        'psnr_mean': np.mean(psnr_values),
        'ssim_best': max(ssim_values),
        'total_epochs': len(epochs_data),
        'best_updates': len(best_epochs)
    }

def analyze_convergence(epochs_data):
    """Analyze convergence behavior"""
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)
    
    psnr_values = [e['psnr_mean'] for e in epochs_data if 'psnr_mean' in e]
    
    # Split into phases
    early = psnr_values[:10]
    middle = psnr_values[10:30] if len(psnr_values) > 30 else psnr_values[10:]
    late = psnr_values[30:] if len(psnr_values) > 30 else []
    
    print(f"\n📊 Training Phases:")
    print(f"   Early (1-10):          {np.mean(early):.2f} ± {np.std(early):.2f} dB")
    if middle:
        print(f"   Middle (11-30):        {np.mean(middle):.2f} ± {np.std(middle):.2f} dB")
    if late:
        print(f"   Late (31+):            {np.mean(late):.2f} ± {np.std(late):.2f} dB")
    
    # Trend analysis
    if len(psnr_values) >= 10:
        last_10_mean = np.mean(psnr_values[-10:])
        first_10_mean = np.mean(psnr_values[:10])
        last_10_std = np.std(psnr_values[-10:])
        
        print(f"\n📈 Trend Analysis:")
        print(f"   First 10 epochs mean:  {first_10_mean:.2f} dB")
        print(f"   Last 10 epochs mean:   {last_10_mean:.2f} dB")
        print(f"   Last 10 epochs std:    {last_10_std:.2f} dB")
        print(f"   Improvement:           {last_10_mean - first_10_mean:+.2f} dB")
        
        if last_10_std > 0.5:
            print(f"   ⚠️  High variance in late epochs - model may be unstable")
        if last_10_mean <= first_10_mean:
            print(f"   ⚠️  No improvement from early to late epochs")

def diagnose_issues(epochs_data, summary):
    """Diagnose potential issues"""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    psnr_values = [e['psnr_mean'] for e in epochs_data if 'psnr_mean' in e]
    
    print(f"\n🔍 Identified Issues:")
    
    # Issue 1: Minimal improvement
    improvement = summary['psnr_final'] - summary['psnr_initial']
    if abs(improvement) < 1.0:
        print(f"   ❌ CRITICAL: Minimal PSNR improvement ({improvement:+.2f} dB)")
        print(f"      - Fine-tuning did not significantly improve model")
        print(f"      - Possible causes:")
        print(f"        * Dataset too similar to pretrained data")
        print(f"        * Learning rate too low")
        print(f"        * Model already converged")
    
    # Issue 2: Negative improvement
    if improvement < 0:
        print(f"   ❌ CRITICAL: Model degraded during fine-tuning ({improvement:+.2f} dB)")
        print(f"      - Final performance worse than initial")
        print(f"      - Possible causes:")
        print(f"        * Catastrophic forgetting")
        print(f"        * Learning rate too high")
        print(f"        * Data distribution mismatch")
    
    # Issue 3: High variance
    psnr_std = np.std(psnr_values)
    if psnr_std > 0.5:
        print(f"   ⚠️  WARNING: High PSNR variance ({psnr_std:.2f} dB)")
        print(f"      - Training is unstable")
        print(f"      - Consider:")
        print(f"        * Lower learning rate")
        print(f"        * Gradient clipping adjustment")
        print(f"        * Batch size increase")
    
    # Issue 4: Low absolute PSNR
    if summary['psnr_best'] < 25.0:
        print(f"   ⚠️  WARNING: Low peak PSNR ({summary['psnr_best']:.2f} dB < 25 dB target)")
        print(f"      - Model not reaching target performance")
        print(f"      - Possible causes:")
        print(f"        * Dataset quality issues (degradation too severe)")
        print(f"        * Model capacity insufficient")
        print(f"        * Training not converged (needs more epochs)")
    
    # Issue 5: Early stopping didn't trigger
    if summary['total_epochs'] == 50:
        print(f"   ℹ️  INFO: Training completed all 50 epochs")
        print(f"      - Early stopping did not trigger")
        print(f"      - Model may benefit from continued training")
    
    # Issue 6: Visual-only mode limitations
    cer_values = [e.get('cer_mean', 1.0) for e in epochs_data]
    if all(c == 1.0 for c in cer_values):
        print(f"   ℹ️  INFO: Visual-only mode (no recognizer)")
        print(f"      - CER/WER metrics disabled")
        print(f"      - Cannot assess text recognition quality")
        print(f"      - Focus on PSNR/SSIM for quality assessment")

def generate_plots(epochs_data, output_dir):
    """Generate analysis plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    psnr_values = [e['psnr_mean'] for e in epochs_data if 'psnr_mean' in e]
    psnr_stds = [e['psnr_std'] for e in epochs_data if 'psnr_std' in e]
    ssim_values = [e['ssim_mean'] for e in epochs_data if 'ssim_mean' in e]
    epochs = list(range(1, len(psnr_values) + 1))
    
    # Plot 1: PSNR progression with error bars
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1.errorbar(epochs, psnr_values, yerr=psnr_stds, fmt='o-', 
                 capsize=3, capthick=1, markersize=4, linewidth=1.5,
                 color='#2E86AB', ecolor='#A23B72', alpha=0.8)
    ax1.axhline(y=max(psnr_values), color='g', linestyle='--', 
                label=f'Best: {max(psnr_values):.2f} dB', alpha=0.7)
    ax1.axhline(y=25.0, color='r', linestyle='--', 
                label='Target: 25.00 dB', alpha=0.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR Progression During Fine-Tuning', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SSIM progression
    ax2.plot(epochs, ssim_values, 'o-', color='#F18F01', 
             markersize=4, linewidth=1.5, alpha=0.8)
    ax2.axhline(y=max(ssim_values), color='g', linestyle='--', 
                label=f'Best: {max(ssim_values):.4f}', alpha=0.7)
    ax2.axhline(y=0.95, color='r', linestyle='--', 
                label='Target: 0.9500', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM Progression During Fine-Tuning', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'training_metrics_progression.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n📊 Plot saved: {plot_path}")
    plt.close()
    
    # Plot 3: Distribution analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(psnr_values, bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(x=np.mean(psnr_values), color='r', linestyle='--', 
                label=f'Mean: {np.mean(psnr_values):.2f} dB', linewidth=2)
    ax1.set_xlabel('PSNR (dB)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('PSNR Distribution Across Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.hist(ssim_values, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(ssim_values), color='r', linestyle='--', 
                label=f'Mean: {np.mean(ssim_values):.4f}', linewidth=2)
    ax2.set_xlabel('SSIM', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('SSIM Distribution Across Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / 'metrics_distribution.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze fine-tuning results')
    parser.add_argument('--log_file', type=str, 
                       default='logbook/dibco_tiled_no_palm_visual_only_20251026_201720.log',
                       help='Path to training log file')
    parser.add_argument('--output_dir', type=str,
                       default='dual_modal_gan/outputs/analysis_dibco_finetuning',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("DIBCO FINE-TUNING ANALYSIS")
    print("=" * 80)
    print(f"Log file: {args.log_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Parse log
    epochs_data = parse_training_log(args.log_file)
    print(f"\n✅ Parsed {len(epochs_data)} epochs from log")
    
    # Analyze performance
    summary = analyze_performance(epochs_data)
    
    # Analyze convergence
    analyze_convergence(epochs_data)
    
    # Diagnose issues
    diagnose_issues(epochs_data, summary)
    
    # Generate plots
    generate_plots(epochs_data, args.output_dir)
    
    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Summary saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
