#!/usr/bin/env python3
"""
Compare Experiment Results - Extract and Compare Metrics
"""

import re
import json
from pathlib import Path
from datetime import datetime

def extract_metrics_from_log(log_file):
    """Extract final metrics from log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract final validation metrics
        psnr_match = re.search(r'📊 PSNR: ([\d.]+)', content)
        ssim_match = re.search(r'SSIM: ([\d.]+)', content)
        cer_match = re.search(r'CER: ([\d.]+)', content)
        wer_match = re.search(r'WER: ([\d.]+)', content)
        
        # Extract final loss values (last occurrence)
        g_loss_matches = re.findall(r'G=([\d.]+)', content)
        d_loss_matches = re.findall(r'D=([\d.]+)', content)
        ctc_matches = re.findall(r'CTC=([\d.]+)', content)
        
        metrics = {
            'psnr': float(psnr_match.group(1)) if psnr_match else None,
            'ssim': float(ssim_match.group(1)) if ssim_match else None,
            'cer': float(cer_match.group(1)) if cer_match else None,
            'wer': float(wer_match.group(1)) if wer_match else None,
            'g_loss': float(g_loss_matches[-1]) if g_loss_matches else None,
            'd_loss': float(d_loss_matches[-1]) if d_loss_matches else None,
            'ctc_loss': float(ctc_matches[-1]) if ctc_matches else None,
        }
        
        return metrics
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def main():
    logbook_dir = Path("/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/logbook")
    
    experiments = {
        'baseline': 'test_enhanced_disc_v2_20251016_170004.log',
        'experiment1': 'experiment1_ctc_annealing_low',
        'experiment2': 'experiment2_ctc_medium',
        'experiment3': 'experiment3_strong_adversarial',
        'experiment4': 'experiment4_balanced_optimal',
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("🔬 EXPERIMENT RESULTS COMPARISON")
    print("="*80 + "\n")
    
    # Extract baseline
    baseline_log = logbook_dir / experiments['baseline']
    if baseline_log.exists():
        results['baseline'] = extract_metrics_from_log(baseline_log)
        print(f"✅ Baseline extracted from: {baseline_log.name}")
    else:
        print(f"⚠️  Baseline log not found: {baseline_log}")
        results['baseline'] = None
    
    # Extract experiments
    for exp_name, exp_pattern in list(experiments.items())[1:]:
        # Find most recent log file matching pattern
        matching_logs = sorted(logbook_dir.glob(f"{exp_pattern}_*.log"))
        if matching_logs:
            latest_log = matching_logs[-1]
            results[exp_name] = extract_metrics_from_log(latest_log)
            print(f"✅ {exp_name} extracted from: {latest_log.name}")
        else:
            print(f"⚠️  {exp_name} log not found (pattern: {exp_pattern}_*.log)")
            results[exp_name] = None
    
    print("\n" + "="*80)
    print("📊 METRICS COMPARISON TABLE")
    print("="*80 + "\n")
    
    # Print header
    print(f"{'Experiment':<25} {'PSNR':<10} {'SSIM':<10} {'CER':<10} {'WER':<10} {'G_Loss':<12} {'D_Loss':<10}")
    print("-" * 97)
    
    # Print baseline
    if results['baseline']:
        b = results['baseline']
        print(f"{'BASELINE (CTC=1.0)':<25} "
              f"{b['psnr']:<10.2f} "
              f"{b['ssim']:<10.4f} "
              f"{b['cer']:<10.4f} "
              f"{b['wer']:<10.4f} "
              f"{b['g_loss']:<12.2f} "
              f"{b['d_loss']:<10.4f}")
    
    # Print experiments
    exp_names = {
        'experiment1': 'Exp1 (CTC=0.1)',
        'experiment2': 'Exp2 (CTC=0.3)',
        'experiment3': 'Exp3 (Adv=5.0)',
        'experiment4': 'Exp4 (Optimal)',
    }
    
    for exp_key, exp_display in exp_names.items():
        if results[exp_key]:
            e = results[exp_key]
            psnr_diff = e['psnr'] - results['baseline']['psnr'] if results['baseline'] else 0
            psnr_indicator = "🔥" if psnr_diff > 5 else "✨" if psnr_diff > 3 else "📈" if psnr_diff > 0 else ""
            
            print(f"{exp_display:<25} "
                  f"{e['psnr']:<10.2f} "
                  f"{e['ssim']:<10.4f} "
                  f"{e['cer']:<10.4f} "
                  f"{e['wer']:<10.4f} "
                  f"{e['g_loss']:<12.2f} "
                  f"{e['d_loss']:<10.4f} {psnr_indicator}")
    
    print("-" * 97)
    
    # Analysis
    print("\n" + "="*80)
    print("🎯 HYPOTHESIS VERIFICATION")
    print("="*80 + "\n")
    
    if results['baseline'] and results['experiment1']:
        baseline_psnr = results['baseline']['psnr']
        exp1_psnr = results['experiment1']['psnr']
        improvement = exp1_psnr - baseline_psnr
        
        print(f"H1: CTC weight 0.1 akan meningkatkan PSNR 5-7 dB")
        print(f"    Result: {improvement:.2f} dB improvement")
        print(f"    {'✅ PROVEN' if 5 <= improvement <= 8 else '❌ NOT PROVEN'}")
        print()
    
    if results['experiment1'] and results['experiment2']:
        exp1_cer = results['experiment1']['cer']
        exp2_cer = results['experiment2']['cer']
        
        print(f"H2: CTC weight 0.3 memberikan CER lebih baik dari 0.1")
        print(f"    Exp1 CER: {exp1_cer:.4f}, Exp2 CER: {exp2_cer:.4f}")
        print(f"    {'✅ PROVEN' if exp2_cer < exp1_cer else '❌ NOT PROVEN'}")
        print()
    
    if results['baseline'] and results['experiment4']:
        baseline_psnr = results['baseline']['psnr']
        exp4_psnr = results['experiment4']['psnr']
        improvement = exp4_psnr - baseline_psnr
        
        print(f"H3: Kombinasi optimal akan memberikan PSNR terbaik (>25 dB)")
        print(f"    Baseline: {baseline_psnr:.2f} dB, Optimal: {exp4_psnr:.2f} dB")
        print(f"    Improvement: {improvement:.2f} dB")
        print(f"    {'✅ PROVEN' if exp4_psnr >= 25 else '❌ NOT PROVEN (but improvement: {improvement:.2f} dB)'}")
        print()
    
    # Find best configuration
    print("="*80)
    print("🏆 BEST CONFIGURATION")
    print("="*80 + "\n")
    
    best_psnr = -1
    best_exp = None
    
    for exp_name, metrics in results.items():
        if metrics and metrics['psnr'] and metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            best_exp = exp_name
    
    if best_exp:
        print(f"Best PSNR: {best_psnr:.2f} dB from {best_exp}")
        
        if best_exp in ['experiment1', 'experiment2', 'experiment3', 'experiment4']:
            config_file = f"configs/{best_exp}_*.json"
            print(f"\nRecommended for full training:")
            print(f"  Config pattern: {config_file}")
            print(f"  Expected final PSNR (50 epochs): {best_psnr + 5:.1f} - {best_psnr + 10:.1f} dB")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80 + "\n")
    
    if best_psnr >= 25:
        print("✅ Hypothesis PROVEN! CTC annealing strategy works!")
        print("   → Proceed with full 50-epoch training using best config")
        print(f"   → Expected final PSNR: {best_psnr + 8:.1f} - {best_psnr + 12:.1f} dB")
    elif best_psnr >= 22:
        print("⚠️  Partial success. Further tuning needed.")
        print("   → Try longer annealing schedule (CTC 0.05 for first 10 epochs)")
        print("   → Or test multi-stage training approach")
    else:
        print("❌ Hypothesis needs refinement.")
        print("   → Consider architectural changes")
        print("   → Or loss normalization approach")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
