#!/usr/bin/env python3
"""
Discriminator Performance Audit Script
Analyze training dynamics to identify discriminator bottlenecks
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_mlflow_metric(metric_file):
    """Read MLflow metric file"""
    values = []
    with open(metric_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamp, value, step = parts[0], parts[1], parts[2] if len(parts) > 2 else 0
                values.append((int(step), float(value)))
    return sorted(values, key=lambda x: x[0])

def analyze_experiment(exp_path):
    """Analyze single MLflow experiment"""
    metrics_dir = exp_path / "metrics"
    if not metrics_dir.exists():
        return None
    
    result = {}
    
    # Read discriminator losses
    disc_loss_file = metrics_dir / "total_disc_loss"
    if disc_loss_file.exists():
        disc_loss_data = read_mlflow_metric(disc_loss_file)
        if disc_loss_data:
            steps, values = zip(*disc_loss_data)
            result['disc_loss'] = {
                'steps': steps,
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'final': values[-1] if values else None,
                'trend': 'decreasing' if values[-1] < values[0] else 'increasing'
            }
    
    # Read generator adversarial loss
    adv_loss_file = metrics_dir / "adversarial_loss"
    if adv_loss_file.exists():
        adv_loss_data = read_mlflow_metric(adv_loss_file)
        if adv_loss_data:
            steps, values = zip(*adv_loss_data)
            result['gen_adv_loss'] = {
                'steps': steps,
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'final': values[-1] if values else None
            }
    
    # Read PSNR for context
    psnr_file = metrics_dir / "val_psnr"
    if psnr_file.exists():
        psnr_data = read_mlflow_metric(psnr_file)
        if psnr_data:
            steps, values = zip(*psnr_data)
            result['psnr'] = {
                'steps': steps,
                'values': values,
                'final': values[-1] if values else None
            }
    
    return result if result else None

def find_recent_experiments(mlruns_dir, n=5):
    """Find N most recent experiments"""
    experiments = []
    
    for exp_dir in Path(mlruns_dir).iterdir():
        if not exp_dir.is_dir() or exp_dir.name == '0':
            continue
        
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            metrics_dir = run_dir / "metrics"
            if metrics_dir.exists():
                # Get last modified time
                mtime = metrics_dir.stat().st_mtime
                experiments.append((mtime, run_dir))
    
    # Sort by modification time and get N most recent
    experiments.sort(reverse=True)
    return [exp[1] for exp in experiments[:n]]

def diagnose_discriminator_health(result):
    """Diagnose discriminator health from metrics"""
    diagnosis = []
    
    if 'disc_loss' not in result:
        return ["⚠️  No discriminator loss data found"]
    
    disc_loss = result['disc_loss']
    
    # Check 1: Discriminator too strong (loss too low)
    if disc_loss['final'] < 0.1:
        diagnosis.append("❌ DISCRIMINATOR TOO STRONG: D-loss = {:.4f} (< 0.1)".format(disc_loss['final']))
        diagnosis.append("   → Generator can't fool discriminator")
        diagnosis.append("   → Generator might give up (mode collapse)")
    
    # Check 2: Discriminator too weak (loss too high)
    elif disc_loss['final'] > 1.0:
        diagnosis.append("❌ DISCRIMINATOR TOO WEAK: D-loss = {:.4f} (> 1.0)".format(disc_loss['final']))
        diagnosis.append("   → Discriminator can't distinguish real vs fake")
        diagnosis.append("   → No useful gradient signal to generator")
    
    # Check 3: Healthy range
    elif 0.3 <= disc_loss['final'] <= 0.7:
        diagnosis.append("✅ DISCRIMINATOR HEALTHY: D-loss = {:.4f} (optimal range)".format(disc_loss['final']))
    
    # Check 4: Marginal
    else:
        diagnosis.append("⚠️  DISCRIMINATOR MARGINAL: D-loss = {:.4f}".format(disc_loss['final']))
    
    # Check loss stability
    if disc_loss['std'] > 0.5:
        diagnosis.append("⚠️  HIGH INSTABILITY: std = {:.4f} (> 0.5)".format(disc_loss['std']))
        diagnosis.append("   → Training is unstable")
    elif disc_loss['std'] < 0.1:
        diagnosis.append("✅ STABLE TRAINING: std = {:.4f} (< 0.1)".format(disc_loss['std']))
    
    # Check generator adversarial loss
    if 'gen_adv_loss' in result:
        gen_adv = result['gen_adv_loss']
        
        # If gen adv loss is very high, gen is struggling
        if gen_adv['final'] > 1.5:
            diagnosis.append("❌ GENERATOR STRUGGLING: Adv-loss = {:.4f} (> 1.5)".format(gen_adv['final']))
            diagnosis.append("   → Generator can't produce realistic outputs")
        elif gen_adv['final'] < 0.3:
            diagnosis.append("⚠️  GENERATOR DOMINATING: Adv-loss = {:.4f} (< 0.3)".format(gen_adv['final']))
            diagnosis.append("   → Discriminator might be too weak")
        else:
            diagnosis.append("✅ GENERATOR COMPETITIVE: Adv-loss = {:.4f}".format(gen_adv['final']))
    
    return diagnosis

def main():
    mlruns_dir = "mlruns"
    
    print("=" * 80)
    print("DISCRIMINATOR PERFORMANCE AUDIT")
    print("=" * 80)
    print()
    
    print("🔍 Searching for recent experiments...")
    recent_exps = find_recent_experiments(mlruns_dir, n=5)
    
    if not recent_exps:
        print("❌ No experiments found in mlruns/")
        return
    
    print(f"✅ Found {len(recent_exps)} recent experiments\n")
    
    all_results = []
    
    for i, exp_path in enumerate(recent_exps, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}: {exp_path.parent.name}/{exp_path.name}")
        print(f"{'='*80}")
        
        result = analyze_experiment(exp_path)
        
        if not result:
            print("⚠️  No discriminator metrics found")
            continue
        
        all_results.append((exp_path, result))
        
        # Print metrics summary
        if 'disc_loss' in result:
            dl = result['disc_loss']
            print(f"\n📊 Discriminator Loss:")
            print(f"   Mean:  {dl['mean']:.4f}")
            print(f"   Std:   {dl['std']:.4f}")
            print(f"   Range: {dl['min']:.4f} - {dl['max']:.4f}")
            print(f"   Final: {dl['final']:.4f}")
            print(f"   Trend: {dl['trend']}")
        
        if 'gen_adv_loss' in result:
            gl = result['gen_adv_loss']
            print(f"\n📊 Generator Adversarial Loss:")
            print(f"   Mean:  {gl['mean']:.4f}")
            print(f"   Std:   {gl['std']:.4f}")
            print(f"   Final: {gl['final']:.4f}")
        
        if 'psnr' in result:
            psnr = result['psnr']
            print(f"\n📊 Validation PSNR:")
            print(f"   Final: {psnr['final']:.2f} dB")
        
        # Diagnosis
        print(f"\n🩺 DIAGNOSIS:")
        diagnosis = diagnose_discriminator_health(result)
        for d in diagnosis:
            print(f"   {d}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    if not all_results:
        print("❌ No valid experiments to analyze")
        return
    
    # Aggregate statistics
    all_disc_finals = [r['disc_loss']['final'] for _, r in all_results if 'disc_loss' in r]
    all_psnr_finals = [r['psnr']['final'] for _, r in all_results if 'psnr' in r]
    
    if all_disc_finals:
        print(f"📈 Discriminator Loss Statistics (across {len(all_disc_finals)} runs):")
        print(f"   Average final D-loss: {np.mean(all_disc_finals):.4f}")
        print(f"   Std deviation:        {np.std(all_disc_finals):.4f}")
        print(f"   Range:                {np.min(all_disc_finals):.4f} - {np.max(all_disc_finals):.4f}")
    
    if all_psnr_finals:
        print(f"\n📈 PSNR Statistics (across {len(all_psnr_finals)} runs):")
        print(f"   Average final PSNR: {np.mean(all_psnr_finals):.2f} dB")
        print(f"   Std deviation:      {np.std(all_psnr_finals):.2f} dB")
        print(f"   Range:              {np.min(all_psnr_finals):.2f} - {np.max(all_psnr_finals):.2f} dB")
    
    # Final recommendation
    print(f"\n🎯 RECOMMENDATION:")
    
    if all_disc_finals:
        avg_d_loss = np.mean(all_disc_finals)
        
        if avg_d_loss < 0.2:
            print("   ❌ CRITICAL: Discriminator is TOO STRONG")
            print("   → Reduce discriminator capacity (fewer layers, filters)")
            print("   → Increase generator capacity")
            print("   → Use one-sided label smoothing (0.9 instead of 1.0 for real)")
            print("   → Add noise to discriminator inputs")
        elif avg_d_loss > 0.8:
            print("   ❌ CRITICAL: Discriminator is TOO WEAK")
            print("   → Increase discriminator capacity")
            print("   → Add residual connections")
            print("   → Improve attention mechanism")
            print("   → Use spectral normalization")
        else:
            print("   ✅ Discriminator balance is ACCEPTABLE")
            print("   → Focus on architectural improvements for both G and D")
            print("   → Current design: Simple CNN + LSTM (137M params)")
            print("   → Potential improvements:")
            print("      • Add ResNet blocks to image branch")
            print("      • Use Bidirectional LSTM (256 units)")
            print("      • Add cross-modal attention")
            print("      • Reduce parameters to ~50-70M")
    
    print("\n✅ Audit completed!")

if __name__ == "__main__":
    main()
