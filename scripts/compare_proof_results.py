#!/usr/bin/env python3
"""
Compare Ground Truth vs Predicted Mode Results
Shows side-by-side comparison of PSNR, CER, SSIM
"""

import json
import os
from pathlib import Path

def load_metrics(checkpoint_dir):
    """Load epoch_info.json from checkpoint directory"""
    epoch_info_path = Path(checkpoint_dir) / "epoch_info.json"
    if not epoch_info_path.exists():
        return None
    
    with open(epoch_info_path) as f:
        return json.load(f)

def main():
    # Experiment paths
    ground_truth_dir = "dual_modal_gan/checkpoints/exp_proof_ground_truth"
    predicted_dir = "dual_modal_gan/checkpoints/exp_proof_predicted"
    v4_optimal_dir = "dual_modal_gan/checkpoints/production_v4_optimal"
    
    print("=" * 80)
    print("PROOF-OF-CONCEPT RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    # Load metrics
    gt_metrics = load_metrics(ground_truth_dir)
    pred_metrics = load_metrics(predicted_dir)
    v4_metrics = load_metrics(v4_optimal_dir)
    
    if gt_metrics is None:
        print("⚠️  Ground truth experiment not completed yet")
        print(f"   Check: {ground_truth_dir}/epoch_info.json")
        return
    
    # Ground truth results
    print("EXPERIMENT B: GROUND TRUTH MODE (PROPOSED FIX)")
    print("-" * 80)
    print(f"Discriminator mode:  ground_truth (perfect labels)")
    print(f"Epochs completed:    {gt_metrics['last_completed_epoch']}/10")
    print(f"Best epoch:          {gt_metrics['best_epoch']}")
    print(f"Best PSNR:           {gt_metrics['best_psnr']:.2f} dB")
    print(f"Best CER:            {gt_metrics['best_cer']*100:.1f}%")
    print(f"Best SSIM:           {gt_metrics.get('best_ssim', 0):.4f}")
    print(f"Combined score:      {gt_metrics.get('best_combined_score', 0):.2f}")
    print()
    
    # Predicted results (if available)
    if pred_metrics:
        print("EXPERIMENT A: PREDICTED MODE (BASELINE)")
        print("-" * 80)
        print(f"Discriminator mode:  predicted (noisy 33% error)")
        print(f"Epochs completed:    {pred_metrics['last_completed_epoch']}/10")
        print(f"Best epoch:          {pred_metrics['best_epoch']}")
        print(f"Best PSNR:           {pred_metrics['best_psnr']:.2f} dB")
        print(f"Best CER:            {pred_metrics['best_cer']*100:.1f}%")
        print(f"Best SSIM:           {pred_metrics.get('best_ssim', 0):.4f}")
        print(f"Combined score:      {pred_metrics.get('best_combined_score', 0):.2f}")
        print()
        
        # Comparison
        print("ΔIMPROVEMENT (Ground Truth - Predicted)")
        print("-" * 80)
        delta_psnr = gt_metrics['best_psnr'] - pred_metrics['best_psnr']
        delta_cer = (pred_metrics['best_cer'] - gt_metrics['best_cer']) * 100
        delta_combined = gt_metrics.get('best_combined_score', 0) - pred_metrics.get('best_combined_score', 0)
        
        print(f"ΔPSNR:               {delta_psnr:+.2f} dB", end="")
        if delta_psnr >= 1.0:
            print(" ✅ SIGNIFICANT!")
        elif delta_psnr >= 0.5:
            print(" ✓ Moderate")
        else:
            print(" ⚠️  Small")
        
        print(f"ΔCER:                {delta_cer:+.1f}%", end="")
        if abs(delta_cer) >= 5.0:
            print(" ✅ SIGNIFICANT!")
        elif abs(delta_cer) >= 2.0:
            print(" ✓ Moderate")
        else:
            print(" ⚠️  Small")
        
        print(f"ΔCombined:           {delta_combined:+.2f}")
        print()
    else:
        print("EXPERIMENT A: PREDICTED MODE (BASELINE)")
        print("-" * 80)
        print("⏳ Not run yet")
        print("   Run: nohup ./scripts/universal_train_from_json.sh configs/exp_proof_predicted_mode.json &")
        print()
    
    # V4 optimal reference
    if v4_metrics:
        print("REFERENCE: PRODUCTION V4 OPTIMAL (80 epochs)")
        print("-" * 80)
        print(f"Discriminator mode:  predicted (noisy 33% error)")
        print(f"Epochs completed:    {v4_metrics['last_completed_epoch']}/100")
        print(f"Best epoch:          {v4_metrics['best_epoch']}")
        print(f"Best PSNR:           {v4_metrics['best_psnr']:.2f} dB")
        print(f"Best CER:            {v4_metrics['best_cer']*100:.1f}%")
        print(f"Best SSIM:           {v4_metrics.get('best_ssim', 0):.4f}")
        print()
        
        # Comparison with V4
        print("GROUND TRUTH @ EPOCH 10 vs V4 @ EPOCH 80")
        print("-" * 80)
        delta_psnr_v4 = gt_metrics['best_psnr'] - v4_metrics['best_psnr']
        delta_cer_v4 = (v4_metrics['best_cer'] - gt_metrics['best_cer']) * 100
        
        print(f"ΔPSNR:               {delta_psnr_v4:+.2f} dB", end="")
        if delta_psnr_v4 >= 0:
            print(" ✅ Ground truth BETTER even at epoch 10!")
        else:
            print(f" (Ground truth needs {-delta_psnr_v4:.2f} dB to match V4)")
        
        print(f"ΔCER:                {delta_cer_v4:+.1f}%", end="")
        if delta_cer_v4 >= 0:
            print(" ✅ Ground truth BETTER even at epoch 10!")
        else:
            print(f" (Ground truth CER higher, expected for early epochs)")
        print()
    
    # Success criteria
    print("=" * 80)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 80)
    
    success_criteria = []
    
    # Criterion 1: PSNR >= 30 dB
    psnr_ok = gt_metrics['best_psnr'] >= 30.0
    success_criteria.append(psnr_ok)
    print(f"1. PSNR >= 30 dB:          {gt_metrics['best_psnr']:.2f} dB", end="")
    print(" ✅ PASS" if psnr_ok else " ❌ FAIL")
    
    # Criterion 2: Better than predicted (if available)
    if pred_metrics:
        better_than_pred = gt_metrics['best_psnr'] > pred_metrics['best_psnr']
        success_criteria.append(better_than_pred)
        delta_psnr = gt_metrics['best_psnr'] - pred_metrics['best_psnr']
        print(f"2. Better than predicted:  +{delta_psnr:.2f} dB", end="")
        print(" ✅ PASS" if better_than_pred else " ❌ FAIL")
    
    # Criterion 3: Fast convergence (best before epoch 10)
    fast_converge = gt_metrics['best_epoch'] <= 10
    success_criteria.append(fast_converge)
    print(f"3. Fast convergence:       Best @ epoch {gt_metrics['best_epoch']}", end="")
    print(" ✅ PASS" if fast_converge else " ⚠️  Need more epochs")
    
    print()
    
    # Overall verdict
    if all(success_criteria):
        print("🎉 OVERALL VERDICT: HYPOTHESIS CONFIRMED!")
        print("   Ground truth mode shows clear advantage")
        print("   Recommend: Proceed with full 150 epoch training")
        print()
        print("Next step:")
        print("  nohup ./scripts/universal_train_from_json.sh \\")
        print("    configs/production_v5_critical_fix_ground_truth.json &")
    elif any(success_criteria):
        print("⚠️  OVERALL VERDICT: PARTIAL SUCCESS")
        print("   Ground truth mode shows promise but needs tuning")
        print("   Recommend: Analyze results and adjust parameters")
    else:
        print("❌ OVERALL VERDICT: HYPOTHESIS REJECTED")
        print("   Ground truth mode did not show expected improvement")
        print("   Recommend: Debug and investigate root cause")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
