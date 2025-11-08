#!/usr/bin/env python3
"""
SKEPTICAL CONFIGURATION AUDIT: DIBCO vs ANRI Successful
Compare DIBCO config with proven ANRI success pattern
"""

import json
import os

def load_config(config_path):
    """Load configuration JSON"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_training_metrics(checkpoint_dir):
    """Load training metrics from successful ANRI checkpoint"""
    metrics_path = os.path.join(checkpoint_dir, "metrics/training_metrics_fp32.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def compare_configs_skeptically():
    """Skeptical comparison between DIBCO and ANRI configs"""
    
    print(f"🔍 SKEPTICAL CONFIGURATION AUDIT")
    print(f"Comparing DIBCO visual-only vs ANRI successful pattern")
    print("=" * 80)
    
    # Load configs
    try:
        dibco_config = load_config('configs/dibco_visual_only_restoration_v1.json')
        anri_config = load_config('configs/anri_finetuning_stage1_adaptation.json')
        anri_metrics = load_training_metrics('dual_modal_gan/checkpoints/anri_finetuning_stage1_full_model_v2')
    except Exception as e:
        print(f"❌ Error loading configs: {e}")
        return
    
    print(f"📊 DIBCO Config: {dibco_config['experiment_name']}")
    print(f"📊 ANRI Config: {anri_config['experiment_name']}")
    
    # Extract key hyperparameters
    dibco_hp = {
        'lr_g': dibco_config.get('lr_g'),
        'lr_d': dibco_config.get('lr_d'),
        'pixel_loss': dibco_config.get('pixel_loss_weight'),
        'adv_loss': dibco_config.get('adv_loss_weight'),
        'perceptual_loss': dibco_config.get('perceptual_loss_weight'),
        'rec_feat_loss': dibco_config.get('rec_feat_loss_weight'),
        'discriminator_mode': dibco_config.get('discriminator_mode'),
        'batch_size': dibco_config.get('batch_size'),
        'epochs': dibco_config.get('epochs'),
        'pretrained': dibco_config.get('pretrained_checkpoint') is not None,
        'dataset_type': 'pure' if 'dibco' in dibco_config.get('tfrecord_path', '') else 'mixed'
    }
    
    anri_hp = {
        'lr_g': anri_config.get('lr_g'),
        'lr_d': anri_config.get('lr_d'),
        'pixel_loss': anri_config.get('pixel_loss_weight'),
        'adv_loss': anri_config.get('adv_loss_weight'),
        'perceptual_loss': anri_config.get('perceptual_loss_weight'),
        'rec_feat_loss': anri_config.get('rec_feat_loss_weight'),
        'discriminator_mode': anri_config.get('discriminator_mode'),
        'batch_size': anri_config.get('batch_size'),
        'epochs': anri_config.get('epochs'),
        'pretrained': anri_config.get('pretrained_checkpoint') is not None,
        'dataset_type': 'mixed'
    }
    
    print(f"\n🔍 CRITICAL COMPARISON:")
    print("-" * 60)
    
    issues = []
    
    # 1. Learning Rate Analysis
    print(f"\n1️⃣ LEARNING RATE COMPARISON:")
    print(f"   ANRI (SUCCESS): lr_g={anri_hp['lr_g']}, lr_d={anri_hp['lr_d']}")
    print(f"   DIBCO (PROPOSED): lr_g={dibco_hp['lr_g']}, lr_d={dibco_hp['lr_d']}")
    
    if dibco_hp['lr_g'] > anri_hp['lr_g']:
        ratio = dibco_hp['lr_g'] / anri_hp['lr_g']
        issues.append(f"Generator LR 2x higher: {ratio:.1f}x (risk: overshooting)")
        print(f"   ⚠️  ISSUE: Generator LR {ratio:.1f}x higher - RISK OF OVERSHOOTING")
    
    if dibco_hp['lr_d'] > anri_hp['lr_d']:
        ratio = dibco_hp['lr_d'] / anri_hp['lr_d']
        issues.append(f"Discriminator LR 2x higher: {ratio:.1f}x (risk: instability)")
        print(f"   ⚠️  ISSUE: Discriminator LR {ratio:.1f}x higher - RISK OF INSTABILITY")
    
    # 2. Loss Weights Analysis
    print(f"\n2️⃣ LOSS WEIGHTS COMPARISON:")
    print(f"   ANRI: pixel={anri_hp['pixel_loss']}, adv={anri_hp['adv_loss']}, perceptual={anri_hp['perceptual_loss']}")
    print(f"   DIBCO: pixel={dibco_hp['pixel_loss']}, adv={dibco_hp['adv_loss']}, perceptual={dibco_hp['perceptual_loss']}")
    
    if dibco_hp['pixel_loss'] > anri_hp['pixel_loss'] * 1.5:
        ratio = dibco_hp['pixel_loss'] / anri_hp['pixel_loss']
        issues.append(f"Pixel loss {ratio:.1f}x higher: may cause overfitting")
        print(f"   ⚠️  ISSUE: Pixel loss {ratio:.1f}x higher - RISK OF OVERFITTING")
    
    if dibco_hp['perceptual_loss'] > anri_hp['perceptual_loss'] * 5:
        ratio = dibco_hp['perceptual_loss'] / max(anri_hp['perceptual_loss'], 1)
        issues.append(f"Perceptual loss {ratio:.1f}x higher: may dominate training")
        print(f"   ⚠️  ISSUE: Perceptual loss {ratio:.1f}x higher - MAY DOMINATE TRAINING")
    
    # 3. Discriminator Mode Analysis
    print(f"\n3️⃣ DISCRIMINATOR MODE COMPARISON:")
    print(f"   ANRI (SUCCESS): {anri_hp['discriminator_mode']}")
    print(f"   DIBCO (PROPOSED): {dibco_hp['discriminator_mode']}")
    
    if dibco_hp['discriminator_mode'] != anri_hp['discriminator_mode']:
        issues.append(f"Different discriminator mode: {dibco_hp['discriminator_mode']} vs {anri_hp['discriminator_mode']}")
        print(f"   ⚠️  ISSUE: Different discriminator mode - UNTESTED COMBINATION")
    
    # 4. Training Strategy Analysis
    print(f"\n4️⃣ TRAINING STRATEGY COMPARISON:")
    print(f"   ANRI: Transfer learning={'YES' if anri_hp['pretrained'] else 'NO'}, Mixed dataset={'YES' if anri_hp['dataset_type']=='mixed' else 'NO'}")
    print(f"   DIBCO: Transfer learning={'YES' if dibco_hp['pretrained'] else 'NO'}, Pure dataset={'YES' if dibco_hp['dataset_type']=='pure' else 'NO'}")
    
    if not dibco_hp['pretrained'] and anri_hp['pretrained']:
        issues.append("No transfer learning: ANRI used transfer learning successfully")
        print(f"   ⚠️  ISSUE: No transfer learning - ANRI succeeded WITH transfer learning")
    
    if dibco_hp['dataset_type'] == 'pure' and anri_hp['dataset_type'] == 'mixed':
        issues.append("Pure dataset: ANRI used mixed dataset for stability")
        print(f"   ⚠️  ISSUE: Pure DIBCO dataset - ANRI used mixed for stability")
    
    # 5. Training Duration
    print(f"\n5️⃣ TRAINING DURATION:")
    print(f"   ANRI (SUCCESS): {anri_hp['epochs']} epochs")
    print(f"   DIBCO (PROPOSED): {dibco_hp['epochs']} epochs")
    
    if dibco_hp['epochs'] > anri_hp['epochs'] * 3:
        issues.append(f"Much longer training: {dibco_hp['epochs']} vs {anri_hp['epochs']} epochs")
        print(f"   ⚠️  ISSUE: {dibco_hp['epochs']} epochs is {dibco_hp['epochs']/anri_hp['epochs']:.1f}x longer - may overfit")
    
    # ANRI Success Metrics (if available)
    if anri_metrics:
        print(f"\n📊 ANRI SUCCESS METRICS:")
        try:
            best_epoch = anri_metrics['epochs'][-1] if anri_metrics['epochs'] else {}
            final_psnr = best_epoch.get('metrics', {}).get('val_psnr', 'N/A')
            print(f"   Final PSNR: {final_psnr}")
            print(f"   Best epoch: {anri_metrics.get('best_epoch', 'N/A')}")
            print(f"   Training time: {anri_metrics.get('hyperparameters', {}).get('epochs', 'N/A')} epochs")
        except:
            print(f"   Metrics not fully available")
    
    # Final Assessment
    print(f"\n" + "=" * 80)
    print(f"🚨 SKEPTICAL ASSESSMENT")
    print(f"=" * 80)
    
    if issues:
        print(f"\n⚠️  IDENTIFIED ISSUES ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. 🔧 REDUCE learning rates to match ANRI success")
        print(f"   2. 🔧 USE discriminator_mode='predicted' (like ANRI)")
        print(f"   3. 🔧 REDUCE loss weights (pixel: 100, perceptual: 15)")
        print(f"   4. 🔧 ADD transfer learning from ANRI checkpoint")
        print(f"   5. 🔧 CONSIDER mixed dataset (70% base + 30% DIBCO)")
        print(f"   6. 🔧 SHORTEN training to 10-15 epochs (like ANRI)")
        
        print(f"\n🎯 IMPROVED CONFIG SUGGESTION:")
        print(f"   lr_g: 5e-6 (vs current 1e-5)")
        print(f"   lr_d: 1e-5 (vs current 2e-5)")
        print(f"   pixel_loss_weight: 100 (vs current 200)")
        print(f"   discriminator_mode: 'predicted' (vs current 'ground_truth')")
        print(f"   pretrained_checkpoint: ANRI checkpoint")
        
        risk_level = "HIGH" if len(issues) >= 4 else "MEDIUM" if len(issues) >= 2 else "LOW"
        print(f"\n🚨 RISK LEVEL: {risk_level}")
        print(f"   Current config may FAIL due to deviation from proven ANRI pattern")
        
    else:
        print(f"\n✅ NO CRITICAL ISSUES FOUND")
        print(f"   Config appears aligned with ANRI success pattern")
    
    return issues

def suggest_conservative_config():
    """Suggest conservative config based on ANRI success"""
    
    print(f"\n" + "="*80)
    print(f"🛠️  CONSERVATIVE CONFIG (Based on ANRI Success)")
    print(f"="*80)
    
    conservative_config = {
        "experiment_name": "dibco_conservative_v1",
        "description": "DIBCO with CONSERVATIVE parameters matching ANRI success pattern",
        
        "generator_version": "enhanced",
        "discriminator_version": "enhanced_v2_fixed",  # Like ANRI
        
        "tfrecord_path": "dual_modal_gan/data/mixed_70base_30dibco.tfrecord",  # Mixed like ANRI
        "charset_path": "real_data_preparation/real_data_charlist.txt",
        "recognizer_weights": "/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5",
        
        "pretrained_checkpoint": "dual_modal_gan/checkpoints/anri_finetuning_stage1_full_model_v2/best_model/ckpt-8",  # Transfer learning!
        "checkpoint_dir": "dual_modal_gan/checkpoints/dibco_conservative_v1",
        "sample_dir": "dual_modal_gan/outputs/samples_dibco_conservative_v1",
        
        "epochs": 10,  # Short like ANRI
        "batch_size": 4,  # Same as ANRI
        "lr_g": 0.000005,  # Same as ANRI
        "lr_d": 0.00001,   # Same as ANRI
        
        "pixel_loss_weight": 100.0,  # Same as ANRI
        "adv_loss_weight": 1.0,
        "rec_feat_loss_weight": 0.0,  # Visual-only but same as ANRI pattern
        "ctc_loss_weight": 0.0,
        "perceptual_loss_weight": 15.0,  # Conservative
        
        "discriminator_mode": "predicted",  # Like ANRI
        
        "early_stopping": {
            "enabled": True,
            "patience": 5,  # Same as ANRI
            "min_delta": 0.01,
            "restore_best_weights": True
        },
        
        "dual_validation": {
            "enabled": True,
            "base_tfrecord": "dual_modal_gan/data/dataset_gan.tfrecord",
            "monitor_base_psnr": True,
            "base_psnr_red_line": 28.0
        }
    }
    
    print(f"✅ Key Changes from Original DIBCO config:")
    print(f"   1. Transfer learning: ANRI checkpoint → DIBCO domain")
    print(f"   2. Mixed dataset: 70% Base + 30% DIBCO (stability)")
    print(f"   3. Conservative LR: 5e-6/1e-5 (match ANRI success)")
    print(f"   4. Discriminator mode: 'predicted' (match ANRI)")
    print(f"   5. Reduced training: 10 epochs (efficient)")
    print(f"   6. Balanced loss weights (match ANRI)")
    
    print(f"\n🎯 Expected Benefits:")
    print(f"   - Follow proven ANRI success pattern")
    print(f"   - Transfer learning from successful model")
    print(f"   - Mixed dataset provides stability")
    print(f"   - Conservative hyperparameters prevent failure")
    
    return conservative_config

def main():
    issues = compare_configs_skeptically()
    conservative_config = suggest_conservative_config()
    
    print(f"\n🚀 FINAL RECOMMENDATION:")
    if issues and len(issues) >= 3:
        print(f"   ⚠️  HIGH RISK: Use CONSERVATIVE config instead of original DIBCO config")
        print(f"   📋 Original config has {len(issues)} issues vs proven ANRI pattern")
        print(f"   🛡️  Conservative config follows ANRI success exactly")
    else:
        print(f"   ✅ Original config may work, but conservative is safer")
    
    return issues

if __name__ == '__main__':
    main()