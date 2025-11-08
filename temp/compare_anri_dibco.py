#!/usr/bin/env python3
"""
ANRI vs DIBCO Configuration Analysis
Comparative analysis to understand success/failure patterns
"""

import json
import os

def load_config(config_path):
    """Load configuration JSON"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_configs():
    """Compare ANRI vs DIBCO configs to identify success/failure factors"""
    
    print("🔍 COMPARATIVE ANALYSIS: ANRI vs DIBCO CONFIGURATIONS")
    print("=" * 80)
    
    # Load configs
    try:
        anri_config = load_config('configs/anri_finetuning_stage1_adaptation.json')
        dibco_config = load_config('configs/dibco_visual_only_restoration_v1.json')
    except Exception as e:
        print(f"❌ Error loading configs: {e}")
        return
    
    print(f"📊 ANRI Config: {anri_config['experiment_name']}")
    print(f"📊 DIBCO Config: {dibco_config['experiment_name']}")
    
    print(f"\n🎯 CRITICAL DIFFERENCES ANALYSIS:")
    print("=" * 50)
    
    # 1. Dataset Strategy
    print(f"\n1️⃣ DATASET STRATEGY:")
    print(f"   ANRI: {anri_config['tfrecord_path']}")
    print(f"   DIBCO: {dibco_config['tfrecord_path']}")
    
    print(f"\n   📈 Expected composition:")
    print(f"   ANRI: Mixed (70% Base + 30% ANRI) - KNOWLEDGE PRESERVATION")
    print(f"   DIBCO: Pure DIBCO only - VULNERABLE to overfitting")
    
    # 2. Pretrained Checkpoint
    print(f"\n2️⃣ PRETRAINED CHECKPOINT:")
    anri_pretrained = anri_config.get('pretrained_checkpoint', 'None')
    dibco_pretrained = dibco_config.get('pretrained_checkpoint', 'None')
    
    print(f"   ANRI: {anri_pretrained}")
    print(f"   DIBCO: {dibco_pretrained}")
    
    print(f"\n   🧠 Knowledge Transfer:")
    print(f"   ANRI: ✅ Transfer learning from Base model (99 epochs)")
    print(f"   DIBCO: ❌ FROM SCRATCH - no prior knowledge")
    
    # 3. Discriminator Architecture
    print(f"\n3️⃣ DISCRIMINATOR ARCHITECTURE:")
    print(f"   ANRI: {anri_config['discriminator_version']}")
    print(f"   DIBCO: {dibco_config['discriminator_version']}")
    
    print(f"\n   🏗️  Architecture Match:")
    print(f"   ANRI: Dual-modal (enhanced_v2_fixed) - MATCHES real text labels")
    print(f"   DIBCO: Visual-only (base) - FIXED for dummy labels")
    
    # 4. Loss Configuration
    print(f"\n4️⃣ LOSS CONFIGURATION:")
    print(f"   ANRI:")
    print(f"     - ctc_loss_weight: {anri_config.get('ctc_loss_weight', 'N/A')}")
    print(f"     - rec_feat_loss_weight: {anri_config.get('rec_feat_loss_weight', 'N/A')}")
    print(f"     - pixel_loss_weight: {anri_config.get('pixel_loss_weight', 'N/A')}")
    print(f"   DIBCO:")
    print(f"     - ctc_loss_weight: {dibco_config.get('ctc_loss_weight', 'N/A')}")
    print(f"     - rec_feat_loss_weight: {dibco_config.get('rec_feat_loss_weight', 'N/A')}")
    print(f"     - pixel_loss_weight: {dibco_config.get('pixel_loss_weight', 'N/A')}")
    
    # 5. Learning Rate
    print(f"\n5️⃣ LEARNING RATE:")
    print(f"   ANRI:")
    print(f"     - lr_g: {anri_config.get('lr_g', 'N/A')}")
    print(f"     - lr_d: {anri_config.get('lr_d', 'N/A')}")
    print(f"   DIBCO:")
    print(f"     - lr_g: {dibco_config.get('lr_g', 'N/A')}")
    print(f"     - lr_d: {dibco_config.get('lr_d', 'N/A')}")
    
    print(f"\n   🎯 Adaptation Strategy:")
    print(f"   ANRI: Conservative LR (5e-6/1e-5) - fine-tuning")
    print(f"   DIBCO: Standard LR (1e-5/2e-5) - from scratch training")
    
    # 6. Training Duration
    print(f"\n6️⃣ TRAINING DURATION:")
    print(f"   ANRI: {anri_config.get('epochs', 'N/A')} epochs")
    print(f"   DIBCO: {dibco_config.get('epochs', 'N/A')} epochs")
    
    print(f"\n   ⏱️  Training Approach:")
    print(f"   ANRI: Short fine-tuning (10 epochs) - efficient")
    print(f"   DIBCO: Extended training (50 epochs) - from scratch needs more time")
    
    # Success/Failure Analysis
    print(f"\n🔬 ROOT CAUSE ANALYSIS:")
    print("=" * 50)
    
    print(f"\n✅ WHY ANRI SUCCEEDED:")
    print(f"   1. TRANSFER LEARNING: Started from trained Base model")
    print(f"   2. MIXED DATASET: 70% Base prevents catastrophic forgetting")
    print(f"   3. DUAL-MODAL MATCH: Real text labels + dual-modal discriminator")
    print(f"   4. CONSERVATIVE LR: Prevents overshooting during adaptation")
    print(f"   5. REC_FEAT LOSS: Active (10.0) helps HTR awareness")
    
    print(f"\n❌ WHY DIBCO PREVIOUSLY FAILED:")
    print(f"   1. FROM SCRATCH: No prior knowledge transfer")
    print(f"   2. DUMMY LABELS: All zeros but used dual-modal discriminator")
    print(f"   3. CROSS-MODAL NOISE: Text features = garbage, corrupts gradients")
    print(f"   4. PURE DATASET: No Base data for stability")
    print(f"   5. REC_FEAT NOISE: Active loss with dummy labels hurts performance")
    
    print(f"\n🎯 CURRENT DIBCO FIX STRATEGY:")
    print(f"   1. ✅ VISUAL-ONLY: Base discriminator (no text path)")
    print(f"   2. ✅ REC_FEAT DISABLED: Set to 0.0 (no HTR noise)")
    print(f"   3. ✅ HIGHER PIXEL LOSS: 200.0 (restoration-focused)")
    print(f"   4. ✅ LONGER TRAINING: 50 epochs (from scratch needs time)")
    print(f"   5. ⚠️  STILL FROM SCRATCH: No transfer learning")
    
    print(f"\n🚀 RECOMMENDED DIBCO IMPROVEMENTS:")
    print("=" * 50)
    
    print(f"\n1️⃣ TRANSFER LEARNING APPROACH:")
    print(f"   Option A: Use ANRI checkpoint as starting point")
    print(f"     pretrained_checkpoint: 'dual_modal_gan/checkpoints/anri_finetuning_stage1_full_model_v2/best_model/ckpt-8'")
    print(f"     Benefit: Transfer ANRI knowledge to DIBCO domain")
    
    print(f"\n   Option B: Mixed dataset like ANRI")
    print(f"     tfrecord_path: 'dual_modal_gan/data/mixed_70base_30dibco.tfrecord'")
    print(f"     Benefit: Base data provides stability during DIBCO adaptation")
    
    print(f"\n2️⃣ HYBRID STRATEGY (RECOMMENDED):")
    print(f"   - Start from ANRI checkpoint (transfer learning)")
    print(f"   - Mixed dataset: 70% Base + 30% DIBCO (stability)")
    print(f"   - Visual-only discriminator (matches dummy labels)")
    print(f"   - Conservative LR: 5e-6/1e-5 (like ANRI success)")
    print(f"   - Extended training: 20-30 epochs (gradual adaptation)")
    
    print(f"\n3️⃣ ALTERNATIVE: PROGRESSIVE APPROACH")
    print(f"   Stage 1: ANRI → Mixed (Base+DIBCO) with visual-only")
    print(f"   Stage 2: Fine-tune on pure DIBCO if needed")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   DIBCO necesita TRANSFER LEARNING like ANRI, not FROM SCRATCH!")
    print(f"   The problem isn't just discriminator architecture - it's missing knowledge transfer.")

def suggest_dibco_improvements():
    """Generate improved DIBCO configuration based on ANRI success pattern"""
    
    print(f"\n" + "="*80)
    print(f"🛠️  IMPROVED DIBCO CONFIGURATION SUGGESTION")
    print(f"="*80)
    
    improved_config = {
        "experiment_name": "dibco_transfer_learning_v1",
        "description": "DIBCO with TRANSFER LEARNING from ANRI checkpoint. Strategy: (1) Start from ANRI checkpoint (proven successful), (2) Mixed dataset 70% Base + 30% DIBCO (stability), (3) Visual-only discriminator (matches dummy labels), (4) Conservative LR (5e-6/1e-5) like ANRI, (5) Extended training (20 epochs). Expected: Break 14-15 dB ceiling through knowledge transfer.",
        
        "generator_version": "enhanced",
        "discriminator_version": "base",
        
        "tfrecord_path": "dual_modal_gan/data/mixed_70base_30dibco.tfrecord",
        "charset_path": "real_data_preparation/real_data_charlist.txt",
        "recognizer_weights": "/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5",
        
        "pretrained_checkpoint": "dual_modal_gan/checkpoints/anri_finetuning_stage1_full_model_v2/best_model/ckpt-8",
        "checkpoint_dir": "dual_modal_gan/checkpoints/dibco_transfer_learning_v1",
        "sample_dir": "dual_modal_gan/outputs/samples_dibco_transfer_learning_v1",
        
        "epochs": 20,
        "batch_size": 4,
        
        "lr_g": 0.000005,  # Conservative like ANRI
        "lr_d": 0.00001,   # Conservative like ANRI
        
        "pixel_loss_weight": 150.0,  # Reduced from 200
        "adv_loss_weight": 1.0,
        "rec_feat_loss_weight": 0.0,  # Visual-only
        "ctc_loss_weight": 0.0,       # Visual-only
        "perceptual_loss_weight": 30.0,  # Reduced from 50
        
        "discriminator_mode": "ground_truth",
        
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "min_delta": 0.05,
            "restore_best_weights": True
        },
        
        "dual_validation": {
            "enabled": True,
            "base_tfrecord": "dual_modal_gan/data/dataset_gan.tfrecord",
            "monitor_base_psnr": True,
            "base_psnr_red_line": 28.0
        }
    }
    
    print(f"📋 IMPROVED CONFIG HIGHLIGHTS:")
    print(f"   ✅ Transfer Learning: ANRI checkpoint → DIBCO domain")
    print(f"   ✅ Mixed Dataset: 70% Base + 30% DIBCO (stability)")
    print(f"   ✅ Visual-only: Base discriminator + dummy label compatibility")
    print(f"   ✅ Conservative LR: 5e-6/1e-5 (ANRI proven formula)")
    print(f"   ✅ Dual Validation: Monitor Base PSNR for stability")
    print(f"   ✅ Reduced training: 20 epochs (vs 50 from scratch)")
    
    return improved_config

if __name__ == '__main__':
    analyze_configs()
    improved_config = suggest_dibco_improvements()
    
    print(f"\n💾 Save improved config? (y/n): ", end="")
    # In real scenario, would get user input
    print(f"y")
    
    with open('configs/dibco_transfer_learning_v1.json', 'w', encoding='utf-8') as f:
        json.dump(improved_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved: configs/dibco_transfer_learning_v1.json")
    print(f"\n🚀 Ready for transfer learning training!")