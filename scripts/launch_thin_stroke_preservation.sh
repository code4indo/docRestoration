#!/bin/bash
# Launcher for thin_stroke_preservation_clean_slate training
# Based on production_v3 config with optimized loss weights for thin stroke preservation
# 
# CRITICAL CHANGES:
# - pixel_loss: 50 → 100 (+2x preserve detail)
# - adv_loss: 3.0 → 1.0 (-3x reduce aggression)  
# - perceptual_loss: 1.0 → 2.0 (+2x maintain structure)
#
# Target: >75% thin stroke preservation (baseline: 43.4%)

CONFIG_FILE="configs/thin_stroke_preservation_clean_slate.json"

echo "🚀 Launching Thin Stroke Preservation Training (Clean Slate)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Config: $CONFIG_FILE"
echo "🎯 Objective: Preserve >75% thin strokes (baseline: 43.4%)"
echo "⚙️  Loss weights: pixel=100, adv=1.0, perceptual=2.0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract config values
CHECKPOINT_DIR="dual_modal_gan/checkpoints/thin_stroke_preservation_clean_slate"
SAMPLE_DIR="dual_modal_gan/outputs/samples_thin_stroke_preservation_clean_slate"
LOG_FILE="logbook/thin_stroke_preservation_clean_slate_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$SAMPLE_DIR"
mkdir -p "logbook"

# Launch training in background
nohup poetry run python dual_modal_gan/scripts/train_enhanced_BACKUP_20251025_181809.py \
  --gpu_id "0" \
  --batch_size 2 \
  --epochs 50 \
  --steps_per_epoch 0 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --tfrecord_path "dual_modal_gan/data/dataset_gan.tfrecord" \
  --charset_path "real_data_preparation/real_data_charlist.txt" \
  --recognizer_weights "/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5" \
  --sample_dir "$SAMPLE_DIR" \
  --generator_version enhanced \
  --discriminator_version enhanced_v2_fixed \
  --pixel_loss_weight 100.0 \
  --adv_loss_weight 1.0 \
  --perceptual_loss_weight 2.0 \
  --rec_feat_loss_weight 8.0 \
  --ctc_loss_weight 0.15 \
  --no_restore \
  --save_best_model_separately \
  --early_stopping \
  --curriculum_aware_early_stopping \
  --patience 25 \
  --min_delta 0.05 \
  --eval_interval 1 \
  --save_interval 2 \
  --use_lr_schedule \
  --warmup_epochs 10 \
  --annealing_epochs 20 \
  --adaptive_loss_balancing \
  --target_ctc_ratio 0.40 \
  --target_visual_ratio 0.60 \
  --adaptation_rate 0.08 \
  --train_split 0.7 \
  --val_split 0.15 \
  --seed 42 > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo ""
echo "✅ Training launched successfully!"
echo "   PID: $TRAIN_PID"
echo "   Log: $LOG_FILE"
echo "   Checkpoints: $CHECKPOINT_DIR"
echo "   Samples: $SAMPLE_DIR"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🔍 Check GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "🛑 Stop training:"
echo "   kill $TRAIN_PID"
echo ""
