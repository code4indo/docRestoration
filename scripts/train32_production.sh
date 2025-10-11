#!/bin/bash

# Production Training Script for train32.py (Pure FP32 - OPTIMIZED)
# Version: 1.0
# Purpose: Train Dual-Modal GAN-HTR with Pure FP32 for optimal quality

echo "=========================================="
echo "🚀 GAN-HTR Training - Pure FP32 (CLOUD GPU)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • Precision: Pure FP32 (NO mixed precision)"
echo "  • Epochs: 150 (max)"
echo "  • Early stopping: Enabled (patience=15, min_delta=0.01)"
echo "  • Steps per epoch: Auto (dataset size / batch size)"
echo "  • Batch size: 16"
echo "  • Loss weights: Pixel=100, CTC=1, Adversarial=2"
echo "  • Target: PSNR ~40, SSIM ~0.99"
echo ""
echo "Expected Results:"
echo "  ✅ Faster convergence dengan batch size besar"
echo "  ✅ Better gradient estimation"
echo "  ✅ Optimal GPU utilization (>80GB)"
echo "  ✅ Early stopping prevents overfitting"
echo "  ✅ Automatic best model restoration"
echo ""
echo "=========================================="
echo ""

# Activate virtual environment (if using poetry)
# poetry shell

# Run training with optimized parameters
# Note: Data mounted di /workspace (volume), code di /workspace/docRestoration
cd /workspace/docRestoration || exit 1

# Use system Python (dependencies already installed in image)
${PYTHON_BIN:-python3} dual_modal_gan/scripts/train32.py \
  --tfrecord_path /workspace/docRestoration/dual_modal_gan/data/dataset_gan.tfrecord \
  --charset_path /workspace/docRestoration/real_data_preparation/real_data_charlist.txt \
  --recognizer_weights /workspace/docRestoration/models/best_htr_recognizer/best_model.weights.h5 \
  --checkpoint_dir /workspace/docRestoration/dual_modal_gan/outputs/checkpoints_fp32_production \
  --sample_dir /workspace/docRestoration/dual_modal_gan/outputs/samples_fp32_production \
  --gpu_id 1 \
  --epochs 150 \
  --batch_size 32 \
  --lr_g 0.0004 \
  --lr_d 0.0004 \
  --pixel_loss_weight 100.0 \
  --ctc_loss_weight 1.0 \
  --adv_loss_weight 2.0 \
  --gradient_clip_norm 1.0 \
  --ctc_loss_clip_max 1000.0 \
  --eval_interval 1 \
  --save_interval 5 \
  --discriminator_mode predicted \
  --cer_weight 0.5 \
  --early_stopping \
  --patience 15 \
  --min_delta 0.01 \
  --restore_best_weights \
  --seed 42

echo ""
echo "=========================================="
echo "✅ Training completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. View MLflow UI: poetry run mlflow ui"
echo "  2. Check metrics: /workspace/docRestoration/dual_modal_gan/outputs/checkpoints_fp32_production/metrics/"
echo "  3. View samples: /workspace/docRestoration/dual_modal_gan/outputs/samples_fp32_production/"
echo ""
