#!/bin/bash

# Production Training Script for train32.py (Pure FP32 - OPTIMIZED)
# Version: 1.0
# Purpose: Train Dual-Modal GAN-HTR with Pure FP32 for optimal quality

echo "=========================================="
echo "🚀 GAN-HTR Training - Pure FP32 (OPTIMIZED)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • Precision: Pure FP32 (NO mixed precision)"
echo "  • Epochs: 100"
echo "  • Steps per epoch: 100"
echo "  • Batch size: 4"
echo "  • Loss weights: Pixel=100, CTC=1, Adversarial=2"
echo "  • Target: PSNR ~40, SSIM ~0.99"
echo ""
echo "Expected Results:"
echo "  ✅ Balanced loss convergence"
echo "  ✅ Stable gradient flow"
echo "  ✅ Superior visual quality vs FP16"
echo ""
echo "=========================================="
echo ""

# Activate virtual environment (if using poetry)
# poetry shell

# Run training with optimized parameters
# Note: Data mounted di /workspace (volume), code di /workspace/docRestoration
cd "$(dirname "$0")/.." || exit 1

# Use system Python (dependencies already installed in image)
${PYTHON_BIN:-python3} dual_modal_gan/scripts/train32.py \
  --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
  --charset_path real_data_preparation/real_data_charlist.txt \
  --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
  --checkpoint_dir dual_modal_gan/outputs/checkpoints_fp32_production \
  --sample_dir dual_modal_gan/outputs/samples_fp32_production \
  --gpu_id 1 \
  --epochs 100 \
  --steps_per_epoch 100 \
  --batch_size 4 \
  --lr_g 0.0002 \
  --lr_d 0.0002 \
  --pixel_loss_weight 100.0 \
  --ctc_loss_weight 1.0 \
  --adv_loss_weight 2.0 \
  --gradient_clip_norm 1.0 \
  --ctc_loss_clip_max 1000.0 \
  --eval_interval 1 \
  --save_interval 5 \
  --discriminator_mode predicted \
  --cer_weight 0.5 \
  --seed 42

echo ""
echo "=========================================="
echo "✅ Training completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. View MLflow UI: poetry run mlflow ui"
echo "  2. Check metrics: dual_modal_gan/outputs/checkpoints_fp32_production/metrics/"
echo "  3. View samples: dual_modal_gan/outputs/samples_fp32_production/"
echo ""
