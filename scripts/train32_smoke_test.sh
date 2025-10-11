#!/bin/bash

# Smoke Test Script for train32.py (Pure FP32)
# Quick validation test (2 epochs) before full training

echo "=========================================="
echo "🧪 GAN-HTR Smoke Test - Pure FP32 (CLOUD GPU)"
echo "=========================================="
echo ""
echo "Purpose: Quick validation (2 epochs)"
echo "  • Fast iteration for testing changes"
echo "  • Validate convergence pattern"
echo "  • Check for errors before full run"
echo "  • Early stopping enabled (patience=3)"
echo ""
echo "========================================="
echo ""

# Ensure we're in the correct directory
cd /workspace/docRestoration || exit 1

# Use system Python (dependencies already installed in image)
# Output redirected to logfile untuk monitoring
LOG_FILE="logbook/smoke_test_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Training output will be saved to: $LOG_FILE"
echo ""

${PYTHON_BIN:-python3} dual_modal_gan/scripts/train32.py \
  --tfrecord_path /workspace/docRestoration/dual_modal_gan/data/dataset_gan.tfrecord \
  --charset_path /workspace/docRestoration/real_data_preparation/real_data_charlist.txt \
  --recognizer_weights /workspace/docRestoration/models/best_htr_recognizer/best_model.weights.h5 \
  --checkpoint_dir /workspace/docRestoration/dual_modal_gan/outputs/checkpoints_fp32_smoke_test \
  --sample_dir /workspace/docRestoration/dual_modal_gan/outputs/samples_fp32_smoke_test \
  --gpu_id 1 \
  --epochs 5 \
  --steps_per_epoch 200 \
  --batch_size 4 \
  --lr_g 0.0002 \
  --lr_d 0.0002 \
  --pixel_loss_weight 1000.0 \
  --ctc_loss_weight 15.0 \
  --adv_loss_weight 1.0 \
  --gradient_clip_norm 1.0 \
  --ctc_loss_clip_max 300.0 \
  --eval_interval 5 \
  --save_interval 5 \
  --discriminator_mode ground_truth \
  --no_restore \
  --early_stopping \
  --patience 3 \
  --min_delta 0.01 \
  --restore_best_weights \
  --seed 42 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "✅ Smoke test completed!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  Metrics: /workspace/docRestoration/dual_modal_gan/outputs/checkpoints_fp32_smoke_test/metrics/training_metrics_fp32.json"
echo "  Samples: /workspace/docRestoration/dual_modal_gan/outputs/samples_fp32_smoke_test/"
echo ""
