#!/bin/bash

# Quick dry-run test for joint training script
# Test 1 step only to verify no errors

set -e

cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration

echo "=================================================="
echo "Joint Training Script - DRY RUN TEST"
echo "=================================================="
echo ""
echo "Testing script with 1 epoch, 2 steps..."
echo ""

# Create temporary config with minimal settings
cat > /tmp/test_joint_config.json << 'EOF'
{
  "tfrecord_path": "dual_modal_gan/data/dataset_gan.tfrecord",
  "charset_path": "real_data_preparation/real_data_charlist.txt",
  "recognizer_weights": "/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5",
  "batch_size": 2,
  "epochs": 1,
  "steps_per_epoch": 2,
  "gpu_id": "1",
  "checkpoint_dir": "/tmp/test_joint_ckpt",
  "sample_dir": "/tmp/test_joint_samples",
  "train_split": 0.7,
  "val_split": 0.15,
  "pixel_loss_weight": 50.0,
  "adv_loss_weight": 3.0,
  "ctc_loss_weight": 1.0,
  "lr_g": 0.0001,
  "lr_d": 0.0001,
  "joint_training_config": {
    "enabled": true,
    "lr_recognizer": 0.0003,
    "baseline_cer": 33.72
  }
}
EOF

mkdir -p /tmp/test_joint_ckpt /tmp/test_joint_samples

poetry run python dual_modal_gan/scripts/train_joint_ablation.py \
    --config /tmp/test_joint_config.json

echo ""
echo "=================================================="
echo "✅ DRY RUN SUCCESSFUL!"
echo "=================================================="
echo "Ready to launch full training with:"
echo "  ./scripts/launch_joint_ablation.sh"
