#!/bin/bash

# PyTorch DDP Multi-GPU Training Launcher
# Usage: ./scripts/train_ddp.sh <config_json>

set -e

CONFIG_FILE="${1:-configs/finetune_thin_stroke_preservation.json}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "  PyTorch DDP Multi-GPU Training"
echo "=========================================="
echo "  Config: $CONFIG_FILE"
echo "  GPUs: 2 (auto-detected)"
echo "=========================================="

# Use torchrun (recommended for PyTorch 1.10+)
poetry run torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    dual_modal_gan/scripts/train_pytorch_ddp.py \
    --config "$CONFIG_FILE" \
    --recognizer_weights /home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5

echo ""
echo "✅ Training completed!"
