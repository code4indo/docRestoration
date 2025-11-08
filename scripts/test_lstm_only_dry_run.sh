#!/bin/bash
# Quick test untuk verify LSTM-only discriminator integration ke train_enhanced.py
# Hanya test 1 epoch untuk speed

set -e  # Exit on error

echo "=================================================="
echo "LSTM-ONLY DISCRIMINATOR - DRY RUN TEST"
echo "Testing integration dengan train_enhanced.py"
echo "=================================================="

cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration

# Test dengan 1 epoch only untuk verifikasi build
/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/.venv/bin/python \
    dual_modal_gan/scripts/train_enhanced.py \
    --config configs/h2a_lstm_only_experiment.json \
    --epochs 1 \
    --batch_size 2 \
    --save_interval 1 \
    --eval_interval 1 \
    2>&1 | tee /tmp/lstm_only_dry_run.log

echo ""
echo "✅ Dry run completed! Check /tmp/lstm_only_dry_run.log"
echo "If successful, ready to launch full 50-epoch training"
