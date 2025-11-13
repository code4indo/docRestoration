#!/bin/bash
# GradNorm Validation - Quick test (5 epochs, 50 steps/epoch)
# Runtime: ~30 minutes
# Purpose: Verify GradNorm implementation works before full training

set -e

PROJECT_ROOT="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration"
CONFIG_FILE="${PROJECT_ROOT}/configs/gradnorm_validation.json"
LOG_FILE="${PROJECT_ROOT}/logs/gradnorm_validation.log"

echo "========================================="
echo "🎯 GradNorm Validation Run"
echo "========================================="
echo "Config: $CONFIG_FILE"
echo "Log: $LOG_FILE"
echo ""
echo "⏱️  Estimated time: 30 minutes"
echo "📊 Epochs: 5 (validation only)"
echo "🔬 Batch size: 4 (faster iteration)"
echo ""
echo "🎯 What to watch:"
echo "   - GradNorm weights should adapt over time"
echo "   - Loss balance should improve"
echo "   - No NaN or Inf values"
echo ""
echo "Starting in 3 seconds..."
sleep 3

cd "$PROJECT_ROOT"

# Activate virtual environment
source .venv/bin/activate

# Run training in background
nohup poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --config_json "$CONFIG_FILE" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "✅ Training started (PID: $TRAIN_PID)"
echo ""
echo "📋 Monitor with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛑 Stop with:"
echo "   kill $TRAIN_PID"
echo ""
echo "⏳ Waiting 5 seconds to check startup..."
sleep 5

# Check if process is still running
if ps -p $TRAIN_PID > /dev/null; then
    echo "✅ Process running successfully"
    echo ""
    echo "📊 Quick status check:"
    tail -n 20 "$LOG_FILE"
else
    echo "❌ Process died! Check log:"
    tail -n 50 "$LOG_FILE"
    exit 1
fi

echo ""
echo "========================================="
echo "🚀 Validation training in progress"
echo "========================================="
