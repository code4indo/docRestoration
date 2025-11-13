#!/bin/bash
# GradNorm Production - Full training (50 epochs, full dataset)
# Runtime: ~8-10 hours
# Purpose: Final training with GradNorm adaptive loss balancing

set -e

PROJECT_ROOT="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration"
CONFIG_FILE="${PROJECT_ROOT}/configs/gradnorm_production.json"
LOG_FILE="${PROJECT_ROOT}/logs/gradnorm_production.log"

echo "========================================="
echo "🚀 GradNorm PRODUCTION Training"
echo "========================================="
echo "Config: $CONFIG_FILE"
echo "Log: $LOG_FILE"
echo ""
echo "⚠️  WARNING: This is FULL production training"
echo "⏱️  Estimated time: 8-10 hours"
echo "💰 Computational cost: HIGH"
echo "📊 Epochs: 50"
echo "🔬 Batch size: 2 (production quality)"
echo ""
echo "🎯 Expected benefits:"
echo "   ✓ Automatic loss weight optimization"
echo "   ✓ Better convergence than static weights"
echo "   ✓ Novel contribution to GAN-HTR field"
echo ""
echo "⚠️  Only proceed if validation was successful!"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Aborted by user"
    exit 0
fi

echo ""
echo "Starting in 5 seconds..."
sleep 5

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
echo "🚀 Production training in progress"
echo "========================================="
echo ""
echo "💡 Tips:"
echo "   - Monitor GradNorm weight evolution in logs"
echo "   - Check MLflow UI for detailed metrics"
echo "   - Compare with production_v3 baseline"
