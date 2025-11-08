#!/bin/bash
# QUICK FIX FOR DIBCO TRANSFER LEARNING
# Fix shape mismatch by using compatible checkpoint

set -e

echo "🔧 DIBCO TRANSFER LEARNING FIX"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "configs/dibco_transfer_learning_v2_compatible.json" ]; then
    echo "❌ Error: Please run from docRestoration root directory"
    exit 1
fi

echo "📋 Problem:"
echo "   ckpt-115 has BatchNorm shape mismatch (64 vs 512 channels)"
echo "   Transfer learning fails → must train from scratch"
echo ""

echo "✅ Solution:"
echo "   Use ckpt-99 instead (BASE architecture compatible)"
echo "   Expected: Transfer learning works + 18-20 dB PSNR"
echo ""

echo "🚀 Launching FIXED training..."
echo "   Config: dibco_transfer_learning_v2_compatible.json"
echo "   Checkpoint: ckpt-99 (compatible)"
echo "   Expected: 5-8 epochs to optimal"
echo ""

# Launch training
nohup ./scripts/universal_train_from_json.sh configs/dibco_transfer_learning_v2_compatible.json > logbook/dibco_transfer_learning_v2_compatible_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "✅ Training launched with PID: $PID"
echo ""
echo "📊 To monitor progress:"
echo "   tail -f logbook/dibco_transfer_learning_v2_compatible_*.log"
echo ""
echo "🎯 Success Criteria:"
echo "   Look for: '✅ Pretrained weights loaded successfully'"
echo "   Expected: PSNR > 18 dB in epoch 5-8"
echo ""

exit 0
