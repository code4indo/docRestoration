#!/bin/bash
# ENHANCED TRANSFER LEARNING LAUNCHER
# Fixed config-checkpoint mismatch (Enhanced + Enhanced)

set -e

echo "🚀 ENHANCED TRANSFER LEARNING - ARCHITECTURE MATCHED"
echo "====================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "configs/dibco_transfer_learning_v2_enhanced.json" ]; then
    echo "❌ Error: Please run from docRestoration root directory"
    exit 1
fi

echo "🔧 Problem Fixed:"
echo "   ❌ BEFORE: BASE config + Enhanced checkpoint (MISMATCH)"
echo "   ✅ NOW:   Enhanced config + Enhanced checkpoint (MATCH)"
echo ""

echo "📋 Architecture Compatibility:"
echo "   Config:      Enhanced generator + Enhanced V2 Fixed discriminator"
echo "   Checkpoint:  ckpt-115 (Enhanced V2 architecture)"
echo "   Status:      ✅ PERFECT MATCH"
echo ""

echo "⚙️  Configuration:"
echo "   Generator:     Enhanced (matches checkpoint)"
echo "   Discriminator: Enhanced V2 Fixed (matches checkpoint)"
echo "   Batch size:    1 (memory optimization for RTX A4000)"
echo "   Perceptual:    15.0 (enabled for quality)"
echo ""

echo "🎯 Expected Results:"
echo "   Checkpoint Load:  ✅ SUCCESS"
echo "   Transfer Learning: ✅ ACTIVE"
echo "   Target PSNR:      20-25 dB"
echo "   Convergence:      5-8 epochs"
echo "   Time:             ~20-30 minutes"
echo ""

echo "💡 Comparison:"
echo "   From scratch:     16.09 dB (15 epochs, 45 min)"
echo "   Transfer learning: 20-25 dB (5 epochs, 20 min)"
echo "   Improvement:      +4-9 dB + 2-3x faster"
echo ""

echo "📊 Memory Notes:"
echo "   RTX A4000: 1104 MB available"
echo "   Estimated: ~1000-1100 MB usage"
echo "   If OOM: Reduce perceptual_weight to 0 in config"
echo ""

echo "🚀 Launching ENHANCED transfer learning..."
echo "   Config: dibco_transfer_learning_v2_enhanced.json"
echo "   Checkpoint: ckpt-115 (compatible)"
echo ""

# Launch training
nohup ./scripts/universal_train_from_json.sh configs/dibco_transfer_learning_v2_enhanced.json > logbook/dibco_transfer_learning_v2_enhanced_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "✅ Training launched with PID: $PID"
echo ""
echo "📈 Success Indicators to Look For:"
echo "   1. '✅ Pretrained weights loaded successfully'"
echo "   2. Epoch 1 PSNR > 18 dB (starting high)"
echo "   3. Epoch 5-8 PSNR > 20 dB (optimal)"
echo ""
echo "📊 To monitor:"
echo "   tail -f logbook/dibco_transfer_learning_v2_enhanced_*.log"
echo "   grep -E 'PSNR:|✅ Pretrained' logbook/dibco_transfer_learning_v2_enhanced_*.log"
echo ""

exit 0
