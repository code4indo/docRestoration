#!/bin/bash
# Launch Production Training V2 - With Range Fix Applied
# Date: 2025-10-21
# Status: CRITICAL BUG FIXED - Ready for clean slate training

echo "================================================================================"
echo "🚀 LAUNCHING PRODUCTION TRAINING V2 - RANGE FIX APPLIED"
echo "================================================================================"
echo ""
echo "⚠️  CRITICAL: All previous training runs are INVALID due to range mismatch bug"
echo "✅ FIX VERIFIED: Generator can now produce black text (pixel range [0,255])"
echo ""
echo "📋 Configuration: configs/full_training_production_v2_range_fixed.json"
echo "   - Generator: enhanced (U-Net with ResBlocks + Attention)"
echo "   - Discriminator: enhanced_v2_fixed (Dual-Modal)"
echo "   - Epochs: 50"
echo "   - Batch Size: 2"
echo "   - GPU: 0"
echo ""
echo "Expected Results:"
echo "   - Validation PSNR: >30 dB"
echo "   - Validation SSIM: >0.95"
echo "   - Text: BLACK (not gray)"
echo "   - CER: <5% at convergence"
echo ""
read -p "Press ENTER to launch training (or Ctrl+C to cancel)..."

# Launch training in background
nohup ./scripts/universal_train_from_json.sh \
  configs/full_training_production_v2_range_fixed.json \
  > /tmp/production_v2_range_fixed.log 2>&1 &

PID=$!

echo ""
echo "✅ Training launched!"
echo "   PID: $PID"
echo "   Log: /tmp/production_v2_range_fixed.log"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f logbook/full_training_production_v2_range_fixed_*.log"
echo ""
echo "🔍 Check samples:"
echo "   ls -lh dual_modal_gan/outputs/samples_full_training_production_v2_range_fixed/"
echo ""
echo "🛑 Stop training:"
echo "   kill $PID"
echo ""
echo "================================================================================"
