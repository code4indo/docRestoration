#!/bin/bash
# PURE DIBCO TRAINING - SOTA COMPARISON
# Prove Enhanced V2 Architecture Superiority over DocEnTR, DE-GAN, Souibgui

set -e

echo "🎯 PURE DIBCO TRAINING - BEAT SOTA CHALLENGE"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "configs/dibco_pure_sota_comparison_v1.json" ]; then
    echo "❌ Error: Please run from docRestoration root directory"
    exit 1
fi

echo "📊 TRAINING STRATEGY:"
echo "   Methodology:   Pure DIBCO leave-one-out (FAIR comparison)"
echo "   Training data: DIBCO 2009-2018 (exclude 2012, 256 samples)"
echo "   Test data:     DIBCO 2012 (14 images, held-out)"
echo "   Pretrained:    Production V3 ckpt-88 (31.5 dB synthetic)"
echo ""

echo "🏆 SOTA TARGETS TO BEAT:"
echo "   DocEnTR:       22.29 dB PSNR, 95.31% F-Measure"
echo "   DE-GAN:        22.00 dB PSNR, 95.18% F-Measure"
echo "   Souibgui:      ~22-23 dB PSNR (H-DIBCO winner)"
echo ""

echo "✨ OUR ARCHITECTURE ADVANTAGES:"
echo "   1. Dual-Modal Discriminator (visual + HTR readability)"
echo "   2. Cross-Modal Attention (visual-text correlation)"
echo "   3. HTR-Oriented Loss (RecFeat guides readable restoration)"
echo "   4. Enhanced V2 Fixed (spatial attention + residuals)"
echo "   5. Perceptual Loss (VGG-based detail preservation)"
echo ""

echo "📈 EXPECTED RESULTS:"
echo "   Zero-shot baseline:  16.46 dB (ckpt-88 without DIBCO)"
echo "   After fine-tuning:   20-23 dB (target >22.5 dB)"
echo "   Improvement:         +6 dB expected"
echo "   Training time:       2-3 hours (10-15 epochs)"
echo ""

echo "🎯 SUCCESS CRITERIA:"
echo "   ✅ Minimum:  PSNR ≥ 20 dB (clear improvement)"
echo "   ✅ Target:   PSNR ≥ 22 dB (match DE-GAN)"
echo "   🏆 Superior: PSNR > 22.5 dB (BEAT ALL SOTA!)"
echo ""

echo "⚙️  CONFIGURATION:"
echo "   Config:        dibco_pure_sota_comparison_v1.json"
echo "   Architecture:  Enhanced + Enhanced V2 Fixed"
echo "   Learning Rate: 1e-6 (G), 5e-6 (D) - conservative"
echo "   Batch Size:    2 (stable gradients)"
echo "   Epochs:        20 max (early stop patience 5)"
echo "   Visual Mode:   CTC=0, RecFeat=10, Perceptual=15"
echo ""

echo "🔍 MONITORING:"
echo "   Watch training:  tail -f logbook/dibco_pure_sota_comparison_v1_*.log"
echo "   Check PSNR:      grep 'Validation PSNR' logbook/dibco_pure_sota_comparison_v1_*.log"
echo "   Best model:      dual_modal_gan/checkpoints/dibco_pure_sota_comparison_v1/best_model/"
echo ""

echo "📊 VALIDATION TRACKING:"
echo "   Every epoch: PSNR/SSIM on DIBCO validation set (15% of 256 samples)"
echo "   Best model:  Saved automatically (highest PSNR)"
echo "   Early stop:  If no improvement for 5 epochs"
echo ""

# Confirm execution
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Ready to launch PURE DIBCO training?"
echo "   This will prove Enhanced V2 superiority over SOTA!"
echo ""
read -p "Press ENTER to start training, or Ctrl+C to cancel..."
echo ""

echo "🚀 Launching PURE DIBCO training..."
echo ""

# Launch training
nohup ./scripts/universal_train_from_json.sh configs/dibco_pure_sota_comparison_v1.json > logbook/dibco_pure_sota_comparison_v1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "✅ Training launched with PID: $PID"
echo ""

# Wait a moment for log file to be created
sleep 3

# Get the latest log file
LOGFILE=$(ls -t logbook/dibco_pure_sota_comparison_v1_*.log 2>/dev/null | head -1)

if [ -n "$LOGFILE" ]; then
    echo "📋 Log file: $LOGFILE"
    echo ""
    echo "🔍 Showing initial training output..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -30 "$LOGFILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📊 To monitor progress:"
    echo "   tail -f $LOGFILE"
    echo ""
    echo "🔍 To check PSNR progression:"
    echo "   grep -E 'Epoch.*PSNR|Validation PSNR' $LOGFILE"
    echo ""
    echo "🎯 Target milestones to watch:"
    echo "   Epoch 5:  PSNR ~19-20 dB (early adaptation)"
    echo "   Epoch 10: PSNR ~21-22 dB (convergence zone)"
    echo "   Epoch 15: PSNR ~22-23 dB (peak performance)"
    echo ""
    echo "🏆 SOTA BEAT WATCH:"
    echo "   When PSNR > 22.00 dB → ✅ Beat DE-GAN!"
    echo "   When PSNR > 22.29 dB → 🎉 Beat DocEnTR!"
    echo "   When PSNR > 22.50 dB → 🏆 NEW SOTA CHAMPION!"
    echo ""
fi

echo "✅ Training is running in background."
echo "   Process will continue even if you close this terminal."
echo ""
echo "📝 Next steps after training completes:"
echo "   1. Check best checkpoint: dual_modal_gan/checkpoints/dibco_pure_sota_comparison_v1/best_model/"
echo "   2. Run inference on DIBCO 2012 test set (14 images)"
echo "   3. Calculate PSNR/SSIM/F-Measure metrics"
echo "   4. Compare with SOTA baselines"
echo "   5. Prepare results for publication 📄"
echo ""

exit 0
