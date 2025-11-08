#!/bin/bash
# MONITOR PURE DIBCO TRAINING FROM SCRATCH
# Real-time progress tracking for SOTA comparison

LOGFILE=$(ls -t logbook/dibco_pure_sota_from_scratch_*.log 2>/dev/null | head -1)

if [ -z "$LOGFILE" ]; then
    echo "❌ No training log found"
    exit 1
fi

echo "📊 PURE DIBCO TRAINING - FROM SCRATCH MONITOR"
echo "=============================================="
echo "Log: $LOGFILE"
echo ""

echo "🎯 GOAL: Beat SOTA with Enhanced V2 Architecture"
echo "   Target: >22.5 dB PSNR on DIBCO 2012"
echo "   SOTA Baselines: DocEnTR (22.29 dB), DE-GAN (22.00 dB)"
echo ""

echo "📋 TRAINING SETUP:"
echo "   Mode: FROM SCRATCH (no pretrained)"
echo "   Data: 100% DIBCO 2009-2018 (256 samples, exclude 2012)"
echo "   LR: 0.0001 (G), 0.0002 (D) with cosine decay"
echo "   Epochs: 60 max (early stop patience 10)"
echo "   Schedule: Warmup 10 → Annealing 30 → Full training"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 REAL-TIME PROGRESS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show PSNR progression
echo "📈 PSNR PROGRESSION (Validation):"
grep -E "PSNR:.*dB|Best model saved" "$LOGFILE" | tail -20

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 CURRENT STATUS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get current epoch
CURRENT_EPOCH=$(grep -oP "Epoch \K\d+/60" "$LOGFILE" | tail -1)
echo "   Current Epoch: $CURRENT_EPOCH"

# Get latest PSNR
LATEST_PSNR=$(grep -oP "PSNR: \K[\d.]+(?= ± )" "$LOGFILE" | tail -1)
if [ -n "$LATEST_PSNR" ]; then
    echo "   Latest PSNR: $LATEST_PSNR dB"
    
    # Compare with SOTA
    if (( $(echo "$LATEST_PSNR > 22.50" | bc -l 2>/dev/null || echo 0) )); then
        echo "   Status: 🏆 BEATING ALL SOTA!"
    elif (( $(echo "$LATEST_PSNR > 22.29" | bc -l 2>/dev/null || echo 0) )); then
        echo "   Status: 🎉 Beat DocEnTR!"
    elif (( $(echo "$LATEST_PSNR > 22.00" | bc -l 2>/dev/null || echo 0) )); then
        echo "   Status: ✅ Beat DE-GAN!"
    elif (( $(echo "$LATEST_PSNR > 18.00" | bc -l 2>/dev/null || echo 0) )); then
        echo "   Status: 📈 Training progressing..."
    else
        echo "   Status: 🔄 Early warmup phase..."
    fi
fi

# Check for issues
if grep -q "nan" "$LOGFILE"; then
    echo "   ⚠️  WARNING: NaN values detected!"
fi

if grep -q "Early stopping" "$LOGFILE" | tail -1 | grep -q "triggered"; then
    echo "   ✅ Training completed (early stopping)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 MONITORING COMMANDS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Watch live updates:"
echo "   tail -f $LOGFILE"
echo ""
echo "Check PSNR only:"
echo "   grep 'PSNR:' $LOGFILE | tail -20"
echo ""
echo "Monitor this script:"
echo "   watch -n 30 ./scripts/monitor_dibco_training.sh"
echo ""
echo "Check if training still running:"
echo "   ps aux | grep dibco_pure_sota"
echo ""

exit 0
