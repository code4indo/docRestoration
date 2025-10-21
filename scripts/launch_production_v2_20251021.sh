#!/bin/bash
# PRODUCTION TRAINING V2 - RANGE FIX APPLIED
# Date: 2025-10-21
# Clean slate training after critical bug fix

CONFIG="configs/production_v2_range_fixed_20251021.json"

echo "================================================================================"
echo "🚀 PRODUCTION TRAINING V2 - RANGE NORMALIZATION FIX"
echo "================================================================================"
echo ""
echo "📋 Configuration: $CONFIG"
echo ""
echo "✅ VERIFIED FIX:"
echo "   - All data normalized to [-1,1] before generator"
echo "   - Generator can produce black text (pixel 0-255)"
echo "   - Test run confirmed: 129,684 black pixels in output"
echo ""
echo "🎯 Training Parameters:"
echo "   - Epochs: 100 (with early stopping)"
echo "   - Batch Size: 2"
echo "   - Steps/Epoch: 2133 (full dataset)"
echo "   - GPU: 0"
echo "   - Patience: 25 epochs"
echo ""
echo "📊 Expected Results:"
echo "   - PSNR: >30 dB (realistic metric)"
echo "   - SSIM: >0.95"
echo "   - Text: BLACK (not gray)"
echo "   - CER: <5% at convergence"
echo "   - Duration: ~6-8 hours (depends on early stopping)"
echo ""
echo "⚠️  IMPORTANT:"
echo "   - Previous checkpoint (ckpt-91) is INVALID"
echo "   - This is clean slate training from epoch 1"
echo "   - All results before 2025-10-21 are unreliable"
echo ""
echo "================================================================================"
echo ""

read -p "Launch training? (Press ENTER to continue, Ctrl+C to cancel) "

echo ""
echo "🚀 Launching training..."

nohup ./scripts/universal_train_from_json.sh "$CONFIG" \
  > /dev/null 2>&1 &

PID=$!
sleep 2

# Get the actual log file name
LOG_FILE=$(ls -t logbook/production_v2_range_fixed_20251021_*.log 2>/dev/null | head -1)

echo ""
echo "✅ Training started!"
echo "   PID: $PID"
echo "   Config: $CONFIG"
if [ -n "$LOG_FILE" ]; then
    echo "   Log: $LOG_FILE"
fi
echo ""
echo "📊 Monitor progress:"
if [ -n "$LOG_FILE" ]; then
    echo "   tail -f $LOG_FILE"
else
    echo "   tail -f logbook/production_v2_range_fixed_20251021_*.log"
fi
echo ""
echo "🔍 Check samples:"
echo "   ls -lh dual_modal_gan/outputs/samples_production_v2_range_fixed_20251021/"
echo ""
echo "📈 View MLflow UI:"
echo "   poetry run mlflow ui"
echo "   Open: http://localhost:5000"
echo ""
echo "🛑 Stop training:"
echo "   kill $PID"
echo ""
echo "✅ Verify black text in samples:"
echo "   poetry run python -c \"import cv2; img=cv2.imread('dual_modal_gan/outputs/samples_production_v2_range_fixed_20251021/comparison_epoch_0001_sample_0.png',0); print(f'Min pixel: {img.min()}, Black pixels: {(img<50).sum()}')\""
echo ""
echo "================================================================================"
echo "⏳ Training in progress... Check log for updates"
echo "================================================================================"
