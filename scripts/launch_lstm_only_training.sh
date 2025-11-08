#!/bin/bash
# LSTM-Only Discriminator - Full 50 Epoch Training
# H2A Ablation Study: Completing the trifecta (CNN-only, LSTM-only, Dual-Modal)
#
# Expected Results:
# - PSNR: ~28-29 dB (WORST, vs CNN-only ~30.6 dB, Dual-Modal ~30.9 dB)
# - CER: ~35-40% (WORST, vs CNN-only ~27.1%, Dual-Modal ~27.1%)
# - Reason: Relies on noisy predicted text (CER ~27% input error)
# - Scientific Value: Validates importance of visual modal
#
# Estimated Time: 25-30 hours
# GPU: RTX A4000 (GPU 0)

set -e  # Exit on error

echo "=========================================="
echo "LSTM-ONLY DISCRIMINATOR TRAINING"
echo "H2A Ablation Study - Text-Only Branch"
echo "=========================================="
echo ""
echo "⏱️  Estimated time: 25-30 hours"
echo "🎯 Expected: WORST performance among 3 variants"
echo "📊 Target: Complete ablation study table"
echo ""

cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration

# Log file untuk monitoring
LOG_DIR="dual_modal_gan/checkpoints/h2a_lstm_only/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Log file: $LOG_FILE"
echo ""
echo "🚀 Starting training in background..."
echo "   Monitor with: tail -f $LOG_FILE"
echo ""

# Launch training di background dengan nohup
nohup /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/.venv/bin/python \
    dual_modal_gan/scripts/train_enhanced.py \
    --config configs/h2a_lstm_only_experiment.json \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✅ Training launched!"
echo "   PID: $TRAIN_PID"
echo "   Config: configs/h2a_lstm_only_experiment.json"
echo "   Checkpoints: dual_modal_gan/checkpoints/h2a_lstm_only"
echo "   Samples: dual_modal_gan/outputs/samples_h2a_lstm_only"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🔍 Check process:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
echo "🛑 Stop training (if needed):"
echo "   kill $TRAIN_PID"
echo ""
echo "⏰ Started at: $(date)"
echo "🎯 Expected completion: $(date -d '+30 hours')"
