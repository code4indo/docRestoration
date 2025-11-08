#!/bin/bash
# Fixed Ablation Study Launcher - Sequential Execution
# Prevents GPU memory conflicts by running one at a time

cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "          🔧 ABLATION STUDIES - SEQUENTIAL (FIXED)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ BUG FIXES APPLIED:"
echo "   1. GPU assignment enforced via tf.config.set_visible_devices"
echo "   2. None check added for prev_epoch_data"
echo "   3. Sequential execution to prevent OOM"
echo ""
echo "📋 EXPERIMENT PLAN:"
echo "   [1] Single-Modal (Image-Only) - GPU 0 - ~40 minutes"
echo "   [2] Dual-Modal (No-CTC) - GPU 1 - ~40 minutes"
echo ""
echo "⏱️  Total time: ~80 minutes (sequential)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Prompt for confirmation
read -p "Start ablation studies sequentially? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 1
fi

echo ""
echo "🚀 Starting Sequential Ablation Studies..."
echo ""

# Experiment 1: Single-Modal (Image-Only)
echo "═══════════════════════════════════════════════════════════════════════════"
echo "[1/2] Single-Modal (Image-Only) - GPU 0"
echo "═══════════════════════════════════════════════════════════════════════════"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE_1="logbook/ablation_single_modal_image_only_${TIMESTAMP}.log"

echo "▶️  Starting training..."
nohup ./scripts/universal_train_from_json.sh configs/ablation_single_modal_image_only.json > "$LOG_FILE_1" 2>&1 &
PID_1=$!
echo "✅ Process started: PID $PID_1"
echo "📝 Log: $LOG_FILE_1"
echo ""

# Wait for first experiment to complete
echo "⏳ Waiting for Single-Modal to complete..."
wait $PID_1
EXIT_CODE_1=$?

if [ $EXIT_CODE_1 -eq 0 ]; then
    echo "✅ Single-Modal completed successfully!"
    
    # Extract final PSNR
    PSNR_1=$(grep -oP 'Epoch 10.*PSNR: \K[0-9]+\.[0-9]+' "$LOG_FILE_1" | tail -1)
    echo "📊 Final PSNR: ${PSNR_1:-N/A} dB"
else
    echo "❌ Single-Modal failed with exit code $EXIT_CODE_1"
    echo "Check log: $LOG_FILE_1"
    exit $EXIT_CODE_1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "[2/2] Dual-Modal (No-CTC) - GPU 1"
echo "═══════════════════════════════════════════════════════════════════════════"

LOG_FILE_2="logbook/ablation_single_modal_no_ctc_${TIMESTAMP}.log"

echo "▶️  Starting training..."
nohup ./scripts/universal_train_from_json.sh configs/ablation_single_modal_no_ctc.json > "$LOG_FILE_2" 2>&1 &
PID_2=$!
echo "✅ Process started: PID $PID_2"
echo "📝 Log: $LOG_FILE_2"
echo ""

# Wait for second experiment to complete
echo "⏳ Waiting for Dual-Modal No-CTC to complete..."
wait $PID_2
EXIT_CODE_2=$?

if [ $EXIT_CODE_2 -eq 0 ]; then
    echo "✅ Dual-Modal No-CTC completed successfully!"
    
    # Extract final PSNR
    PSNR_2=$(grep -oP 'Epoch 10.*PSNR: \K[0-9]+\.[0-9]+' "$LOG_FILE_2" | tail -1)
    echo "📊 Final PSNR: ${PSNR_2:-N/A} dB"
else
    echo "❌ Dual-Modal No-CTC failed with exit code $EXIT_CODE_2"
    echo "Check log: $LOG_FILE_2"
    exit $EXIT_CODE_2
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "          ✅ ALL ABLATION STUDIES COMPLETED!"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 RESULTS SUMMARY:"
echo "   Single-Modal (Image-Only):  ${PSNR_1:-N/A} dB"
echo "   Dual-Modal (No-CTC):        ${PSNR_2:-N/A} dB"
echo ""
echo "📝 Logs:"
echo "   [1] $LOG_FILE_1"
echo "   [2] $LOG_FILE_2"
echo ""
echo "🎯 Next Steps:"
echo "   1. Compare with Dual-Modal Full: 20.23 dB (GT) / 20.07 dB (Pred)"
echo "   2. Calculate improvement: Δ PSNR = Full - Single-Modal"
echo "   3. Update paper with novelty claims"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
