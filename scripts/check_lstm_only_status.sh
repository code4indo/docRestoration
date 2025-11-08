#!/bin/bash
# Quick monitor script untuk LSTM-only training

echo "=========================================="
echo "LSTM-ONLY TRAINING STATUS"
echo "=========================================="
echo ""

# Check if process running
PID=1507939
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Process RUNNING (PID: $PID)"
    ps aux | grep $PID | grep -v grep | head -1
    echo ""
else
    echo "❌ Process STOPPED or CRASHED"
    echo "   Last known PID: $PID"
    echo ""
    echo "Check log for errors:"
    echo "   tail -100 dual_modal_gan/checkpoints/h2a_lstm_only/logs/training_20251102_143129.log | grep -E 'ERROR|Exception|OOM'"
    exit 1
fi

# Get latest progress from log
LOG_FILE="dual_modal_gan/checkpoints/h2a_lstm_only/logs/training_20251102_143129.log"

echo "📊 LATEST PROGRESS:"
echo "---"
tail -50 "$LOG_FILE" | grep -E "Epoch [0-9]+:" | tail -1
echo ""

echo "🏆 BEST METRICS (if any):"
echo "---"
grep "NEW BEST" "$LOG_FILE" | tail -5
echo ""

echo "⏰ TIMING:"
echo "---"
echo "Started: Nov 2, 2025 14:31 WIB"
echo "Current: $(date)"
echo "Expected completion: Nov 5, 2025 ~14:00 WIB"
echo ""

echo "📝 Full log:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🔍 Detailed progress:"
echo "   tail -100 $LOG_FILE | grep -E 'Epoch|PSNR|CER'"
