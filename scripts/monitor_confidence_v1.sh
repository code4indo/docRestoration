#!/bin/bash
# Confidence Build Monitoring Script
# Created: 2025-10-19

LOG_FILE="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/logs/confidence_gen_v1_20251019_133453.log"
PID_FILE="/tmp/confidence_gen_v1.pid"

echo "======================================"
echo "CONFIDENCE BUILD - GENERATOR V1 FALLBACK"
echo "======================================"
echo ""
echo "📊 Experiment: Test if Generator V2 is root cause of white dots"
echo "🎯 Strategy: Fallback to Generator V1 (proven) + Discriminator V2 Fixed"
echo "⏱️  Duration: ~1.5 hours (15 epochs × 50 steps)"
echo ""

if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null; then
        echo "✅ Training RUNNING (PID: $PID)"
    else
        echo "❌ Training STOPPED (PID: $PID not found)"
    fi
else
    echo "⚠️  PID file not found"
fi

echo ""
echo "📝 Latest log entries:"
echo "--------------------------------------"
tail -15 "$LOG_FILE"
echo ""
echo "--------------------------------------"
echo "🔍 Monitor live: tail -f $LOG_FILE"
echo "🛑 Stop training: kill $(cat $PID_FILE 2>/dev/null || echo 'N/A')"
echo "======================================"
