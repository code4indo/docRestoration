#!/bin/bash

# Quick monitoring script untuk Quick Win experiment
LOG_FILE="logs/quickwin_ctc_epoch1_20251106_174258.log"

echo "═══════════════════════════════════════════════════════════════"
echo "📊 QUICK WIN EXPERIMENT - LIVE MONITORING"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if training is running
PID=1282394
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Training process RUNNING (PID: $PID)"
else
    echo "⚠️  Training process NOT FOUND (may have finished or crashed)"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "🔍 CTC Weight Schedule (Critical Validation)"
echo "─────────────────────────────────────────────────────────────"

# Extract CTC weights per epoch
grep -E "Epoch [0-9]+/10.*CTC_w=" "$LOG_FILE" | tail -10

echo ""
echo "Expected:"
echo "  Epoch 1: CTC_w=1.67 (annealing 1/3)"
echo "  Epoch 2: CTC_w=3.33 (annealing 2/3)"
echo "  Epoch 3: CTC_w=5.00 (annealing 3/3)"
echo "  Epoch 4+: CTC_w=5.00 (full strength)"

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "📈 Validation Metrics (Last 5 Epochs)"
echo "─────────────────────────────────────────────────────────────"

# Extract validation metrics
grep -A 10 "🎯 Validation Results" "$LOG_FILE" | tail -50

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "💻 GPU Status"
echo "─────────────────────────────────────────────────────────────"

nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F, '{printf "GPU %s: %s | Util: %s%% | Mem: %s/%s MB\n", $1, $2, $3, $4, $5}'

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "⏱️  Last Activity (Last 10 lines)"
echo "─────────────────────────────────────────────────────────────"

tail -10 "$LOG_FILE"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Commands:"
echo "  Live log: tail -f $LOG_FILE"
echo "  Kill:     kill $PID"
echo "  Refresh:  watch -n 30 './scripts/monitor_quickwin.sh'"
echo ""
