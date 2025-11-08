#!/bin/bash

# Monitor DIBCO Pure SOTA Training (Fixed Config)
# Usage: ./scripts/monitor_dibco_v2.sh

LOG_FILE="logbook/dibco_pure_sota_comparison_v1_20251101_074648.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 DIBCO PURE SOTA TRAINING - REAL-TIME MONITOR (Fixed Config)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if training is running
if pgrep -f "dibco_pure_sota_comparison_v1" > /dev/null; then
    echo "✅ Training Status: RUNNING"
    PID=$(pgrep -f "dibco_pure_sota_comparison_v1" | head -1)
    echo "   PID: $PID"
    echo "   Uptime: $(ps -p $PID -o etime= | tr -d ' ')"
else
    echo "❌ Training Status: STOPPED"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 PSNR PROGRESSION (Target: >22.5 dB to beat SOTA)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract PSNR for each epoch
grep "Best Metric (psnr_only):" "$LOG_FILE" | awk '{print $5, $6}' | nl -v1 -w2 -s". " | while read line; do
    epoch=$(echo "$line" | awk '{print $1}')
    psnr=$(echo "$line" | awk '{print $2}')
    
    # Color coding based on PSNR value
    if (( $(echo "$psnr >= 22.5" | bc -l) )); then
        echo "  🏆 Epoch $epoch $psnr (SOTA BEATEN!)"
    elif (( $(echo "$psnr >= 22.0" | bc -l) )); then
        echo "  🥈 Epoch $epoch $psnr (DE-GAN level)"
    elif (( $(echo "$psnr >= 20.0" | bc -l) )); then
        echo "  🔥 Epoch $epoch $psnr (Very good)"
    elif (( $(echo "$psnr >= 15.0" | bc -l) )); then
        echo "  ⭐ Epoch $epoch $psnr (Good)"
    elif (( $(echo "$psnr >= 10.0" | bc -l) )); then
        echo "  ✓  Epoch $epoch $psnr (Warming up)"
    else
        echo "     Epoch $epoch $psnr (Early phase)"
    fi
done

echo ""

# Current best
BEST_PSNR=$(grep "Best Metric (psnr_only):" "$LOG_FILE" | tail -1 | awk '{print $5}')
BEST_EPOCH=$(grep "Best Epoch:" "$LOG_FILE" | tail -1 | awk '{print $3}' | tr -d ',')

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 CURRENT STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Best PSNR: ${BEST_PSNR} dB (Epoch ${BEST_EPOCH})"
echo "   Gap to DE-GAN (22.00): $(echo "22.00 - $BEST_PSNR" | bc) dB"
echo "   Gap to DocEnTR (22.29): $(echo "22.29 - $BEST_PSNR" | bc) dB"
echo "   Gap to Target (22.50): $(echo "22.50 - $BEST_PSNR" | bc) dB"

# Early stopping status
PATIENCE=$(grep "Patience Counter:" "$LOG_FILE" | tail -1 | awk '{print $3}')
ES_STATUS=$(grep "Early Stopping:" "$LOG_FILE" | tail -1 | awk -F'Early Stopping: ' '{print $2}')

echo ""
echo "   Early Stopping: $ES_STATUS"
echo "   Patience: $PATIENCE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 LATEST LOG (Last 10 lines)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -10 "$LOG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Auto-refresh: watch -n 30 './scripts/monitor_dibco_v2.sh'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
