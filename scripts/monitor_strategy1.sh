#!/bin/bash
# Monitor training progress for Strategy 1

# Auto-detect latest log file
LOG_FILE=$(ls -t /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/logbook/test_strategy1_extreme_rebalance_*.log 2>/dev/null | head -1)

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   MONITORING STRATEGI 1 TRAINING PROGRESS                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   STRATEGI 1: EXTREME REBALANCING - LIVE MONITORING           ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📄 Log file: $LOG_FILE"
    echo "⏰ Last update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 LATEST TRAINING METRICS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Get latest step info
    tail -100 "$LOG_FILE" | grep -E "Step [0-9]+/200" | tail -5
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 LOSS STATISTICS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Calculate D_loss statistics
    D_LOSS_MIN=$(grep "D_loss:" "$LOG_FILE" | grep -oP "D_loss: \K[0-9.]+" | sort -n | head -1)
    D_LOSS_MAX=$(grep "D_loss:" "$LOG_FILE" | grep -oP "D_loss: \K[0-9.]+" | sort -n | tail -1)
    D_LOSS_AVG=$(grep "D_loss:" "$LOG_FILE" | grep -oP "D_loss: \K[0-9.]+" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    D_LOSS_LATEST=$(grep "D_loss:" "$LOG_FILE" | tail -1 | grep -oP "D_loss: \K[0-9.]+")
    
    echo "D_loss:"
    echo "  Current:  ${D_LOSS_LATEST:-N/A}"
    echo "  Min:      ${D_LOSS_MIN:-N/A}"
    echo "  Max:      ${D_LOSS_MAX:-N/A}"
    echo "  Average:  ${D_LOSS_AVG:-N/A}"
    echo "  Baseline: 1.23-1.71 (stuck)"
    echo ""
    
    # Show latest loss components
    echo "Latest Loss Components:"
    tail -100 "$LOG_FILE" | grep -E "Adv:|Pixel:|RecFeat:|CTC:" | tail -4
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 TARGET vs CURRENT:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  TARGET: D_loss → 0.5-0.7 (ideal)"
    echo "  CURRENT: D_loss → ${D_LOSS_AVG:-N/A}"
    echo ""
    
    if [ ! -z "$D_LOSS_AVG" ] && [ "$D_LOSS_AVG" != "N/A" ]; then
        if (( $(echo "$D_LOSS_AVG < 0.7" | bc -l) )); then
            echo "  ✅ D_loss IMPROVED! Mendekati target!"
        elif (( $(echo "$D_LOSS_AVG < 1.0" | bc -l) )); then
            echo "  ⚡ D_loss membaik, tapi belum optimal"
        else
            echo "  ⏳ D_loss masih tinggi, perlu lebih banyak step"
        fi
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done
