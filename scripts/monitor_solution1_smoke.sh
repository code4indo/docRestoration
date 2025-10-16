#!/bin/bash
# Real-time monitoring for Solution 1 Smoke Test
# Shows: Epoch progress, PSNR, SSIM, CER, WER

LOG_FILE=$(ls -t logbook/solution1_smoke_test_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No smoke test log found"
    exit 1
fi

echo "🔬 SOLUTION 1 SMOKE TEST - REAL-TIME MONITOR"
echo "============================================"
echo "📄 Log: $LOG_FILE"
echo ""
echo "🎯 Success Criteria: PSNR >27.0 dB (improvement from 26.65 dB)"
echo "⏱️  Expected Duration: ~1.5 hours (5 epochs)"
echo ""
echo "Press Ctrl+C to stop monitoring (training continues in background)"
echo ""

# Function to extract metrics
extract_metrics() {
    tail -n 100 "$LOG_FILE" | grep -A 8 "Validation Metrics" | tail -9
}

# Monitor loop
while true; do
    clear
    echo "🔬 SOLUTION 1 SMOKE TEST - LIVE METRICS"
    echo "========================================"
    echo ""
    
    # Current epoch
    CURRENT_EPOCH=$(tail -n 50 "$LOG_FILE" | grep -oP "Epoch \K[0-9]+(?=/5)" | tail -1)
    if [ -n "$CURRENT_EPOCH" ]; then
        echo "📊 Current: Epoch $CURRENT_EPOCH/5"
    else
        echo "📊 Status: Initializing..."
    fi
    
    # Latest metrics
    echo ""
    echo "📈 Latest Validation Metrics:"
    echo "----------------------------"
    extract_metrics
    
    # Check for completion
    if grep -q "Training Finished" "$LOG_FILE"; then
        echo ""
        echo "✅ SMOKE TEST COMPLETED!"
        echo ""
        echo "📊 FINAL RESULTS:"
        tail -n 100 "$LOG_FILE" | grep -A 8 "Validation Metrics" | tail -20
        break
    fi
    
    # Check for errors
    if grep -qi "error\|exception\|failed" "$LOG_FILE" | tail -5; then
        echo ""
        echo "⚠️  POTENTIAL ISSUE DETECTED - Check log for details"
    fi
    
    echo ""
    echo "🔄 Refreshing in 30 seconds... (Ctrl+C to exit monitor)"
    sleep 30
done
