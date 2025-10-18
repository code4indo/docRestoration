#!/bin/bash
# Monitor Fast Fail Comparison: Adaptive vs Baseline Training
# Usage: ./scripts/monitor_fast_fail_comparison.sh

ADAPTIVE_LOG="logs/adaptive_5epochs_20251018_102456.log"
BASELINE_LOG="logs/baseline_5epochs_20251018_102549.log"

echo "=========================================="
echo "FAST FAIL COMPARISON - Training Monitor"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Check if both logs exist
if [ ! -f "$ADAPTIVE_LOG" ]; then
    echo "❌ Adaptive log not found: $ADAPTIVE_LOG"
    exit 1
fi

if [ ! -f "$BASELINE_LOG" ]; then
    echo "❌ Baseline log not found: $BASELINE_LOG"
    exit 1
fi

# Function to extract latest metrics
extract_metrics() {
    local logfile=$1
    local experiment=$2
    
    echo "=== $experiment ==="
    
    # Current epoch
    local current_epoch=$(tail -100 "$logfile" | grep -oP "Epoch \K[0-9]+" | tail -1)
    echo "Current Epoch: ${current_epoch:-0}/5"
    
    # Latest PSNR
    local latest_psnr=$(tail -200 "$logfile" | grep -oP "psnr: \K[0-9]+\.[0-9]+" | tail -1)
    echo "Latest PSNR: ${latest_psnr:-N/A} dB"
    
    # Latest CER
    local latest_cer=$(tail -200 "$logfile" | grep -oP "cer: \K[0-9]+\.[0-9]+" | tail -1)
    echo "Latest CER: ${latest_cer:-N/A}"
    
    # For adaptive: Check loss balance ratio
    if [ "$experiment" == "ADAPTIVE" ]; then
        local ctc_ratio=$(tail -200 "$logfile" | grep -oP "CTC ratio: \K[0-9]+\.[0-9]+%" | tail -1)
        local visual_ratio=$(tail -200 "$logfile" | grep -oP "Visual ratio: \K[0-9]+\.[0-9]+%" | tail -1)
        echo "Loss Balance: CTC ${ctc_ratio:-N/A} / Visual ${visual_ratio:-N/A}"
        echo "Target: 65% CTC / 35% Visual"
    fi
    
    # Training speed
    local iter_speed=$(tail -50 "$logfile" | grep -oP "[0-9]+\.[0-9]+s/it" | tail -1)
    echo "Speed: ${iter_speed:-N/A}"
    
    # Check for errors
    local errors=$(tail -100 "$logfile" | grep -i "error\|exception\|failed" | wc -l)
    if [ $errors -gt 0 ]; then
        echo "⚠️  $errors error(s) detected in last 100 lines"
    else
        echo "✅ No errors detected"
    fi
    
    echo ""
}

# Extract metrics for both experiments
extract_metrics "$ADAPTIVE_LOG" "ADAPTIVE"
extract_metrics "$BASELINE_LOG" "BASELINE"

# Comparison summary
echo "=== COMPARISON SUMMARY ==="

adaptive_psnr=$(tail -200 "$ADAPTIVE_LOG" | grep -oP "psnr: \K[0-9]+\.[0-9]+" | tail -1)
baseline_psnr=$(tail -200 "$BASELINE_LOG" | grep -oP "psnr: \K[0-9]+\.[0-9]+" | tail -1)

if [ -n "$adaptive_psnr" ] && [ -n "$baseline_psnr" ]; then
    diff=$(echo "$adaptive_psnr - $baseline_psnr" | bc -l)
    echo "PSNR Difference: $diff dB (Adaptive - Baseline)"
    
    # Check if adaptive is winning
    if (( $(echo "$diff > 0" | bc -l) )); then
        echo "✅ Adaptive is winning by $diff dB"
    elif (( $(echo "$diff < 0" | bc -l) )); then
        echo "⚠️  Baseline is ahead by ${diff#-} dB"
    else
        echo "🟡 Both at same PSNR"
    fi
else
    echo "Waiting for first PSNR measurements..."
fi

echo ""
echo "=== PROGRESS CHECK ==="
adaptive_epoch=$(tail -100 "$ADAPTIVE_LOG" | grep -oP "Epoch \K[0-9]+" | tail -1)
baseline_epoch=$(tail -100 "$BASELINE_LOG" | grep -oP "Epoch \K[0-9]+" | tail -1)

echo "Adaptive progress: ${adaptive_epoch:-0}/5 epochs"
echo "Baseline progress: ${baseline_epoch:-0}/5 epochs"

# Estimate completion time (assuming 45 min per epoch)
if [ -n "$adaptive_epoch" ] && [ "$adaptive_epoch" -gt 0 ]; then
    remaining_epochs=$((5 - adaptive_epoch))
    estimated_hours=$(echo "$remaining_epochs * 0.75" | bc -l)
    echo "Estimated time remaining: ~${estimated_hours} hours"
fi

echo ""
echo "=========================================="
echo "Next check recommended in 1 hour"
echo "=========================================="
