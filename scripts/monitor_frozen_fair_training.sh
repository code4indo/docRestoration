#!/bin/bash

# Monitor Frozen Fair Ablation Training
# Usage: ./scripts/monitor_frozen_fair_training.sh

echo "=============================================="
echo "FROZEN RECOGNIZER FAIR ABLATION TRAINING"
echo "Config: ablation_frozen_fair_20epochs_100steps.json"
echo "Expected Duration: ~30 minutes (20 epochs, 100 steps/epoch)"
echo "=============================================="
echo ""

# Find the latest log file
LOG_FILE=$(ls -t logs/ablation_frozen_fair_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No log file found!"
    exit 1
fi

echo "📝 Monitoring: $LOG_FILE"
echo ""

# Check if process is running
PROCESS=$(ps aux | grep "train_enhanced.py --config configs/ablation_frozen_fair" | grep -v grep)
if [ -z "$PROCESS" ]; then
    echo "⚠️  Training process NOT running!"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$LOG_FILE"
    exit 1
else
    echo "✅ Training process is running (PID: $(echo $PROCESS | awk '{print $2}'))"
fi

echo ""
echo "=========================================="
echo "📊 TRAINING PROGRESS"
echo "=========================================="

# Extract key metrics
echo ""
echo "🔍 Latest Metrics:"
grep -E "Epoch [0-9]+/[0-9]+" "$LOG_FILE" | tail -5
echo ""

# Check for CER metrics
echo "📈 CER Tracking:"
grep -E "(Val|Test) CER:" "$LOG_FILE" | tail -10
echo ""

# Check for PSNR/SSIM
echo "🎯 Quality Metrics:"
grep -E "(PSNR|SSIM):" "$LOG_FILE" | tail -5
echo ""

# Check for early stopping
echo "⏰ Early Stopping Status:"
grep -E "Early Stopping|patience|Stopping" "$LOG_FILE" | tail -5
echo ""

# Show last 15 lines of log
echo "=========================================="
echo "📋 RECENT LOG ENTRIES (last 15 lines):"
echo "=========================================="
tail -15 "$LOG_FILE"
echo ""

echo "=========================================="
echo "🔄 Auto-refresh: watch -n 10 'bash scripts/monitor_frozen_fair_training.sh'"
echo "📊 Full log: tail -f $LOG_FILE"
echo "=========================================="
