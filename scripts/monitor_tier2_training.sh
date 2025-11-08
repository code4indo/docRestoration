#!/bin/bash
# Monitor Tier 2 Training Progress
# Usage: ./scripts/monitor_tier2_training.sh

LOGFILE=$(ls -t logbook/tier2_improved_dibco_finetuning_*.log 2>/dev/null | head -1)

if [ -z "$LOGFILE" ]; then
    echo "❌ No tier2 training log found"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║             TIER 2 TRAINING MONITORING                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Log file: $LOGFILE"
echo "📊 Last updated: $(stat -c %y "$LOGFILE" | cut -d'.' -f1)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show epoch progress
echo "📈 EPOCH PROGRESS:"
tail -200 "$LOGFILE" | grep -E "^Epoch [0-9]+/" | tail -10
echo ""

# Show latest PSNR/SSIM
echo "🎯 LATEST METRICS:"
tail -100 "$LOGFILE" | grep -E "val.*PSNR|val.*SSIM|Best val" | tail -10
echo ""

# Show early stopping status
echo "⏱️  EARLY STOPPING STATUS:"
tail -100 "$LOGFILE" | grep -E "Early Stopping|Patience|Counter" | tail -5
echo ""

# Check if still running
PID=$(ps aux | grep -E "train_enhanced.*tier2_improved" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✅ Training ACTIVE (PID: $PID)"
    ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')
    echo "   Elapsed time: $ELAPSED"
else
    echo "⚠️  Training STOPPED or COMPLETED"
    # Check if completed successfully
    if tail -50 "$LOGFILE" | grep -q "Training completed successfully"; then
        echo "   Status: ✅ COMPLETED"
    elif tail -50 "$LOGFILE" | grep -q "Early stopping triggered"; then
        echo "   Status: ⏹️  EARLY STOPPED"
    elif tail -50 "$LOGFILE" | grep -q "Error"; then
        echo "   Status: ❌ ERROR"
    else
        echo "   Status: ⏹️  INTERRUPTED"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 To view full log: tail -f $LOGFILE"
echo "📈 To view MLflow UI: poetry run mlflow ui"
echo "🔍 To see detailed progress: tail -100 $LOGFILE | grep -E 'Epoch|PSNR|SSIM'"
