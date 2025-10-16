#!/bin/bash

# HPO Monitoring Script
# Monitor progress of Bayesian Optimization for loss weights

PROJECT_ROOT="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration"
LOG_FILE=$(ls -t "$PROJECT_ROOT"/logbook/hpo_correct_*.log 2>/dev/null | head -1)
PID_FILE="$PROJECT_ROOT/hpo_pid.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No HPO log file found"
    exit 1
fi

if [ -f "$PID_FILE" ]; then
    HPO_PID=$(cat "$PID_FILE")
    if ps -p "$HPO_PID" > /dev/null 2>&1; then
        echo "✅ HPO is running (PID: $HPO_PID)"
    else
        echo "⚠️  HPO process not found (PID: $HPO_PID)"
        echo "   Process may have completed or crashed"
    fi
else
    echo "⚠️  PID file not found"
fi

echo ""
echo "📊 Monitoring: $LOG_FILE"
echo "================================"
echo ""

# Show last 50 lines
tail -50 "$LOG_FILE"

echo ""
echo "================================"
echo ""
echo "Commands:"
echo "  📖 View full log:     tail -f $LOG_FILE"
echo "  📈 Count trials:      grep 'Trial.*Results' $LOG_FILE | wc -l"
echo "  🏆 Best so far:       grep 'Best Objective' $LOG_FILE | tail -1"
echo "  🛑 Stop HPO:          kill $HPO_PID"
echo "  📊 View dashboard:    poetry run optuna-dashboard sqlite:///hpo_study.db"
echo ""
