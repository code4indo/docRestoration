#!/bin/bash

# Quick HPO Progress Checker
# Alternative to Optuna dashboard

LOG_FILE="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/logbook/hpo_correct_20251015_085755.log"

echo "=========================================="
echo "📊 HPO PROGRESS SUMMARY"
echo "=========================================="
echo ""

# Count trials
COMPLETED=$(grep "Trial.*finished with value" "$LOG_FILE" 2>/dev/null | wc -l)
echo "✅ Completed Trials: $COMPLETED / 30"
echo ""

# Best trial so far
echo "🏆 Best Trial:"
grep "Best is trial" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/\[I [0-9-]* [0-9:,]*\] //'
echo ""

# Last 3 completed trials
echo "📈 Last 3 Completed Trials:"
grep "Trial.*finished with value" "$LOG_FILE" 2>/dev/null | tail -3 | while read line; do
    trial_num=$(echo "$line" | grep -oP 'Trial \K[0-9]+')
    score=$(echo "$line" | grep -oP 'value: \K[0-9.-]+')
    params=$(echo "$line" | grep -oP "parameters: \{[^}]+\}" | sed "s/parameters: //")
    echo "  Trial $trial_num: Score=$score"
    echo "    $params"
done
echo ""

# Check if still running
PID=$(cat /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/hpo_pid.txt 2>/dev/null)
if [ ! -z "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
    echo "⚙️  Status: RUNNING (PID: $PID)"
    
    # Estimate remaining time
    REMAINING=$((30 - COMPLETED))
    EST_MINUTES=$((REMAINING * 2))
    echo "⏱️  Estimated remaining: ~$EST_MINUTES minutes ($REMAINING trials)"
else
    echo "⚠️  Status: NOT RUNNING"
fi

echo ""
echo "=========================================="
echo ""
echo "Commands:"
echo "  📖 View full log:  tail -f $LOG_FILE"
echo "  🔄 Refresh:        ./scripts/check_hpo_progress.sh"
echo "  🛑 Stop HPO:       kill $PID"
echo ""
