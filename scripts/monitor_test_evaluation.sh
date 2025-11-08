#!/bin/bash
# Monitor test set evaluation progress

LOG_FILE="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/logs/test_evaluation_v4_optimal.log"
OUTPUT_FILE="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/results/test_set_evaluation_production_v4_optimal.json"

echo "================================================================"
echo " MONITORING TEST SET EVALUATION - Production V4 Optimal"
echo "================================================================"
echo ""

while true; do
    clear
    echo "================================================================"
    echo " TEST EVALUATION MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    echo ""
    
    # Check if process is running
    if ps aux | grep -v grep | grep "evaluate_test_set.py" > /dev/null; then
        echo "✅ Evaluation process: RUNNING"
        PROC_INFO=$(ps aux | grep -v grep | grep "evaluate_test_set.py" | awk '{print "   CPU: " $3 "% | Memory: " $4 "% | Time: " $10}')
        echo "$PROC_INFO"
    else
        echo "⚠️  Evaluation process: NOT RUNNING"
    fi
    
    echo ""
    echo "--- Latest Log Output (last 20 lines) ---"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
    
    echo ""
    echo "--- Result File Status ---"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✅ Results file exists:"
        ls -lh "$OUTPUT_FILE"
        echo ""
        echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    else
        echo "⏳ Results file not yet created"
    fi
    
    echo ""
    echo "================================================================"
    echo "Press Ctrl+C to exit monitoring | Refreshing in 10s..."
    echo "================================================================"
    
    sleep 10
done
