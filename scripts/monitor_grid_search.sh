#!/bin/bash
# Monitor parallel grid search progress

echo "==================================================================="
echo "PARALLEL GRID SEARCH MONITOR"
echo "==================================================================="
echo ""

# Check if process running
echo "📊 Process Status:"
ps aux | grep mini_grid_search | grep -v grep | grep -v monitor || echo "  ❌ Not running"
echo ""

# Check GPU usage
echo "🎮 GPU Usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
echo ""

# Check results
echo "📈 Results so far:"
if [ -f dual_modal_gan/docs/grid_search_results/grid_search_results_raw.csv ]; then
    TOTAL=$(wc -l < dual_modal_gan/docs/grid_search_results/grid_search_results_raw.csv)
    COMPLETED=$((TOTAL - 1))  # Minus header
    echo "  Completed: $COMPLETED/27"
    
    if [ $COMPLETED -gt 0 ]; then
        echo ""
        echo "  Latest results:"
        tail -5 dual_modal_gan/docs/grid_search_results/grid_search_results_raw.csv | column -t -s,
    fi
else
    echo "  No results yet"
fi
echo ""

# Check training logs
echo "📝 Active Training Logs:"
ls -lth dual_modal_gan/docs/grid_search_results/training_log_*.txt 2>/dev/null | head -5 || echo "  No logs yet"
echo ""

echo "==================================================================="
echo "Run: watch -n 30 bash scripts/monitor_grid_search.sh"
echo "==================================================================="
