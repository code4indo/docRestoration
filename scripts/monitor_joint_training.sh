#!/bin/bash

# Quick monitoring script for joint training ablation study

LOG_FILE="logs/ablation_joint_training/joint_training_20251111_134724.log"
PID=535812

echo "=================================================="
echo "Joint Training Ablation Study - Quick Monitor"
echo "=================================================="
echo ""

# Check if process is running
if ps -p $PID > /dev/null; then
    ELAPSED=$(ps -p $PID -o etime=)
    echo "✅ Process Status: RUNNING (PID: $PID)"
    echo "   Elapsed Time: $ELAPSED"
else
    echo "❌ Process Status: NOT RUNNING"
    echo "   Check log for completion or errors"
fi
echo ""

# GPU usage
echo "GPU Usage:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep $PID || echo "   No GPU usage found (process may have ended)"
echo ""

# Latest metrics from log
echo "Latest Training Metrics:"
echo "------------------------"
tail -100 "$LOG_FILE" | grep -E "Epoch [0-9]|Step [0-9]|CER:|PSNR:|Forgetting" | tail -10
echo ""

# Training progress estimate
TOTAL_STEPS=2000  # 20 epochs × 100 steps
CURRENT_STEP=$(tail -100 "$LOG_FILE" | grep -oE "Step [0-9]+" | tail -1 | grep -oE "[0-9]+")

if [ ! -z "$CURRENT_STEP" ]; then
    CURRENT_EPOCH=$(tail -100 "$LOG_FILE" | grep -oE "Epoch [0-9]+/[0-9]+" | tail -1 | awk -F'/' '{print $1}' | grep -oE "[0-9]+")
    TOTAL_EPOCHS=20
    
    if [ ! -z "$CURRENT_EPOCH" ]; then
        PROGRESS=$((CURRENT_EPOCH * 5))  # 5% per epoch
        echo "Progress Estimate: Epoch $CURRENT_EPOCH/$TOTAL_EPOCHS (~$PROGRESS%)"
    fi
fi
echo ""

echo "Commands:"
echo "  View live log:  tail -f $LOG_FILE"
echo "  GPU monitor:    watch -n 1 nvidia-smi"
echo "  Kill training:  kill $PID"
echo "=================================================="
