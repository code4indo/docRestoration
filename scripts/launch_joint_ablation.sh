#!/bin/bash

# Joint Training Ablation Study Launcher - PRODUCTION
# Full 20 epochs training to validate frozen recognizer superiority

set -e

CONFIG_FILE="configs/ablation_joint_training_vs_frozen.json"
SCRIPT_PATH="dual_modal_gan/scripts/train_joint_ablation.py"
LOG_DIR="logs/ablation_joint_training"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/joint_training_${TIMESTAMP}.log"

echo "=================================================="
echo "🚀 Joint Training Ablation Study - FULL TRAINING"
echo "=================================================="
echo ""
echo "Config: 20 epochs, 100 steps/epoch"
echo "Expected: Catastrophic forgetting (CER 33.72% → >40%)"
echo "Duration: ~3-4 hours"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Check files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "✅ Config: $CONFIG_FILE"
echo "✅ Script: $SCRIPT_PATH"
echo "✅ Log: $LOG_FILE"
echo ""

# GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
echo ""

# Launch
echo "🚀 Launching joint training (background process)..."
echo ""
nohup poetry run python "$SCRIPT_PATH" --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo ""
echo "✅ TRAINING STARTED (PID: $TRAIN_PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Expected completion: $(date -d '+4 hours' '+%Y-%m-%d %H:%M')"
echo "=================================================="

