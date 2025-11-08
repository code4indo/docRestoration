#!/bin/bash

# Fine-tuning Launcher for Thin Stroke Preservation Model
# 
# This script:
# 1. Creates fine-tuning dataset from full-page documents
# 2. Launches fine-tuning with pretrained checkpoint
# 3. Runs in background with logging

set -e  # Exit on error

echo "=========================================="
echo "FINE-TUNING LAUNCHER"
echo "=========================================="
echo ""

# Configuration
GT_DIR="DokumenRusak/manual_restoration/gt"
DEG_DIR="DokumenRusak/manual_restoration/deg"
OUTPUT_DIR="dual_modal_gan/data/finetuning"
CONFIG_PATH="configs/thin_stroke_preservation_v1_finetuning.json"
LOG_DIR="logs/finetuning"

# Create log directory
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/finetuning_${TIMESTAMP}.log"

echo "📋 Step 1: Creating fine-tuning dataset"
echo "   GT directory:   $GT_DIR"
echo "   DEG directory:  $DEG_DIR"
echo "   Output:         $OUTPUT_DIR"
echo ""

# Check if dataset already exists
if [ -f "$OUTPUT_DIR/finetuning_train.tfrecord" ]; then
    echo "⚠️  Fine-tuning dataset already exists!"
    read -p "   Recreate dataset? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing old dataset..."
        rm -rf "$OUTPUT_DIR"
    else
        echo "   Using existing dataset"
    fi
fi

# Create dataset if not exists
if [ ! -f "$OUTPUT_DIR/finetuning_train.tfrecord" ]; then
    echo "   Running dataset creation..."
    poetry run python scripts/create_finetuning_strips.py \
        --gt_dir "$GT_DIR" \
        --deg_dir "$DEG_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --augment
    
    if [ $? -ne 0 ]; then
        echo "❌ Dataset creation failed!"
        exit 1
    fi
    echo "   ✅ Dataset created successfully"
else
    echo "   ✅ Using existing dataset"
fi

echo ""
echo "📋 Step 2: Verifying pretrained checkpoint"

# Extract pretrained checkpoint path from config
PRETRAINED_CKPT=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['pretrained_checkpoint'])")
echo "   Checkpoint: $PRETRAINED_CKPT"

if [ ! -f "${PRETRAINED_CKPT}.index" ]; then
    echo "❌ Pretrained checkpoint not found: ${PRETRAINED_CKPT}.index"
    echo "   Please train base model first or update config"
    exit 1
fi

echo "   ✅ Checkpoint verified"
echo ""

echo "📋 Step 3: Launching fine-tuning"
echo "   Config:     $CONFIG_PATH"
echo "   Log file:   $LOG_FILE"
echo ""

# Ask for confirmation
read -p "🚀 Start fine-tuning in background? (Y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "   Starting training..."
    echo "   Log: tail -f $LOG_FILE"
    echo ""
    
    # Launch training in background
    nohup ./scripts/universal_train_from_json.sh "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &
    
    TRAIN_PID=$!
    echo "   ✅ Training started (PID: $TRAIN_PID)"
    echo "   📊 Monitor: tail -f $LOG_FILE"
    echo "   🛑 Stop:    kill $TRAIN_PID"
    echo ""
    echo "   Waiting 5 seconds to check if training started successfully..."
    sleep 5
    
    # Check if process is still running
    if ps -p $TRAIN_PID > /dev/null; then
        echo "   ✅ Training is running"
        echo ""
        echo "   View initial log:"
        tail -n 20 "$LOG_FILE"
    else
        echo "   ❌ Training failed to start! Check log:"
        tail -n 50 "$LOG_FILE"
        exit 1
    fi
else
    echo "   Fine-tuning cancelled"
fi

echo ""
echo "=========================================="
echo "FINE-TUNING SETUP COMPLETE"
echo "=========================================="
