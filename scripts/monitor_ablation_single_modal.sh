#!/bin/bash

# Monitor Ablation Single-Modal Training Progress
# Created: 2025-11-06

LOG_FILE="logs/ablation_single_modal_20251106_004001.log"
CHECKPOINT_DIR="dual_modal_gan/checkpoints/ablation_single_modal"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 ABLATION SINGLE-MODAL TRAINING MONITOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Check process status
echo "📊 Process Status:"
if ps aux | grep "train_enhanced.py" | grep "ablation_single_modal" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "train_enhanced.py" | grep "ablation_single_modal" | grep -v grep | awk '{print $2}')
    CPU=$(ps aux | grep "train_enhanced.py" | grep "ablation_single_modal" | grep -v grep | awk '{print $3}')
    MEM=$(ps aux | grep "train_enhanced.py" | grep "ablation_single_modal" | grep -v grep | awk '{print $4}')
    echo "   ✅ Training RUNNING"
    echo "   PID: $PID | CPU: ${CPU}% | MEM: ${MEM}%"
else
    echo "   ❌ Training NOT RUNNING"
fi
echo ""

# 2. Latest epoch info
echo "📈 Latest Training Progress:"
if [ -f "$LOG_FILE" ]; then
    CURRENT_EPOCH=$(grep "Epoch [0-9]*/100" "$LOG_FILE" | tail -1 | sed 's/.*Epoch \([0-9]*\)\/100.*/\1/')
    if [ -n "$CURRENT_EPOCH" ]; then
        echo "   Current Epoch: $CURRENT_EPOCH/100"
        
        # Extract latest metrics
        LATEST_G=$(grep "G=" "$LOG_FILE" | tail -1 | sed 's/.*G=\([0-9.]*\).*/\1/')
        LATEST_D=$(grep "D=" "$LOG_FILE" | tail -1 | sed 's/.*D=\([0-9.]*\).*/\1/')
        LATEST_PIX=$(grep "Pix=" "$LOG_FILE" | tail -1 | sed 's/.*Pix=\([0-9.]*\).*/\1/')
        
        echo "   Latest Losses: G=$LATEST_G | D=$LATEST_D | Pix=$LATEST_PIX"
    else
        echo "   ⏳ Training initializing..."
    fi
else
    echo "   ❌ Log file not found: $LOG_FILE"
fi
echo ""

# 3. Validation metrics
echo "📊 Best Validation Metrics:"
if [ -d "$CHECKPOINT_DIR" ]; then
    if [ -f "$CHECKPOINT_DIR/epoch_info.json" ]; then
        BEST_EPOCH=$(jq -r '.best_epoch // "N/A"' "$CHECKPOINT_DIR/epoch_info.json")
        BEST_PSNR=$(jq -r '.best_psnr // "N/A"' "$CHECKPOINT_DIR/epoch_info.json")
        BEST_CER=$(jq -r '.best_cer // "N/A"' "$CHECKPOINT_DIR/epoch_info.json")
        
        echo "   Best Epoch: $BEST_EPOCH"
        echo "   Best PSNR: $BEST_PSNR dB"
        echo "   Best CER: $BEST_CER%"
    else
        echo "   ⏳ No validation results yet"
    fi
else
    echo "   ⏳ Checkpoint directory not created yet"
fi
echo ""

# 4. Recent validation results
echo "📈 Recent Validation Results:"
if [ -f "$LOG_FILE" ]; then
    echo "   Last 3 validation epochs:"
    grep -E "Val PSNR|Val CER" "$LOG_FILE" | tail -6 | sed 's/^/   /'
else
    echo "   ⏳ No validation results yet"
fi
echo ""

# 5. Early stopping status
echo "🛡️ Early Stopping Status:"
if [ -f "$LOG_FILE" ]; then
    PATIENCE=$(grep "Patience:" "$LOG_FILE" | tail -1 | sed 's/.*Patience: \([0-9]*\)\/25.*/\1/')
    if [ -n "$PATIENCE" ]; then
        echo "   Current Patience: $PATIENCE/25"
        if [ "$PATIENCE" -gt 15 ]; then
            echo "   ⚠️  WARNING: High patience counter (may stop soon)"
        elif [ "$PATIENCE" -gt 10 ]; then
            echo "   ℹ️  Moderate patience counter"
        else
            echo "   ✅ Low patience counter (training healthy)"
        fi
    else
        echo "   ⏳ No early stopping info yet"
    fi
else
    echo "   ❌ Log file not found"
fi
echo ""

# 6. GPU utilization
echo "🖥️  GPU Utilization:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    while IFS=, read -r idx name util mem_used mem_total; do
        echo "   GPU $idx: $name"
        echo "   Util: $util | Memory: $mem_used / $mem_total"
    done
else
    echo "   ⚠️  nvidia-smi not available"
fi
echo ""

# 7. Estimated time remaining
echo "⏱️  Estimated Time:"
if [ -f "$LOG_FILE" ]; then
    # Get iteration speed
    SPEED=$(grep "it/s" "$LOG_FILE" | tail -1 | sed 's/.*\([0-9.]*\)it\/s.*/\1/')
    if [ -n "$SPEED" ] && [ -n "$CURRENT_EPOCH" ]; then
        EPOCHS_REMAINING=$((100 - CURRENT_EPOCH))
        STEPS_PER_EPOCH=1658
        TOTAL_STEPS=$((EPOCHS_REMAINING * STEPS_PER_EPOCH))
        SECONDS_REMAINING=$(echo "scale=0; $TOTAL_STEPS / $SPEED" | bc)
        HOURS=$(($SECONDS_REMAINING / 3600))
        MINUTES=$((($SECONDS_REMAINING % 3600) / 60))
        
        echo "   Speed: ${SPEED} it/s"
        echo "   Epochs Remaining: $EPOCHS_REMAINING"
        echo "   Estimated Time: ${HOURS}h ${MINUTES}m"
    else
        echo "   ⏳ Calculating..."
    fi
else
    echo "   ❌ Unable to estimate"
fi
echo ""

# 8. Quick comparison with dual-modal baseline
echo "📊 Comparison with Dual-Modal (production_v4_optimal):"
echo "   Target to beat:"
echo "   - PSNR: 30.86 dB (dual-modal)"
echo "   - CER: 27.07%"
echo "   - Statistical threshold: Δ PSNR ≥ 0.53 dB for significance"
echo ""

# 9. Show latest log tail
echo "📝 Latest Log (last 10 lines):"
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE" | sed 's/^/   /'
else
    echo "   ❌ Log file not found"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Commands:"
echo "   Watch log: tail -f $LOG_FILE"
echo "   Kill training: kill -9 $PID"
echo "   View MLflow: poetry run mlflow ui"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
