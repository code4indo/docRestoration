#!/bin/bash
# Quick progress check for ablation studies

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              ABLATION STUDY PROGRESS CHECK                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if processes running
echo "🔍 Process Status:"
PROC_COUNT=$(ps aux | grep train_enhanced | grep -v grep | wc -l)
if [ $PROC_COUNT -eq 2 ]; then
    echo "   ✅ Both experiments running ($PROC_COUNT processes)"
elif [ $PROC_COUNT -eq 1 ]; then
    echo "   ⚠️  Only 1 experiment running"
elif [ $PROC_COUNT -eq 0 ]; then
    echo "   ❌ No experiments running (training completed or stopped)"
else
    echo "   ⚠️  $PROC_COUNT processes found (unexpected)"
fi
echo ""

# GPU usage
echo "💻 GPU Status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "   GPU %s: %s / %s (%s utilization)\n", $1, $2, $3, $4}'
echo ""

# Single-Modal progress
echo "📊 Single-Modal (Image-Only):"
LOG1="logbook/ablation_single_modal_image_only_20251106_140603.log"
if [ -f "$LOG1" ]; then
    LAST_EPOCH=$(grep -oP 'Epoch \K[0-9]+(?=/10)' "$LOG1" | tail -1)
    LAST_PSNR=$(grep -oP 'PSNR.*?: \K[0-9]+\.[0-9]+' "$LOG1" | tail -1)
    if [ -n "$LAST_EPOCH" ]; then
        echo "   Epoch: $LAST_EPOCH/10"
        if [ -n "$LAST_PSNR" ]; then
            echo "   Latest PSNR: $LAST_PSNR dB"
        fi
    else
        echo "   Status: Starting..."
    fi
else
    echo "   ❌ Log file not found"
fi
echo ""

# Dual-Modal No-CTC progress
echo "📊 Dual-Modal (No CTC):"
LOG2="logbook/ablation_single_modal_no_ctc_20251106_140603.log"
if [ -f "$LOG2" ]; then
    LAST_EPOCH=$(grep -oP 'Epoch \K[0-9]+(?=/10)' "$LOG2" | tail -1)
    LAST_PSNR=$(grep -oP 'PSNR.*?: \K[0-9]+\.[0-9]+' "$LOG2" | tail -1)
    if [ -n "$LAST_EPOCH" ]; then
        echo "   Epoch: $LAST_EPOCH/10"
        if [ -n "$LAST_PSNR" ]; then
            echo "   Latest PSNR: $LAST_PSNR dB"
        fi
    else
        echo "   Status: Starting..."
    fi
else
    echo "   ❌ Log file not found"
fi
echo ""

# Estimated time remaining
if [ -n "$LAST_EPOCH" ] && [ "$LAST_EPOCH" -gt 0 ]; then
    REMAINING=$((10 - LAST_EPOCH))
    TIME_MIN=$((REMAINING * 4))
    echo "⏱️  Estimated remaining: ~$TIME_MIN minutes ($REMAINING epochs × 4 min/epoch)"
fi
echo ""

echo "──────────────────────────────────────────────────────────────────────────"
echo "Run this script again: ./check_ablation_progress.sh"
echo "Live logs: tail -f logbook/ablation_single_modal_*_20251106_140603.log"
echo "──────────────────────────────────────────────────────────────────────────"
