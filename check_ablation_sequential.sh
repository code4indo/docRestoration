#!/bin/bash
# Quick progress checker for sequential ablation studies

echo "═══════════════════════════════════════════════════════════════════════════"
echo "          📊 ABLATION STUDY PROGRESS (Sequential Mode)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Check running processes
RUNNING_COUNT=$(ps aux | grep train_enhanced | grep -v grep | wc -l)

if [ $RUNNING_COUNT -eq 0 ]; then
    echo "🔍 Process Status: ⚠️  No training processes running"
    echo ""
    echo "📋 Check if experiments completed or crashed:"
    echo "   ls -lth logbook/ablation_* | head -5"
else
    echo "🔍 Process Status: ✅ $RUNNING_COUNT training process(es) running"
    echo ""
    ps aux | grep train_enhanced | grep -v grep | awk '{print "   PID:", $2, "- Command:", $11, $12, $13, $14}'
fi

echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo "💻 GPU Status:"
echo "───────────────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | column -t -s,

echo ""
echo "───────────────────────────────────────────────────────────────────────────"
echo "📈 Latest Log Files:"
echo "───────────────────────────────────────────────────────────────────────────"

# Find latest ablation logs
LATEST_LOGS=$(ls -t logbook/ablation_single_modal_*_2025*.log 2>/dev/null | head -2)

if [ -z "$LATEST_LOGS" ]; then
    echo "⚠️  No recent ablation logs found"
else
    for LOG in $LATEST_LOGS; do
        echo ""
        echo "📝 $(basename $LOG)"
        
        # Extract experiment name
        if [[ $LOG == *"image_only"* ]]; then
            EXP_NAME="Single-Modal (Image-Only)"
        elif [[ $LOG == *"no_ctc"* ]]; then
            EXP_NAME="Dual-Modal (No-CTC)"
        else
            EXP_NAME="Unknown"
        fi
        
        # Get current epoch
        CURRENT_EPOCH=$(grep -oP 'Epoch \K[0-9]+(?=/10)' "$LOG" | tail -1)
        
        # Get latest PSNR
        LATEST_PSNR=$(grep -oP 'PSNR: \K[0-9]+\.[0-9]+' "$LOG" | tail -1)
        
        # Check for errors
        ERROR_COUNT=$(grep -i "error\|exception\|traceback" "$LOG" | wc -l)
        
        if [ -n "$CURRENT_EPOCH" ]; then
            echo "   🎯 Experiment: $EXP_NAME"
            echo "   📊 Progress: Epoch $CURRENT_EPOCH/10"
            
            if [ -n "$LATEST_PSNR" ]; then
                echo "   📈 Latest PSNR: $LATEST_PSNR dB"
            fi
            
            if [ $ERROR_COUNT -gt 0 ]; then
                echo "   ⚠️  Errors detected: $ERROR_COUNT (check log for details)"
            else
                echo "   ✅ No errors detected"
            fi
            
            # Calculate remaining time (4 min per epoch)
            REMAINING_EPOCHS=$((10 - CURRENT_EPOCH))
            REMAINING_MIN=$((REMAINING_EPOCHS * 4))
            echo "   ⏱️  Estimated remaining: ~$REMAINING_MIN minutes"
        else
            echo "   ⚠️  No epoch data found (may still be initializing)"
            
            # Check file size to see if training started
            SIZE=$(stat -c%s "$LOG" 2>/dev/null || echo "0")
            if [ $SIZE -gt 1000 ]; then
                echo "   📄 Log size: $(numfmt --to=iec-i --suffix=B $SIZE)"
            fi
        fi
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "🔍 Monitor Commands:"
echo "   tail -f logbook/ablation_single_modal_image_only_*.log  # Single-Modal"
echo "   tail -f logbook/ablation_single_modal_no_ctc_*.log      # Dual-Modal No-CTC"
echo "   watch -n 10 './check_ablation_sequential.sh'            # Auto-refresh"
echo ""
