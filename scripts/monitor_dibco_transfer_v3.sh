#!/bin/bash

# Monitor DIBCO Transfer Learning from Production V3 (FIXED - Using Best Model ckpt-88)
# Contingency Plan 1.1: Transfer Learning Boost

LOG_FILE=$(ls -t logbook/dibco_transfer_from_production_v3_fixed*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No training log found!"
    exit 1
fi

echo "================================================================================"
echo "🚀 DIBCO TRANSFER LEARNING - CONTINGENCY PLAN 1.1"
echo "================================================================================"
echo ""
echo "📋 Training Config:"
echo "   Experiment: dibco_transfer_from_production_v3_fixed"
echo "   Strategy: Transfer Learning from Synthetic → DIBCO Fine-tuning"
echo "   Pretrained: production_v3/best_model/ckpt-88 (BEST, epoch 44, PSNR 30.91 dB)"
echo "   ❌ Previous ERROR: Used ckpt-96 (latest, overfitted) → Result 20.01 dB (WORSE)"
echo "   ✅ FIXED: Now using ckpt-88 (best_model)"
echo "   From-Scratch Baseline: 21.71 dB (plateaued)"
echo "   Expected Boost: +1.5-2.0 dB → 23.2-23.7 dB"
echo ""
echo "🎯 TARGETS:"
echo "   Minimum: 22.5 dB (beat DocEnTR 22.29 dB)"
echo "   Target:  23.0 dB (new SOTA with margin)"
echo "   Ambitious: 24.0 dB (significant improvement)"
# Check if training is running
PID=$(ps aux | grep "train_enhanced.py.*dibco_transfer_from_production_v3_fixed" | grep -v grep | awk '{print $2}')
# Check if training is running
PID=$(ps aux | grep "train_enhanced.py.*dibco_transfer_from_production_v3" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "⚠️  Training Status: STOPPED or COMPLETED"
else
    UPTIME=$(ps -p $PID -o etime= | tr -d ' ')
    echo "✅ Training Status: RUNNING (PID: $PID, Uptime: $UPTIME)"
fi

echo ""
echo "📊 PROGRESS:"
echo "--------------------------------------------------------------------------------"

# Extract latest epoch info
LATEST_EPOCH=$(grep -E "^Epoch [0-9]+/30" "$LOG_FILE" | tail -1 | sed 's/Epoch \([0-9]*\)\/30.*/\1/')
if [ -z "$LATEST_EPOCH" ]; then
    echo "   ⏳ Training initializing... (no epochs completed yet)"
else
    echo "   Current: Epoch $LATEST_EPOCH/30"
    
    # Get best PSNR so far
    BEST_PSNR=$(grep "Best Metric (psnr_only):" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+' | head -1)
    BEST_EPOCH=$(grep "Best Metric (psnr_only):" "$LOG_FILE" | tail -1 | grep -oP 'Epoch \K\d+')
    
    if [ -n "$BEST_PSNR" ]; then
        echo "   Best PSNR: ${BEST_PSNR} dB (Epoch ${BEST_EPOCH:-N/A})"
        
        # Calculate gaps to targets
        GAP_DE_GAN=$(python3 -c "print(f'{22.0 - $BEST_PSNR:.2f}')")
        GAP_DOCENTR=$(python3 -c "print(f'{22.29 - $BEST_PSNR:.2f}')")
        GAP_TARGET=$(python3 -c "print(f'{22.5 - $BEST_PSNR:.2f}')")
        GAP_AMBITIOUS=$(python3 -c "print(f'{24.0 - $BEST_PSNR:.2f}')")
        
        echo ""
        echo "📏 GAP TO SOTA TARGETS:"
        
        # DE-GAN
        if (( $(echo "$BEST_PSNR >= 22.0" | bc -l) )); then
            echo "   ✅ DE-GAN (22.00 dB):    BEATEN by ${GAP_DE_GAN#-} dB"
        else
            echo "   ⏳ DE-GAN (22.00 dB):    Gap ${GAP_DE_GAN} dB"
        fi
        
        # DocEnTR
        if (( $(echo "$BEST_PSNR >= 22.29" | bc -l) )); then
            echo "   ✅ DocEnTR (22.29 dB):   BEATEN by ${GAP_DOCENTR#-} dB"
        else
            echo "   ⏳ DocEnTR (22.29 dB):   Gap ${GAP_DOCENTR} dB"
        fi
        
        # Target 22.5
        if (( $(echo "$BEST_PSNR >= 22.5" | bc -l) )); then
            echo "   ✅ Target (22.50 dB):    ACHIEVED! Excess ${GAP_TARGET#-} dB"
        else
            echo "   ⏳ Target (22.50 dB):    Gap ${GAP_TARGET} dB"
        fi
        
        # Ambitious 24.0
        if (( $(echo "$BEST_PSNR >= 24.0" | bc -l) )); then
            echo "   🎉 Ambitious (24.00 dB): ACHIEVED! Excess ${GAP_AMBITIOUS#-} dB"
        else
            echo "   🎯 Ambitious (24.00 dB): Gap ${GAP_AMBITIOUS} dB"
        fi
        
        # Comparison with from-scratch baseline
        BASELINE_PSNR=21.71
        BOOST=$(python3 -c "print(f'{$BEST_PSNR - $BASELINE_PSNR:.2f}')")
        echo ""
        echo "📈 TRANSFER LEARNING BOOST:"
        echo "   From-Scratch Baseline: ${BASELINE_PSNR} dB"
        echo "   Transfer Learning:     ${BEST_PSNR} dB"
        if (( $(echo "$BOOST > 0" | bc -l) )); then
            echo "   Improvement:           +${BOOST} dB ✅"
        else
            echo "   Status:                ${BOOST} dB ⚠️  (worse than baseline)"
        fi
    fi
    
    # Check early stopping status
    PATIENCE=$(grep "Patience Counter:" "$LOG_FILE" | tail -1 | grep -oP '\d+/\d+')
    if [ -n "$PATIENCE" ]; then
        echo ""
        echo "🛑 Early Stopping: Patience ${PATIENCE}"
    fi
fi

echo ""
echo "================================================================================"
echo "📝 RECENT PSNR PROGRESSION (Last 10 Epochs):"
echo "--------------------------------------------------------------------------------"

grep "Best Metric (psnr_only):" "$LOG_FILE" | tail -10 | while read line; do
    EPOCH=$(echo "$line" | grep -oP 'Epoch \K\d+')
    PSNR=$(echo "$line" | grep -oP '\d+\.\d+' | head -1)
    printf "   Epoch %2s: %s dB\n" "$EPOCH" "$PSNR"
done

echo "================================================================================"
echo ""
echo "🔄 To auto-refresh every 30s:"
echo "   watch -n 30 './scripts/monitor_dibco_transfer_v3.sh'"
echo ""
echo "📊 To view detailed log:"
echo "   tail -f $LOG_FILE"
echo ""
echo "================================================================================"
