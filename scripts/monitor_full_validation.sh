#!/usr/bin/env bash

# ════════════════════════════════════════════════════════════════════════════
# FULL VALIDATION VERIFICATION MONITOR
# Monitors QuickWin FULL 710 validation experiment to verify PSNR legitimacy
# ════════════════════════════════════════════════════════════════════════════

EXPERIMENT="exp_quickwin_ctc_epoch1_FULL_VALIDATION"
CHECKPOINT_DIR="dual_modal_gan/checkpoints/${EXPERIMENT}"
EPOCH_FILE="epoch_info.json"
LOG_DIR="logs"

clear
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║    🔬 FULL VALIDATION VERIFICATION MONITOR 🔬                       ║"
echo "║                                                                      ║"
echo "║    Experiment: QuickWin CTC Epoch 1 - FULL 710 Validation          ║"
echo "║    Purpose: Verify if PSNR 23.09 is REAL or sampling bias          ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ═══ CRITICAL COMPARISON ═══
echo "📊 COMPARISON WITH SAMPLED VALIDATION:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "QuickWin OPTIMIZED (sampled 400):"
echo "  • PSNR: 23.09 dB"
echo "  • CER:  0.3141"
echo "  • SSIM: 0.9547"
echo "  • Validation: 400 samples (56% coverage)"
echo ""
echo "FULL VALIDATION (current, 710 samples):"
echo "  • Status: RUNNING..."
echo "  • Validation: 710 samples (100% coverage)"
echo "  • Expected: PSNR ~20 dB if sampling bias, ~23 dB if legitimate"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ═══ GPU STATUS ═══
echo "🖥️  GPU STATUS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits | awk -F', ' '
BEGIN {
    printf "%-5s %-20s %10s %8s %10s\n", "GPU", "Name", "Memory", "GPU", "Temp"
    printf "%-5s %-20s %10s %8s %10s\n", "===", "====", "======", "===", "===="
}
{
    mem_pct = ($3 / $4) * 100
    printf "%-5s %-20s %5s/%4s MB (%3.0f%%)   %3s%%   %3s°C\n", 
        $1, substr($2, 1, 18), $3, $4, mem_pct, $5, $6
}'
echo ""

# ═══ PROCESS STATUS ═══
echo "⚙️  TRAINING PROCESS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ps aux | grep -q "[t]rain_enhanced.py.*FULL_VALIDATION"; then
    ps aux | grep "[t]rain_enhanced.py.*FULL_VALIDATION" | awk '{
        cpu = $3
        mem = $4
        time = $10
        printf "Status: ✅ RUNNING\n"
        printf "CPU:    %.1f%%\n", cpu
        printf "Memory: %.1f%%\n", mem
        printf "Time:   %s\n", time
    }'
else
    echo "Status: ❌ NOT RUNNING"
fi
echo ""

# ═══ TRAINING PROGRESS ═══
if [ -f "${EPOCH_FILE}" ]; then
    echo "📈 TRAINING PROGRESS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    LAST_EPOCH=$(jq -r '.last_completed_epoch // "N/A"' "${EPOCH_FILE}")
    BEST_PSNR=$(jq -r '.best_psnr // "N/A"' "${EPOCH_FILE}")
    BEST_CER=$(jq -r '.best_cer // "N/A"' "${EPOCH_FILE}")
    BEST_EPOCH=$(jq -r '.best_epoch // "N/A"' "${EPOCH_FILE}")
    
    echo "Last Completed Epoch: ${LAST_EPOCH}/10"
    echo "Best PSNR:            ${BEST_PSNR} dB (Epoch ${BEST_EPOCH})"
    echo "Best CER:             ${BEST_CER} (Epoch ${BEST_EPOCH})"
    echo ""
    
    # ═══ CRITICAL VERIFICATION ═══
    if [ "$BEST_PSNR" != "N/A" ] && [ "$BEST_PSNR" != "null" ]; then
        echo "🔍 VERIFICATION RESULT:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        PSNR_DIFF=$(echo "$BEST_PSNR - 23.09" | bc -l)
        PSNR_ABS=$(echo "$PSNR_DIFF" | awk '{if ($1 < 0) print -$1; else print $1}')
        
        echo "PSNR Full Validation:   ${BEST_PSNR} dB"
        echo "PSNR Sampled (400):     23.09 dB"
        echo "Difference:             ${PSNR_DIFF} dB"
        echo ""
        
        # Check if difference is significant (> 1.0 dB threshold)
        if (( $(echo "$PSNR_ABS < 1.0" | bc -l) )); then
            echo "✅ VERDICT: PSNR improvement is LEGITIMATE!"
            echo "   (Difference < 1.0 dB → sampling bias negligible)"
        elif (( $(echo "$BEST_PSNR < 21.0" | bc -l) )); then
            echo "❌ VERDICT: PSNR improvement was SAMPLING BIAS!"
            echo "   (PSNR dropped from 23.09 → ${BEST_PSNR} dB with full validation)"
        else
            echo "⚠️  VERDICT: PARTIAL sampling bias"
            echo "   (PSNR ${BEST_PSNR} dB is between 20-23 dB range)"
        fi
        echo ""
    fi
else
    echo "⏳ Waiting for first epoch to complete..."
    echo ""
fi

# ═══ LATEST LOG EXCERPT ═══
echo "📝 LATEST LOG (last 20 lines):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LATEST_LOG=$(find ${LOG_DIR} -name "*FULL_VALIDATION*.log" -o -name "*quickwin*.log" 2>/dev/null | \
    xargs ls -t 2>/dev/null | head -n 1)

if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    tail -n 20 "$LATEST_LOG" | grep -E "Epoch|Validation|PSNR|CER|batch|GPU" || \
    tail -n 20 "$LATEST_LOG"
else
    # Fallback to nohup.out if log file not found
    if [ -f "nohup.out" ]; then
        tail -n 20 nohup.out | grep -E "Epoch|Validation|PSNR|CER|batch|GPU|validating" || \
        echo "No training output found yet..."
    else
        echo "No log file found yet. Training initializing..."
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "💡 TIP: Run 'watch -n 10 ./scripts/monitor_full_validation.sh' for live updates"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
