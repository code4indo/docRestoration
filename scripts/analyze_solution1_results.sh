#!/bin/bash
# Automatic Decision Script for Solution 1 Smoke Test
# Analyzes results and recommends next action

LOG_FILE=$(ls -t logbook/solution1_smoke_test_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No smoke test log found"
    exit 1
fi

echo "🔍 SOLUTION 1 SMOKE TEST - RESULT ANALYSIS"
echo "=========================================="
echo ""

# Check if training completed
if ! grep -q "Training Finished\|Phase 6" "$LOG_FILE"; then
    echo "⚠️  Smoke test still running or incomplete"
    echo "   Current status:"
    tail -n 5 "$LOG_FILE" | grep -E "Epoch|PSNR"
    echo ""
    echo "   Wait for completion before analysis"
    exit 1
fi

# Extract all PSNR values from validation metrics
echo "📊 Extracting PSNR metrics..."
PSNR_VALUES=$(grep -oP "PSNR: \K[0-9]+\.[0-9]+" "$LOG_FILE")

# Get best PSNR
BEST_PSNR=$(echo "$PSNR_VALUES" | sort -rn | head -1)
BASELINE_PSNR=26.65

echo ""
echo "📈 PSNR PROGRESSION:"
echo "-------------------"
echo "$PSNR_VALUES" | nl -w2 -s". Epoch " | sed 's/^/  /'

echo ""
echo "🎯 RESULTS SUMMARY:"
echo "------------------"
echo "  Baseline (Epoch 22): $BASELINE_PSNR dB"
echo "  Best (Smoke Test):   $BEST_PSNR dB"

# Calculate improvement
IMPROVEMENT=$(echo "$BEST_PSNR - $BASELINE_PSNR" | bc -l)
echo "  Improvement:         $IMPROVEMENT dB"

# Decision logic
echo ""
echo "🤔 DECISION ANALYSIS:"
echo "--------------------"

# Success threshold: >27 dB
if (( $(echo "$BEST_PSNR > 27.0" | bc -l) )); then
    echo "  Status: ✅ SUCCESS"
    echo "  Verdict: Solution 1 shows improvement!"
    echo ""
    echo "📋 RECOMMENDED ACTION:"
    echo "  → Run full training with Solution 1 (50 epochs)"
    echo "  → Expected final PSNR: 28-30 dB"
    echo ""
    echo "💻 Execute:"
    echo "  bash scripts/train32_solution1_full.sh"
    
elif (( $(echo "$BEST_PSNR > 26.65" | bc -l) )); then
    echo "  Status: ⚠️  MARGINAL IMPROVEMENT"
    echo "  Verdict: Slight improvement, but may not reach 35 dB target"
    echo ""
    echo "📋 RECOMMENDED ACTION:"
    echo "  OPTION 1: Try full training (optimistic, 40% success)"
    echo "  OPTION 2: Escalate to Solution 3 (conservative, safer)"
    echo ""
    echo "💻 Execute (choose one):"
    echo "  bash scripts/train32_solution1_full.sh  # Try full training"
    echo "  # OR implement Solution 3 (architecture upgrade)"
    
else
    echo "  Status: ❌ NO IMPROVEMENT"
    echo "  Verdict: LR scheduling alone insufficient"
    echo ""
    echo "📋 RECOMMENDED ACTION:"
    echo "  → Escalate to Solution 3 (Architecture Upgrade)"
    echo "  → Potential changes:"
    echo "     - Deeper U-Net generator (more skip connections)"
    echo "     - Multi-scale discriminator"
    echo "     - Attention mechanisms"
    echo "     - Different loss combinations"
    echo ""
    echo "⚠️  Note: Architecture changes require significant dev time"
    echo "         Consider consulting with advisor before proceeding"
fi

echo ""
echo "📝 Full metrics available in: $LOG_FILE"
echo ""

# Extract full validation metrics for final epoch
echo "📊 FINAL EPOCH METRICS:"
echo "----------------------"
tail -n 100 "$LOG_FILE" | grep -A 8 "Validation Metrics" | tail -9

echo ""
echo "🔬 Analysis complete"
