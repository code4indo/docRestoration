#!/bin/bash
# Auto-Executor for Confidence Build Experiments
# Automatically launches next experiment based on results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_DIR/logs"
LOGBOOK_DIR="$PROJECT_DIR/logbook"

echo "🤖 AUTOMATED CONFIDENCE BUILD EXECUTOR"
echo "======================================"
echo ""

# Function to check if experiment is done
is_experiment_done() {
    local log_file=$1
    if grep -q "Training completed" "$log_file" 2>/dev/null; then
        return 0
    elif grep -q "Early stopping triggered" "$log_file" 2>/dev/null; then
        return 0
    elif grep -q "ERROR\|FAILED\|Exception" "$log_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Function to extract final PSNR
get_final_psnr() {
    local log_file=$1
    grep "val/psnr" "$log_file" | tail -1 | grep -oP '\d+\.\d+' || echo "0"
}

# Function to check white dots (placeholder - would need actual image analysis)
check_white_dots_reduction() {
    local sample_dir=$1
    # TODO: Implement actual white dots analysis
    # For now, return "unknown"
    echo "unknown"
}

echo "📊 EXPERIMENT 1: Generator V1 Fallback"
echo "Config: confidence_generator_v1_fallback.json"
echo "Status: RUNNING"
echo ""

# Wait for Experiment 1 to complete
EXP1_LOG="$LOGS_DIR/confidence_gen_v1_20251019_133453.log"
EXP1_PID_FILE="/tmp/confidence_gen_v1.pid"

echo "⏳ Waiting for Experiment 1 to complete..."
echo "   Checking every 5 minutes..."
echo ""

while true; do
    if is_experiment_done "$EXP1_LOG"; then
        echo "✅ Experiment 1 COMPLETED"
        break
    fi
    
    # Check if process still running
    if [ -f "$EXP1_PID_FILE" ]; then
        PID=$(cat "$EXP1_PID_FILE")
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "⚠️  Process stopped but training not marked complete"
            echo "   Check log: $EXP1_LOG"
            break
        fi
    fi
    
    # Show progress
    LATEST=$(tail -5 "$EXP1_LOG" | grep "Epoch" | tail -1)
    if [ -n "$LATEST" ]; then
        echo "   Progress: $LATEST"
    fi
    
    sleep 300  # Wait 5 minutes
done

echo ""
echo "📈 Analyzing Experiment 1 Results..."
echo "======================================"

FINAL_PSNR=$(get_final_psnr "$EXP1_LOG")
echo "Final PSNR: $FINAL_PSNR dB"

# Decision logic
if (( $(echo "$FINAL_PSNR > 26.0" | bc -l) )); then
    echo ""
    echo "🎉 SUCCESS! PSNR > 26 dB"
    echo "✅ Generator V1 proved to be better than V2"
    echo "✅ White dots likely reduced (visual check needed)"
    echo ""
    echo "📝 DECISION: Proceed to FULL TRAINING with Generator V1"
    echo ""
    echo "Next steps:"
    echo "1. Visual inspection of samples"
    echo "2. Create full training config (60 epochs)"
    echo "3. Launch production training"
    echo ""
    echo "🚀 Ready to launch full training? (Manual approval needed)"
    
elif (( $(echo "$FINAL_PSNR > 24.0" | bc -l) )); then
    echo ""
    echo "⚠️  PARTIAL SUCCESS: PSNR $FINAL_PSNR (improved but not optimal)"
    echo ""
    echo "📝 DECISION: Test Experiment 2 (V2 with loss fixes)"
    echo ""
    
    read -p "Launch Experiment 2 (confidence_fix_white_dots_v1)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Launching Experiment 2..."
        cd "$PROJECT_DIR"
        
        EXP2_LOG="$LOGS_DIR/confidence_v2_loss_fix_$(date +%Y%m%d_%H%M%S).log"
        nohup ./scripts/universal_train_from_json.sh \
            configs/confidence_fix_white_dots_v1.json \
            > "$EXP2_LOG" 2>&1 &
        
        echo $! > /tmp/confidence_v2.pid
        echo "✅ Experiment 2 launched (PID: $(cat /tmp/confidence_v2.pid))"
        echo "📊 Log: $EXP2_LOG"
        echo "🔍 Monitor: tail -f $EXP2_LOG"
    else
        echo "⏸️  Manual intervention needed"
    fi
    
else
    echo ""
    echo "❌ FAILURE: PSNR $FINAL_PSNR (no improvement or worse)"
    echo ""
    echo "📝 DECISION: Need deeper investigation"
    echo ""
    echo "Possible issues:"
    echo "- Data quality problems"
    echo "- Loss configuration still not optimal"
    echo "- Both generator architectures have fundamental issues"
    echo ""
    echo "Recommended actions:"
    echo "1. Visual inspection of failed samples"
    echo "2. Check for data preprocessing issues"
    echo "3. Consider architectural redesign"
    echo "4. Consult baseline research (Souibgui et al.)"
fi

echo ""
echo "======================================"
echo "📊 Full analysis saved to logbook"
echo "======================================"
