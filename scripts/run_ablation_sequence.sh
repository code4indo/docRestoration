#!/bin/bash
#############################################################################
# ABLATION STUDY - INCREMENTAL LOSS COMPONENT ANALYSIS
# Systematically adds loss components to measure individual contributions
#############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UNIVERSAL_TRAINER="$SCRIPT_DIR/universal_train_from_json.sh"

echo "========================================="
echo "  INCREMENTAL ABLATION STUDY SEQUENCE   "
echo "========================================="
echo ""
echo "Timeline: ~9 hours (25 epochs × 5 experiments)"
echo "GPU: Will use GPU 1 (check availability first!)"
echo ""

# Check if universal trainer exists
if [ ! -f "$UNIVERSAL_TRAINER" ]; then
    echo "❌ ERROR: Universal trainer not found at $UNIVERSAL_TRAINER"
    exit 1
fi

# Make it executable
chmod +x "$UNIVERSAL_TRAINER"

# Define experiments in order
experiments=(
    "configs/ablation_01_pixel_only.json"
    "configs/ablation_02_pixel_adv.json"
    "configs/ablation_03_pixel_adv_perc.json"
    "configs/ablation_04_pixel_adv_perc_ctc.json"
    "configs/ablation_05_full.json"
)

experiment_names=(
    "01_PIXEL_ONLY"
    "02_PIXEL+ADV"
    "03_PIXEL+ADV+PERC"
    "04_PIXEL+ADV+PERC+CTC"
    "05_FULL (with RECFEAT)"
)

# Verify all configs exist
echo "🔍 Verifying experiment configurations..."
for i in "${!experiments[@]}"; do
    config="${experiments[$i]}"
    name="${experiment_names[$i]}"
    
    if [ ! -f "$PROJECT_ROOT/$config" ]; then
        echo "❌ ERROR: Config not found: $config"
        exit 1
    fi
    
    echo "  ✅ $name: $config"
done
echo ""

# Run experiments sequentially
for i in "${!experiments[@]}"; do
    config="${experiments[$i]}"
    name="${experiment_names[$i]}"
    log_file="logs/ablation_$(printf "%02d" $((i+1)))_training.log"
    
    echo "========================================="
    echo "EXPERIMENT $((i+1))/5: $name"
    echo "========================================="
    echo "Config: $config"
    echo "Log: $log_file"
    echo ""
    
    # Create logs directory if not exists
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Launch training (foreground - wait for completion)
    echo "🚀 Starting training at $(date)..."
    cd "$PROJECT_ROOT"
    
    # Run with nohup but wait for completion
    nohup "$UNIVERSAL_TRAINER" "$config" > "$log_file" 2>&1 &
    PID=$!
    
    echo "   Process PID: $PID"
    echo "   Monitoring log: tail -f $log_file"
    echo ""
    
    # Wait for this experiment to finish
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ EXPERIMENT $name FAILED (exit code: $EXIT_CODE)"
        echo "   Check log: $log_file"
        echo ""
        echo "Do you want to continue with next experiment? (y/n)"
        read -r response
        if [ "$response" != "y" ]; then
            echo "Aborting ablation sequence."
            exit 1
        fi
    else
        echo "✅ EXPERIMENT $name COMPLETED at $(date)"
        echo ""
    fi
    
    # Brief pause between experiments
    if [ $i -lt $((${#experiments[@]} - 1)) ]; then
        echo "⏳ Waiting 10 seconds before next experiment..."
        sleep 10
        echo ""
    fi
done

echo "========================================="
echo "  ✅ ABLATION STUDY SEQUENCE COMPLETE   "
echo "========================================="
echo ""
echo "Summary logs:"
for i in "${!experiments[@]}"; do
    log_file="logs/ablation_$(printf "%02d" $((i+1)))_training.log"
    name="${experiment_names[$i]}"
    echo "  - $name: $log_file"
done
echo ""
echo "🎯 Next steps:"
echo "  1. Extract best metrics from each experiment"
echo "  2. Create ablation table for paper"
echo "  3. Analyze incremental contributions"
echo ""
