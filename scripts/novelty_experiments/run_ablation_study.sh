#!/bin/bash
# Master Script for Ablation Study: Baseline vs Iterative Refinement

echo "=================================================================="
echo "🔬 ABLATION STUDY: Baseline vs Iterative Refinement"
echo "=================================================================="

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "📍 Project root: $PROJECT_ROOT"
echo "📅 Start time: $(date)"
echo ""

# Function to run experiment and check status
run_experiment() {
    local exp_name=$1
    local script_path=$2
    local log_file="$PROJECT_ROOT/logbook/ablation_master_$exp_name.log"
    
    echo "🚀 Running experiment: $exp_name"
    echo "   Script: $script_path"
    echo "   Log: $log_file"
    echo "   Start: $(date)"
    
    # Run the experiment
    if bash "$script_path" > "$log_file" 2>&1; then
        echo "   ✅ SUCCESS: $exp_name completed"
        echo "   End: $(date)"
        return 0
    else
        echo "   ❌ FAILED: $exp_name failed with exit code $?"
        echo "   End: $(date)"
        return 1
    fi
}

# Create summary log
SUMMARY_LOG="$PROJECT_ROOT/logbook/ablation_study_summary.log"
echo "Ablation Study Summary - $(date)" > "$SUMMARY_LOG"
echo "============================================" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

# Run experiments
echo "📋 Experiment Queue:"
echo "   1. Baseline (Tanpa Iterative Refinement)"
echo "   2. Iterative Refinement (Dengan Attention-Guided Refinement)"
echo ""

# Experiment 1: Baseline
echo "--------------------------------------------------"
echo "📊 EXPERIMENT 1/2: BASELINE"
echo "--------------------------------------------------"
if run_experiment "baseline" "$SCRIPT_DIR/ablation_baseline.sh"; then
    echo "✅ Experiment 1 (Baseline) completed successfully" >> "$SUMMARY_LOG"
else
    echo "❌ Experiment 1 (Baseline) failed" >> "$SUMMARY_LOG"
    echo "⚠️  Continuing to next experiment..."
fi

echo ""

# Experiment 2: Iterative Refinement
echo "--------------------------------------------------"
echo "📊 EXPERIMENT 2/2: ITERATIVE REFINEMENT"
echo "--------------------------------------------------"
if run_experiment "iterative_refinement" "$SCRIPT_DIR/ablation_iterative_refinement.sh"; then
    echo "✅ Experiment 2 (Iterative Refinement) completed successfully" >> "$SUMMARY_LOG"
else
    echo "❌ Experiment 2 (Iterative Refinement) failed" >> "$SUMMARY_LOG"
fi

echo ""
echo "=================================================================="
echo "🎉 ABLATION STUDY COMPLETED!"
echo "=================================================================="
echo "📅 End time: $(date)"
echo ""
echo "📊 Summary log: $SUMMARY_LOG"
echo "📁 Experiment logs: $PROJECT_ROOT/logbook/ablation_*.log"
echo "📁 Results: $PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_ablation/"
echo "🖼️ Samples: $PROJECT_ROOT/dual_modal_gan/outputs/samples_ablation/"
echo ""
echo "📋 Next Steps:"
echo "   1. Compare metrics between baseline and iterative refinement"
echo "   2. Analyze visual quality improvements"
echo "   3. Evaluate CER/WER improvements"
echo "   4. Document findings in research paper"
echo ""
echo "🚀 View results: poetry run mlflow ui"
echo "   Then open: http://localhost:5000"