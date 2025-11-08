#!/bin/bash
# H2A Experiment Launcher - Dual-Modal vs Single-Modal Comparison
#
# Purpose: Membuktikan hipotesis H2A bahwa dual-modal discriminator (CNN+LSTM)
#          menghasilkan CER lebih rendah dibanding single-modal (CNN-only)
#
# Experimental Design:
# - Treatment Group: Dual-Modal Discriminator (enhanced_v2_fixed)
# - Control Group: Single-Modal Discriminator (CNN-only)
# - Metric: Character Error Rate (CER) pada validation set (n=710)
# - Statistical Test: Paired t-test (α = 0.05)
# - Effect Size: Cohen's d > 0.5 (medium effect)
#
# Expected Results:
# - Dual-Modal: Lower CER (better text readability)
# - Single-Modal: Higher CER (worse text readability)
# - Statistical significance: p < 0.05
# - Effect size: d > 0.5
#
# Timeline:
# 1. Train Dual-Modal model (~6 hours)
# 2. Train Single-Modal model (~6 hours)
# 3. Statistical Analysis (~30 minutes)
# Total: ~13 hours

set -e

# Configuration
DUAL_MODAL_CONFIG="configs/h2a_dual_modal_experiment.json"
SINGLE_MODAL_CONFIG="configs/h2a_single_modal_experiment.json"
ANALYSIS_SCRIPT="scripts/h2a_statistical_analysis.py"

# Timestamps
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="h2a_experiment_${TIMESTAMP}"
LOG_DIR="${EXPERIMENT_DIR}/logs"
RESULTS_DIR="${EXPERIMENT_DIR}/results"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    H2A HYPOTHESIS VALIDATION EXPERIMENT                  ║"
echo "║                  Dual-Modal vs Single-Modal Discriminator                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Hypothesis: Dual-Modal (CNN+LSTM) → Lower CER than Single-Modal (CNN-only)"
echo "📊 Sample Size: n=710 (validation set)"
echo "🔬 Statistical Test: Paired t-test (α = 0.05)"
echo "📏 Effect Size Target: Cohen's d > 0.5 (medium effect)"
echo "⏱️  Estimated Time: ~13 hours total"
echo ""

# Create experiment directory structure
mkdir -p "${EXPERIMENT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "📁 Experiment Directory: ${EXPERIMENT_DIR}"
echo "📋 Logs will be saved to: ${LOG_DIR}"
echo "📊 Results will be saved to: ${RESULTS_DIR}"
echo ""

# Check configurations exist
if [ ! -f "$DUAL_MODAL_CONFIG" ]; then
    echo "❌ Error: Dual-modal config not found: $DUAL_MODAL_CONFIG"
    exit 1
fi

if [ ! -f "$SINGLE_MODAL_CONFIG" ]; then
    echo "❌ Error: Single-modal config not found: $SINGLE_MODAL_CONFIG"
    exit 1
fi

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "❌ Error: Analysis script not found: $ANALYSIS_SCRIPT"
    exit 1
fi

echo "✅ Configuration files validated"
echo ""

# ==============================================================================
# PHASE 1: Train Dual-Modal Discriminator (Treatment Group)
# ==============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: Training Dual-Modal Discriminator (Treatment Group)           ║"
echo "║  Architecture: CNN + LSTM + Cross-Modal Attention                       ║"
echo "║  Expected: Lower CER (better text readability)                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

DUAL_MODAL_LOG="${LOG_DIR}/phase1_dual_modal_training.log"
DUAL_MODAL_CKPT="dual_modal_gan/checkpoints/h2a_dual_modal"

echo "🚀 Starting dual-modal training..."
echo "   Config: $DUAL_MODAL_CONFIG"
echo "   Log file: $DUAL_MODAL_LOG"
echo "   Checkpoint: $DUAL_MODAL_CKPT"
echo ""

# Clean previous checkpoints untuk clean slate
if [ -d "$DUAL_MODAL_CKPT" ]; then
    echo "🧹 Cleaning previous dual-modal checkpoints..."
    rm -rf "$DUAL_MODAL_CKPT"
fi

# Launch training
nohup ./scripts/universal_train_from_json.sh "$DUAL_MODAL_CONFIG" > "$DUAL_MODAL_LOG" 2>&1 &
DUAL_PID=$!

echo "📋 Training started (PID: $DUAL_PID)"
echo "⏱️  Estimated completion: ~6 hours"
echo "📊 Monitoring progress: tail -f $DUAL_MODAL_LOG"
echo ""

# Wait for completion
wait $DUAL_PID
DUAL_EXIT=$?

if [ $DUAL_EXIT -eq 0 ]; then
    echo "✅ Phase 1 completed successfully!"
else
    echo "❌ Phase 1 failed with exit code: $DUAL_EXIT"
    echo "📋 Check log: $DUAL_MODAL_LOG"
    exit 1
fi

echo ""
echo "⏸️  Phase 1 completed. Ready for Phase 2."
echo ""

# ==============================================================================
# PHASE 2: Train Single-Modal Discriminator (Control Group)
# ==============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Training Single-Modal Discriminator (Control Group)           ║"
echo "║  Architecture: CNN-Only (no LSTM, no text processing)                   ║"
echo "║  Expected: Higher CER (worse text readability)                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

SINGLE_MODAL_LOG="${LOG_DIR}/phase2_single_modal_training.log"
SINGLE_MODAL_CKPT="dual_modal_gan/checkpoints/h2a_single_modal"

echo "🚀 Starting single-modal training..."
echo "   Config: $SINGLE_MODAL_CONFIG"
echo "   Log file: $SINGLE_MODAL_LOG"
echo "   Checkpoint: $SINGLE_MODAL_CKPT"
echo ""

# Clean previous checkpoints untuk clean slate
if [ -d "$SINGLE_MODAL_CKPT" ]; then
    echo "🧹 Cleaning previous single-modal checkpoints..."
    rm -rf "$SINGLE_MODAL_CKPT"
fi

# Launch training
nohup ./scripts/universal_train_from_json.sh "$SINGLE_MODAL_CONFIG" > "$SINGLE_MODAL_LOG" 2>&1 &
SINGLE_PID=$!

echo "📋 Training started (PID: $SINGLE_PID)"
echo "⏱️  Estimated completion: ~6 hours"
echo "📊 Monitoring progress: tail -f $SINGLE_MODAL_LOG"
echo ""

# Wait for completion
wait $SINGLE_PID
SINGLE_EXIT=$?

if [ $SINGLE_EXIT -eq 0 ]; then
    echo "✅ Phase 2 completed successfully!"
else
    echo "❌ Phase 2 failed with exit code: $SINGLE_EXIT"
    echo "📋 Check log: $SINGLE_MODAL_LOG"
    exit 1
fi

echo ""
echo "⏸️  Phase 2 completed. Ready for statistical analysis."
echo ""

# ==============================================================================
# PHASE 3: Statistical Analysis
# ==============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: Statistical Analysis & Hypothesis Validation                  ║"
echo "║  Test: Paired t-test (α = 0.05)                                         ║"
echo "║  Effect Size: Cohen's d > 0.5                                           ║"
echo "║  H0: μ_dual = μ_single                                                  ║"
echo "║  H1: μ_dual < μ_single (dual-modal better)                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

ANALYSIS_LOG="${LOG_DIR}/phase3_statistical_analysis.log"

echo "🔬 Starting statistical analysis..."
echo "   Dual-Modal results: $DUAL_MODAL_CKPT"
echo "   Single-Modal results: $SINGLE_MODAL_CKPT"
echo "   Analysis script: $ANALYSIS_SCRIPT"
echo "   Log file: $ANALYSIS_LOG"
echo ""

# Run statistical analysis
poetry run python "$ANALYSIS_SCRIPT" \
    --dual_modal_checkpoint "$DUAL_MODAL_CKPT" \
    --single_modal_checkpoint "$SINGLE_MODAL_CKPT" \
    --output_dir "$RESULTS_DIR" \
    > "$ANALYSIS_LOG" 2>&1

ANALYSIS_EXIT=$?

if [ $ANALYSIS_EXIT -eq 0 ]; then
    echo "✅ Phase 3 completed successfully!"
else
    echo "❌ Phase 3 failed with exit code: $ANALYSIS_EXIT"
    echo "📋 Check log: $ANALYSIS_LOG"
    exit 1
fi

# ==============================================================================
# FINAL RESULTS
# ==============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                         EXPERIMENT COMPLETED                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📁 Experiment Directory: ${EXPERIMENT_DIR}"
echo "📋 All logs saved to: ${LOG_DIR}"
echo "📊 Results saved to: ${RESULTS_DIR}"
echo ""

# Extract key results dari analysis log
if [ -f "$ANALYSIS_LOG" ]; then
    echo "🔍 H2A HYPOTHESIS TEST RESULTS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Look for conclusion in the log
    if grep -q "H2A: SUPPORTED" "$ANALYSIS_LOG"; then
        echo "🏆 CONCLUSION: ✅ H2A HYPOTHESIS SUPPORTED"
        echo "   Dual-Modal discriminator significantly outperforms Single-Modal"
        grep -A 5 "HYPOTHESIS CONCLUSION" "$ANALYSIS_LOG" | head -10
    elif grep -q "H2A: NOT SUPPORTED" "$ANALYSIS_LOG"; then
        echo "🏆 CONCLUSION: ❌ H2A HYPOTHESIS NOT SUPPORTED"
        echo "   Insufficient evidence untuk conclude dual-modal superiority"
    else
        echo "🔍 Please check analysis results in: $ANALYSIS_LOG"
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

echo ""
echo "📋 Detailed results available in:"
echo "   • Training logs: ${LOG_DIR}"
echo "   • Analysis report: ${RESULTS_DIR}"
echo "   • Statistical plots: ${RESULTS_DIR}/h2a_statistical_analysis.png"
echo ""

echo "✅ H2A Experiment completed successfully!"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review statistical analysis results"
echo "   2. Check visualization plots"
echo "   3. Update paper dengan empirical evidence"
echo "   4. Prepare publication-ready figures"

exit 0