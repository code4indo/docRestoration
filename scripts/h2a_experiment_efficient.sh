#!/bin/bash
# H2A Experiment Launcher (EFFICIENT VERSION)
# 
# STRATEGI EFISIEN:
# 1. Gunakan model dual-modal yang SUDAH ADA (production_v3) - HEMAT 12-24 JAM!
# 2. Train HANYA single-modal sebagai control group
# 3. Extract per-sample CER dari kedua model
# 4. Statistical analysis dengan paired t-test
#
# Timeline:
# 1. Evaluate existing dual-modal (~30 min)
# 2. Train single-modal (~12 hours)
# 3. Evaluate single-modal (~30 min)
# 4. Statistical analysis (~10 min)
# Total: ~13 hours (vs 24-48 hours kalau train both from scratch!)

set -e

# Configuration
DUAL_MODAL_CHECKPOINT="dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15"
SINGLE_MODAL_CONFIG="configs/h2a_single_modal_experiment.json"
TFRECORD_PATH="dual_modal_gan/data/dataset_gan.tfrecord"
CHARSET_PATH="real_data_preparation/real_data_charlist.txt"
RECOGNIZER_WEIGHTS="/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5"
ANALYSIS_SCRIPT="scripts/h2a_statistical_analysis_v2.py"

# Timestamps
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="h2a_experiment_${TIMESTAMP}"
LOG_DIR="${EXPERIMENT_DIR}/logs"
RESULTS_DIR="${EXPERIMENT_DIR}/results"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              H2A HYPOTHESIS VALIDATION EXPERIMENT (EFFICIENT)            ║"
echo "║                  Dual-Modal vs Single-Modal Discriminator                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Hypothesis: Dual-Modal (CNN+LSTM) → Lower CER than Single-Modal (CNN-only)"
echo "📊 Sample Size: n=710 (validation set)"
echo "🔬 Statistical Test: Paired t-test (α = 0.05)"
echo "📏 Effect Size Target: Cohen's d > 0.5 (medium effect)"
echo ""
echo "💡 EFFICIENT STRATEGY:"
echo "   ✅ Use existing dual-modal model (production_v3) - SAVES 12-24 HOURS!"
echo "   ✅ Train ONLY single-modal as control group"
echo "   ✅ Extract per-sample CER for statistical analysis"
echo "   ⏱️  Estimated Time: ~13 hours (vs 24-48h if training both)"
echo ""

# Create experiment directory structure
mkdir -p "${EXPERIMENT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "📁 Experiment Directory: ${EXPERIMENT_DIR}"
echo "📋 Logs: ${LOG_DIR}"
echo "📊 Results: ${RESULTS_DIR}"
echo ""

# Verify existing dual-modal checkpoint
if [ ! -d "$DUAL_MODAL_CHECKPOINT" ]; then
    echo "❌ Error: Dual-modal checkpoint not found: $DUAL_MODAL_CHECKPOINT"
    echo "   Available checkpoints:"
    ls -d dual_modal_gan/checkpoints/*/ 2>/dev/null || echo "   No checkpoints found"
    exit 1
fi

echo "✅ Dual-modal checkpoint verified: $DUAL_MODAL_CHECKPOINT"

# Check epoch info
if [ -f "$DUAL_MODAL_CHECKPOINT/epoch_info.json" ]; then
    echo "   📊 Checkpoint info:"
    cat "$DUAL_MODAL_CHECKPOINT/epoch_info.json" | grep -E "epoch|cer|psnr" | head -5
fi
echo ""

# Check configurations exist
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
# PHASE 1: Evaluate Existing Dual-Modal Model (Treatment Group)
# ==============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: Evaluate Existing Dual-Modal Model (Treatment Group)          ║"
echo "║  Model: production_v3_academic_split_70_15_15                           ║"
echo "║  Architecture: CNN + LSTM + Cross-Modal Attention                       ║"
echo "║  Expected: Lower CER (better text readability)                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

DUAL_MODAL_CER="${RESULTS_DIR}/dual_modal_cer.json"
DUAL_MODAL_EVAL_LOG="${LOG_DIR}/phase1_dual_modal_evaluation.log"

echo "🔍 Evaluating dual-modal model..."
echo "   Checkpoint: $DUAL_MODAL_CHECKPOINT"
echo "   Output: $DUAL_MODAL_CER"
echo "   Log: $DUAL_MODAL_EVAL_LOG"
echo ""

# Run evaluation
poetry run python scripts/h2a_evaluate_model.py \
    --checkpoint "$DUAL_MODAL_CHECKPOINT" \
    --tfrecord "$TFRECORD_PATH" \
    --charset "$CHARSET_PATH" \
    --recognizer "$RECOGNIZER_WEIGHTS" \
    --discriminator_type dual_modal \
    --output "$DUAL_MODAL_CER" \
    --gpu 0 \
    > "$DUAL_MODAL_EVAL_LOG" 2>&1

DUAL_EVAL_EXIT=$?

if [ $DUAL_EVAL_EXIT -eq 0 ]; then
    echo "✅ Phase 1 completed successfully!"
    
    # Show summary
    if [ -f "$DUAL_MODAL_CER" ]; then
        echo ""
        echo "📊 DUAL-MODAL CER STATISTICS:"
        python3 -c "
import json
with open('$DUAL_MODAL_CER', 'r') as f:
    data = json.load(f)
stats = data['statistics']
print(f\"   Mean CER:   {stats['mean']:.4f} ± {stats['std']:.4f}\")
print(f\"   Median CER: {stats['median']:.4f}\")
print(f\"   Range:      [{stats['min']:.4f}, {stats['max']:.4f}]\")
print(f\"   Samples:    {data['metadata']['num_samples']}\")
"
    fi
else
    echo "❌ Phase 1 failed with exit code: $DUAL_EVAL_EXIT"
    echo "📋 Check log: $DUAL_MODAL_EVAL_LOG"
    exit 1
fi

echo ""
echo "⏸️  Phase 1 completed. Ready for Phase 2."
echo ""

# ==============================================================================
# PHASE 2: Train Single-Modal Discriminator (Control Group)
# ==============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Train Single-Modal Discriminator (Control Group)              ║"
echo "║  Architecture: CNN-Only (no LSTM, no text processing)                   ║"
echo "║  Expected: Higher CER (worse text readability)                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

SINGLE_MODAL_LOG="${LOG_DIR}/phase2_single_modal_training.log"
SINGLE_MODAL_CKPT="dual_modal_gan/checkpoints/h2a_single_modal"

echo "🚀 Starting single-modal training..."
echo "   Config: $SINGLE_MODAL_CONFIG"
echo "   Log: $SINGLE_MODAL_LOG"
echo "   Checkpoint: $SINGLE_MODAL_CKPT"
echo ""

# Clean previous checkpoints untuk clean slate
if [ -d "$SINGLE_MODAL_CKPT" ]; then
    echo "🧹 Cleaning previous single-modal checkpoints..."
    rm -rf "$SINGLE_MODAL_CKPT"
fi

# Launch training in background
nohup poetry run ./scripts/universal_train_from_json.sh "$SINGLE_MODAL_CONFIG" \
    > "$SINGLE_MODAL_LOG" 2>&1 &
SINGLE_PID=$!

echo "📋 Training started (PID: $SINGLE_PID)"
echo "⏱️  Estimated completion: ~12 hours"
echo "📊 Monitor progress: tail -f $SINGLE_MODAL_LOG"
echo ""
echo "⚠️  Training running in background. Waiting for completion..."
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
echo "⏸️  Phase 2 completed. Ready for Phase 3."
echo ""

# ==============================================================================
# PHASE 3: Evaluate Single-Modal Model
# ==============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: Evaluate Single-Modal Model                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

SINGLE_MODAL_CER="${RESULTS_DIR}/single_modal_cer.json"
SINGLE_MODAL_EVAL_LOG="${LOG_DIR}/phase3_single_modal_evaluation.log"

echo "🔍 Evaluating single-modal model..."
echo "   Checkpoint: $SINGLE_MODAL_CKPT"
echo "   Output: $SINGLE_MODAL_CER"
echo "   Log: $SINGLE_MODAL_EVAL_LOG"
echo ""

# Run evaluation
poetry run python scripts/h2a_evaluate_model.py \
    --checkpoint "$SINGLE_MODAL_CKPT" \
    --tfrecord "$TFRECORD_PATH" \
    --charset "$CHARSET_PATH" \
    --recognizer "$RECOGNIZER_WEIGHTS" \
    --discriminator_type single_modal \
    --output "$SINGLE_MODAL_CER" \
    --gpu 0 \
    > "$SINGLE_MODAL_EVAL_LOG" 2>&1

SINGLE_EVAL_EXIT=$?

if [ $SINGLE_EVAL_EXIT -eq 0 ]; then
    echo "✅ Phase 3 completed successfully!"
    
    # Show summary
    if [ -f "$SINGLE_MODAL_CER" ]; then
        echo ""
        echo "📊 SINGLE-MODAL CER STATISTICS:"
        python3 -c "
import json
with open('$SINGLE_MODAL_CER', 'r') as f:
    data = json.load(f)
stats = data['statistics']
print(f\"   Mean CER:   {stats['mean']:.4f} ± {stats['std']:.4f}\")
print(f\"   Median CER: {stats['median']:.4f}\")
print(f\"   Range:      [{stats['min']:.4f}, {stats['max']:.4f}]\")
print(f\"   Samples:    {data['metadata']['num_samples']}\")
"
    fi
else
    echo "❌ Phase 3 failed with exit code: $SINGLE_EVAL_EXIT"
    echo "📋 Check log: $SINGLE_MODAL_EVAL_LOG"
    exit 1
fi

echo ""
echo "⏸️  Phase 3 completed. Ready for statistical analysis."
echo ""

# ==============================================================================
# PHASE 4: Statistical Analysis
# ==============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 4: Statistical Analysis & Hypothesis Validation                  ║"
echo "║  Test: Paired t-test (α = 0.05)                                         ║"
echo "║  Effect Size: Cohen's d > 0.5                                           ║"
echo "║  H0: μ_dual = μ_single                                                  ║"
echo "║  H1: μ_dual < μ_single (dual-modal better)                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

ANALYSIS_LOG="${LOG_DIR}/phase4_statistical_analysis.log"

echo "🔬 Starting statistical analysis..."
echo "   Dual-Modal CER: $DUAL_MODAL_CER"
echo "   Single-Modal CER: $SINGLE_MODAL_CER"
echo "   Analysis script: $ANALYSIS_SCRIPT"
echo "   Log: $ANALYSIS_LOG"
echo ""

# Run statistical analysis
poetry run python "$ANALYSIS_SCRIPT" \
    --dual_modal_cer "$DUAL_MODAL_CER" \
    --single_modal_cer "$SINGLE_MODAL_CER" \
    --output_dir "$RESULTS_DIR" \
    > "$ANALYSIS_LOG" 2>&1

ANALYSIS_EXIT=$?

if [ $ANALYSIS_EXIT -eq 0 ]; then
    echo "✅ Phase 4 completed successfully!"
else
    echo "❌ Phase 4 failed with exit code: $ANALYSIS_EXIT"
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
echo "📋 All logs: ${LOG_DIR}"
echo "📊 Results: ${RESULTS_DIR}"
echo ""

# Extract key results
if [ -f "$ANALYSIS_LOG" ]; then
    echo "🔍 H2A HYPOTHESIS TEST RESULTS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Show conclusion
    grep -A 10 "HYPOTHESIS CONCLUSION" "$ANALYSIS_LOG" 2>/dev/null || \
        echo "   📋 Check detailed results in: $ANALYSIS_LOG"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

echo ""
echo "📋 Detailed results:"
echo "   • Dual-Modal CER:     ${DUAL_MODAL_CER}"
echo "   • Single-Modal CER:   ${SINGLE_MODAL_CER}"
echo "   • Training logs:      ${LOG_DIR}"
echo "   • Analysis report:    ${RESULTS_DIR}"
echo "   • Statistical plots:  ${RESULTS_DIR}/h2a_statistical_analysis.png"
echo ""

echo "✅ H2A Experiment completed successfully!"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review statistical analysis results in ${RESULTS_DIR}"
echo "   2. Check visualization plots"
echo "   3. Update paper dengan empirical evidence"
echo "   4. Prepare publication-ready figures"
echo ""
echo "💾 Time saved by using existing dual-modal model: ~12-24 hours!"

exit 0
