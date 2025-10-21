#!/bin/bash

# ============================================================================
# Production Training Launcher - Academic Protocol (70/15/15 Split)
# ============================================================================
# 
# Purpose: Launch production training with proper train/val/test split
# Test set: LOCKED for final evaluation (never accessed during training)
# 
# Author: AI/ML Engineer
# Date: 2025-10-21
# ============================================================================

echo "============================================================================"
echo "🚀 PRODUCTION TRAINING - ACADEMIC PROTOCOL"
echo "============================================================================"
echo ""
echo "📊 Dataset Split:"
echo "   Train:      3,317 samples (70%) - for learning"
echo "   Validation:   710 samples (15%) - for hyperparameter tuning"
echo "   Test:         712 samples (15%) - LOCKED for final evaluation"
echo ""
echo "⚠️  CRITICAL:"
echo "   - Test set will NOT be used during training"
echo "   - Test evaluation happens ONCE after training"
echo "   - Use scripts/evaluate_test_set.py for final evaluation"
echo ""
echo "============================================================================"
echo ""

# Configuration
CONFIG_FILE="configs/production_v3_academic_split_70_15_15.json"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/production_v3_academic_split_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p ${LOG_DIR}

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "📁 Configuration:"
echo "   Config:  ${CONFIG_FILE}"
echo "   Log:     ${LOG_FILE}"
echo ""

# Show key config parameters
echo "🔧 Key Parameters:"
grep -E '"experiment_name"|"generator_version"|"discriminator_version"|"epochs"|"batch_size"|"train_split"|"val_split"' ${CONFIG_FILE} | sed 's/^/   /'
echo ""

# Confirmation
read -p "🚦 Ready to start training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled."
    exit 1
fi

echo ""
echo "============================================================================"
echo "🏃 Starting Training..."
echo "============================================================================"
echo ""

# Launch training in background with nohup
nohup ./scripts/universal_train_from_json.sh ${CONFIG_FILE} > ${LOG_FILE} 2>&1 &

TRAIN_PID=$!

echo "✅ Training launched!"
echo "   PID: ${TRAIN_PID}"
echo "   Log: ${LOG_FILE}"
echo ""
echo "📊 Monitor training with:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "🔍 Check GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "⏹️  Stop training (if needed):"
echo "   kill ${TRAIN_PID}"
echo ""
echo "============================================================================"
echo "⚠️  REMEMBER: Test set evaluation AFTER training completes"
echo "============================================================================"
echo ""
echo "After training finishes, run test evaluation ONCE:"
echo ""
echo "  poetry run python scripts/evaluate_test_set.py \\"
echo "    --checkpoint dual_modal_gan/checkpoints/production_v3_academic_split/best_model \\"
echo "    --tfrecord dual_modal_gan/data/dataset_gan.tfrecord \\"
echo "    --output results/test_set_results_production_v3.json \\"
echo "    --train_split 0.7 \\"
echo "    --val_split 0.15"
echo ""
echo "============================================================================"
echo ""
