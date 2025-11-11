#!/bin/bash
# ABLATION STUDY: Joint Training vs Frozen Recognizer
# Purpose: Prove frozen recognizer superiority through empirical evidence
# Expected: Joint training shows catastrophic forgetting, frozen maintains stability

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  ABLATION STUDY: Joint Training vs Frozen Recognizer             ║"
echo "║  Hypothesis: Joint training causes gradient conflicts & forgetting║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
CONFIG_FILE="configs/ablation_joint_training_vs_frozen.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/ablation_joint_training"
LOG_FILE="${LOG_DIR}/joint_training_${TIMESTAMP}.log"

# Create log directory
mkdir -p "${LOG_DIR}"

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "📋 Experimental Setup:"
echo "   Config: ${CONFIG_FILE}"
echo "   Epochs: 20 (limited for quick validation)"
echo "   Steps/epoch: 100 (limited for efficiency)"
echo "   Mode: Joint Training (recognizer trainable=True)"
echo "   Baseline CER: 33.72% (frozen pre-trained)"
echo "   Expected CER: >40% (degradation due to forgetting)"
echo ""

echo "⚠️  WARNING: This experiment is expected to show:"
echo "   1. Catastrophic forgetting (CER drift from 33.72%)"
echo "   2. Gradient conflicts (loss oscillations)"
echo "   3. Training instability (convergence issues)"
echo ""

read -p "Continue with joint training experiment? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 1
fi

echo ""
echo "🚀 Launching joint training experiment..."
echo "   Log file: ${LOG_FILE}"
echo ""

# Check if recognizer_joint_trainable.py exists
JOINT_RECOGNIZER="dual_modal_gan/src/models/recognizer_joint_trainable.py"
if [ ! -f "${JOINT_RECOGNIZER}" ]; then
    echo "❌ Error: Joint trainable recognizer not found: ${JOINT_RECOGNIZER}"
    echo "   Please ensure recognizer_joint_trainable.py is created"
    exit 1
fi

echo "✅ Joint trainable recognizer found"
echo "✅ Starting training process..."
echo ""

# Launch training in background with nohup
nohup poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --config "${CONFIG_FILE}" \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!

echo "✅ Training launched in background (PID: ${TRAIN_PID})"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "🔍 Check status:"
echo "   ps aux | grep ${TRAIN_PID}"
echo ""
echo "🛑 Stop training:"
echo "   kill ${TRAIN_PID}"
echo ""
echo "📈 View TensorBoard (after training starts):"
echo "   poetry run tensorboard --logdir dual_modal_gan/checkpoints/ablation_joint_training"
echo ""

# Wait a few seconds and show initial output
sleep 5
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄 Initial training output:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
head -n 50 "${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Experiment launched successfully"
echo "   Full log: ${LOG_FILE}"
echo ""
