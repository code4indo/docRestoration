#!/bin/bash

# QUICK PROOF-OF-CONCEPT: Ground Truth Mode Only
# Run GROUND_TRUTH experiment first to prove the concept
# If successful (PSNR >30 dB @ epoch 10), then we know the fix works!
# Author: Senior ML Engineer
# Date: 2025-11-06

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

CONFIG="configs/exp_proof_ground_truth_mode.json"
LOG_FILE="logs/exp_proof_ground_truth_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================="
echo "QUICK PROOF-OF-CONCEPT: GROUND TRUTH MODE"
echo "=============================================="
echo "Hypothesis: Ground truth gives +2-3 dB PSNR improvement"
echo "Expected PSNR @ epoch 10: 30-31 dB"
echo "Duration: ~10-15 minutes"
echo "GPU: 1"
echo "=============================================="
echo ""
echo "Config: ${CONFIG}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Starting training..."
echo ""

# Create log directory
mkdir -p logs

# Run training in background
nohup poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --config "${CONFIG}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started with PID: ${PID}"
echo ""
echo "Monitor training with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Or watch metrics:"
echo "  watch -n 5 'tail -n 30 ${LOG_FILE} | grep -E \"Epoch|PSNR|CER\"'"
echo ""
echo "Expected timeline:"
echo "  Epoch 1-2:  Warmup (visual only)"
echo "  Epoch 3-5:  CTC annealing"
echo "  Epoch 6-10: Full dual-modal power"
echo ""
echo "Check results at epoch 10:"
echo "  cat dual_modal_gan/checkpoints/exp_proof_ground_truth/epoch_info.json"
echo ""
