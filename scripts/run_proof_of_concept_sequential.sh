#!/bin/bash

# PROOF-OF-CONCEPT EXPERIMENT: Ground Truth vs Predicted Mode
# Hypothesis: Ground truth mode gives +2-3 dB PSNR improvement
# Author: Senior ML Engineer Audit
# Date: 2025-11-06

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

LOG_DIR="logs/exp_proof_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "PROOF-OF-CONCEPT EXPERIMENT"
echo "=============================================="
echo "Hypothesis: discriminator_mode='ground_truth' gives +2-3 dB PSNR improvement"
echo "GPU: 1"
echo "Duration: ~10-15 minutes per experiment"
echo "Total time: ~30 minutes (sequential)"
echo "=============================================="
echo ""

# Function to run experiment
run_experiment() {
    local config_file=$1
    local exp_name=$2
    local log_file="${LOG_DIR}/${exp_name}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${exp_name}..."
    echo "Config: ${config_file}"
    echo "Log: ${log_file}"
    echo ""
    
    nohup poetry run python dual_modal_gan/scripts/train_enhanced.py \
        --config "${config_file}" \
        > "${log_file}" 2>&1 &
    
    local pid=$!
    echo "PID: ${pid}"
    echo "${pid}" > "${LOG_DIR}/${exp_name}.pid"
    
    # Wait for completion
    while kill -0 $pid 2>/dev/null; do
        sleep 10
        # Show last line of log
        tail -n 1 "${log_file}" 2>/dev/null || true
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${exp_name} completed!"
    echo ""
}

# EXPERIMENT A: Predicted mode (baseline)
echo "=============================================="
echo "EXPERIMENT A: PREDICTED MODE (BASELINE)"
echo "=============================================="
run_experiment \
    "configs/exp_proof_predicted_mode.json" \
    "exp_a_predicted"

# Wait 30 seconds for GPU cooldown
echo "Waiting 30 seconds for GPU cooldown..."
sleep 30

# EXPERIMENT B: Ground truth mode (proposed fix)
echo "=============================================="
echo "EXPERIMENT B: GROUND TRUTH MODE (PROPOSED)"
echo "=============================================="
run_experiment \
    "configs/exp_proof_ground_truth_mode.json" \
    "exp_b_ground_truth"

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=============================================="
echo ""
echo "Results location:"
echo "- Logs: ${LOG_DIR}/"
echo "- Exp A (predicted): dual_modal_gan/checkpoints/exp_proof_predicted/"
echo "- Exp B (ground_truth): dual_modal_gan/checkpoints/exp_proof_ground_truth/"
echo ""
echo "To compare results:"
echo "  python -c \""
echo "import json"
echo "a = json.load(open('dual_modal_gan/checkpoints/exp_proof_predicted/epoch_info.json'))"
echo "b = json.load(open('dual_modal_gan/checkpoints/exp_proof_ground_truth/epoch_info.json'))"
echo "print(f'PREDICTED:    PSNR={a[\\\"best_psnr\\\"]:.2f} dB, CER={a[\\\"best_cer\\\"]*100:.1f}%')"
echo "print(f'GROUND_TRUTH: PSNR={b[\\\"best_psnr\\\"]:.2f} dB, CER={b[\\\"best_cer\\\"]*100:.1f}%')"
echo "print(f'IMPROVEMENT:  ΔPSNR={(b[\\\"best_psnr\\\"]-a[\\\"best_psnr\\\"]):.2f} dB')"
echo "\""
echo ""
echo "Expected results:"
echo "  PREDICTED:    PSNR ~ 28-29 dB"
echo "  GROUND_TRUTH: PSNR ~ 30-31 dB"
echo "  IMPROVEMENT:  ΔPSNR ~ +2-3 dB ← PROOF OF CONCEPT!"
echo ""
