#!/bin/bash

# ========================================
# EXPERIMENT SERIES: CTC DOMINANCE HYPOTHESIS
# ========================================
# Tujuan: Membuktikan bahwa CTC loss dominance (97%) menyebabkan PSNR rendah
# Solusi: Test berbagai CTC weight untuk menemukan balance optimal
# 
# Each experiment: 1 epoch × 1000 steps = ~20 minutes
# Total time: ~80 minutes (4 experiments)
# ========================================

set -e

REPO_ROOT="/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration"
cd "$REPO_ROOT"

echo "=========================================="
echo "🧪 EXPERIMENT SERIES - CTC DOMINANCE TEST"
echo "=========================================="
echo ""
echo "Hypothesis to prove:"
echo "1. CTC weight 1.0 → CTC dominates 97% → PSNR ~19 dB (BASELINE)"
echo "2. CTC weight 0.1 → Image quality fokus → PSNR ~25 dB (+6 dB)"
echo "3. CTC weight 0.3 → Balanced → PSNR ~23 dB, CER better"
echo "4. Strong adversarial → Better image quality → PSNR ~24 dB"
echo "5. Optimal combination → BEST PSNR → ~26 dB"
echo ""
echo "=========================================="
echo ""

# Fungsi untuk run experiment
run_experiment() {
    local exp_num=$1
    local config_file=$2
    local exp_name=$3
    
    echo ""
    echo "=========================================="
    echo "🔬 EXPERIMENT $exp_num: $exp_name"
    echo "=========================================="
    echo "Config: $config_file"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Run training
    poetry run bash scripts/run_training_from_json.sh "$config_file"
    
    echo ""
    echo "✅ Experiment $exp_num completed!"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Wait 10 seconds before next experiment
    echo "⏳ Waiting 10 seconds before next experiment..."
    sleep 10
}

# ========================================
# EXPERIMENT 1: CTC Weight 0.1 (Low)
# ========================================
# Expected: PSNR 24-26 dB, CER 0.6-0.7
# Proof: Reducing CTC dominance improves image quality
run_experiment 1 \
    "configs/experiment1_ctc_annealing_low.json" \
    "CTC Weight 0.1 - Low CTC Dominance"

# ========================================
# EXPERIMENT 2: CTC Weight 0.3 (Medium)
# ========================================
# Expected: PSNR 22-24 dB, CER 0.4-0.5
# Proof: Medium CTC gives balanced image+text quality
run_experiment 2 \
    "configs/experiment2_ctc_medium.json" \
    "CTC Weight 0.3 - Balanced Approach"

# ========================================
# EXPERIMENT 3: Strong Adversarial (Adv 5.0)
# ========================================
# Expected: PSNR 23-25 dB, D_loss lower
# Proof: Stronger discriminator improves generator quality
run_experiment 3 \
    "configs/experiment3_strong_adversarial.json" \
    "Strong Adversarial Weight 5.0"

# ========================================
# EXPERIMENT 4: Optimal Combination
# ========================================
# Expected: PSNR 25-27 dB (BEST)
# Proof: Optimal weight combination gives best PSNR
run_experiment 4 \
    "configs/experiment4_balanced_optimal.json" \
    "Optimal Balanced Combination"

# ========================================
# SUMMARY
# ========================================
echo ""
echo "=========================================="
echo "🎉 ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "📊 RESULTS SUMMARY:"
echo ""
echo "Please check logs in logbook/ for detailed metrics:"
echo "  - experiment1_ctc_annealing_low_*.log"
echo "  - experiment2_ctc_medium_*.log"
echo "  - experiment3_strong_adversarial_*.log"
echo "  - experiment4_balanced_optimal_*.log"
echo ""
echo "Expected Results to Verify:"
echo "┌────────────┬─────────────┬──────────────┬─────────────┐"
echo "│ Experiment │ CTC Weight  │ Expected PSNR│ Expected CER│"
echo "├────────────┼─────────────┼──────────────┼─────────────┤"
echo "│ Baseline   │ 1.0         │ 18.91 dB     │ 0.34        │"
echo "│ Exp 1      │ 0.1         │ 24-26 dB     │ 0.6-0.7     │"
echo "│ Exp 2      │ 0.3         │ 22-24 dB     │ 0.4-0.5     │"
echo "│ Exp 3      │ 0.5, Adv 5  │ 23-25 dB     │ 0.4-0.5     │"
echo "│ Exp 4      │ 0.2, Opt    │ 25-27 dB ✨  │ 0.5-0.6     │"
echo "└────────────┴─────────────┴──────────────┴─────────────┘"
echo ""
echo "🔍 To extract final metrics:"
echo "   tail -50 logbook/experiment*_*.log | grep PSNR"
echo ""
echo "📈 To compare results:"
echo "   poetry run python scripts/compare_experiment_results.py"
echo ""
echo "=========================================="
echo "Total runtime: ~80 minutes (4 × 20 min)"
echo "=========================================="
