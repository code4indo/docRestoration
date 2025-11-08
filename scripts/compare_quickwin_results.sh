#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# COMPARE QUICK WIN RESULTS vs BASELINE
# ═══════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Checkpoints
SINGLE_MODAL_DIR="dual_modal_gan/checkpoints/ablation_single_modal_image_only"
DUAL_GT_V2_DIR="dual_modal_gan/checkpoints/exp_proof_gt_v2_balanced"
QUICKWIN_DIR="dual_modal_gan/checkpoints/exp_quickwin_ctc_from_epoch1"

# Results files
SINGLE_MODAL_METRICS="${SINGLE_MODAL_DIR}/metrics/training_metrics_fp32_final.json"
DUAL_GT_V2_METRICS="${DUAL_GT_V2_DIR}/metrics/training_metrics_fp32_final.json"
QUICKWIN_METRICS="${QUICKWIN_DIR}/metrics/training_metrics_fp32_final.json"

SINGLE_MODAL_POSTHOC="${SINGLE_MODAL_DIR}/posthoc_cer_evaluation_FIXED_V2.json"
DUAL_GT_V2_POSTHOC="${DUAL_GT_V2_DIR}/posthoc_cer_evaluation_FIXED_V2.json"
QUICKWIN_POSTHOC="${QUICKWIN_DIR}/posthoc_cer_evaluation.json"

# ═══════════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════════

cat << 'EOF'

╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              📊 QUICK WIN HYPOTHESIS TEST - RESULTS 📊                   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

EOF

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🔬 HYPOTHESIS REMINDER${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Problem:${NC}"
echo "  Dual-modal baseline shows only 3% CER improvement"
echo ""
echo -e "${YELLOW}Root Cause Hypothesis:${NC}"
echo "  CTC loss inactive during critical learning phase (Epoch 1-2)"
echo ""
echo -e "${YELLOW}Solution Tested:${NC}"
echo "  1. warmup_epochs: 2 → 0 (CTC active from Epoch 1)"
echo "  2. pixel_loss_weight: 50.0 → 20.0"
echo "  3. ctc_loss_weight: 2.0 → 5.0"
echo "  4. adversarial_loss_weight: 2.0 → 3.0"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Check Files Exist
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📁 CHECKING DATA FILES${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

check_file() {
    local file=$1
    local label=$2
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} ${label}"
        return 0
    else
        echo -e "${RED}✗${NC} ${label} - ${RED}NOT FOUND${NC}"
        echo "   Expected: ${file}"
        return 1
    fi
}

FILES_OK=true

check_file "$SINGLE_MODAL_METRICS" "Single-Modal Training Metrics" || FILES_OK=false
check_file "$DUAL_GT_V2_METRICS" "Dual-GT-V2 Training Metrics" || FILES_OK=false
check_file "$QUICKWIN_METRICS" "QuickWin Training Metrics" || FILES_OK=false

echo ""
echo -e "${BLUE}Post-hoc Evaluation Results (optional):${NC}"
check_file "$SINGLE_MODAL_POSTHOC" "Single-Modal Post-hoc" || true
check_file "$DUAL_GT_V2_POSTHOC" "Dual-GT-V2 Post-hoc" || true
check_file "$QUICKWIN_POSTHOC" "QuickWin Post-hoc" || true

if [ "$FILES_OK" = false ]; then
    echo ""
    echo -e "${RED}❌ Missing required training metrics files!${NC}"
    echo -e "${YELLOW}Please ensure all experiments have completed training.${NC}"
    exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Extract Training Metrics (Best Epoch)
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📈 TRAINING RESULTS (Best Validation Epoch)${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

extract_best_metrics() {
    local metrics_file=$1
    python3 << EOF
import json
import sys

try:
    with open('${metrics_file}', 'r') as f:
        data = json.load(f)
    
    # Find best validation epoch
    best_psnr = 0
    best_epoch = None
    
    for epoch_data in data.get('epochs', []):
        if 'validation' in epoch_data:
            val = epoch_data['validation']
            if val.get('psnr', 0) > best_psnr:
                best_psnr = val['psnr']
                best_epoch = epoch_data
    
    if best_epoch:
        val = best_epoch['validation']
        print(f"{best_epoch['epoch']}|{val.get('psnr', 0):.2f}|{val.get('ssim', 0):.4f}|{val.get('cer', 1.0):.4f}|{val.get('wer', 1.0):.4f}")
    else:
        print("N/A|0.00|0.0000|1.0000|1.0000")
except Exception as e:
    print(f"ERROR|0.00|0.0000|1.0000|1.0000", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
EOF
}

# Extract metrics
SINGLE_METRICS=$(extract_best_metrics "$SINGLE_MODAL_METRICS")
DUAL_GT_METRICS=$(extract_best_metrics "$DUAL_GT_V2_METRICS")
QUICKWIN_METRICS=$(extract_best_metrics "$QUICKWIN_METRICS")

# Parse into variables
IFS='|' read -r SM_EPOCH SM_PSNR SM_SSIM SM_CER SM_WER <<< "$SINGLE_METRICS"
IFS='|' read -r DG_EPOCH DG_PSNR DG_SSIM DG_CER DG_WER <<< "$DUAL_GT_METRICS"
IFS='|' read -r QW_EPOCH QW_PSNR QW_SSIM QW_CER QW_WER <<< "$QUICKWIN_METRICS"

# Display table
echo -e "${BOLD}┌─────────────────────┬───────┬───────────┬─────────┬─────────┬─────────┐${NC}"
echo -e "${BOLD}│ Experiment          │ Epoch │ PSNR (dB) │ SSIM    │ CER     │ WER     │${NC}"
echo -e "${BOLD}├─────────────────────┼───────┼───────────┼─────────┼─────────┼─────────┤${NC}"
printf "│ %-19s │ %5s │ %9s │ %7s │ %7s │ %7s │\n" "Single-Modal" "$SM_EPOCH" "$SM_PSNR" "$SM_SSIM" "$SM_CER" "$SM_WER"
printf "│ %-19s │ %5s │ %9s │ %7s │ %7s │ %7s │\n" "Dual-GT-V2" "$DG_EPOCH" "$DG_PSNR" "$DG_SSIM" "$DG_CER" "$DG_WER"
printf "│ %-19s │ %5s │ %9s │ %7s │ %7s │ %7s │\n" "QuickWin (CTC E1)" "$QW_EPOCH" "$QW_PSNR" "$QW_SSIM" "$QW_CER" "$QW_WER"
echo -e "${BOLD}└─────────────────────┴───────┴───────────┴─────────┴─────────┴─────────┘${NC}"

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Calculate Improvements
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📊 CER IMPROVEMENT ANALYSIS${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Calculate improvements using Python
python3 << EOF
import sys

# Parse values
try:
    sm_cer = float("${SM_CER}")
    dg_cer = float("${DG_CER}")
    qw_cer = float("${QW_CER}")
    
    # Calculate improvements
    dg_improvement = (sm_cer - dg_cer) / sm_cer * 100 if sm_cer > 0 else 0
    qw_improvement = (sm_cer - qw_cer) / sm_cer * 100 if sm_cer > 0 else 0
    
    # Absolute differences
    dg_abs = sm_cer - dg_cer
    qw_abs = sm_cer - qw_cer
    
    print(f"Single-Modal (Baseline):  CER = {sm_cer:.4f}")
    print(f"")
    print(f"Dual-GT-V2 (Old):         CER = {dg_cer:.4f}")
    print(f"  → Absolute improvement: {dg_abs:.4f}")
    print(f"  → Relative improvement: {dg_improvement:.1f}%")
    if dg_improvement < 5:
        print(f"  → Status: ❌ Marginal benefit (< 5%)")
    elif dg_improvement < 10:
        print(f"  → Status: ⚠️  Modest benefit (5-10%)")
    else:
        print(f"  → Status: ✅ Significant benefit (≥ 10%)")
    
    print(f"")
    print(f"QuickWin (CTC from E1):   CER = {qw_cer:.4f}")
    print(f"  → Absolute improvement: {qw_abs:.4f}")
    print(f"  → Relative improvement: {qw_improvement:.1f}%")
    if qw_improvement < 5:
        print(f"  → Status: ❌ Marginal benefit (< 5%)")
    elif qw_improvement < 10:
        print(f"  → Status: ⚠️  Modest benefit (5-10%)")
    else:
        print(f"  → Status: ✅ Significant benefit (≥ 10%) 🎉")
    
    print(f"")
    
    # QuickWin vs Dual-GT-V2 comparison
    qw_vs_dg = (dg_cer - qw_cer) / dg_cer * 100 if dg_cer > 0 else 0
    print(f"QuickWin vs Dual-GT-V2:")
    print(f"  → CER change: {qw_cer - dg_cer:+.4f}")
    if qw_vs_dg > 5:
        print(f"  → Improvement: +{qw_vs_dg:.1f}% ✅ QuickWin BETTER!")
    elif qw_vs_dg > 0:
        print(f"  → Improvement: +{qw_vs_dg:.1f}% (marginal)")
    elif qw_vs_dg > -5:
        print(f"  → Change: {qw_vs_dg:.1f}% (similar)")
    else:
        print(f"  → Regression: {qw_vs_dg:.1f}% ❌ QuickWin WORSE")
    
    print(f"")
    print(f"═══════════════════════════════════════════════════════════════")
    
    # Hypothesis test
    if qw_improvement >= 10:
        print(f"🎉 HYPOTHESIS CONFIRMED! 🎉")
        print(f"")
        print(f"QuickWin achieves ≥10% CER improvement over single-modal.")
        print(f"Root cause identified: CTC loss MUST be active from Epoch 1.")
        print(f"")
        print(f"Recommendation: Use QuickWin settings for production training.")
    elif qw_improvement >= 5:
        print(f"⚠️  HYPOTHESIS PARTIALLY CONFIRMED")
        print(f"")
        print(f"QuickWin shows 5-10% improvement - better than baseline (3%)")
        print(f"but below target (10%). Early CTC activation helps, but")
        print(f"additional improvements may be needed (better recognizer, etc.)")
    else:
        print(f"❌ HYPOTHESIS REJECTED")
        print(f"")
        print(f"QuickWin does NOT achieve significant improvement.")
        print(f"Problem is NOT solely in curriculum learning schedule.")
        print(f"Need to investigate other root causes:")
        print(f"  1. Discriminator effectiveness")
        print(f"  2. Recognizer quality (33.72% CER baseline)")
        print(f"  3. Architecture limitations")
        print(f"  4. Dataset difficulty")
    
except ValueError as e:
    print(f"Error parsing CER values: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# CTC Weight Verification
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🔍 CTC WEIGHT SCHEDULE VERIFICATION${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Dual-GT-V2 (Baseline):${NC}"
python3 << 'EOF'
import json

with open('dual_modal_gan/checkpoints/exp_proof_gt_v2_balanced/metrics/training_metrics_fp32_final.json', 'r') as f:
    data = json.load(f)

for i, epoch in enumerate(data['epochs'][:5], 1):
    ctc_w = epoch.get('current_ctc_weight', 0.0)
    phase = epoch.get('phase', 'unknown')
    print(f"  Epoch {i}: CTC_weight = {ctc_w:.2f} ({phase})")
EOF

echo ""
echo -e "${BLUE}QuickWin (CTC from E1):${NC}"
python3 << 'EOF'
import json
import os

metrics_file = 'dual_modal_gan/checkpoints/exp_quickwin_ctc_from_epoch1/metrics/training_metrics_fp32_final.json'

if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    for i, epoch in enumerate(data['epochs'][:5], 1):
        ctc_w = epoch.get('current_ctc_weight', 0.0)
        phase = epoch.get('phase', 'unknown')
        print(f"  Epoch {i}: CTC_weight = {ctc_w:.2f} ({phase})")
else:
    print("  Training metrics not found - training may still be running")
EOF

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Next Steps
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📋 NEXT STEPS${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}1. If Hypothesis CONFIRMED (≥10% improvement):${NC}"
echo "   - Document findings in research paper"
echo "   - Use QuickWin settings for future training"
echo "   - Run more extensive validation (larger test set)"
echo ""

echo -e "${YELLOW}2. If Hypothesis PARTIALLY CONFIRMED (5-10%):${NC}"
echo "   - Investigate additional improvements:"
echo "     • Better recognizer (fine-tune to <20% CER)"
echo "     • Text-specific discriminator loss"
echo "     • Attention-guided generation"
echo ""

echo -e "${YELLOW}3. If Hypothesis REJECTED (<5%):${NC}"
echo "   - Conduct deeper root cause analysis:"
echo "     • Discriminator effectiveness study"
echo "     • Recognizer quality investigation"
echo "     • Architecture redesign"
echo "     • Dataset difficulty assessment"
echo ""

echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ ANALYSIS COMPLETE${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
