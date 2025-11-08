#!/bin/bash
# ==============================================================================
# ABLATION STUDY - Parallel Training for Novelty Claim
# ==============================================================================
# Purpose: Prove dual-modal superiority over single-modal baseline
# Expected Results:
#   - Single-Modal (Image-Only): ~18.5 dB PSNR
#   - Dual-Modal No-CTC: ~19.0 dB PSNR  
#   - Dual-Modal Full (GT V2): 20.23 dB PSNR ✅ (already completed)
#   - Dual-Modal Full (Pred V2): 20.07 dB PSNR ✅ (already completed)
# Novelty Claim: Dual-modal achieves +1.5 to +2.0 dB improvement
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}  ABLATION STUDY: Single-Modal vs Dual-Modal (Novelty Claim)${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

# Check if configs exist
if [ ! -f "configs/ablation_single_modal_image_only.json" ]; then
    echo -e "${RED}ERROR: Config ablation_single_modal_image_only.json not found!${NC}"
    exit 1
fi

if [ ! -f "configs/ablation_single_modal_no_ctc.json" ]; then
    echo -e "${RED}ERROR: Config ablation_single_modal_no_ctc.json not found!${NC}"
    exit 1
fi

# Check if launcher script exists
if [ ! -f "scripts/universal_train_from_json.sh" ]; then
    echo -e "${RED}ERROR: Launcher scripts/universal_train_from_json.sh not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All config files found${NC}"
echo ""

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG_IMAGE_ONLY="configs/ablation_single_modal_image_only.json"
CONFIG_NO_CTC="configs/ablation_single_modal_no_ctc.json"

LOG_IMAGE_ONLY="logbook/ablation_single_modal_image_only_$(date +%Y%m%d_%H%M%S).log"
LOG_NO_CTC="logbook/ablation_single_modal_no_ctc_$(date +%Y%m%d_%H%M%S).log"

# ==============================================================================
# Training Plan Display
# ==============================================================================

echo -e "${YELLOW}Training Plan:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 1: Single-Modal (Image-Only)"
echo "  - Config: $CONFIG_IMAGE_ONLY"
echo "  - GPU: 0"
echo "  - Discriminator: single_modal (NO text)"
echo "  - Expected PSNR: ~18.5 dB"
echo "  - Log: $LOG_IMAGE_ONLY"
echo ""
echo "Experiment 2: Dual-Modal (No CTC Loss)"
echo "  - Config: $CONFIG_NO_CTC"
echo "  - GPU: 1"
echo "  - Discriminator: enhanced_v2_fixed (dual-modal)"
echo "  - CTC Loss: DISABLED"
echo "  - Expected PSNR: ~19.0-19.5 dB"
echo "  - Log: $LOG_NO_CTC"
echo ""
echo "Baseline (Already Completed):"
echo "  - GT V2 Balanced: 20.23 dB PSNR ✅"
echo "  - Pred V2 Balanced: 20.07 dB PSNR ✅"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ==============================================================================
# Estimate Training Time
# ==============================================================================

echo -e "${BLUE}Estimated Training Time:${NC}"
echo "  - Each experiment: ~40 minutes (10 epochs × 4 min/epoch)"
echo "  - Eval interval: Every 2 epochs (2x faster than dual-modal)"
echo "  - Total parallel time: ~40 minutes (both run simultaneously)"
echo "  - Sequential would take: ~80 minutes"
echo ""

# ==============================================================================
# Confirmation
# ==============================================================================

read -p "$(echo -e ${YELLOW}Start ablation studies in parallel? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted by user${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🚀 Starting ablation studies...${NC}"
echo ""

# ==============================================================================
# Launch Experiment 1: Single-Modal (Image-Only) on GPU 0
# ==============================================================================

echo -e "${BLUE}[GPU 0]${NC} Launching Single-Modal (Image-Only)..."
nohup ./scripts/universal_train_from_json.sh "$CONFIG_IMAGE_ONLY" > "$LOG_IMAGE_ONLY" 2>&1 &
PID_IMAGE_ONLY=$!
echo -e "${GREEN}✅ [GPU 0] PID: $PID_IMAGE_ONLY${NC}"
echo -e "   Log: tail -f $LOG_IMAGE_ONLY"
echo ""

# Wait 5 seconds to ensure first process initialized
sleep 5

# ==============================================================================
# Launch Experiment 2: Dual-Modal (No CTC) on GPU 1
# ==============================================================================

echo -e "${BLUE}[GPU 1]${NC} Launching Dual-Modal (No CTC)..."
nohup ./scripts/universal_train_from_json.sh "$CONFIG_NO_CTC" > "$LOG_NO_CTC" 2>&1 &
PID_NO_CTC=$!
echo -e "${GREEN}✅ [GPU 1] PID: $PID_NO_CTC${NC}"
echo -e "   Log: tail -f $LOG_NO_CTC"
echo ""

# ==============================================================================
# Monitor Progress
# ==============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}Training Started!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Process IDs:"
echo "  [GPU 0] Single-Modal (Image-Only): PID $PID_IMAGE_ONLY"
echo "  [GPU 1] Dual-Modal (No CTC): PID $PID_NO_CTC"
echo ""
echo "Monitor commands:"
echo "  watch -n 30 'nvidia-smi'                    # GPU usage"
echo "  tail -f $LOG_IMAGE_ONLY     # GPU 0 log"
echo "  tail -f $LOG_NO_CTC         # GPU 1 log"
echo ""
echo "Check if running:"
echo "  ps aux | grep train_enhanced"
echo "  kill $PID_IMAGE_ONLY  # Stop GPU 0"
echo "  kill $PID_NO_CTC      # Stop GPU 1"
echo ""
echo -e "${GREEN}Experiments running in background...${NC}"
echo -e "${BLUE}Expected completion: ~40 minutes${NC}"
echo ""

# Save PIDs to file for later reference
echo "$PID_IMAGE_ONLY" > /tmp/ablation_image_only.pid
echo "$PID_NO_CTC" > /tmp/ablation_no_ctc.pid

echo "PID files saved:"
echo "  /tmp/ablation_image_only.pid"
echo "  /tmp/ablation_no_ctc.pid"
echo ""

# ==============================================================================
# Optional: Wait for completion
# ==============================================================================

read -p "$(echo -e ${YELLOW}Wait for training to complete? [y/N]:${NC} )" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}Waiting for both experiments to complete...${NC}"
    echo "You can safely Ctrl+C - training will continue in background"
    echo ""
    
    wait $PID_IMAGE_ONLY
    echo -e "${GREEN}✅ [GPU 0] Single-Modal (Image-Only) completed!${NC}"
    
    wait $PID_NO_CTC
    echo -e "${GREEN}✅ [GPU 1] Dual-Modal (No CTC) completed!${NC}"
    
    echo ""
    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${GREEN}  ABLATION STUDIES COMPLETED!${NC}"
    echo -e "${GREEN}========================================================================${NC}"
    echo ""
    echo "Results locations:"
    echo "  - Single-Modal: dual_modal_gan/checkpoints/ablation_single_modal_image_only/"
    echo "  - No CTC: dual_modal_gan/checkpoints/ablation_single_modal_no_ctc/"
    echo ""
    echo "Next steps:"
    echo "  1. python scripts/compare_ablation_results.py"
    echo "  2. Check final PSNR in logs"
    echo "  3. Compile results for paper Table"
    echo ""
else
    echo ""
    echo -e "${BLUE}Training continues in background${NC}"
    echo "Check progress: tail -f $LOG_IMAGE_ONLY"
    echo ""
fi
