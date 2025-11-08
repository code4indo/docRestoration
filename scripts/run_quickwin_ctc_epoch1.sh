#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# QUICK WIN EXPERIMENT: CTC from Epoch 1 + Rebalanced Losses
# ═══════════════════════════════════════════════════════════════════════════
#
# HYPOTHESIS TEST:
# - Problem: Current dual-modal shows only 3% CER improvement
# - Root Cause: CTC loss inactive during critical learning phase (Epoch 1-2)
# - Solution: Activate CTC from Epoch 1 + rebalance losses
# - Expected: ≥10% CER improvement over single-modal
#
# CHANGES vs BASELINE (exp_proof_gt_v2_balanced):
# 1. warmup_epochs: 2 → 0 (CTC active from Epoch 1!)
# 2. pixel_loss_weight: 50.0 → 20.0 (60% reduction)
# 3. ctc_loss_weight: 2.0 → 5.0 (150% increase)
# 4. adversarial_loss_weight: 2.0 → 3.0 (50% increase)
#
# TARGET:
# - Single-Modal CER: 0.323
# - Dual-Modal Baseline CER: 0.314 (3% better)
# - Dual-Modal QuickWin CER: ≤0.290 (≥10% better) ← SUCCESS!
#
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Config
CONFIG_FILE="configs/exp_quickwin_ctc_from_epoch1.json"
LAUNCHER_SCRIPT="scripts/universal_train_from_json.sh"
LOG_DIR="logs"
CHECKPOINT_DIR="dual_modal_gan/checkpoints/exp_quickwin_ctc_from_epoch1"

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/quickwin_ctc_epoch1_${TIMESTAMP}.log"

# ═══════════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════════

cat << 'EOF'

╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║            🚀 QUICK WIN EXPERIMENT: HYPOTHESIS TESTING 🚀                ║
║                                                                           ║
║              CTC from Epoch 1 + Rebalanced Loss Weights                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

EOF

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📋 EXPERIMENT DETAILS${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Config File:${NC} ${CONFIG_FILE}"
echo -e "${BLUE}Checkpoint:${NC} ${CHECKPOINT_DIR}"
echo -e "${BLUE}Log File:${NC} ${LOG_FILE}"
echo -e "${BLUE}GPU:${NC} 0"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Show Hypothesis
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🔬 HYPOTHESIS BEING TESTED${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${RED}Problem:${NC}"
echo "  Current dual-modal shows only 3% CER improvement (0.314 vs 0.323)"
echo ""
echo -e "${RED}Root Cause Analysis:${NC}"
echo "  1. CTC loss weight = 0.0 during Epoch 1-2 (warmup phase)"
echo "  2. Pixel loss dominates (93% contribution)"
echo "  3. Generator learns purely from pixel reconstruction"
echo "  4. Text supervision activated too late (Epoch 3+)"
echo ""
echo -e "${GREEN}Proposed Solution:${NC}"
echo "  1. Set warmup_epochs: 2 → 0 (CTC active from Epoch 1)"
echo "  2. Reduce pixel_loss_weight: 50.0 → 20.0 (60% reduction)"
echo "  3. Increase ctc_loss_weight: 2.0 → 5.0 (150% increase)"
echo "  4. Increase adversarial_loss_weight: 2.0 → 3.0 (50% increase)"
echo ""
echo -e "${BLUE}Expected CTC Weight Schedule:${NC}"
echo "  Epoch 1: CTC_weight = 5.0 × (1/3) = 1.67 (annealing phase)"
echo "  Epoch 2: CTC_weight = 5.0 × (2/3) = 3.33"
echo "  Epoch 3: CTC_weight = 5.0 × (3/3) = 5.00"
echo "  Epoch 4+: CTC_weight = 5.0 (full strength)"
echo ""
echo -e "${BLUE}Expected Loss Contribution:${NC}"
echo "  Baseline: Pixel 93%, CTC 4%, Adv 3%"
echo "  QuickWin: Pixel 71%, CTC 18%, Adv 11%"
echo ""
echo -e "${GREEN}Success Criteria:${NC}"
echo "  CER ≤ 0.290 (≥10% improvement over single-modal 0.323)"
echo ""
echo -e "${RED}Failure Criteria:${NC}"
echo "  CER > 0.310 (hypothesis rejected - problem not in curriculum)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}✅ PRE-FLIGHT CHECKS${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Config file exists"

# Check launcher script
if [ ! -f "$LAUNCHER_SCRIPT" ]; then
    echo -e "${RED}❌ Launcher script not found: ${LAUNCHER_SCRIPT}${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Launcher script exists"

# Check if already trained
if [ -d "$CHECKPOINT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Checkpoint directory already exists: ${CHECKPOINT_DIR}${NC}"
    echo -e "   This experiment may have been run before."
    read -p "   Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted by user.${NC}"
        exit 1
    fi
fi

# Create log directory
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓${NC} Log directory ready"

# Make launcher executable
chmod +x "$LAUNCHER_SCRIPT"
echo -e "${GREEN}✓${NC} Launcher script is executable"

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Confirmation
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}⏱️  ESTIMATED DURATION${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Training: ~280 seconds/epoch × 10 epochs = ~47 minutes"
echo "  Evaluation: ~5 minutes"
echo "  Total: ~50-55 minutes"
echo ""

echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 READY TO LAUNCH${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
read -p "Start Quick Win experiment? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted by user.${NC}"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# Launch Training
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🔥 LAUNCHING TRAINING${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Command:${NC}"
echo "  nohup ./${LAUNCHER_SCRIPT} ${CONFIG_FILE} &"
echo ""
echo -e "${BLUE}Logging to:${NC} ${LOG_FILE}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to abort launch (30 second countdown)...${NC}"
echo ""

# Countdown
for i in {30..1}; do
    echo -ne "  Launching in ${i} seconds...\r"
    sleep 1
done
echo -e "\n"

# Launch in background with nohup
echo -e "${GREEN}▶️  Starting training in background...${NC}"
nohup ./"${LAUNCHER_SCRIPT}" "${CONFIG_FILE}" > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

# Wait a bit to check if process started successfully
sleep 3

if ps -p $TRAIN_PID > /dev/null; then
    echo -e "${GREEN}✅ Training process started successfully!${NC}"
    echo -e "${GREEN}   PID: ${TRAIN_PID}${NC}"
else
    echo -e "${RED}❌ Training process failed to start!${NC}"
    echo -e "${RED}   Check log file: ${LOG_FILE}${NC}"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# Monitoring Instructions
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📊 MONITORING INSTRUCTIONS${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}1. Watch live log:${NC}"
echo "   tail -f ${LOG_FILE}"
echo ""
echo -e "${BLUE}2. Monitor GPU:${NC}"
echo "   watch -n 5 nvidia-smi"
echo ""
echo -e "${BLUE}3. Check training metrics:${NC}"
echo "   # After Epoch 2, check if CTC weight is active:"
echo "   grep 'current_ctc_weight' ${LOG_FILE}"
echo ""
echo -e "${BLUE}4. Kill training (if needed):${NC}"
echo "   kill ${TRAIN_PID}"
echo ""
echo -e "${BLUE}5. Expected CTC weight progression:${NC}"
echo "   Epoch 1: ~1.67 (annealing 1/3)"
echo "   Epoch 2: ~3.33 (annealing 2/3)"
echo "   Epoch 3: ~5.00 (annealing 3/3)"
echo "   Epoch 4+: 5.00 (full)"
echo ""
echo -e "${YELLOW}⚠️  CRITICAL: Watch for CTC weight at Epoch 1!${NC}"
echo "   If still 0.0, hypothesis test INVALID - curriculum bug exists"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Wait and Monitor Initial Progress
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}⏳ INITIAL PROGRESS CHECK (30 seconds)${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Waiting 30 seconds for training to initialize..."
sleep 30

echo ""
echo -e "${BLUE}Last 20 lines of log:${NC}"
tail -n 20 "${LOG_FILE}"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ QUICK WIN EXPERIMENT LAUNCHED SUCCESSFULLY${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Training is running in background (PID: ${TRAIN_PID})${NC}"
echo -e "${YELLOW}Monitor with: tail -f ${LOG_FILE}${NC}"
echo ""
echo -e "${CYAN}Expected completion: ~50-55 minutes${NC}"
echo ""
echo -e "${BLUE}Next steps after training completes:${NC}"
echo "  1. Run post-hoc CER evaluation"
echo "  2. Compare CER with baseline (0.323 single, 0.314 dual-GT)"
echo "  3. If CER ≤ 0.290 → Hypothesis CONFIRMED! 🎉"
echo "  4. If CER > 0.310 → Hypothesis REJECTED - need deeper investigation"
echo ""
