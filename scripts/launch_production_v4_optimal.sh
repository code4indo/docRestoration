#!/bin/bash

# ============================================================================
# PRODUCTION V4 TRAINING LAUNCHER - OPTIMAL EXPERIMENT 04 CONFIGURATION
# ============================================================================
# Based on: Ablation Study Experiment 04 (Pixel + Adv + Perc + CTC)
# Evidence: Best CER 29.66%, PSNR 24.76 dB, SSIM 0.9627, Combined Score 24.16
# Decision: RecFeat DISABLED (proven counterproductive, CER +0.50% degradation)
# Target: 100 epochs production training for Q1 journal publication
# ============================================================================

set -e  # Exit on error

# Configuration
CONFIG_FILE="configs/production_v4_optimal_exp04_config.json"
LOG_DIR="logs"
LOGBOOK_DIR="logbook"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/production_v4_optimal_${TIMESTAMP}.log"
LOGBOOK_FILE="${LOGBOOK_DIR}/production_v4_optimal_${TIMESTAMP}.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create directories if not exist
mkdir -p "${LOG_DIR}"
mkdir -p "${LOGBOOK_DIR}"

# Print header
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${MAGENTA}PRODUCTION V4 TRAINING - OPTIMAL EXPERIMENT 04 CONFIGURATION${NC}  ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📅 Start Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${BLUE}📄 Config:${NC} ${CONFIG_FILE}"
echo -e "${BLUE}📝 Log File:${NC} ${LOG_FILE}"
echo -e "${BLUE}📓 Logbook:${NC} ${LOGBOOK_FILE}"
echo ""

# Verify config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}❌ ERROR: Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

# Print configuration summary
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}📊 ABLATION STUDY EVIDENCE (Experiment 04):${NC}"
echo -e "${YELLOW}───────────────────────────────────────────────────────────────────${NC}"
echo -e "  ${CYAN}Configuration:${NC} Pixel + Adversarial + Perceptual + CTC"
echo -e "  ${CYAN}RecFeat Status:${NC} ${RED}DISABLED${NC} (rec_feat_loss_weight = 0.0)"
echo ""
echo -e "  ${GREEN}✓${NC} CER:  ${GREEN}29.66%${NC} (BEST among all configurations)"
echo -e "  ${GREEN}✓${NC} PSNR: ${GREEN}24.76 ± 4.71 dB${NC} (competitive)"
echo -e "  ${GREEN}✓${NC} SSIM: ${GREEN}0.9627 ± 0.0294${NC} (competitive)"
echo -e "  ${GREEN}✓${NC} Combined Score: ${GREEN}24.16${NC} (HIGHEST)"
echo ""
echo -e "  ${CYAN}vs Experiment 05 (Full):${NC}"
echo -e "    ${GREEN}▼${NC} CER: -0.50% better (29.66% vs 30.16%)"
echo -e "    ${YELLOW}≈${NC} PSNR: +0.01 dB (negligible)"
echo -e "    ${YELLOW}≈${NC} SSIM: -0.0002 (negligible)"
echo -e "    ${GREEN}▲${NC} Memory: 15-20% saving"
echo -e "    ${GREEN}▲${NC} Speed: 5-10% faster"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎯 PRODUCTION V4 TRAINING PARAMETERS:${NC}"
echo -e "${YELLOW}───────────────────────────────────────────────────────────────────${NC}"
echo -e "  ${CYAN}Epochs:${NC} 100 (extended for production quality)"
echo -e "  ${CYAN}Batch Size:${NC} 2"
echo -e "  ${CYAN}Learning Rate:${NC} 2e-4 (Generator & Discriminator)"
echo -e "  ${CYAN}LR Schedule:${NC} Enabled (warmup: 15, annealing: 30)"
echo -e "  ${CYAN}Early Stopping:${NC} Enabled (patience: 30, min_delta: 0.03)"
echo -e "  ${CYAN}Data Split:${NC} 70/15/15 (train/val/test)"
echo ""
echo -e "  ${CYAN}Loss Weights:${NC}"
echo -e "    • Pixel:      50.0"
echo -e "    • Adversarial: 3.0"
echo -e "    • Perceptual:  1.0"
echo -e "    • CTC:         0.15 (auto-balanced)"
echo -e "    • ${RED}RecFeat:     0.0 (DISABLED)${NC}"
echo ""
echo -e "  ${CYAN}Adaptive Balancing:${NC} Enabled (CTC:Visual = 40:60)"
echo -e "  ${CYAN}Gradient Clipping:${NC} norm=1.0, CTC_max=400.0"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🔬 EXPECTED PERFORMANCE (100 epochs):${NC}"
echo -e "${YELLOW}───────────────────────────────────────────────────────────────────${NC}"
echo -e "  ${CYAN}PSNR:${NC} ~30-31 dB (estimated)"
echo -e "  ${CYAN}SSIM:${NC} ~0.980-0.985 (estimated)"
echo -e "  ${CYAN}CER:${NC}  ~27-29% (estimated 5-7% improvement vs production_v3)"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Confirmation prompt
echo -e "${YELLOW}⚠️  WARNING: This will start 100-epoch training (estimated 40-50 hours)${NC}"
echo -e "${YELLOW}⚠️  Config: Experiment 04 optimal (4-component loss, NO RecFeat)${NC}"
echo -e "${YELLOW}⚠️  GPU: Will use GPU 0 (ensure availability)${NC}"
echo ""
read -p "$(echo -e ${CYAN}Continue with production training? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Training cancelled by user${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Starting Production V4 Training...${NC}"
echo ""

# Launch training in background with nohup
echo -e "${BLUE}🚀 Launching training process...${NC}"

nohup poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --config "${CONFIG_FILE}" \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!

echo -e "${GREEN}✓ Training process launched (PID: ${TRAIN_PID})${NC}"
echo -e "${BLUE}📝 Log file:${NC} ${LOG_FILE}"
echo -e "${BLUE}📊 Monitor progress:${NC} tail -f ${LOG_FILE}"
echo ""

# Wait a few seconds and check if process is still running
sleep 5

if ps -p ${TRAIN_PID} > /dev/null; then
    echo -e "${GREEN}✓ Training process confirmed running${NC}"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎯 NEXT STEPS:${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "  1. Monitor training: ${YELLOW}tail -f ${LOG_FILE}${NC}"
    echo -e "  2. Check GPU usage:  ${YELLOW}watch -n 1 nvidia-smi${NC}"
    echo -e "  3. View samples:     ${YELLOW}ls -lht dual_modal_gan/outputs/samples_production_v4_optimal/${NC}"
    echo -e "  4. Expected duration: ${YELLOW}~40-50 hours (100 epochs)${NC}"
    echo -e "  5. Best model saved: ${YELLOW}dual_modal_gan/checkpoints/production_v4_optimal/best_model/${NC}"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}📚 RESEARCH DOCUMENTATION:${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "  • Ablation analysis:  ${YELLOW}catatan/ANALISIS_MENDALAM_EKSPERIMEN_04_OPTIMAL.md${NC}"
    echo -e "  • RecFeat analysis:   ${YELLOW}catatan/ANALISIS_MENDALAM_RECFEAT_INEFFECTIVENESS.md${NC}"
    echo -e "  • Visual quality Q&A: ${YELLOW}catatan/CLARIFICATION_EXP05_VISUAL_QUALITY_VS_HTR_PERFORMANCE.md${NC}"
    echo -e "  • Paper section:      ${YELLOW}Paper/main/jatniko_id.tex (Section V.C.4)${NC}"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✨ Production V4 training successfully launched!${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
else
    echo -e "${RED}❌ ERROR: Training process failed to start${NC}"
    echo -e "${YELLOW}Check log file for errors: ${LOG_FILE}${NC}"
    exit 1
fi

# Create initial logbook entry
cat > "${LOGBOOK_FILE}" << EOF
═══════════════════════════════════════════════════════════════════
PRODUCTION V4 TRAINING LOG - OPTIMAL EXPERIMENT 04 CONFIGURATION
═══════════════════════════════════════════════════════════════════

Start Time: $(date '+%Y-%m-%d %H:%M:%S')
Config: ${CONFIG_FILE}
Training PID: ${TRAIN_PID}
Log File: ${LOG_FILE}

───────────────────────────────────────────────────────────────────
CONFIGURATION SUMMARY
───────────────────────────────────────────────────────────────────

Loss Configuration: Experiment 04 Optimal (4-component)
  ✓ Pixel Loss:       50.0
  ✓ Adversarial Loss:  3.0
  ✓ Perceptual Loss:   1.0
  ✓ CTC Loss:          0.15 (adaptive balancing)
  ✗ RecFeat Loss:      0.0 (DISABLED - proven counterproductive)

Ablation Study Evidence (15 epochs):
  • CER:  29.66% (BEST among all configurations)
  • PSNR: 24.76 ± 4.71 dB (competitive)
  • SSIM: 0.9627 ± 0.0294 (competitive)
  • Combined Score: 24.16 (HIGHEST)

Why Experiment 04 is Optimal:
  1. Best HTR performance (primary objective)
  2. Pareto optimal in multi-objective space
  3. Clean gradient flow (no RecFeat interference)
  4. 15-20% memory saving vs Exp 05
  5. 5-10% faster training vs Exp 05
  6. Consistent across diverse degradation types

Training Parameters:
  • Epochs: 100 (extended for production quality)
  • Batch Size: 2
  • Learning Rate: 2e-4 (G & D)
  • LR Schedule: Warmup 15 epochs, Annealing 30 epochs
  • Early Stopping: Patience 30, Min Delta 0.03
  • Data Split: 70/15/15 (train/val/test)
  • Adaptive Balancing: CTC:Visual = 40:60

Expected Performance (100 epochs):
  • PSNR: ~30-31 dB
  • SSIM: ~0.980-0.985
  • CER:  ~27-29% (5-7% improvement vs production_v3)

───────────────────────────────────────────────────────────────────
TRAINING PROGRESS
───────────────────────────────────────────────────────────────────

[Training started - monitor ${LOG_FILE} for updates]

═══════════════════════════════════════════════════════════════════
EOF

echo -e "${GREEN}✓ Logbook created: ${LOGBOOK_FILE}${NC}"
echo ""
echo -e "${BLUE}Happy training! 🚀${NC}"
echo ""
