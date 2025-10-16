#!/bin/bash
# Smoke Test for Cross-Modal Contrastive Loss Implementation

echo "=================================================================="
echo "🔥 Smoke Test: Contrastive Loss Implementation"
echo "=================================================================="

# Setup paths and python command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="poetry run python"

echo "📍 Project root: $PROJECT_ROOT"
echo "🐍 Python command: $PYTHON_CMD"

# Define unique output directories
EXP_NAME="smoke_test_contrastive"
CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_smoke_test/$EXP_NAME"
SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_smoke_test/$EXP_NAME"
LOG_FILE="$PROJECT_ROOT/logbook/smoke_test_contrastive.log"

mkdir -p "$CHECKPOINT_DIR" "$SAMPLE_DIR"
echo "📝 Log file: $LOG_FILE"

# Change to project root to run
cd "$PROJECT_ROOT" || exit 1

# Run the training with smoke test parameters
$PYTHON_CMD dual_modal_gan/scripts/train32.py \
  --epochs 1 \
  --steps_per_epoch 10 \
  --pixel_loss_weight 200.0 \
  --rec_feat_loss_weight 5.0 \
  --adv_loss_weight 3.0 \
  --contrastive_loss_weight 1.0 \
  --batch_size 2 \
  --no_restore \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --sample_dir "$SAMPLE_DIR" 2>&1 | tee "$LOG_FILE"

echo "✅ Smoke test $EXP_NAME finished."
