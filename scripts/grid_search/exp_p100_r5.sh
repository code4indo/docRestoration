#!/bin/bash
# Grid Search Experiment: pixel_loss_weight=100.0, rec_feat_loss_weight=5.0

echo "=================================================================="
echo "🔬 Grid Search: pixel_weight=100.0, rec_feat_weight=5.0"
echo "=================================================================="

# Setup paths and python command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_CMD="poetry run python"

echo "📍 Project root: $PROJECT_ROOT"
echo "🐍 Python command: $PYTHON_CMD"

# Define unique output directories
EXP_NAME="p100_r5"
CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_grid_search/$EXP_NAME"
SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_grid_search/$EXP_NAME"
LOG_FILE="$PROJECT_ROOT/logbook/grid_search_$EXP_NAME.log"

mkdir -p "$CHECKPOINT_DIR" "$SAMPLE_DIR"
echo "📝 Log file: $LOG_FILE"

# Change to project root to run
cd "$PROJECT_ROOT" || exit 1

# Run the training with specific parameters for this experiment
$PYTHON_CMD dual_modal_gan/scripts/train32.py \
  --epochs 10 \
  --steps_per_epoch 100 \
  --pixel_loss_weight 100.0 \
  --rec_feat_loss_weight 5.0 \
  --adv_loss_weight 3.0 \
  --batch_size 2 \
  --no_restore \
  --early_stopping \
  --patience 15 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --sample_dir "$SAMPLE_DIR" 2>&1 | tee "$LOG_FILE"

echo "✅ Experiment $EXP_NAME finished."
