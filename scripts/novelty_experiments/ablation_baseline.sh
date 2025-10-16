#!/bin/bash
# Ablation Study: Baseline (Tanpa Iterative Refinement)

echo "=================================================================="
echo "🔬 Ablation Study: Baseline (Tanpa Iterative Refinement)"
echo "=================================================================="

# Setup paths and python command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_CMD="poetry run python"

echo "📍 Project root: $PROJECT_ROOT"
echo "🐍 Python command: $PYTHON_CMD"

# Define unique output directories
EXP_NAME="ablation_baseline"
CHECKPOINT_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/checkpoints_ablation/$EXP_NAME"
SAMPLE_DIR="$PROJECT_ROOT/dual_modal_gan/outputs/samples_ablation/$EXP_NAME"
LOG_FILE="$PROJECT_ROOT/logbook/ablation_$EXP_NAME.log"

mkdir -p "$CHECKPOINT_DIR" "$SAMPLE_DIR"
echo "📝 Log file: $LOG_FILE"

# Change to project root to run
cd "$PROJECT_ROOT" || exit 1

# Run training with baseline parameters (same as contrastive experiment for valid comparison)
echo "🚀 Starting baseline training (Contrastive Loss OFF - same as previous experiment)..."

$PYTHON_CMD dual_modal_gan/scripts/train32.py \
  --epochs 10 \
  --steps_per_epoch 100 \
  --pixel_loss_weight 200.0 \
  --rec_feat_loss_weight 5.0 \
  --adv_loss_weight 3.0 \
  --contrastive_loss_weight 1.0 \
  --ctc_loss_weight 1.0 \
  --batch_size 4 \
  --no_restore \
  --early_stopping \
  --patience 15 \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --sample_dir "$SAMPLE_DIR" \
  --eval_interval 1 \
  --save_interval 2 \
  --discriminator_mode predicted \
  --gpu_id 1 2>&1 | tee "$LOG_FILE"

echo "✅ Ablation baseline experiment finished."
echo "📊 Results saved to: $CHECKPOINT_DIR"
echo "🖼️ Samples saved to: $SAMPLE_DIR"
echo "📝 Log file: $LOG_FILE"