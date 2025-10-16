#!/bin/bash
# Solution 1: LR Scheduling - SMOKE TEST (5 epochs)
# Purpose: Quick validation if LR scheduling improves PSNR
# Expected: PSNR improvement within 5 epochs, otherwise escalate to Solution 3

echo "🔬 SOLUTION 1: LR SCHEDULING - SMOKE TEST"
echo "========================================"
echo "Strategy: Cosine Annealing LR Decay"
echo "Test Duration: 5 epochs (~1.5 hours)"
echo "Success Criteria: PSNR >27 dB (improvement from 26.65 dB)"
echo "Baseline: Epoch 22 checkpoint (PSNR 26.65 dB)"
echo ""

# Create checkpoint directory for Solution 1
CKPT_DIR="dual_modal_gan/outputs/checkpoints_fp32_solution1_smoke"
mkdir -p "$CKPT_DIR"

# Copy best checkpoint from previous training (Epoch 22)
echo "📦 Copying Epoch 22 checkpoint (best model)..."
PREV_CKPT="dual_modal_gan/outputs/checkpoints_fp32_full_unlimited"
if [ -d "$PREV_CKPT" ]; then
    cp -r "$PREV_CKPT"/* "$CKPT_DIR/" 2>/dev/null || true
    echo "✅ Checkpoint copied from: $PREV_CKPT"
else
    echo "⚠️  No previous checkpoint found, starting fresh"
fi

# Activate virtual environment
source .venv/bin/activate

# Run smoke test with Solution 1
poetry run python dual_modal_gan/scripts/train32.py \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 1 \
    --checkpoint_dir "$CKPT_DIR" \
    --sample_dir dual_modal_gan/outputs/samples_solution1_smoke \
    --max_checkpoints 2 \
    --epochs 5 \
    --steps_per_epoch 0 \
    --batch_size 2 \
    --pixel_loss_weight 200.0 \
    --rec_feat_loss_weight 5.0 \
    --adv_loss_weight 1.5 \
    --ctc_loss_weight 1.0 \
    --use_lr_schedule \
    --lr_decay_epochs 5 \
    --lr_g 0.0002 \
    --lr_d 0.0002 \
    --lr_alpha 0.0 \
    --early_stopping \
    --patience 20 \
    --min_delta 0.01 \
    --warmup_epochs 0 \
    --annealing_epochs 0 \
    --seed 42

echo ""
echo "🎯 SMOKE TEST COMPLETED"
echo "======================="
echo "Check results:"
echo "  1. Best PSNR in last 5 epochs"
echo "  2. If PSNR >27 dB → Solution 1 WORKS, run full training"
echo "  3. If PSNR ≤26.65 dB → Solution 1 FAILED, escalate to Solution 3"
echo ""
echo "Next steps:"
echo "  - SUCCESS: Run scripts/train32_solution1_full.sh (50 epochs)"
echo "  - FAILURE: Implement Solution 3 (Architecture upgrade)"
