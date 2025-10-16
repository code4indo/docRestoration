#!/bin/bash
# ============================================================================
# FIXED Solution 1 Smoke Test: LR Scheduling (5 epochs fast validation)
# ============================================================================
# FIX: lr_alpha changed from 0.0 → 0.00002 (10% of initial LR)
# REASON: lr_alpha=0.0 caused NaN at step 1762 due to LR decay to zero
# 
# Date: 2025-10-16 10:20 WIB
# Problem: NaN values after 83% of epoch 1 (step 1762/2133)
# Root Cause: Cosine decay with alpha=0.0 → LR decayed too fast → numerical instability
# Solution: Set alpha=0.00002 (10% of initial 0.0002) to keep minimum trainable LR
# ============================================================================

set -e

echo "🔧 SOLUTION 1 (FIXED): LR Scheduling Smoke Test"
echo "   Duration: 5 epochs (~1.5 hours)"
echo "   Purpose: Fast validation of LR scheduling effectiveness"
echo "   FIX: lr_alpha=0.00002 (was 0.0) to prevent NaN"
echo ""

# Create checkpoint directory
CKPT_DIR="dual_modal_gan/checkpoints/solution1_smoke_fixed"
mkdir -p "$CKPT_DIR"

# Check if we should restore from existing checkpoint
if [ -f "${CKPT_DIR}/ckpt-5.index" ]; then
    echo "📂 Found existing checkpoint ckpt-5, will continue training..."
    echo "model_checkpoint_path: \"ckpt-5\"" > "$CKPT_DIR/checkpoint"
    echo "all_model_checkpoint_paths: \"ckpt-5\"" >> "$CKPT_DIR/checkpoint"
else
    # ✅ Use latest clean checkpoint (ckpt-9 from fp32 training)
    echo "📂 No existing checkpoint, using clean checkpoint from fp32 training (ckpt-9)..."
    BASELINE_CKPT="dual_modal_gan/outputs/checkpoints_fp32/ckpt-9"
    if [ -f "${BASELINE_CKPT}.index" ]; then
        cp "${BASELINE_CKPT}."* "$CKPT_DIR/"
        # Rename to ckpt-0 so training starts fresh with LR schedule
        for file in ${CKPT_DIR}/ckpt-9.*; do
            newfile=$(echo $file | sed 's/ckpt-9/ckpt-0/')
            mv "$file" "$newfile"
        done
        echo "✅ Checkpoint copied and renamed to ckpt-0"
    else
        echo "❌ ERROR: Baseline checkpoint not found!"
        echo "   Looking for: ${BASELINE_CKPT}.index"
        echo "   Available checkpoints:"
        find dual_modal_gan/outputs -name "ckpt-*.index" | head -5
        exit 1
    fi

    # Update checkpoint file to point to ckpt-0
    echo "model_checkpoint_path: \"ckpt-0\"" > "$CKPT_DIR/checkpoint"
    echo "all_model_checkpoint_paths: \"ckpt-0\"" >> "$CKPT_DIR/checkpoint"
fi

echo ""
echo "🚀 Starting FIXED smoke test training..."
echo "   ⚠️ CRITICAL FIX: lr_alpha=0.00002 (not 0.0)"
echo "   Expected LR range: 0.0002 → 0.00002 (10x reduction, still trainable)"
echo ""

# Run training in background
nohup poetry run python dual_modal_gan/scripts/train32.py \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 1 \
    --checkpoint_dir "$CKPT_DIR" \
    --sample_dir dual_modal_gan/outputs/samples_solution1_smoke_fixed \
    --max_checkpoints 2 \
    --epochs 10 \
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
    --lr_alpha 0.00002 \
    --early_stopping \
    --patience 20 \
    --min_delta 0.01 \
    --warmup_epochs 0 \
    --annealing_epochs 0 \
    --seed 42 \
    > logbook/solution1_smoke_test_FIXED_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAIN_PID=$!
echo "✅ Training started with PID: $TRAIN_PID"
echo "📊 Monitor progress: tail -f logbook/solution1_smoke_test_FIXED_*.log"
echo ""
echo "🔍 Expected Results (if LR scheduling works):"
echo "   - Epoch 1-2: PSNR should be >= 26.65 dB (baseline)"
echo "   - Epoch 3-5: PSNR should improve to > 27.0 dB"
echo "   - No NaN values (LR stays above 0.00002)"
echo ""
echo "🧪 Success Criteria:"
echo "   ✅ Best PSNR > 27.0 dB → Run full 50 epoch training"
echo "   ⚠️ Best PSNR 26.65-27.0 dB → Marginal, consider Solution 3"
echo "   ❌ Best PSNR ≤ 26.65 dB or NaN → Escalate to Solution 3"
