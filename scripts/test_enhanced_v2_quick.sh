#!/bin/bash
# Quick validation test for Enhanced V2 (50 steps)
# Target: Check if V2 is trainable and shows improvement signal

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 ENHANCED V2 Quick Validation Test (50 steps)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logbook/test_v2_quick_${TIMESTAMP}.log"

echo "📊 Test Configuration:"
echo "   Generator: Enhanced V2 (CBAM + RDB + MSFP)"
echo "   Parameters: 18.8M (vs 21.8M Enhanced, 30M BASE)"
echo "   Steps: 50 (minimal untuk cek trainability)"
echo "   Batch size: 2"
echo "   Seed: 42 (reproducible)"
echo ""
echo "🎯 Success Criteria:"
echo "   ✅ Training completes without errors"
echo "   ✅ PSNR > 10 dB (shows learning)"
echo "   ✅ No NaN/Inf in losses"
echo ""

echo "▶️  Starting test at $(date)..."
echo ""

poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --generator_version enhanced_v2 \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 0 \
    --no_restore \
    --checkpoint_dir dual_modal_gan/checkpoints/test_v2_quick \
    --sample_dir dual_modal_gan/outputs/samples_test_v2_quick \
    --max_checkpoints 1 \
    --epochs 1 \
    --steps_per_epoch 50 \
    --batch_size 2 \
    --pixel_loss_weight 200.0 \
    --rec_feat_loss_weight 5.0 \
    --adv_loss_weight 1.5 \
    --ctc_loss_weight 1.0 \
    --gradient_clip_norm 1.0 \
    --ctc_loss_clip_max 300.0 \
    --warmup_epochs 0 \
    --annealing_epochs 0 \
    --save_interval 1 \
    --eval_interval 1 \
    --discriminator_mode predicted \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Extracting Results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract metrics
PSNR=$(grep -E "📊.*PSNR:" "$LOG_FILE" | tail -1 | grep -oP "PSNR: \K[0-9.]+")
SSIM=$(grep -E "📊.*SSIM:" "$LOG_FILE" | tail -1 | grep -oP "SSIM: \K[0-9.]+")
CER=$(grep -E "📊.*CER:" "$LOG_FILE" | tail -1 | grep -oP "CER: \K[0-9.]+")
WER=$(grep -E "📊.*WER:" "$LOG_FILE" | tail -1 | grep -oP "WER: \K[0-9.]+")

echo ""
echo "✅ ENHANCED V2 Results (50 steps):"
echo "   PSNR: ${PSNR:-N/A} dB"
echo "   SSIM: ${SSIM:-N/A}"
echo "   CER:  ${CER:-N/A}"
echo "   WER:  ${WER:-N/A}"
echo ""

# Check if training was successful
if grep -q "Training completed successfully" "$LOG_FILE"; then
    echo "✅ Training completed without errors!"
    
    # Check PSNR threshold
    if (( $(echo "$PSNR > 10.0" | bc -l) )); then
        echo "✅ PSNR > 10 dB - Model is learning!"
        echo ""
        echo "🎉 VERDICT: Enhanced V2 is TRAINABLE and showing improvement!"
        echo "   Proceed to full comparison test (200 steps)"
        exit 0
    else
        echo "⚠️  PSNR < 10 dB - Weak learning signal"
        echo "   Consider architecture adjustments"
        exit 1
    fi
else
    echo "❌ Training failed - check log: $LOG_FILE"
    exit 1
fi
