#!/bin/bash

###############################################################################
# Comparison Test: BASE vs ENHANCED Generator
# Purpose: Quick smoke test to compare generator architectures
# Duration: ~10-15 minutes (1 epoch each, 200 steps)
###############################################################################

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║         Generator Architecture Comparison Test                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Objective: Compare ENHANCED (Residual+Attention) vs BASE (U-Net)"
echo "⚙️  Config: 1 epoch, 200 steps, identical hyperparameters"
echo "📊 Metrics: PSNR, SSIM, CER, Training Time"
echo ""

# --- Configuration ---
CHECKPOINT_BASE="dual_modal_gan/checkpoints/compare_base"
CHECKPOINT_ENHANCED="dual_modal_gan/checkpoints/compare_enhanced"
SAMPLE_BASE="dual_modal_gan/outputs/samples_compare_base"
SAMPLE_ENHANCED="dual_modal_gan/outputs/samples_compare_enhanced"
LOGDIR="logbook"

# Create directories
mkdir -p "$CHECKPOINT_BASE" "$CHECKPOINT_ENHANCED"
mkdir -p "$SAMPLE_BASE" "$SAMPLE_ENHANCED"
mkdir -p "$LOGDIR"

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

###############################################################################
# Phase 1: Test ENHANCED Generator (NEW Architecture) 🚀
###############################################################################

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 1/2: Testing ENHANCED Generator (Residual + Attention) 🚀        ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo "   Checkpoint: $CHECKPOINT_ENHANCED"
echo "   Samples: $SAMPLE_ENHANCED"
echo ""

# Run ENHANCED generator test
nohup poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --generator_version enhanced \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 0 \
    --no_restore \
    --checkpoint_dir "$CHECKPOINT_ENHANCED" \
    --sample_dir "$SAMPLE_ENHANCED" \
    --max_checkpoints 1 \
    --epochs 1 \
    --steps_per_epoch 200 \
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
    > "$LOGDIR/compare_enhanced_${TIMESTAMP}.log" 2>&1 &

ENHANCED_PID=$!
echo "✅ ENHANCED generator test started (PID: $ENHANCED_PID)"
echo "📊 Monitor: tail -f $LOGDIR/compare_enhanced_${TIMESTAMP}.log"
echo ""

# Wait for ENHANCED to complete
echo "⏳ Waiting for ENHANCED generator test to complete..."
wait $ENHANCED_PID
ENHANCED_EXIT=$?

if [ $ENHANCED_EXIT -ne 0 ]; then
    echo "❌ ENHANCED generator test FAILED (exit code: $ENHANCED_EXIT)"
    echo "   Check log: $LOGDIR/compare_enhanced_${TIMESTAMP}.log"
    exit 1
fi

echo "✅ ENHANCED generator test completed successfully!"
echo ""

# Extract ENHANCED metrics
ENHANCED_PSNR=$(grep -oP "val_psnr: \K[0-9.]+" "$LOGDIR/compare_enhanced_${TIMESTAMP}.log" | tail -1)
ENHANCED_SSIM=$(grep -oP "val_ssim: \K[0-9.]+" "$LOGDIR/compare_enhanced_${TIMESTAMP}.log" | tail -1)
ENHANCED_CER=$(grep -oP "val_cer: \K[0-9.]+" "$LOGDIR/compare_enhanced_${TIMESTAMP}.log" | tail -1)
ENHANCED_TIME=$(grep -oP "Epoch 1 completed in \K[0-9.]+" "$LOGDIR/compare_enhanced_${TIMESTAMP}.log")

echo "📊 ENHANCED Generator Results:"
echo "   PSNR: ${ENHANCED_PSNR:-N/A} dB 🚀"
echo "   SSIM: ${ENHANCED_SSIM:-N/A}"
echo "   CER:  ${ENHANCED_CER:-N/A}"
echo "   Time: ${ENHANCED_TIME:-N/A}s"
echo ""

sleep 5  # Cool down GPU

###############################################################################
# Phase 2: Test BASE Generator (Standard U-Net)
###############################################################################

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 2/2: Testing BASE Generator (Standard U-Net)                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo "   Checkpoint: $CHECKPOINT_BASE"
echo "   Samples: $SAMPLE_BASE"
echo ""

# Run BASE generator test
nohup poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --generator_version base \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 0 \
    --no_restore \
    --checkpoint_dir "$CHECKPOINT_BASE" \
    --sample_dir "$SAMPLE_BASE" \
    --max_checkpoints 1 \
    --epochs 1 \
    --steps_per_epoch 200 \
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
    > "$LOGDIR/compare_base_${TIMESTAMP}.log" 2>&1 &

BASE_PID=$!
echo "✅ BASE generator test started (PID: $BASE_PID)"
echo "📊 Monitor: tail -f $LOGDIR/compare_base_${TIMESTAMP}.log"
echo ""

# Wait for BASE to complete
echo "⏳ Waiting for BASE generator test to complete..."
wait $BASE_PID
BASE_EXIT=$?

if [ $BASE_EXIT -ne 0 ]; then
    echo "❌ BASE generator test FAILED (exit code: $BASE_EXIT)"
    echo "   Check log: $LOGDIR/compare_base_${TIMESTAMP}.log"
    exit 1
fi

echo "✅ BASE generator test completed successfully!"
echo ""

# Extract BASE metrics
BASE_PSNR=$(grep -oP "val_psnr: \K[0-9.]+" "$LOGDIR/compare_base_${TIMESTAMP}.log" | tail -1)
BASE_SSIM=$(grep -oP "val_ssim: \K[0-9.]+" "$LOGDIR/compare_base_${TIMESTAMP}.log" | tail -1)
BASE_CER=$(grep -oP "val_cer: \K[0-9.]+" "$LOGDIR/compare_base_${TIMESTAMP}.log" | tail -1)
BASE_TIME=$(grep -oP "Epoch 1 completed in \K[0-9.]+" "$LOGDIR/compare_base_${TIMESTAMP}.log")

echo "📊 BASE Generator Results:"
echo "   PSNR: ${BASE_PSNR:-N/A} dB"
echo "   SSIM: ${BASE_SSIM:-N/A}"
echo "   CER:  ${BASE_CER:-N/A}"
echo "   Time: ${BASE_TIME:-N/A}s"
echo ""

###############################################################################
# Comparison Analysis & Verdict
###############################################################################

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                         FINAL COMPARISON                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ -z "$BASE_PSNR" ] || [ -z "$ENHANCED_PSNR" ]; then
    echo "❌ Cannot generate comparison - missing PSNR values"
    echo "   BASE PSNR: ${BASE_PSNR:-MISSING}"
    echo "   ENHANCED PSNR: ${ENHANCED_PSNR:-MISSING}"
    echo ""
    echo "   Please check the log files:"
    echo "   - $LOGDIR/compare_base_${TIMESTAMP}.log"
    echo "   - $LOGDIR/compare_enhanced_${TIMESTAMP}.log"
    exit 1
else
    # Calculate deltas
    PSNR_DELTA=$(echo "$ENHANCED_PSNR - $BASE_PSNR" | bc)
    PSNR_PERCENT=$(echo "scale=2; ($ENHANCED_PSNR - $BASE_PSNR) / $BASE_PSNR * 100" | bc)
    
    SSIM_DELTA=$(echo "$ENHANCED_SSIM - $BASE_SSIM" | bc)
    SSIM_PERCENT=$(echo "scale=2; ($ENHANCED_SSIM - $BASE_SSIM) / $BASE_SSIM * 100" | bc)
    
    CER_DELTA=$(echo "$ENHANCED_CER - $BASE_CER" | bc)
    CER_PERCENT=$(echo "scale=2; ($ENHANCED_CER - $BASE_CER) / $BASE_CER * 100" | bc)
    
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                    COMPARISON SUMMARY                                ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 PSNR Comparison:"
    echo "   ENHANCED: $ENHANCED_PSNR dB 🚀"
    echo "   BASE:     $BASE_PSNR dB"
    echo "   Δ PSNR:   $PSNR_DELTA dB (${PSNR_PERCENT}%)"
    echo ""
    echo "📊 SSIM Comparison:"
    echo "   ENHANCED: $ENHANCED_SSIM"
    echo "   BASE:     $BASE_SSIM"
    echo "   Δ SSIM:   $SSIM_DELTA (${SSIM_PERCENT}%)"
    echo ""
    echo "📊 CER Comparison:"
    echo "   ENHANCED: $ENHANCED_CER"
    echo "   BASE:     $BASE_CER"
    echo "   Δ CER:    $CER_DELTA (${CER_PERCENT}%)"
    echo ""
    
if [ ! -z "$BASE_TIME" ] && [ ! -z "$ENHANCED_TIME" ]; then
    TIME_DELTA=$(echo "$ENHANCED_TIME - $BASE_TIME" | bc)
    TIME_PERCENT=$(echo "scale=2; ($ENHANCED_TIME - $BASE_TIME) / $BASE_TIME * 100" | bc)
    
    echo "⏱️  Training Time Comparison:"
    echo "   ENHANCED: ${ENHANCED_TIME}s 🚀"
    echo "   BASE:     ${BASE_TIME}s"
    echo "   Δ Time:   ${TIME_DELTA}s (${TIME_PERCENT}% overhead)"
    echo ""
fi

echo "📁 Artifacts Generated:"
echo "   ENHANCED logs: $LOGDIR/compare_enhanced_${TIMESTAMP}.log 🚀"
echo "   BASE logs:     $LOGDIR/compare_base_${TIMESTAMP}.log"
echo "   ENHANCED checkpoint: $CHECKPOINT_ENHANCED/ 🚀"
echo "   BASE checkpoint: $CHECKPOINT_BASE/"
echo "   ENHANCED samples: $SAMPLE_ENHANCED/ 🚀"
echo "   BASE samples: $SAMPLE_BASE/"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                          VERDICT                                     ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Decision logic based on PSNR delta
    if (( $(echo "$PSNR_DELTA > 2.0" | bc -l) )); then
        echo "✅ ENHANCED GENERATOR WINS! 🎉"
        echo ""
        echo "   🚀 Significant PSNR improvement: +${PSNR_DELTA} dB"
        echo "   ✅ Recommendation: Use ENHANCED generator for full training"
        echo ""
        echo "   Next Steps:"
        echo "   1. Run full 50-epoch training with ENHANCED generator"
        echo "   2. Use unlimited data (--steps_per_epoch 0)"
        echo "   3. Monitor for overfitting with validation metrics"
        echo ""
        echo "   Command:"
        echo "   bash scripts/train32_solution1_smoke_test_FIXED.sh \\"
        echo "     --generator_version enhanced \\"
        echo "     --epochs 50 \\"
        echo "     --steps_per_epoch 0"
        
    elif (( $(echo "$PSNR_DELTA >= 0.5" | bc -l) )); then
        echo "⚠️  MARGINAL IMPROVEMENT"
        echo ""
        echo "   📊 PSNR improvement: +${PSNR_DELTA} dB"
        echo "   ⚠️  Benefit is marginal - cost-benefit analysis needed"
        echo ""
        echo "   Considerations:"
        echo "   - Training time overhead: ${TIME_PERCENT}%"
        echo "   - Model complexity increase: ~30% more parameters"
        echo "   - SSIM improvement: ${SSIM_PERCENT}%"
        echo ""
        echo "   Options:"
        echo "   1. If resources allow → Use ENHANCED (potential long-term gain)"
        echo "   2. If resources limited → Stick with BASE (proven stable)"
        echo "   3. Try longer training (5 epochs) to see if gap widens"
        
    else
        echo "❌ NO SIGNIFICANT BENEFIT"
        echo ""
        echo "   📊 PSNR delta: ${PSNR_DELTA} dB (< 0.5 dB threshold)"
        echo "   ❌ Recommendation: Stick with BASE generator"
        echo ""
        echo "   Analysis:"
        echo "   - Residual blocks + Attention gates did NOT improve performance"
        echo "   - Training time overhead: ${TIME_PERCENT}%"
        echo "   - Additional complexity not justified"
        echo ""
        echo "   Possible reasons:"
        echo "   1. Dataset too small (only 200 steps) for architecture benefits"
        echo "   2. Hyperparameters not tuned for ENHANCED architecture"
        echo "   3. Bug in generator_enhanced.py implementation"
        echo ""
        echo "   Next Steps:"
        echo "   1. Review generator_enhanced.py architecture"
        echo "   2. Run longer test (5 epochs) to verify"
        echo "   3. If still no benefit → Use BASE for full training"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    COMPARISON TEST COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Review detailed logs:"
echo "   ENHANCED: tail -100 $LOGDIR/compare_enhanced_${TIMESTAMP}.log"
echo "   BASE:     tail -100 $LOGDIR/compare_base_${TIMESTAMP}.log"
echo ""
echo "🖼️  View sample images:"
echo "   ENHANCED: ls -lh $SAMPLE_ENHANCED/"
echo "   BASE:     ls -lh $SAMPLE_BASE/"
echo ""
