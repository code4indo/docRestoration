#!/bin/bash
# Test ENHANCED V1 dengan 2 epochs (400 steps total)
# Baseline 1 epoch: PSNR=13.30 dB
# Target: PSNR > 18 dB untuk membuktikan V1 lebih baik dengan training lebih lama

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 ENHANCED V1 Extended Training (2 Epochs x 200 Steps)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logbook/test_v1_2epochs_${TIMESTAMP}.log"

echo "📊 Training Configuration:"
echo "   Generator: ENHANCED V1 (ResBlocks + Attention)"
echo "   Parameters: 21.8M"
echo "   Epochs: 2 (double dari baseline)"
echo "   Steps per epoch: 200"
echo "   Total steps: 400"
echo "   Batch size: 2"
echo "   Seed: 42 (reproducible)"
echo ""
echo "📈 Baseline (1 epoch, 200 steps):"
echo "   PSNR: 13.30 dB"
echo "   SSIM: 0.7951"
echo "   CER:  0.6214"
echo "   WER:  0.7281"
echo ""
echo "🎯 Target (2 epochs, 400 steps):"
echo "   PSNR > 18 dB (35% improvement)"
echo "   SSIM > 0.85"
echo "   CER < 0.50 (better HTR)"
echo ""

echo "▶️  Starting training at $(date)..."
echo ""

poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --generator_version enhanced \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 0 \
    --no_restore \
    --checkpoint_dir dual_modal_gan/checkpoints/test_v1_2epochs \
    --sample_dir dual_modal_gan/outputs/samples_test_v1_2epochs \
    --max_checkpoints 2 \
    --epochs 2 \
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
    2>&1 | tee "$LOG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Extracting Results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract epoch 1 metrics (for comparison)
EPOCH1_PSNR=$(grep -E "📊.*PSNR:" "$LOG_FILE" | sed -n '1p' | grep -oP "PSNR: \K[0-9.]+")
EPOCH1_SSIM=$(grep -E "📊.*SSIM:" "$LOG_FILE" | sed -n '1p' | grep -oP "SSIM: \K[0-9.]+")
EPOCH1_CER=$(grep -E "📊.*CER:" "$LOG_FILE" | sed -n '1p' | grep -oP "CER: \K[0-9.]+")

# Extract epoch 2 metrics (final)
EPOCH2_PSNR=$(grep -E "📊.*PSNR:" "$LOG_FILE" | tail -1 | grep -oP "PSNR: \K[0-9.]+")
EPOCH2_SSIM=$(grep -E "📊.*SSIM:" "$LOG_FILE" | tail -1 | grep -oP "SSIM: \K[0-9.]+")
EPOCH2_CER=$(grep -E "📊.*CER:" "$LOG_FILE" | tail -1 | grep -oP "CER: \K[0-9.]+")
EPOCH2_WER=$(grep -E "📊.*WER:" "$LOG_FILE" | tail -1 | grep -oP "WER: \K[0-9.]+")

# Baseline
BASELINE_PSNR=13.30
BASELINE_SSIM=0.7951
BASELINE_CER=0.6214
BASELINE_WER=0.7281

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║           ENHANCED V1 Training Progress Analysis                       ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║ Metric  │  Baseline  │  Epoch 1   │  Epoch 2   │  Δ vs Baseline       ║"
echo "║         │  (prev)    │  (200 st)  │  (400 st)  │                      ║"
echo "╠═════════╪════════════╪════════════╪════════════╪══════════════════════╣"
printf "║ PSNR    │  %6.2f dB │  %6.2f dB │  %6.2f dB │  %+6.2f dB (%+5.1f%%)  ║\n" \
    $BASELINE_PSNR ${EPOCH1_PSNR:-0} ${EPOCH2_PSNR:-0} \
    $(echo "$EPOCH2_PSNR - $BASELINE_PSNR" | bc) \
    $(echo "($EPOCH2_PSNR - $BASELINE_PSNR) / $BASELINE_PSNR * 100" | bc -l)
printf "║ SSIM    │   %6.4f  │   %6.4f  │   %6.4f  │  %+7.4f (%+5.1f%%)  ║\n" \
    $BASELINE_SSIM ${EPOCH1_SSIM:-0} ${EPOCH2_SSIM:-0} \
    $(echo "$EPOCH2_SSIM - $BASELINE_SSIM" | bc) \
    $(echo "($EPOCH2_SSIM - $BASELINE_SSIM) / $BASELINE_SSIM * 100" | bc -l)
printf "║ CER     │   %6.4f  │   %6.4f  │   %6.4f  │  %+7.4f (%+5.1f%%)  ║\n" \
    $BASELINE_CER ${EPOCH1_CER:-0} ${EPOCH2_CER:-0} \
    $(echo "$EPOCH2_CER - $BASELINE_CER" | bc) \
    $(echo "($EPOCH2_CER - $BASELINE_CER) / $BASELINE_CER * 100" | bc -l)
printf "║ WER     │   %6.4f  │      -     │   %6.4f  │  %+7.4f (%+5.1f%%)  ║\n" \
    $BASELINE_WER ${EPOCH2_WER:-0} \
    $(echo "$EPOCH2_WER - $BASELINE_WER" | bc) \
    $(echo "($EPOCH2_WER - $BASELINE_WER) / $BASELINE_WER * 100" | bc -l)
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Verdict
if (( $(echo "$EPOCH2_PSNR >= 18.0" | bc -l) )); then
    echo "🏆 SUCCESS: V1 mencapai PSNR > 18 dB dengan 2 epochs!"
    echo "   Extended training TERBUKTI efektif"
    echo "   Recommendation: Lanjutkan ke 5-10 epochs untuk target PSNR > 20"
    exit 0
elif (( $(echo "$EPOCH2_PSNR > $BASELINE_PSNR + 2.0" | bc -l) )); then
    echo "✅ GOOD: PSNR meningkat signifikan (+$(echo "$EPOCH2_PSNR - $BASELINE_PSNR" | bc) dB)"
    echo "   V1 menunjukkan improvement dengan extended training"
    echo "   Recommendation: Test dengan 5 epochs untuk trajectory analysis"
    exit 0
else
    echo "⚠️  MARGINAL: Improvement kurang dari +2 dB"
    echo "   Pertimbangkan hyperparameter tuning atau data augmentation"
    exit 1
fi
