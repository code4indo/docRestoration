#!/bin/bash
# FULL comparison test: ENHANCED vs ENHANCED_V2 (200 steps each)
# Fair comparison untuk menilai arsitektur V2

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚔️  ENHANCED V1 vs V2 Comparison (200 steps each)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "📊 Baseline (from previous test):"
echo "   ENHANCED V1: PSNR=13.30, SSIM=0.7951, CER=0.6214"
echo ""
echo "🎯 Target for V2:"
echo "   PSNR > 15 dB → Significant improvement"
echo "   PSNR 14-15 dB → Marginal improvement"
echo "   PSNR < 14 dB → V1 is better"
echo ""

# Test V2
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Phase 1/1] Testing ENHANCED V2 (CBAM + RDB + MSFP)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LOG_V2="logbook/compare_v2_full_${TIMESTAMP}.log"

poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --generator_version enhanced_v2 \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 0 \
    --no_restore \
    --checkpoint_dir dual_modal_gan/checkpoints/compare_v2_full \
    --sample_dir dual_modal_gan/outputs/samples_compare_v2_full \
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
    2>&1 | tee "$LOG_V2"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Extracting Results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract V2 metrics
V2_PSNR=$(grep -E "📊.*PSNR:" "$LOG_V2" | tail -1 | grep -oP "PSNR: \K[0-9.]+")
V2_SSIM=$(grep -E "📊.*SSIM:" "$LOG_V2" | tail -1 | grep -oP "SSIM: \K[0-9.]+")
V2_CER=$(grep -E "📊.*CER:" "$LOG_V2" | tail -1 | grep -oP "CER: \K[0-9.]+")
V2_WER=$(grep -E "📊.*WER:" "$LOG_V2" | tail -1 | grep -oP "WER: \K[0-9.]+")

# V1 baseline
V1_PSNR=13.30
V1_SSIM=0.7951
V1_CER=0.6214
V1_WER=0.7281

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ENHANCED V1 vs V2 Comparison                  ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║ Metric  │  V1 (21.8M)  │  V2 (18.8M)  │   Δ Delta         ║"
echo "╠═════════╪══════════════╪══════════════╪═══════════════════╣"
printf "║ PSNR    │  %7.2f dB  │  %7.2f dB  │ %+7.2f dB      ║\n" $V1_PSNR $V2_PSNR $(echo "$V2_PSNR - $V1_PSNR" | bc)
printf "║ SSIM    │    %7.4f  │    %7.4f  │   %+7.4f      ║\n" $V1_SSIM $V2_SSIM $(echo "$V2_SSIM - $V1_SSIM" | bc)
printf "║ CER     │    %7.4f  │    %7.4f  │   %+7.4f      ║\n" $V1_CER $V2_CER $(echo "$V2_CER - $V1_CER" | bc)
printf "║ WER     │    %7.4f  │    %7.4f  │   %+7.4f      ║\n" $V1_WER $V2_WER $(echo "$V2_WER - $V1_WER" | bc)
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Verdict
DELTA_PSNR=$(echo "$V2_PSNR - $V1_PSNR" | bc)

if (( $(echo "$DELTA_PSNR >= 2.0" | bc -l) )); then
    echo "🏆 VERDICT: V2 WINS - Use for full training!"
    echo "   Δ PSNR: +${DELTA_PSNR} dB (significant improvement)"
    exit 0
elif (( $(echo "$DELTA_PSNR >= 0.5" | bc -l) )); then
    echo "⚖️  VERDICT: MARGINAL - Consider cost-benefit"
    echo "   Δ PSNR: +${DELTA_PSNR} dB (small improvement, 13% fewer params)"
    exit 0
else
    echo "❌ VERDICT: V1 is BETTER - Stick with V1"
    echo "   Δ PSNR: ${DELTA_PSNR} dB (V2 underperforms)"
    exit 1
fi
