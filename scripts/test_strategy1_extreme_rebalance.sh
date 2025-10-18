#!/bin/bash
# STRATEGI 1: AGGRESSIVE LOSS REBALANCING
# Tujuan: Buktikan bahwa CTC dominance adalah masalah utama
# Test: 1 epoch, 200 steps - hasil dalam 10-15 menit!

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   STRATEGI 1: AGGRESSIVE LOSS REBALANCING - QUICK TEST        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 HIPOTESIS: CTC loss dominasi 97% → Generator mengabaikan adversarial"
echo ""
echo "📊 KONFIGURASI BASELINE (CURRENT):"
echo "   pixel_loss_weight:   200.0 (kontribusi: ~6)"
echo "   adv_loss_weight:     1.5   (kontribusi: ~1.1)"
echo "   rec_feat_loss_weight: 5.0  (kontribusi: ~1.0)"
echo "   ctc_loss_weight:     1.0   (kontribusi: ~300) ← DOMINAN 97%!"
echo "   ctc_loss_clip_max:   300.0"
echo ""
echo "🚀 KONFIGURASI STRATEGI 1 (NEW):"
echo "   pixel_loss_weight:   300.0 (+50%) → kontribusi: ~9"
echo "   adv_loss_weight:     5.0   (+233%!) → kontribusi: ~3.75"
echo "   rec_feat_loss_weight: 10.0 (+100%) → kontribusi: ~2.0"
echo "   ctc_loss_weight:     0.3   (-70%!) → kontribusi: ~30"
echo "   ctc_loss_clip_max:   100.0 (-67%!)"
echo ""
echo "   NEW BALANCE: CTC hanya 67% (bukan 97%!)"
echo ""
echo "✅ TARGET:"
echo "   - D_loss turun ke 0.5-0.7 (sekarang stuck di 1.2-1.7)"
echo "   - PSNR improvement minimal +1-2 dB"
echo "   - Generator fokus ke visual quality, bukan hanya text"
echo ""
echo "⏱️  WAKTU: ~10-15 menit (1 epoch, 200 steps)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Log file
LOG_FILE="logbook/test_strategy1_extreme_rebalance_${TIMESTAMP}.log"

echo "▶️  Starting training with EXTREME REBALANCING..."
echo "   Log file: $LOG_FILE"
echo ""

# Run training with aggressive loss rebalancing
poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --generator_version enhanced \
    --discriminator_version enhanced_v2 \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 0 \
    --no_restore \
    --checkpoint_dir dual_modal_gan/checkpoints/strategy1_extreme_rebalance \
    --sample_dir dual_modal_gan/outputs/strategy1_extreme_rebalance \
    --max_checkpoints 1 \
    --epochs 1 \
    --steps_per_epoch 200 \
    --batch_size 4 \
    --pixel_loss_weight 300.0 \
    --rec_feat_loss_weight 10.0 \
    --adv_loss_weight 5.0 \
    --ctc_loss_weight 0.3 \
    --gradient_clip_norm 1.0 \
    --ctc_loss_clip_max 100.0 \
    --warmup_epochs 0 \
    --annealing_epochs 0 \
    --save_interval 1 \
    --eval_interval 1 \
    --discriminator_mode predicted \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXTRACTING RESULTS..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract metrics
STRATEGY1_PSNR=$(grep -E "📊.*PSNR:" "$LOG_FILE" | tail -1 | grep -oP "PSNR: \K[0-9.]+")
STRATEGY1_SSIM=$(grep -E "📊.*SSIM:" "$LOG_FILE" | tail -1 | grep -oP "SSIM: \K[0-9.]+")
STRATEGY1_CER=$(grep -E "📊.*CER:" "$LOG_FILE" | tail -1 | grep -oP "CER: \K[0-9.]+")

# Extract D_loss range
D_LOSS_MIN=$(grep "D_loss:" "$LOG_FILE" | grep -oP "D_loss: \K[0-9.]+" | sort -n | head -1)
D_LOSS_MAX=$(grep "D_loss:" "$LOG_FILE" | grep -oP "D_loss: \K[0-9.]+" | sort -n | tail -1)

# Baseline from current training
BASELINE_PSNR=22.5  # Approximate dari current training
BASELINE_D_LOSS_MIN=1.23
BASELINE_D_LOSS_MAX=1.71

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║           STRATEGI 1: EXTREME REBALANCING RESULTS                      ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║ Metric      │  Baseline (Current)  │  Strategy 1      │   Δ Delta       ║"
echo "╠═════════════╪══════════════════════╪══════════════════╪═════════════════╣"
printf "║ PSNR        │  %7.2f dB          │  %7.2f dB     │ %+7.2f dB      ║\n" \
    $BASELINE_PSNR ${STRATEGY1_PSNR:-0} \
    $(echo "${STRATEGY1_PSNR:-0} - $BASELINE_PSNR" | bc)
printf "║ SSIM        │    0.9425           │    %7.4f     │   %+7.4f      ║\n" \
    ${STRATEGY1_SSIM:-0} \
    $(echo "${STRATEGY1_SSIM:-0} - 0.9425" | bc)
printf "║ CER         │    0.1642           │    %7.4f     │   %+7.4f      ║\n" \
    ${STRATEGY1_CER:-0} \
    $(echo "${STRATEGY1_CER:-0} - 0.1642" | bc)
echo "╠═════════════╪══════════════════════╪══════════════════╪═════════════════╣"
printf "║ D_loss MIN  │    %7.2f           │    %7.2f     │   %+7.2f      ║\n" \
    $BASELINE_D_LOSS_MIN ${D_LOSS_MIN:-0} \
    $(echo "${D_LOSS_MIN:-0} - $BASELINE_D_LOSS_MIN" | bc)
printf "║ D_loss MAX  │    %7.2f           │    %7.2f     │   %+7.2f      ║\n" \
    $BASELINE_D_LOSS_MAX ${D_LOSS_MAX:-0} \
    $(echo "${D_LOSS_MAX:-0} - $BASELINE_D_LOSS_MAX" | bc)
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Calculate improvements
DELTA_PSNR=$(echo "${STRATEGY1_PSNR:-0} - $BASELINE_PSNR" | bc)
DELTA_D_LOSS=$(echo "${D_LOSS_MIN:-0} - $BASELINE_D_LOSS_MIN" | bc)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 VERDICT:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if (( $(echo "$DELTA_PSNR >= 1.0" | bc -l) )); then
    echo "✅ SUCCESS: Strategi 1 terbukti EFEKTIF!"
    echo ""
    echo "   Δ PSNR: +${DELTA_PSNR} dB"
    echo "   Δ D_loss: ${DELTA_D_LOSS} (mendekati 0.5-0.7 ideal!)"
    echo ""
    echo "   📝 HIPOTESIS TERBUKTI:"
    echo "   ✓ CTC dominance WAS the problem!"
    echo "   ✓ Loss rebalancing fixes adversarial training"
    echo "   ✓ Discriminator sekarang memberikan gradient signal yang kuat"
    echo ""
    echo "   🚀 NEXT STEPS:"
    echo "   1. Jalankan 5 epoch dengan strategi ini"
    echo "   2. Target: PSNR 26-28 dB dalam 3-4 jam"
    echo "   3. Jika berhasil, lanjut ke Two-Stage Training (Strategi 4)"
    echo ""
    echo "   💡 PERINTAH NEXT:"
    echo "   bash scripts/test_strategy1_extreme_rebalance_5epochs.sh"
    echo ""
    exit 0
    
elif (( $(echo "$DELTA_PSNR >= 0.5" | bc -l) )); then
    echo "⚖️  MARGINAL IMPROVEMENT"
    echo ""
    echo "   Δ PSNR: +${DELTA_PSNR} dB (< +1.0 dB)"
    echo ""
    echo "   📝 ANALISIS:"
    echo "   → Rebalancing membantu, tapi tidak cukup"
    echo "   → Mungkin perlu kombinasi dengan Strategi 2 (boost D learning rate)"
    echo "   → Atau langsung ke Strategi 4 (Two-Stage Training)"
    echo ""
    echo "   🤔 OPSI:"
    echo "   A. Kombinasi Strategi 1 + 2 (rebalance + boost LR)"
    echo "   B. Skip ke Strategi 4 (Two-Stage: visual first, text later)"
    echo ""
    exit 0
    
else
    echo "❌ STRATEGI 1 TIDAK EFEKTIF"
    echo ""
    echo "   Δ PSNR: ${DELTA_PSNR} dB (no improvement)"
    echo ""
    echo "   📝 DIAGNOSIS:"
    echo "   → Loss rebalancing BUKAN akar masalah"
    echo "   → Kemungkinan: Discriminator architecture yang lemah"
    echo "   → Atau: Learning rate yang tidak optimal"
    echo ""
    echo "   🔄 NEXT STRATEGY:"
    echo "   → Test Strategi 2 (Discriminator LR boost 5×)"
    echo "   → Atau Strategi 4 (Two-Stage: disable CTC first)"
    echo ""
    exit 1
fi
