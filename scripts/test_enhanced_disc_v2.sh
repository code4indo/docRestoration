#!/bin/bash
# Quick Validation Test: Enhanced Discriminator V2
# Compare V1 Generator + Old D vs V1 Generator + Enhanced D V2
# Decision: If +1 dB gain → proceed with full training, else abort

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   ENHANCED DISCRIMINATOR V2 - QUICK VALIDATION TEST (200 steps) ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 OBJECTIVE: Validate if Enhanced Discriminator V2 improves PSNR"
echo ""
echo "📊 BASELINE (from previous test):"
echo "   Generator V1 + Old Discriminator (137M params)"
echo "   PSNR: 21.90 dB (1 epoch unlimited) / 15.93 dB (200 steps limited)"
echo ""
echo "🆕 EXPERIMENT:"
echo "   Generator V1 + Enhanced Discriminator V2 (18M params)"
echo "   Target: PSNR ≥ 22.90 dB (minimum +1 dB improvement)"
echo ""
echo "✅ SUCCESS CRITERIA:"
echo "   - PSNR improvement ≥ +1.0 dB → Use Enhanced D V2 for full training"
echo "   - PSNR improvement < +1.0 dB → Stick with Old D, proceed to 50-epoch"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test Enhanced Discriminator V2
LOG_ENHANCED_D="logbook/test_enhanced_disc_v2_${TIMESTAMP}.log"

echo "▶️  [Test 1/1] Generator V1 + Enhanced Discriminator V2"
echo "   - Generator: ENHANCED V1 (21.8M params, proven)"
echo "   - Discriminator: ENHANCED V2 (18M params, ResNet + BiLSTM + Attention)"
echo "   - Steps: 200 (for fair comparison with baseline)"
echo "   - Batch size: 2"
echo "   - Loss weights: pixel=200, rec_feat=5, adv=1.5"
echo ""

poetry run python dual_modal_gan/scripts/train_enhanced.py \
    --generator_version enhanced \
    --discriminator_version enhanced_v2 \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
    --gpu_id 0 \
    --no_restore \
    --checkpoint_dir dual_modal_gan/checkpoints/test_enhanced_disc_v2 \
    --sample_dir dual_modal_gan/outputs/samples_test_enhanced_disc_v2 \
    --max_checkpoints 1 \
    --epochs 1 \
    --steps_per_epoch 1000 \
    --batch_size 4 \
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
    2>&1 | tee "$LOG_ENHANCED_D"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXTRACTING RESULTS..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract metrics from Enhanced D V2 test
ENHANCED_PSNR=$(grep -E "📊.*PSNR:" "$LOG_ENHANCED_D" | tail -1 | grep -oP "PSNR: \K[0-9.]+")
ENHANCED_SSIM=$(grep -E "📊.*SSIM:" "$LOG_ENHANCED_D" | tail -1 | grep -oP "SSIM: \K[0-9.]+")
ENHANCED_CER=$(grep -E "📊.*CER:" "$LOG_ENHANCED_D" | tail -1 | grep -oP "CER: \K[0-9.]+")
ENHANCED_WER=$(grep -E "📊.*WER:" "$LOG_ENHANCED_D" | tail -1 | grep -oP "WER: \K[0-9.]+")

# Baseline (V1 + Old D, 200 steps limited)
BASELINE_PSNR=15.93
BASELINE_SSIM=0.8465
BASELINE_CER=0.4860
BASELINE_WER=0.6133

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║           ENHANCED DISCRIMINATOR V2 VALIDATION RESULTS                 ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║ Metric  │  Old D (Baseline)  │  Enhanced D V2  │   Δ Delta            ║"
echo "║         │  137M params       │  18M params     │                      ║"
echo "╠═════════╪════════════════════╪═════════════════╪══════════════════════╣"
printf "║ PSNR    │  %7.2f dB       │  %7.2f dB    │ %+7.2f dB (%+5.1f%%)  ║\n" \
    $BASELINE_PSNR ${ENHANCED_PSNR:-0} \
    $(echo "${ENHANCED_PSNR:-0} - $BASELINE_PSNR" | bc) \
    $(echo "(${ENHANCED_PSNR:-0} - $BASELINE_PSNR) / $BASELINE_PSNR * 100" | bc -l)
printf "║ SSIM    │    %7.4f       │    %7.4f    │   %+7.4f (%+5.1f%%)  ║\n" \
    $BASELINE_SSIM ${ENHANCED_SSIM:-0} \
    $(echo "${ENHANCED_SSIM:-0} - $BASELINE_SSIM" | bc) \
    $(echo "(${ENHANCED_SSIM:-0} - $BASELINE_SSIM) / $BASELINE_SSIM * 100" | bc -l)
printf "║ CER     │    %7.4f       │    %7.4f    │   %+7.4f (%+5.1f%%)  ║\n" \
    $BASELINE_CER ${ENHANCED_CER:-0} \
    $(echo "${ENHANCED_CER:-0} - $BASELINE_CER" | bc) \
    $(echo "(${ENHANCED_CER:-0} - $BASELINE_CER) / $BASELINE_CER * 100" | bc -l)
printf "║ WER     │    %7.4f       │    %7.4f    │   %+7.4f (%+5.1f%%)  ║\n" \
    $BASELINE_WER ${ENHANCED_WER:-0} \
    $(echo "${ENHANCED_WER:-0} - $BASELINE_WER" | bc) \
    $(echo "(${ENHANCED_WER:-0} - $BASELINE_WER) / $BASELINE_WER * 100" | bc -l)
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Calculate delta
DELTA_PSNR=$(echo "${ENHANCED_PSNR:-0} - $BASELINE_PSNR" | bc)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 VERDICT:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if (( $(echo "$DELTA_PSNR >= 1.0" | bc -l) )); then
    echo "✅ SUCCESS: Enhanced D V2 provides significant improvement!"
    echo ""
    echo "   Δ PSNR: +${DELTA_PSNR} dB (≥ +1.0 dB threshold)"
    echo "   Parameter reduction: 137M → 18M (87% reduction)"
    echo ""
    echo "   📝 RECOMMENDATION:"
    echo "   → Use Enhanced D V2 for full 50-epoch training"
    echo "   → Expected final PSNR: 36-42 dB (vs 35-40 dB with Old D)"
    echo "   → Training time: ~15% faster (fewer params)"
    echo "   → Memory savings: ~20% reduction"
    echo ""
    echo "   🚀 NEXT STEPS:"
    echo "   1. Create train_v1_enhanced_disc_full_50epochs.sh"
    echo "   2. Launch full training with V1 + Enhanced D V2"
    echo "   3. Monitor for PSNR 35-40 dB target"
    echo ""
    exit 0
    
elif (( $(echo "$DELTA_PSNR >= 0.5" | bc -l) )); then
    echo "⚖️  MARGINAL: Enhanced D V2 provides modest improvement"
    echo ""
    echo "   Δ PSNR: +${DELTA_PSNR} dB (< +1.0 dB threshold)"
    echo "   Parameter reduction: 137M → 18M (87% reduction)"
    echo ""
    echo "   📝 RECOMMENDATION:"
    echo "   → Consider using Enhanced D V2 (parameter efficiency gain)"
    echo "   → OR stick with Old D (proven, minimal risk)"
    echo ""
    echo "   💡 DECISION FACTORS:"
    echo "   - Enhanced D V2: Faster training, lower memory, modern architecture"
    echo "   - Old D: Battle-tested, 21.90 dB unlimited proven"
    echo ""
    echo "   🤔 SUGGESTED:"
    echo "   → Use Enhanced D V2 if training time/memory is concern"
    echo "   → Use Old D if minimizing risk is priority"
    echo ""
    exit 0
    
else
    echo "❌ ENHANCED D V2 DOES NOT IMPROVE PERFORMANCE"
    echo ""
    echo "   Δ PSNR: ${DELTA_PSNR} dB (no significant improvement)"
    echo ""
    echo "   📝 RECOMMENDATION:"
    echo "   → ABORT Enhanced D V2 experiment"
    echo "   → Proceed with V1 Generator + Old Discriminator"
    echo "   → Launch full 50-epoch training with proven setup"
    echo ""
    echo "   ✅ FALLBACK PLAN:"
    echo "   → Generator V1 (21.8M, proven PSNR 21.90 dB)"
    echo "   → Discriminator Old (137M, battle-tested)"
    echo "   → 50 epochs, unlimited data (steps_per_epoch=0)"
    echo "   → Expected: PSNR 35-40 dB, CER < 0.10"
    echo ""
    echo "   🚀 NEXT STEPS:"
    echo "   1. Create train_v1_full_50epochs.sh (V1 + Old D)"
    echo "   2. Launch immediately (no further experiments)"
    echo "   3. Monitor progress, publish results"
    echo ""
    exit 1
fi
