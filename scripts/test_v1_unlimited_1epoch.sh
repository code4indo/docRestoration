#!/bin/bash

# ============================================================================
# UNLIMITED DATA EXPERIMENT - V1 ENHANCED GENERATOR (1 Epoch Proof-of-Concept)
# ============================================================================
# Hypothesis: steps_per_epoch=200 limit wastes 91.6% training data
# Solution: Remove limit (steps_per_epoch=0) to use FULL dataset
# Expected: Massive PSNR improvement from full data exposure
# ============================================================================

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logbook/test_v1_unlimited_${TIMESTAMP}.log"

echo "╔════════════════════════════════════════════════════════════════╗" | tee -a "$LOGFILE"
echo "║     UNLIMITED DATA EXPERIMENT - V1 (1 Epoch Proof-of-Concept) ║" | tee -a "$LOGFILE"
echo "╠════════════════════════════════════════════════════════════════╣" | tee -a "$LOGFILE"
echo "║ Configuration:                                                 ║" | tee -a "$LOGFILE"
echo "║ - Generator: ENHANCED V1 (21.8M params, proven architecture)  ║" | tee -a "$LOGFILE"
echo "║ - Steps per epoch: 0 (UNLIMITED - auto-calculate from data)   ║" | tee -a "$LOGFILE"
echo "║ - Epochs: 1 (proof-of-concept, validate hypothesis)           ║" | tee -a "$LOGFILE"
echo "║ - Batch size: 2                                               ║" | tee -a "$LOGFILE"
echo "║ - Loss weights: pixel=200, rec_feat=5, adv=1.5 (Grid Search)  ║" | tee -a "$LOGFILE"
echo "║                                                                ║" | tee -a "$LOGFILE"
echo "║ Expected dataset utilization:                                 ║" | tee -a "$LOGFILE"
echo "║ - LIMITED (200 steps): 400/4,739 samples = 8.4% only          ║" | tee -a "$LOGFILE"
echo "║ - UNLIMITED (auto):    ~4,266/4,739 samples = 90% utilization ║" | tee -a "$LOGFILE"
echo "║ - Steps expected: ~2,133 (vs 200 limited = 10.67× more!)      ║" | tee -a "$LOGFILE"
echo "╚════════════════════════════════════════════════════════════════╝" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "⏱️  Start time: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Activate virtual environment
source .venv/bin/activate

# Run training with UNLIMITED data
poetry run python dual_modal_gan/scripts/train_enhanced.py \
  --generator_version enhanced \
  --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
  --charset_path real_data_preparation/real_data_charlist.txt \
  --recognizer_weights models/best_htr_recognizer/best_model.weights.h5 \
  --gpu_id 0 \
  --no_restore \
  --checkpoint_dir dual_modal_gan/checkpoints/unlimited_v1_test \
  --sample_dir dual_modal_gan/outputs/unlimited_v1_test \
  --max_checkpoints 2 \
  --epochs 1 \
  --steps_per_epoch 0 \
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
  2>&1 | tee "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "⏱️  End time: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Extract results
echo "╔════════════════════════════════════════════════════════════════╗" | tee -a "$LOGFILE"
echo "║                 EXPERIMENT RESULTS SUMMARY                     ║" | tee -a "$LOGFILE"
echo "╚════════════════════════════════════════════════════════════════╝" | tee -a "$LOGFILE"

# Extract epoch 1 metrics
EPOCH1_PSNR=$(grep "Epoch 1/1" "$LOGFILE" | grep -oP 'PSNR: \K[0-9.]+' | tail -1)
EPOCH1_SSIM=$(grep "Epoch 1/1" "$LOGFILE" | grep -oP 'SSIM: \K[0-9.]+' | tail -1)
EPOCH1_CER=$(grep "Epoch 1/1" "$LOGFILE" | grep -oP 'CER: \K[0-9.]+' | tail -1)
EPOCH1_WER=$(grep "Epoch 1/1" "$LOGFILE" | grep -oP 'WER: \K[0-9.]+' | tail -1)
EPOCH1_TIME=$(grep "Epoch 1/1" "$LOGFILE" | grep -oP '\K[0-9.]+(?=s)' | tail -1)

# Extract steps actually executed
ACTUAL_STEPS=$(grep -oP 'Epoch 1/1.*?(\d+)/(\d+)' "$LOGFILE" | grep -oP '\d+/\K\d+' | head -1)

echo "" | tee -a "$LOGFILE"
echo "📊 EPOCH 1 METRICS (UNLIMITED DATA):" | tee -a "$LOGFILE"
echo "├─ Steps executed: ${ACTUAL_STEPS:-N/A} (vs 200 limited)" | tee -a "$LOGFILE"
echo "├─ PSNR: ${EPOCH1_PSNR:-N/A} dB" | tee -a "$LOGFILE"
echo "├─ SSIM: ${EPOCH1_SSIM:-N/A}" | tee -a "$LOGFILE"
echo "├─ CER:  ${EPOCH1_CER:-N/A}" | tee -a "$LOGFILE"
echo "├─ WER:  ${EPOCH1_WER:-N/A}" | tee -a "$LOGFILE"
echo "└─ Time: ${EPOCH1_TIME:-N/A} seconds" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Comparison with LIMITED data (from previous test)
echo "📊 COMPARISON: UNLIMITED vs LIMITED DATA:" | tee -a "$LOGFILE"
echo "┌─────────────┬───────────────┬──────────────┬────────────────┐" | tee -a "$LOGFILE"
echo "│ Metric      │ LIMITED       │ UNLIMITED    │ Improvement    │" | tee -a "$LOGFILE"
echo "│             │ (200 steps)   │ (auto steps) │                │" | tee -a "$LOGFILE"
echo "├─────────────┼───────────────┼──────────────┼────────────────┤" | tee -a "$LOGFILE"
echo "│ Steps       │ 200           │ ${ACTUAL_STEPS:-N/A}        │ $(echo "scale=2; ${ACTUAL_STEPS:-0}/200" | bc)× more    │" | tee -a "$LOGFILE"
echo "│ PSNR        │ 15.93 dB      │ ${EPOCH1_PSNR:-N/A} dB   │ +$(echo "scale=2; ${EPOCH1_PSNR:-0}-15.93" | bc) dB        │" | tee -a "$LOGFILE"
echo "│ SSIM        │ 0.8465        │ ${EPOCH1_SSIM:-N/A}      │ +$(echo "scale=4; ${EPOCH1_SSIM:-0}-0.8465" | bc)       │" | tee -a "$LOGFILE"
echo "│ CER         │ 0.4860        │ ${EPOCH1_CER:-N/A}       │ $(echo "scale=4; ${EPOCH1_CER:-0}-0.4860" | bc)        │" | tee -a "$LOGFILE"
echo "│ WER         │ 0.6133        │ ${EPOCH1_WER:-N/A}       │ $(echo "scale=4; ${EPOCH1_WER:-0}-0.6133" | bc)        │" | tee -a "$LOGFILE"
echo "└─────────────┴───────────────┴──────────────┴────────────────┘" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Verdict
echo "🎯 VERDICT:" | tee -a "$LOGFILE"
if (( $(echo "${EPOCH1_PSNR:-0} > 18.0" | bc -l) )); then
    echo "   ✅ HYPOTHESIS CONFIRMED! Unlimited data significantly improves PSNR!" | tee -a "$LOGFILE"
    echo "   ✅ Data starvation WAS the bottleneck (91.6% data waste penalty)" | tee -a "$LOGFILE"
    echo "   📈 Proceed with FULL training (50 epochs, unlimited steps)" | tee -a "$LOGFILE"
elif (( $(echo "${EPOCH1_PSNR:-0} > 16.0" | bc -l) )); then
    echo "   ⚠️  Improvement detected but not as significant as expected" | tee -a "$LOGFILE"
    echo "   🔍 Investigate: May need architecture changes + unlimited data" | tee -a "$LOGFILE"
else
    echo "   ❌ Hypothesis REJECTED: Data quantity not the primary issue" | tee -a "$LOGFILE"
    echo "   🔍 Root cause is architectural or loss function, not data starvation" | tee -a "$LOGFILE"
fi

echo "" | tee -a "$LOGFILE"
echo "📝 Full log saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "✅ Experiment completed!" | tee -a "$LOGFILE"
