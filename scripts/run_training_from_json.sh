#!/bin/bash
# STRATEGI 1: AGGRESSIVE LOSS REBALANCING (JSON-based config)
# Membaca konfigurasi dari configs/strategy1_extreme_rebalance.json

set -e

CONFIG_FILE="${1:-configs/strategy1_extreme_rebalance.json}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file.json]"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   STRATEGI 1: AGGRESSIVE LOSS REBALANCING (JSON Config)       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📄 Loading configuration from: $CONFIG_FILE"
echo ""

# Parse JSON using Python
PARSED=$(poetry run python -c "
import json
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# Print configuration summary
print('🎯 EXPERIMENT:', config.get('experiment_name', 'N/A'))
print('📝 DESCRIPTION:', config.get('description', 'N/A'))
print('')
print('📊 KONFIGURASI LOSS WEIGHTS:')
print(f\"   pixel_loss_weight:   {config.get('pixel_loss_weight', 'N/A')}\")
print(f\"   adv_loss_weight:     {config.get('adv_loss_weight', 'N/A')}\")
print(f\"   rec_feat_loss_weight: {config.get('rec_feat_loss_weight', 'N/A')}\")
print(f\"   ctc_loss_weight:     {config.get('ctc_loss_weight', 'N/A')}\")
print('')
print('🏃 TRAINING:')
print(f\"   epochs:              {config.get('epochs', 'N/A')}\")
print(f\"   steps_per_epoch:     {config.get('steps_per_epoch', 'N/A')}\")
print(f\"   batch_size:          {config.get('batch_size', 'N/A')}\")
print('')

# For backward compatibility with old structure
weights = config.get('loss_weights', config)  # Fallback to config itself for flat structure
model = config.get('model', {})
data = config.get('data', {})
optimizer = config.get('optimizer', {})
loss_cfg = config.get('loss_config', {})
training = config.get('training', config)  # Fallback to config itself
checkpoints = config.get('checkpoints', {})
schedule = config.get('schedule', {})
discriminator = config.get('discriminator', {})
early_stopping = config.get('early_stopping', {})
metrics = config.get('metrics', {})

args = []
# Use direct config values if available, else fall back to nested structure
args.append(f\"--generator_version {config.get('generator_type', model.get('generator_version', 'enhanced'))}\")
args.append(f\"--discriminator_version {config.get('discriminator_type', model.get('discriminator_version', 'enhanced_v2'))}\")
args.append(f\"--tfrecord_path {data.get('tfrecord_path', 'dual_modal_gan/data/dataset_gan.tfrecord')}\")
args.append(f\"--charset_path {data.get('charset_path', 'real_data_preparation/real_data_charlist.txt')}\")
args.append(f\"--recognizer_weights {data.get('recognizer_weights', 'models/best_htr_recognizer/best_model.weights.h5')}\")
args.append(f\"--gpu_id {config.get('gpu_id', training.get('gpu_id', '0'))}\")
if config.get('no_restore', training.get('no_restore', False)):
    args.append('--no_restore')
args.append(f\"--checkpoint_dir dual_modal_gan/checkpoints/test_{config.get('experiment_name', 'experiment')}\")
args.append(f\"--sample_dir dual_modal_gan/outputs/samples_test_{config.get('experiment_name', 'experiment')}\")
args.append(f\"--max_checkpoints {checkpoints.get('max_checkpoints', 1)}\")
args.append(f\"--epochs {config.get('epochs', training.get('epochs'))}\")
args.append(f\"--steps_per_epoch {config.get('steps_per_epoch', training.get('steps_per_epoch', 0))}\")
args.append(f\"--batch_size {config.get('batch_size', training.get('batch_size'))}\")
args.append(f\"--pixel_loss_weight {config.get('pixel_loss_weight', weights.get('pixel_loss_weight'))}\")
args.append(f\"--rec_feat_loss_weight {config.get('rec_feat_loss_weight', weights.get('rec_feat_loss_weight'))}\")
args.append(f\"--adv_loss_weight {config.get('adv_loss_weight', weights.get('adv_loss_weight'))}\")
args.append(f\"--ctc_loss_weight {config.get('ctc_loss_weight', weights.get('ctc_loss_weight'))}\")
args.append(f\"--contrastive_loss_weight {config.get('contrastive_loss_weight', weights.get('contrastive_loss_weight', 0.0))}\")
args.append(f\"--gradient_clip_norm {loss_cfg.get('gradient_clip_norm', 5.0)}\")
args.append(f\"--ctc_loss_clip_max {loss_cfg.get('ctc_loss_clip_max', 500.0)}\")
args.append(f\"--warmup_epochs {schedule.get('warmup_epochs', 0)}\")
args.append(f\"--annealing_epochs {schedule.get('annealing_epochs', 0)}\")
args.append(f\"--save_interval {checkpoints.get('save_interval', 1)}\")
args.append(f\"--eval_interval {checkpoints.get('eval_interval', 1)}\")
args.append(f\"--discriminator_mode {config.get('discriminator_mode', discriminator.get('mode', 'predicted'))}\")
args.append(f\"--seed {config.get('seed', training.get('seed', 42))}\")
args.append(f\"--lr_g {config.get('learning_rate', optimizer.get('lr_g', 0.0001))}\")
args.append(f\"--lr_d {config.get('discriminator_learning_rate', optimizer.get('lr_d', 0.00005))}\")
if schedule.get('use_lr_schedule', False):
    args.append('--use_lr_schedule')
args.append(f\"--lr_decay_epochs {schedule.get('lr_decay_epochs', 50)}\")
args.append(f\"--lr_alpha {schedule.get('lr_alpha', 0.0)}\")
args.append(f\"--cer_weight {metrics.get('cer_weight', 0.5)}\")
if early_stopping.get('enabled', False):
    args.append('--early_stopping')
args.append(f\"--patience {early_stopping.get('patience', 15)}\")
args.append(f\"--min_delta {early_stopping.get('min_delta', 0.01)}\")
if early_stopping.get('restore_best_weights', True):
    args.append('--restore_best_weights')

print(' '.join(args))
" 2>&1)

# Extract command from parsed output
EXPERIMENT_NAME=$(echo "$PARSED" | grep "🎯 EXPERIMENT:" | cut -d' ' -f3-)
CMD_ARGS=$(echo "$PARSED" | tail -1)

# Display config summary
echo "$PARSED" | head -n -1
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Log file
LOG_FILE="logbook/test_${EXPERIMENT_NAME}_${TIMESTAMP}.log"

echo "▶️  Starting training..."
echo "   Log file: $LOG_FILE"
echo ""

# Run training
poetry run python dual_modal_gan/scripts/train_enhanced.py $CMD_ARGS \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXTRACTING RESULTS..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract metrics
STRATEGY1_PSNR=$(grep -E "📊.*PSNR:" "$LOG_FILE" | tail -1 | grep -oP "PSNR: \K[0-9.]+" || echo "0")
STRATEGY1_SSIM=$(grep -E "📊.*SSIM:" "$LOG_FILE" | tail -1 | grep -oP "SSIM: \K[0-9.]+" || echo "0")
STRATEGY1_CER=$(grep -E "📊.*CER:" "$LOG_FILE" | tail -1 | grep -oP "CER: \K[0-9.]+" || echo "0")

# Extract D_loss range
D_LOSS_MIN=$(grep "D_loss:" "$LOG_FILE" | grep -oP "D_loss: \K[0-9.]+" | sort -n | head -1 || echo "0")
D_LOSS_MAX=$(grep "D_loss:" "$LOG_FILE" | grep -oP "D_loss: \K[0-9.]+" | sort -n | tail -1 || echo "0")
D_LOSS_AVG=$(grep "D_loss:" "$LOG_FILE" | grep -oP "D_loss: \K[0-9.]+" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

# Extract G_loss components
ADV_LOSS_AVG=$(grep "Adv:" "$LOG_FILE" | grep -oP "Adv: \K[0-9.]+" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
PIXEL_LOSS_AVG=$(grep "Pixel:" "$LOG_FILE" | grep -oP "Pixel: \K[0-9.]+" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
RECFEAT_LOSS_AVG=$(grep "RecFeat:" "$LOG_FILE" | grep -oP "RecFeat: \K[0-9.]+" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
CTC_LOSS_AVG=$(grep "CTC:" "$LOG_FILE" | grep -oP "CTC: \K[0-9.]+" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

# Baseline from current training (running now)
BASELINE_PSNR=22.5
BASELINE_SSIM=0.9425
BASELINE_CER=0.1642
BASELINE_D_LOSS_AVG=1.47

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║           STRATEGI 1: EXTREME REBALANCING RESULTS                      ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║ Metric      │  Baseline (Current)  │  Strategy 1      │   Δ Delta       ║"
echo "╠═════════════╪══════════════════════╪══════════════════╪═════════════════╣"
printf "║ PSNR        │  %7.2f dB          │  %7.2f dB     │ %+7.2f dB      ║\n" \
    $BASELINE_PSNR $STRATEGY1_PSNR \
    $(echo "$STRATEGY1_PSNR - $BASELINE_PSNR" | bc)
printf "║ SSIM        │    %7.4f           │    %7.4f     │   %+7.4f      ║\n" \
    $BASELINE_SSIM $STRATEGY1_SSIM \
    $(echo "$STRATEGY1_SSIM - $BASELINE_SSIM" | bc)
printf "║ CER         │    %7.4f           │    %7.4f     │   %+7.4f      ║\n" \
    $BASELINE_CER $STRATEGY1_CER \
    $(echo "$STRATEGY1_CER - $BASELINE_CER" | bc)
echo "╠═════════════╪══════════════════════╪══════════════════╪═════════════════╣"
printf "║ D_loss AVG  │    %7.2f           │    %7.2f     │   %+7.2f      ║\n" \
    $BASELINE_D_LOSS_AVG $D_LOSS_AVG \
    $(echo "$D_LOSS_AVG - $BASELINE_D_LOSS_AVG" | bc)
printf "║ D_loss MIN  │    1.23              │    %7.2f     │   %+7.2f      ║\n" \
    $D_LOSS_MIN \
    $(echo "$D_LOSS_MIN - 1.23" | bc)
printf "║ D_loss MAX  │    1.71              │    %7.2f     │   %+7.2f      ║\n" \
    $D_LOSS_MAX \
    $(echo "$D_LOSS_MAX - 1.71" | bc)
echo "╠═════════════╪══════════════════════╪══════════════════╪═════════════════╣"
echo "║ LOSS COMPONENTS CONTRIBUTION (Average):                                ║"
echo "╠═════════════╪══════════════════════╪══════════════════╪═════════════════╣"
printf "║ Adv Loss    │    ~1.1 (0.36%%)      │    %7.2f     │                 ║\n" $ADV_LOSS_AVG
printf "║ Pixel Loss  │    ~6.0 (2%%)          │    %7.2f     │                 ║\n" $PIXEL_LOSS_AVG
printf "║ RecFeat Loss│    ~1.0 (0.3%%)        │    %7.2f     │                 ║\n" $RECFEAT_LOSS_AVG
printf "║ CTC Loss    │    ~300 (97%%)         │    %7.2f     │                 ║\n" $CTC_LOSS_AVG
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Calculate improvements
DELTA_PSNR=$(echo "$STRATEGY1_PSNR - $BASELINE_PSNR" | bc)
DELTA_D_LOSS=$(echo "$D_LOSS_AVG - $BASELINE_D_LOSS_AVG" | bc)

# Calculate new contribution percentages
NEW_TOTAL=$(echo "$ADV_LOSS_AVG + $PIXEL_LOSS_AVG + $RECFEAT_LOSS_AVG + $CTC_LOSS_AVG" | bc)
if [ $(echo "$NEW_TOTAL > 0" | bc) -eq 1 ]; then
    CTC_PERCENTAGE=$(echo "scale=1; $CTC_LOSS_AVG / $NEW_TOTAL * 100" | bc)
    ADV_PERCENTAGE=$(echo "scale=1; $ADV_LOSS_AVG / $NEW_TOTAL * 100" | bc)
    echo "📊 NEW LOSS BALANCE:"
    echo "   CTC contributes: ${CTC_PERCENTAGE}% (was 97%)"
    echo "   Adv contributes: ${ADV_PERCENTAGE}% (was 0.36%)"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 VERDICT:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if (( $(echo "$DELTA_PSNR >= 1.0" | bc -l) )); then
    echo "✅ SUCCESS: Strategi 1 terbukti EFEKTIF!"
    echo ""
    echo "   Δ PSNR: +${DELTA_PSNR} dB"
    echo "   Δ D_loss: ${DELTA_D_LOSS}"
    echo ""
    echo "   📝 HIPOTESIS TERBUKTI:"
    echo "   ✓ CTC dominance WAS the problem!"
    echo "   ✓ Loss rebalancing fixes adversarial training"
    echo "   ✓ Discriminator sekarang memberikan gradient signal yang kuat"
    echo ""
    echo "   🚀 NEXT STEPS:"
    echo "   1. Buat config untuk 5 epoch dengan strategi ini"
    echo "   2. Target: PSNR 26-28 dB dalam 3-4 jam"
    echo ""
    exit 0
    
elif (( $(echo "$DELTA_PSNR >= 0.5" | bc -l) )); then
    echo "⚖️  MARGINAL IMPROVEMENT"
    echo ""
    echo "   Δ PSNR: +${DELTA_PSNR} dB"
    echo ""
    echo "   🤔 OPSI:"
    echo "   A. Kombinasi Strategi 1 + 2 (rebalance + boost LR)"
    echo "   B. Skip ke Strategi 4 (Two-Stage Training)"
    echo ""
    exit 0
    
else
    echo "❌ STRATEGI 1 TIDAK EFEKTIF"
    echo ""
    echo "   Δ PSNR: ${DELTA_PSNR} dB"
    echo ""
    echo "   🔄 NEXT STRATEGY:"
    echo "   → Test Strategi 2 (Discriminator LR boost 5×)"
    echo "   → Atau Strategi 4 (Two-Stage: disable CTC first)"
    echo ""
    exit 1
fi
