#!/bin/bash
# UNIVERSAL TRAINING LAUNCHER FROM JSON CONFIG
# Dapat membaca SEMUA format JSON config (flat, nested, mixed)
# Usage: ./universal_train_from_json.sh <path/to/config.json>

set -e

# Check if config file is provided
if [ -z "$1" ]; then
    echo "❌ Error: Config file not provided"
    echo "Usage: $0 <path/to/config.json>"
    exit 1
fi

CONFIG_FILE="$1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         UNIVERSAL TRAINING LAUNCHER FROM JSON CONFIG          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📄 Config file: $CONFIG_FILE"
echo "🕐 Timestamp: $TIMESTAMP"
echo ""

# Parse JSON and build command using Python
CMD_OUTPUT=$(poetry run python -c "
import json
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)
    
    # Function to get value from nested or flat structure
    def get_value(config, keys, default=None):
        '''Get value from nested dict using list of keys, fallback to flat structure'''
        # Try nested first
        current = config
        for key in keys[:-1]:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                # Fallback to flat structure with last key
                return config.get(keys[-1], default)
        return current.get(keys[-1], default) if isinstance(current, dict) else default
    
    # Extract experiment name for logging
    exp_name = config.get('experiment_name', 'experiment')
    
    # Build training arguments
    training_args = []
    
    # Model architecture - map architecture names to valid choices
    gen_arch = get_value(config, ['model', 'generator', 'architecture'], 'enhanced')
    # Map common names to valid choices
    gen_map = {'unet': 'enhanced', 'enhanced': 'enhanced', 'enhanced_v2': 'enhanced_v2', 'base': 'base'}
    gen_version = gen_map.get(gen_arch, 'enhanced')
    
    disc_arch = get_value(config, ['model', 'discriminator', 'architecture'], 'enhanced_v2')
    disc_map = {'patchgan': 'enhanced_v2', 'enhanced_v2': 'enhanced_v2', 'base': 'base'}
    disc_version = disc_map.get(disc_arch, 'enhanced_v2')
    
    training_args.append(f\"--generator_version {gen_version}\")
    training_args.append(f\"--discriminator_version {disc_version}\")
    
    # Data paths - use correct defaults
    training_args.append(f\"--tfrecord_path {get_value(config, ['data', 'tfrecord_path'], 'dual_modal_gan/data/dataset_gan.tfrecord')}\")
    training_args.append(f\"--charset_path {get_value(config, ['data', 'charset_path'], 'real_data_preparation/real_data_charlist.txt')}\")
    training_args.append(f\"--recognizer_weights {get_value(config, ['data', 'recognizer_weights'], 'models/best_htr_recognizer/best_model.weights.h5')}\")
    
    # GPU configuration
    training_args.append(f\"--gpu_id {get_value(config, ['training', 'gpu_id'], '0')}\")
    
    # Checkpoint and output directories
    training_args.append(f\"--checkpoint_dir dual_modal_gan/checkpoints/{exp_name}\")
    training_args.append(f\"--sample_dir dual_modal_gan/outputs/samples_{exp_name}\")
    training_args.append(f\"--max_checkpoints {get_value(config, ['checkpoints', 'max_checkpoints'], 1)}\")
    
    # Training hyperparameters
    epochs = get_value(config, ['training', 'epochs'], 1)
    steps_per_epoch = get_value(config, ['training', 'steps_per_epoch'], 500)
    batch_size = get_value(config, ['training', 'batch_size'], 2)
    
    training_args.append(f\"--epochs {epochs}\")
    training_args.append(f\"--steps_per_epoch {steps_per_epoch}\")
    training_args.append(f\"--batch_size {batch_size}\")
    
    # Learning rates
    lr_g = get_value(config, ['training', 'learning_rate_g'], 0.0002)
    lr_d = get_value(config, ['training', 'learning_rate_d'], 0.0002)
    training_args.append(f\"--lr_g {lr_g}\")
    training_args.append(f\"--lr_d {lr_d}\")
    
    # Gradient clipping
    grad_clip = get_value(config, ['training', 'gradient_clip'], 1.0)
    training_args.append(f\"--gradient_clip_norm {grad_clip}\")
    
    # Loss weights
    pixel_weight = get_value(config, ['loss_weights', 'pixel_loss_weight'], 200.0)
    adv_weight = get_value(config, ['loss_weights', 'adv_loss_weight'], 1.5)
    recfeat_weight = get_value(config, ['loss_weights', 'rec_feat_loss_weight'], 5.0)
    ctc_weight = get_value(config, ['loss_weights', 'ctc_loss_weight'], 1.0)
    percep_weight = get_value(config, ['loss_weights', 'perceptual_loss_weight'], 0.0)
    
    training_args.append(f\"--pixel_loss_weight {pixel_weight}\")
    training_args.append(f\"--adv_loss_weight {adv_weight}\")
    training_args.append(f\"--rec_feat_loss_weight {recfeat_weight}\")
    training_args.append(f\"--ctc_loss_weight {ctc_weight}\")
    training_args.append(f\"--perceptual_loss_weight {percep_weight}\")
    training_args.append(f\"--contrastive_loss_weight {get_value(config, ['loss_weights', 'contrastive_loss_weight'], 0.0)}\")
    
    # Loss configuration
    ctc_clip = get_value(config, ['loss_config', 'ctc_loss_clip_max'], 300.0)
    training_args.append(f\"--ctc_loss_clip_max {ctc_clip}\")
    
    # Other settings
    training_args.append(f\"--discriminator_mode {get_value(config, ['discriminator', 'mode'], 'predicted')}\")
    training_args.append(f\"--seed {get_value(config, ['training', 'seed'], 42)}\")
    
    # Early stopping (optional)
    if get_value(config, ['early_stopping', 'enabled'], False):
        training_args.append('--early_stopping')
        training_args.append(f\"--patience {get_value(config, ['early_stopping', 'patience'], 15)}\")
        training_args.append(f\"--min_delta {get_value(config, ['early_stopping', 'min_delta'], 0.01)}\")
    
    # LR schedule (optional)
    if get_value(config, ['training', 'use_lr_schedule'], False):
        training_args.append('--use_lr_schedule')
        training_args.append(f\"--lr_decay_epochs {get_value(config, ['training', 'lr_decay_epochs'], 50)}\")
        training_args.append(f\"--lr_alpha {get_value(config, ['training', 'lr_alpha'], 0.0)}\")
    
    # Checkpoint resume (optional)
    if get_value(config, ['training', 'resume_from_checkpoint'], False):
        training_args.append('--resume')
    
    # No restore flag (optional - for clean slate training)
    if get_value(config, ['training', 'no_restore'], False):
        training_args.append('--no_restore')
    
    # Print configuration summary
    print('=' * 80)
    print('CONFIGURATION SUMMARY:')
    print('=' * 80)
    print(f'Experiment: {exp_name}')
    print(f'Description: {config.get(\"description\", \"N/A\")}')
    print('')
    print(f'Training:')
    print(f'  Epochs: {epochs}')
    print(f'  Steps per epoch: {steps_per_epoch}')
    print(f'  Batch size: {batch_size}')
    print(f'  Learning rates: G={lr_g}, D={lr_d}')
    resume_flag = get_value(config, ['training', 'resume_from_checkpoint'], False)
    if resume_flag:
        print(f'  Resume from checkpoint: ✅ ENABLED')
    print('')
    print(f'Loss Weights:')
    print(f'  Pixel: {pixel_weight}')
    print(f'  Adversarial: {adv_weight}')
    print(f'  RecFeat: {recfeat_weight}')
    print(f'  CTC: {ctc_weight}')
    if percep_weight > 0:
        print(f'  VGG Perceptual: {percep_weight} 🎨')
    print('')
    print(f'Loss Config:')
    print(f'  Gradient clip: {grad_clip}')
    print(f'  CTC clip max: {ctc_clip}')
    print('=' * 80)
    print('')
    
    # Output: experiment_name|command_args
    print(f'{exp_name}|{\" \".join(training_args)}')
    
except Exception as e:
    print(f'ERROR: Failed to parse config: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

# Check if parsing was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to parse config file"
    echo "$CMD_OUTPUT"
    exit 1
fi

# Extract experiment name and command args
EXPERIMENT_NAME=$(echo "$CMD_OUTPUT" | tail -1 | cut -d'|' -f1)
CMD_ARGS=$(echo "$CMD_OUTPUT" | tail -1 | cut -d'|' -f2-)

# Display configuration summary (everything except last line)
echo "$CMD_OUTPUT" | head -n -1

# Log file
LOG_FILE="logbook/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

echo "▶️  Starting training..."
echo "📝 Log file: $LOG_FILE"
echo ""

# Execute training
poetry run python dual_modal_gan/scripts/train_enhanced.py $CMD_ARGS \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Extract final metrics
    echo ""
    echo "📊 FINAL METRICS:"
    grep -E "📊.*PSNR:|SSIM:|CER:" "$LOG_FILE" | tail -3
    
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "📋 Last 20 lines of log:"
    tail -20 "$LOG_FILE"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit $EXIT_CODE
