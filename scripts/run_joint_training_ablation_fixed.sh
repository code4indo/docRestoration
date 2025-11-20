#!/bin/bash
################################################################################
# Joint Training Ablation Study - Launcher Script (FIXED VERSION)
# 
# This script runs the CORRECT joint training experiment where the recognizer
# is trainable, allowing us to observe catastrophic forgetting.
#
# Expected outcomes:
# - CER will degrade from baseline 33.72% (catastrophic forgetting)
# - Training instability due to gradient conflicts
# - Validates superiority of frozen recognizer approach
################################################################################

set -e  # Exit on error

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."
pwd

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/joint_training_ablation"
LOG_FILE="${LOG_DIR}/joint_training_fixed_${TIMESTAMP}.log"
CHECKPOINT_DIR="dual_modal_gan/checkpoints/joint_training_ablation_${TIMESTAMP}"
SAMPLE_DIR="dual_modal_gan/outputs/samples_joint_training_${TIMESTAMP}"

# Joint Training Parameters
EPOCHS=20
BATCH_SIZE=1  # Reduced from 2 to prevent OOM
STEPS_PER_EPOCH=100
EVAL_INTERVAL=5  # Validate every 5 epochs to reduce overhead
GPU_ID=0

# Create log directory
mkdir -p "${LOG_DIR}"

echo "================================================================================
🚀 JOINT TRAINING ABLATION STUDY - PROPER IMPLEMENTATION
================================================================================
⚠️  WARNING: This is the CORRECT joint training implementation!
   - Recognizer IS trainable (NOT frozen)
   - Expected: Catastrophic forgetting (CER >> 33.72%)
   - Purpose: Validate frozen recognizer superiority

📅 Timestamp: ${TIMESTAMP}
📁 Log file: ${LOG_FILE}
📊 Config: ${EPOCHS} epochs, batch_size=${BATCH_SIZE}, ${STEPS_PER_EPOCH} steps/epoch
================================================================================
"

# Run training in background with nohup
# CRITICAL FIX: Use -u flag for unbuffered Python output (same as frozen training)
echo "🔄 Starting training (running in background)..."

nohup poetry run python -u dual_modal_gan/scripts/train_joint_training_ablation.py \
    --tfrecord_path dual_modal_gan/data/dataset_gan.tfrecord \
    --charset_path real_data_preparation/real_data_charlist.txt \
    --recognizer_weights /home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5 \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --sample_dir "${SAMPLE_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --steps_per_epoch ${STEPS_PER_EPOCH} \
    --eval_interval ${EVAL_INTERVAL} \
    --gpu_id ${GPU_ID} \
    --train_split 0.7 \
    --val_split 0.15 \
    --lr_g 0.0001 \
    --lr_d 0.0001 \
    --lr_r 0.0003 \
    --pixel_loss_weight 50.0 \
    --adv_loss_weight 3.0 \
    --rec_feat_loss_weight 1.0 \
    --ctc_loss_weight 1.0 \
    --gradient_clip_norm 1.0 \
    --ctc_loss_clip_max 400.0 \
    --baseline_cer 33.72 \
    --seed 42 \
    2>&1 | tee "${LOG_FILE}" &

TRAIN_PID=$!

echo "✅ Training started with PID: ${TRAIN_PID}"
echo ""
echo "📊 To monitor progress, run:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "🛑 To stop training, run:"
echo "   kill ${TRAIN_PID}"
echo ""
echo "================================================================================
⏳ Training is running in background...
   Check log file for progress: ${LOG_FILE}
================================================================================
"

# Save PID to file
echo "${TRAIN_PID}" > "${LOG_DIR}/training.pid"
echo "💾 PID saved to: ${LOG_DIR}/training.pid"
