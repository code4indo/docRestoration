#!/bin/bash
"""
Fine-tuning Configuration for DIBCO (WITHOUT text labels)
Visual-Only Mode: Pixel + Perceptual + Adversarial losses
"""

# Configuration
CHECKPOINT_PRETRAINED="dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model/ckpt-99"
TFRECORD_DIBCO_2013="dibco_datasets/dibco_2013_train.tfrecord"  # To be created
OUTPUT_DIR="dual_modal_gan/checkpoints/dibco_finetuned"

# Visual-Only Fine-tuning (NO text labels needed)
poetry run python dual_modal_gan/scripts/train_enhanced.py \
  --tfrecord_path $TFRECORD_DIBCO_2013 \
  --checkpoint_dir $OUTPUT_DIR \
  --resume_from $CHECKPOINT_PRETRAINED \
  \
  --discriminator_mode predicted \
  \
  --ctc_loss_weight 0.0 \
  --rec_feat_loss_weight 0.0 \
  --pixel_loss_weight 1.0 \
  --adversarial_loss_weight 0.3 \
  --perceptual_loss_weight 0.5 \
  \
  --batch_size 4 \
  --num_epochs 10 \
  --learning_rate_g 0.00005 \
  --learning_rate_d 0.00005 \
  \
  --generator_version enhanced \
  --discriminator_version enhanced_v2_fixed \
  \
  --gpu_id 0 \
  --seed 42

# Notes:
# 1. NO recognizer_weights needed (CTC=0, RecFeat=0)
# 2. Discriminator uses PREDICTED text from recognizer (jika masih ada)
#    atau pure visual jika recognizer=None
# 3. This is STANDARD practice for document binarization!
# 4. Academically VALID for DIBCO benchmark
