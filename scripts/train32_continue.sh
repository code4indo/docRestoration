#!/bin/bash

# Activate poetry environment  
source /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration/.venv/bin/activate

# Run training WITHOUT warm-up (skip warm-up phase)
poetry run python dual_modal_gan/scripts/train32.py \
    --epochs 50 \
    --steps_per_epoch 0 \
    --batch_size 2 \
    --pixel_loss_weight 200.0 \
    --rec_feat_loss_weight 5.0 \
    --adv_loss_weight 1.5 \
    --early_stopping \
    --patience 15 \
    --warmup_epochs 0 \
    --annealing_epochs 0

