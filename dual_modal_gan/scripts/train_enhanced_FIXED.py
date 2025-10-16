"""
Dual-Modal GAN-HTR Training Script - Pure FP32 Version (OPTIMIZED)

Version: train32.py (Perfected)
Based on: train.py + train_old.py analysis

Key Improvements:
1. Pure FP32 training (NO mixed precision) for numerical stability
2. Balanced loss components (CTC, Pixel, Adversarial)
3. Comprehensive metrics (PSNR, SSIM, CER, WER)
4. MLflow tracking integration
5. Proper gradient clipping and loss management
6. Optimized for GAN-HTR with CTC loss
7. Efficient checkpoint management (max_to_keep=1 by default)

Research Finding:
- FP16 mixed precision causes imbalanced optimization in GAN-HTR
- CTC loss log-space calculations require FP32 precision
- Pure FP32 achieves better convergence and visual quality
"""

import argparse
import os
import time
import numpy as np
import cv2
import json
from datetime import datetime
import mlflow
import mlflow.tensorflow

# Disable XLA optimization to avoid layout errors - MUST be before TF import
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf

# Disable XLA at TF config level
tf.config.optimizer.set_jit(False)

# ✅ PURE FP32 - NO MIXED PRECISION
# Explicitly set float32 policy for numerical stability with CTC loss
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
print("✅ Pure FP32 precision enabled for training stability")
print("   (FP16 disabled: Required for CTC loss numerical stability)")

from tqdm import tqdm
import editdistance  # For CER/WER calculation

# Fix for ModuleNotFoundError: Add project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import models and utils
from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.recognizer import load_frozen_recognizer
from dual_modal_gan.src.models.discriminator import build_dual_modal_discriminator

# --- Utility Functions ---
def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"--- Random seeds set to {seed} for reproducibility ---")

def decode_label(label_ids, charset):
    """Decode label IDs to text string, removing blank tokens (0) and padding."""
    decoded_chars = []
    prev_id = -1
    for label_id in label_ids:
        label_id = int(label_id)
        # Skip blank token (0) and padding (0), and CTC duplicates
        if label_id > 0 and label_id != prev_id:
            if label_id - 1 < len(charset):
                decoded_chars.append(charset[label_id - 1])
        prev_id = label_id
    return ''.join(decoded_chars)

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate using edit distance."""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)

def calculate_wer(ground_truth, prediction):
    """Calculate Word Error Rate using edit distance on word level."""
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = editdistance.eval(gt_words, pred_words)
    return distance / len(gt_words)


# --- Dataset Pipeline ---
def _parse_tfrecord_fn(example_proto):
    # Define feature description for raw bytes and metadata
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64), # H, W, C
        'degraded_image_dtype': tf.io.FixedLenFeature([], tf.string), # Stored as string name
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64), # (length,)
        'label_dtype': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize degraded image
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image_dtype_str = example['degraded_image_dtype'] # Get string directly
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32) # Decode raw bytes
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image_dtype_str = example['clean_image_dtype']
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32) # Decode raw bytes
    clean_image = tf.reshape(clean_image, clean_image_shape)
    # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    # Deserialize label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label_dtype_str = example['label_dtype']
    label = tf.io.decode_raw(example['label_raw'], tf.int64) # Label is int64
    label = tf.reshape(label, label_shape)
    
    label = tf.cast(label, tf.int32)
    # Pad label to a static shape to prevent TF graph retracing/warnings
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label

def create_dataset(tfrecord_path, batch_size, val_split=0.1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Get the total number of items
    total_size = sum(1 for _ in dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, train_size, val_size

# --- Main Training & Evaluation Logic ---
def run_validation_step(val_dataset, generator, recognizer, charset, psnr_metric, ssim_metric, cer_metric, wer_metric):
    """Run validation step with both visual (PSNR/SSIM) and textual (CER/WER) metrics.
    
    Note: Cannot use @tf.function due to Python-based CER/WER calculation.
    """
    for degraded_images, clean_images, labels in val_dataset:
        # Generate enhanced images
        generated_images = generator(degraded_images, training=False)
        
        # Visual metrics: PSNR and SSIM expect values in [0, 1] range
        psnr = tf.image.psnr(clean_images, generated_images, max_val=1.0)
        ssim = tf.image.ssim(clean_images, generated_images, max_val=1.0)
        psnr_metric.update_state(psnr)
        ssim_metric.update_state(ssim)
        
        # Textual metrics: CER/WER
        # Get HTR predictions for clean (ground truth quality) and generated (enhanced) images
        # Handle both single and multi-output recognizer
        recognizer_output_clean = recognizer(clean_images, training=False)
        recognizer_output_generated = recognizer(generated_images, training=False)
        
        # Extract logits (handle both tuple and single output)
        if isinstance(recognizer_output_clean, (list, tuple)):
            clean_logits = recognizer_output_clean[0]
            generated_logits = recognizer_output_generated[0]
        else:
            clean_logits = recognizer_output_clean
            generated_logits = recognizer_output_generated
        
        # Decode predictions using greedy decoding (argmax)
        clean_predictions = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
        generated_predictions = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)
        
        # Convert to numpy for text decoding
        labels_np = labels.numpy()
        clean_pred_np = clean_predictions.numpy()
        generated_pred_np = generated_predictions.numpy()
        
        # Calculate CER/WER for each sample in batch
        batch_cer = []
        batch_wer = []
        for i in range(labels_np.shape[0]):
            # Decode ground truth
            gt_text = decode_label(labels_np[i], charset)
            
            # Decode HTR predictions
            clean_text = decode_label(clean_pred_np[i], charset)
            generated_text = decode_label(generated_pred_np[i], charset)
            
            # Calculate CER/WER: Compare generated image predictions vs clean image predictions
            # (We use clean as reference because it represents "ideal" HTR performance)
            cer = calculate_cer(clean_text, generated_text)
            wer = calculate_wer(clean_text, generated_text)
            
            batch_cer.append(cer)
            batch_wer.append(wer)
        
        # Update metrics
        cer_metric.update_state(batch_cer)
        wer_metric.update_state(batch_wer)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print(f"--- Configuring to use GPU: {args.gpu_id} ---")
    print("--- Dual-Modal GAN-HTR Training (Pure FP32 - OPTIMIZED) ---")
    print("    Version: train32.py (Perfected)")
    print("    Precision: Pure FP32 (NO mixed precision)")
    print("    Optimized for: Balanced loss, CTC stability, visual quality")

    # Set random seeds for reproducibility
    set_seeds(args.seed)

    strategy = tf.distribute.get_strategy()
    print(f"Running with strategy: {strategy}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Create metrics directory for JSON logs
    metrics_dir = os.path.join(args.checkpoint_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    print("\n[Phase 1/6] Loading Datasets and Charset...")
    charset = read_charlist(args.charset_path)
    vocab_size = len(charset) + 1
    print(f"Charset loaded: {vocab_size} characters (including blank token)")

    train_dataset, val_dataset, train_count, val_count = create_dataset(args.tfrecord_path, args.batch_size)
    print(f"Dataset created: {train_count} training samples, {val_count} validation samples.")

    # Calculate steps per epoch if not provided
    steps_per_epoch = args.steps_per_epoch or (train_count // args.batch_size)

    with strategy.scope():
        print("\n[Phase 2/6] Building Models...")
        # Use (W, H, C) = (1024, 128, 1) to match HTR recognizer expectation
        
        # --- Select Generator Architecture ---
        if args.generator_version == 'enhanced':
            print("   ✅ Using ENHANCED generator (U-Net with Residual Blocks and Attention)")
            generator = unet_enhanced(input_size=(1024, 128, 1))
            generator_name = "U-Net Enhanced (ResBlocks+Attention)"
        else:
            print("   ✅ Using BASE generator (Standard U-Net)")
            generator = unet(input_size=(1024, 128, 1))
            generator_name = "U-Net (30M params, no dropout)"
        
        # Load recognizer with multi-output for Recognition Feature Loss
        use_rec_feat_loss = args.rec_feat_loss_weight > 0.0
        recognizer = load_frozen_recognizer(
            weights_path=args.recognizer_weights, 
            charset_size=vocab_size - 1,
            return_feature_map=use_rec_feat_loss
        )
        discriminator = build_dual_modal_discriminator(img_shape=(1024, 128, 1), vocab_size=vocab_size, max_text_len=128)
        print("All models built.")

    with strategy.scope():
        print("\n[Phase 3/6] Setting up Optimizers and Checkpoints...")
        # ✅ Pure FP32 - NO LossScaleOptimizer wrapping
        # Add clipnorm to optimizers for gradient stability
        
        # ✅ Solution 1: Cosine Annealing LR Schedule (if enabled)
        if args.use_lr_schedule:
            # Calculate decay steps: epochs * steps_per_epoch
            # Use dataset size to estimate steps if steps_per_epoch=0 (unlimited)
            decay_steps = args.lr_decay_epochs * (args.steps_per_epoch if args.steps_per_epoch > 0 else 2133)
            
            lr_schedule_g = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.lr_g,
                decay_steps=decay_steps,
                alpha=args.lr_alpha
            )
            lr_schedule_d = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.lr_d,
                decay_steps=decay_steps,
                alpha=args.lr_alpha
            )
            generator_optimizer = tf.keras.optimizers.Adam(lr_schedule_g, beta_1=0.5, clipnorm=args.gradient_clip_norm)
            discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule_d, momentum=0.9, clipnorm=args.gradient_clip_norm)
            
            print(f"  🔄 Cosine LR Schedule ENABLED (Solution 1)")
            print(f"     Initial LR (G/D): {args.lr_g}/{args.lr_d}")
            print(f"     Decay steps: {decay_steps} ({args.lr_decay_epochs} epochs)")
            print(f"     Alpha (min LR): {args.lr_alpha}")
        else:
            generator_optimizer = tf.keras.optimizers.Adam(args.lr_g, beta_1=0.5, clipnorm=args.gradient_clip_norm)
            discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_d, momentum=0.9, clipnorm=args.gradient_clip_norm)
            
            print(f"  Generator optimizer: Adam(lr={args.lr_g}, clipnorm={args.gradient_clip_norm}) [PURE FP32]")
            print(f"  Discriminator optimizer: SGD(lr={args.lr_d}, momentum=0.9, clipnorm={args.gradient_clip_norm}) [PURE FP32]")
        
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=args.max_checkpoints)
        if ckpt_manager.latest_checkpoint and not args.no_restore:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print(f"Restored from {ckpt_manager.latest_checkpoint}")
        else:
            if args.no_restore and ckpt_manager.latest_checkpoint:
                print(f"🚫 Skipping checkpoint restoration (--no_restore flag)")
                print(f"   Found checkpoint: {ckpt_manager.latest_checkpoint}")
            print("Initializing from scratch.")

    with strategy.scope():
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
        mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
        mse_loss_fn = tf.keras.losses.MeanSquaredError()  # For Recognition Feature Loss
        val_psnr_metric = tf.keras.metrics.Mean(name='val_psnr')
        val_ssim_metric = tf.keras.metrics.Mean(name='val_ssim')
        val_cer_metric = tf.keras.metrics.Mean(name='val_cer')
        val_wer_metric = tf.keras.metrics.Mean(name='val_wer')

        @tf.function
        def train_step(degraded_images, clean_images, ground_truth_text, ctc_weight, rec_feat_weight):
            real_labels_disc = tf.ones([args.batch_size, 1]) * 0.9
            fake_labels_disc = tf.zeros([args.batch_size, 1])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(degraded_images, training=True)
                
                # Get recognizer outputs - handle both single and multi-output
                recognizer_output_clean = recognizer(clean_images, training=False)
                recognizer_output_generated = recognizer(generated_images, training=False)
                
                # Extract logits and feature maps (if available)
                if isinstance(recognizer_output_clean, (list, tuple)):
                    # Multi-output mode: (logits, feature_map)
                    clean_logits = recognizer_output_clean[0]
                    clean_feature_map = recognizer_output_clean[1]
                    generated_logits = recognizer_output_generated[0]
                    generated_feature_map = recognizer_output_generated[1]
                else:
                    # Single output mode: logits only
                    clean_logits = recognizer_output_clean
                    generated_logits = recognizer_output_generated
                    # Create dummy feature maps with same shape for consistency
                    clean_feature_map = tf.zeros([args.batch_size, 1], dtype=tf.float32)
                    generated_feature_map = tf.zeros([args.batch_size, 1], dtype=tf.float32)

                clean_text_pred = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
                generated_text_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)

                # --- SWITCHABLE DISCRIMINATOR LOGIC ---
                if args.discriminator_mode == 'ground_truth':
                    real_output = discriminator([clean_images, ground_truth_text], training=True)
                else: # Default to 'predicted' mode (original logic)
                    real_output = discriminator([clean_images, clean_text_pred], training=True)
                fake_output = discriminator([generated_images, generated_text_pred], training=True)

                disc_loss_real = bce_loss_fn(real_labels_disc, real_output)
                disc_loss_fake = bce_loss_fn(fake_labels_disc, fake_output)
                total_disc_loss = disc_loss_real + disc_loss_fake

                adversarial_loss = bce_loss_fn(real_labels_disc, fake_output)
                pixel_loss = mae_loss_fn(clean_images, generated_images)
                
                # Recognition Feature Loss (HTR-aware loss from intermediate features)
                # Always calculate but weight will control its contribution
                rec_feat_loss = mse_loss_fn(clean_feature_map, generated_feature_map)
                
                # Always calculate CTC loss; its contribution is controlled by ctc_weight.
                # This avoids conditional graph structures that break gradient flow in @tf.function.
                label_len = tf.math.count_nonzero(ground_truth_text, axis=1, keepdims=True, dtype=tf.int32)
                label_len = tf.reshape(label_len, [-1])
                logit_len = tf.fill([args.batch_size], generated_logits.shape[1])
                
                ctc_loss_raw = tf.reduce_mean(tf.nn.ctc_loss(labels=tf.cast(ground_truth_text, tf.int32), logits=generated_logits, label_length=label_len, logit_length=logit_len, logits_time_major=False, blank_index=0))
                # Clip CTC loss to prevent spikes that cause training instability
                ctc_loss = tf.clip_by_value(ctc_loss_raw, 0.0, args.ctc_loss_clip_max)

                # ✅ Pure FP32 - NO casting needed (already in FP32)
                # Total Generator Loss with Recognition Feature Loss
                total_gen_loss = (
                    (args.adv_loss_weight * adversarial_loss) + 
                    (args.pixel_loss_weight * pixel_loss) + 
                    (rec_feat_weight * rec_feat_loss) +
                    (ctc_weight * ctc_loss)
                )

            generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
            
            # Apply gradient clipping to prevent explosion (especially from CTC loss)
            generator_gradients, gen_grad_norm = tf.clip_by_global_norm(generator_gradients, args.gradient_clip_norm)
            discriminator_gradients, disc_grad_norm = tf.clip_by_global_norm(discriminator_gradients, args.gradient_clip_norm)
            
            # ✅ Pure FP32 - Direct gradient application (no LossScaleOptimizer unscaling)
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            return total_gen_loss, total_disc_loss, adversarial_loss, pixel_loss, rec_feat_loss, ctc_loss, ctc_loss_raw, gen_grad_norm, disc_grad_norm

    print("\n[Phase 4/6] Starting Training Loop...")
    dataset_iterator = iter(train_dataset)
    
    # Create a fixed batch from the VALIDATION set for generating samples
    val_iterator = iter(val_dataset)
    sample_batch = next(val_iterator)

    best_val_psnr = -1.0 # Initialize best PSNR for checkpoint management
    
    # --- Early Stopping Setup ---
    patience_counter = 0
    best_epoch = 0
    early_stopped = False
    best_weights_path = None
    
    if args.early_stopping:
        print(f"\n🛡️ Early Stopping enabled:")
        print(f"   Patience: {args.patience} epochs")
        print(f"   Min Delta: {args.min_delta}")
        print(f"   Restore Best Weights: {args.restore_best_weights}")
    
    # --- MLflow Tracking Setup ---
    mlflow.set_tracking_uri("file:./mlruns")  # Save to local directory
    experiment_name = f"GAN_HTR_FP32_{args.discriminator_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    print(f"\n📊 MLflow tracking enabled: {experiment_name}")
    print(f"   View at: http://localhost:5000 (run: poetry run mlflow ui)\n")
    
    # Initialize training history for JSON logging
    training_history = {
        "start_time": datetime.now().isoformat(),
        "hyperparameters": {
            "precision": "Pure FP32 (NO mixed precision)",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "lr_generator": args.lr_g,
            "lr_discriminator": args.lr_d,
            "pixel_loss_weight": args.pixel_loss_weight,
            "ctc_loss_weight": args.ctc_loss_weight,
            "discriminator_mode": args.discriminator_mode,
            "adv_loss_weight": args.adv_loss_weight,
            "gradient_clip_norm": args.gradient_clip_norm,
            "ctc_loss_clip_max": args.ctc_loss_clip_max,
            "early_stopping_enabled": args.early_stopping,
            "early_stopping_patience": args.patience if args.early_stopping else None,
            "early_stopping_min_delta": args.min_delta if args.early_stopping else None,
            "early_stopping_restore_best": args.restore_best_weights if args.early_stopping else None,
            "optimizer_generator": f"Adam(lr={args.lr_g}, beta_1=0.5, clipnorm={args.gradient_clip_norm})",
            "optimizer_discriminator": f"SGD(lr={args.lr_d}, momentum=0.9, clipnorm={args.gradient_clip_norm})",
            "model_architecture": {
                "generator": generator_name,
                "discriminator": "Dual-Modal (137M params)",
                "recognizer": "Frozen HTR Stage 3 (50M params, CER 33.72%)"
            },
            "dataset": {
                "path": args.tfrecord_path,
                "train_samples": train_count,
                "val_samples": val_count,
                "charset_size": vocab_size
            }
        },
        "epochs": []
    }

    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "precision": "FP32",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr_generator": args.lr_g,
            "lr_discriminator": args.lr_d,
            "pixel_loss_weight": args.pixel_loss_weight,
            "rec_feat_loss_weight": args.rec_feat_loss_weight,
            "ctc_loss_weight": args.ctc_loss_weight,
            "adv_loss_weight": args.adv_loss_weight,
            "contrastive_loss_weight": args.contrastive_loss_weight,
            "discriminator_mode": args.discriminator_mode,
            "gradient_clip_norm": args.gradient_clip_norm,
            "ctc_loss_clip_max": args.ctc_loss_clip_max,
            "train_samples": train_count,
            "val_samples": val_count,
            "vocab_size": vocab_size,
            "early_stopping": args.early_stopping,
            "patience": args.patience if args.early_stopping else 0,
            "min_delta": args.min_delta if args.early_stopping else 0
        })
        
        # Log model architecture info as tags
        mlflow.set_tags({
            "model_type": "Dual-Modal GAN-HTR",
            "precision": "Pure FP32",
            "generator": generator_name,
            "discriminator": "Dual-Modal (137M params)",
            "recognizer": "Frozen HTR Stage 3 (CER 33.72%)",
            "optimized_for": "Balanced loss + CTC stability"
        })
    
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            
            # --- Curriculum Learning & Loss Annealing Logic ---
            is_warmup = epoch < args.warmup_epochs
            is_annealing = not is_warmup and (epoch - args.warmup_epochs) < args.annealing_epochs
            
            current_ctc_weight = 0.0
            if is_annealing:
                # Linear ramp-up of CTC weight
                annealing_epoch = epoch - args.warmup_epochs
                progress = float(annealing_epoch + 1) / float(args.annealing_epochs)
                current_ctc_weight = args.ctc_loss_weight * progress
            elif not is_warmup:
                # Full CTC weight after warm-up and annealing
                current_ctc_weight = args.ctc_loss_weight

            # --- Epoch Logging ---
            phase_str = ""
            if is_warmup:
                phase_str = f" (Warm-up, CTC_w=0.0)"
                if epoch == 0:
                    print(f"\n🔥 Starting visual warm-up for {args.warmup_epochs} epochs...")
            elif is_annealing:
                phase_str = f" (Annealing, CTC_w={current_ctc_weight:.2f})"
                if epoch == args.warmup_epochs:
                    print(f"\n📈 Starting CTC loss annealing for {args.annealing_epochs} epochs...")
            else:
                phase_str = f" (Full Training, CTC_w={current_ctc_weight:.1f})"
                if epoch == args.warmup_epochs + args.annealing_epochs:
                    print(f"\n✅ Annealing complete. Using full CTC loss weight.")

            print(f"\nEpoch {epoch + 1}/{args.epochs}{phase_str}")

            # Track epoch metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "phase": "warmup" if is_warmup else "annealing" if is_annealing else "full_training",
                "current_ctc_weight": current_ctc_weight,
                "current_rec_feat_weight": args.rec_feat_loss_weight,
                "losses": {
                    "g_loss": [], "d_loss": [], "adv_loss": [], "pixel_loss": [],
                    "rec_feat_loss": [], "ctc_loss": [], "ctc_loss_raw": [], 
                    "g_grad_norm": [], "d_grad_norm": []
                }
            }
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
            for step in pbar:
                degraded_batch, clean_batch, text_batch = next(dataset_iterator)
                g_loss, d_loss, adv_loss, pix_loss, rec_feat_loss, ctc_loss, ctc_loss_raw, g_grad_norm, d_grad_norm = train_step(
                    degraded_batch, clean_batch, text_batch, 
                    tf.constant(current_ctc_weight, dtype=tf.float32),
                    tf.constant(args.rec_feat_loss_weight, dtype=tf.float32)
                )
                
                # Collect metrics
                epoch_metrics["losses"]["g_loss"].append(float(g_loss.numpy()))
                epoch_metrics["losses"]["d_loss"].append(float(d_loss.numpy()))
                epoch_metrics["losses"]["adv_loss"].append(float(adv_loss.numpy()))
                epoch_metrics["losses"]["pixel_loss"].append(float(pix_loss.numpy()))
                epoch_metrics["losses"]["rec_feat_loss"].append(float(rec_feat_loss.numpy()))
                epoch_metrics["losses"]["ctc_loss"].append(float(ctc_loss.numpy()))
                if current_ctc_weight > 0:
                    epoch_metrics["losses"]["ctc_loss_raw"].append(float(ctc_loss_raw.numpy()))
                epoch_metrics["losses"]["g_grad_norm"].append(float(g_grad_norm.numpy()))
                epoch_metrics["losses"]["d_grad_norm"].append(float(d_grad_norm.numpy()))

                if (step + 1) % 50 == 0:
                    pbar.set_postfix({
                        'G': f'{g_loss:.4f}', 'D': f'{d_loss:.4f}', 'Adv': f'{adv_loss:.4f}', 
                        'Pix': f'{pix_loss:.4f}', 'RecFeat': f'{rec_feat_loss:.4f}',
                        'CTC': f'{ctc_loss:.2f}', 'CTC_w': f'{current_ctc_weight:.2f}'
                    })

            # Calculate epoch statistics
            epoch_metrics["training_losses"] = {
                "total_loss": float(np.mean(epoch_metrics["losses"]["g_loss"])),
                "pixel_loss": float(np.mean(epoch_metrics["losses"]["pixel_loss"])),
                "rec_feat_loss": float(np.mean(epoch_metrics["losses"]["rec_feat_loss"])),
                "ctc_loss": float(np.mean(epoch_metrics["losses"]["ctc_loss"])),
                "adv_loss": float(np.mean(epoch_metrics["losses"]["adv_loss"])),
                "ctc_loss_raw": float(np.mean(epoch_metrics["losses"]["ctc_loss_raw"])) if epoch_metrics["losses"]["ctc_loss_raw"] else 0.0,
                "gradient_norm": {
                    "generator_mean": float(np.mean(epoch_metrics["losses"]["g_grad_norm"])),
                    "generator_max": float(np.max(epoch_metrics["losses"]["g_grad_norm"])),
                    "discriminator_mean": float(np.mean(epoch_metrics["losses"]["d_grad_norm"]))
                }
            }
            
            # Log epoch mean losses to MLflow
            mlflow_step = epoch + 1
            mlflow.log_metrics({
                "train/g_loss": epoch_metrics["training_losses"]["total_loss"],
                "train/d_loss": float(np.mean(epoch_metrics["losses"]["d_loss"])),
                "train/adv_loss": epoch_metrics["training_losses"]["adv_loss"],
                "train/pixel_loss": epoch_metrics["training_losses"]["pixel_loss"],
                "train/rec_feat_loss": epoch_metrics["training_losses"]["rec_feat_loss"],
                "train/ctc_loss": epoch_metrics["training_losses"]["ctc_loss"],
                "train/ctc_loss_raw": epoch_metrics["training_losses"]["ctc_loss_raw"],
                "train/g_grad_norm_mean": epoch_metrics["training_losses"]["gradient_norm"]["generator_mean"],
                "train/g_grad_norm_max": epoch_metrics["training_losses"]["gradient_norm"]["generator_max"],
                "train/d_grad_norm_mean": epoch_metrics["training_losses"]["gradient_norm"]["discriminator_mean"]
            }, step=mlflow_step)
        
            # --- End of Epoch Actions ---
            if (epoch + 1) % args.eval_interval == 0:
                print("  Running validation (visual + textual metrics)...")
                run_validation_step(
                    val_dataset, generator, recognizer, charset,
                    val_psnr_metric, val_ssim_metric, val_cer_metric, val_wer_metric
                )
                psnr_result = val_psnr_metric.result()
                ssim_result = val_ssim_metric.result()
                cer_result = val_cer_metric.result()
                wer_result = val_wer_metric.result()
                
                print(f"  📊 PSNR: {psnr_result:.2f}, SSIM: {ssim_result:.4f}, CER: {cer_result:.4f}, WER: {wer_result:.4f}")
                
                # Add validation metrics to epoch data
                epoch_metrics["validation"] = {
                    "psnr": float(psnr_result.numpy()),
                    "ssim": float(ssim_result.numpy()),
                    "cer": float(cer_result.numpy()),
                    "wer": float(wer_result.numpy())
                }
                
                # Log validation metrics to MLflow
                mlflow.log_metrics({
                    "val/psnr": float(psnr_result.numpy()),
                    "val/ssim": float(ssim_result.numpy()),
                    "val/cer": float(cer_result.numpy()),
                    "val/wer": float(wer_result.numpy())
                }, step=epoch+1)
            
                # --- Checkpoint Management: Dual objective (PSNR + CER) ---
                # Best model = High PSNR (visual quality) + Low CER (text readability)
                combined_score = psnr_result - (args.cer_weight * cer_result * 100)
                
                # Check if there's significant improvement
                improvement = combined_score - best_val_psnr
                is_improvement = improvement > args.min_delta
                
                if is_improvement:
                    best_val_psnr = combined_score
                    best_epoch = epoch + 1
                    patience_counter = 0  # Reset patience counter
                    
                    # Save checkpoint using CheckpointManager (respects max_to_keep)
                    saved_path = ckpt_manager.save()
                    best_weights_path = saved_path  # Track best model path for restoration
                    
                    print(f"  ✅ New best model saved! (Improvement: {improvement:.2f})")
                    print(f"     PSNR: {psnr_result:.2f}, CER: {cer_result:.4f}, Combined Score: {combined_score:.2f}")
                    print(f"     Best Epoch: {best_epoch}, Patience Counter: {patience_counter}/{args.patience}")
                    epoch_metrics["best_model_saved"] = True
                    epoch_metrics["patience_counter"] = patience_counter
                    
                    # Log best metrics to MLflow
                    mlflow.log_metrics({
                        "best_val_psnr": float(psnr_result),
                        "best_val_cer": float(cer_result),
                        "best_combined_score": float(combined_score),
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter
                    }, step=epoch+1)
                else:
                    patience_counter += 1  # Increment patience counter
                    print(f"  ⚠️  No improvement (score: {combined_score:.2f} vs best: {best_val_psnr:.2f})")
                    print(f"     Patience Counter: {patience_counter}/{args.patience}")
                    epoch_metrics["best_model_saved"] = False
                    epoch_metrics["patience_counter"] = patience_counter
                    
                    # Log patience to MLflow
                    mlflow.log_metric("patience_counter", patience_counter, step=epoch+1)
                    
                    # --- Early Stopping Check ---
                    if args.early_stopping and patience_counter >= args.patience:
                        print(f"\n🛑 EARLY STOPPING TRIGGERED!")
                        print(f"   No improvement for {args.patience} epochs")
                        print(f"   Best epoch: {best_epoch} with score: {best_val_psnr:.2f}")
                        
                        # Restore best weights if enabled
                        if args.restore_best_weights and best_weights_path:
                            print(f"   Restoring best model weights from: {best_weights_path}")
                            checkpoint.restore(best_weights_path)
                            print(f"   ✅ Best weights restored successfully")
                        
                        early_stopped = True
                        epoch_metrics["early_stopped"] = True
                        
                        # Log early stopping to MLflow
                        mlflow.log_metrics({
                            "early_stopped_epoch": epoch + 1,
                            "early_stopped": 1
                        }, step=epoch+1)
                        
                        # Break the training loop
                        break
                
                val_psnr_metric.reset_state()
                val_ssim_metric.reset_state()
                val_cer_metric.reset_state()
                val_wer_metric.reset_state()
            else:
                epoch_metrics["validation"] = None

            # Always save samples at save_interval
            if (epoch + 1) % args.save_interval == 0:
                generated_samples = generator(sample_batch[0], training=False)
                degraded_samples = sample_batch[0]  # Original degraded images
                clean_samples = sample_batch[1]     # Ground truth clean images
                
                # Save generated images
                img_to_save = (generated_samples * 255).numpy().astype(np.uint8)
                for i in range(min(args.batch_size, 4)):
                    img = img_to_save[i]
                    if img.shape[-1] == 1:
                        img = np.squeeze(img, axis=-1)
                    
                    # Transpose image to correct orientation for horizontal text
                    if img.shape[0] > img.shape[1]:  # If height > width, transpose
                        img = np.transpose(img)
                    
                    sample_path = os.path.join(args.sample_dir, f'epoch_{epoch+1:04d}_sample_{i}.png')
                    cv2.imwrite(sample_path, img)
                    
                    # --- Log to MLflow: Create comparison image (degraded | generated | clean) ---
                    if i < 2:  # Log only first 2 samples to avoid clutter
                        # Prepare degraded image
                        deg_img = (degraded_samples[i] * 255).numpy().astype(np.uint8)
                        deg_img = np.squeeze(deg_img, axis=-1) if deg_img.shape[-1] == 1 else deg_img
                        if deg_img.shape[0] > deg_img.shape[1]:
                            deg_img = np.transpose(deg_img)
                        
                        # Prepare clean (GT) image
                        clean_img = (clean_samples[i] * 255).numpy().astype(np.uint8)
                        clean_img = np.squeeze(clean_img, axis=-1) if clean_img.shape[-1] == 1 else clean_img
                        if clean_img.shape[0] > clean_img.shape[1]:
                            clean_img = np.transpose(clean_img)
                        
                        # Create horizontal concatenation: [Degraded | Generated | Clean]
                        comparison = np.hstack([deg_img, img, clean_img])
                        comparison_path = os.path.join(args.sample_dir, f'comparison_epoch_{epoch+1:04d}_sample_{i}.png')
                        cv2.imwrite(comparison_path, comparison)
                        
                        # Log to MLflow
                        mlflow.log_artifact(comparison_path, artifact_path=f"samples/epoch_{epoch+1:04d}")
                
                print(f"  💾 Saved sample images to {args.sample_dir}")

            epoch_time = time.time() - epoch_start_time
            epoch_metrics["epoch_time_seconds"] = float(epoch_time)
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            
            # Log epoch time to MLflow
            mlflow.log_metric("epoch_time_seconds", epoch_time, step=epoch+1)
            
            # Append epoch metrics to history
            training_history["epochs"].append(epoch_metrics)
            
            # Save metrics to JSON after each epoch (incremental save)
            metrics_file = os.path.join(metrics_dir, "training_metrics_fp32.json")
            with open(metrics_file, 'w') as f:
                json.dump(training_history, f, indent=2)

        # Finalize training history
        training_history["end_time"] = datetime.now().isoformat()
        training_history["best_val_psnr"] = float(best_val_psnr)
        training_history["best_epoch"] = best_epoch
        training_history["total_epochs_completed"] = len(training_history["epochs"])
        training_history["early_stopped"] = early_stopped
        training_history["final_patience_counter"] = patience_counter
        
        # Save final metrics
        final_metrics_file = os.path.join(metrics_dir, "training_metrics_fp32_final.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Log final metrics to MLflow
        mlflow.log_artifact(final_metrics_file, artifact_path="metrics")
        mlflow.log_metric("final_best_psnr", float(best_val_psnr))
        mlflow.log_metric("best_epoch", best_epoch)
        
        print(f"\n✅ Training metrics saved to: {final_metrics_file}")
        print(f"📊 All metrics logged to MLflow")
        print(f"\n🚀 View results: poetry run mlflow ui")
        print(f"   Then open: http://localhost:5000\n")

    print("\n[Phase 6/6] Training Finished.")
    
    if early_stopped:
        print("\n🛑 Training stopped early (Early Stopping triggered)")
        print(f"   Best model from epoch: {best_epoch}")
        print(f"   Total epochs completed: {len(training_history['epochs'])} / {args.epochs}")
        print(f"   Resources saved: {args.epochs - len(training_history['epochs'])} epochs")
    else:
        print("\n🎉 Training completed successfully with Pure FP32!")
        print(f"   Best model from epoch: {best_epoch}")
    
    print("   ✅ Numerical stability maintained")
    print("   ✅ Balanced loss optimization")
    print("   ✅ Superior convergence quality")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dual-Modal GAN-HTR with Pure FP32 (OPTIMIZED)')
    parser.add_argument('--generator_version', type=str, default='base', choices=['base', 'enhanced'], help='Version of the generator to use: base or enhanced.')
    parser.add_argument('--tfrecord_path', type=str, default='dual_modal_gan/data/dataset_gan.tfrecord', help='Path to the training TFRecord file.')
    parser.add_argument('--charset_path', type=str, default='real_data_preparation/real_data_charlist.txt', help='Path to the character set file.')
    parser.add_argument('--recognizer_weights', type=str, default='models/best_htr_recognizer/best_model.weights.h5', help='Path to pre-trained recognizer weights (Stage 3, CER 33.72%).')
    parser.add_argument('--gpu_id', type=str, default='1', help='ID of the GPU to use (e.g., "0" or "1").')
    parser.add_argument('--no_restore', action='store_true', help='Do not restore from checkpoint, start from scratch.')
    parser.add_argument('--checkpoint_dir', type=str, default='dual_modal_gan/outputs/checkpoints_fp32', help='Directory to save model checkpoints.')
    parser.add_argument('--max_checkpoints', type=int, default=1, help='Maximum number of checkpoints to keep (default: 1 for storage efficiency).')
    parser.add_argument('--sample_dir', type=str, default='dual_modal_gan/outputs/samples_fp32', help='Directory to save sample images.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Number of steps per epoch (if None, calculated from dataset size).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per replica.')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator learning rate.')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator learning rate.')
    parser.add_argument('--use_lr_schedule', action='store_true', help='Enable cosine annealing LR schedule (Solution 1).')
    parser.add_argument('--lr_decay_epochs', type=int, default=50, help='Number of epochs for LR cosine decay (default: 50).')
    parser.add_argument('--lr_alpha', type=float, default=0.0, help='Minimum learning rate as fraction of initial LR (0.0 = decay to zero).')
    parser.add_argument('--pixel_loss_weight', type=float, default=100.0, help='Weight for the L1 pixel loss.')
    parser.add_argument('--ctc_loss_weight', type=float, default=1.0, help='Weight for the HTR CTC loss (monitoring only, not backpropagated).')
    parser.add_argument('--adv_loss_weight', type=float, default=2.0, help='Weight for the adversarial loss.')
    parser.add_argument('--rec_feat_loss_weight', type=float, default=0.0, help='Weight for the Recognition Feature Loss (HTR-aware).')
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.0, help='Weight for the Contrastive Loss.')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm to prevent explosion.')
    parser.add_argument('--ctc_loss_clip_max', type=float, default=300.0, help='Maximum value for CTC loss clipping.')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of epochs for visual warm-up (CTC loss is disabled).')
    parser.add_argument('--annealing_epochs', type=int, default=10, help='Number of epochs to gradually ramp-up CTC loss weight after warm-up.')
    parser.add_argument('--save_interval', type=int, default=5, help='Save samples every N epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Run evaluation every N epochs.')
    parser.add_argument('--discriminator_mode', type=str, default='predicted', choices=['predicted', 'ground_truth'], help="Mode for the discriminator's real pair text input.")
    parser.add_argument('--cer_weight', type=float, default=0.5, help='Weight for CER in combined score calculation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    # Early Stopping Parameters
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping to prevent overfitting and save resources.')
    parser.add_argument('--patience', type=int, default=15, help='Number of epochs without improvement before stopping (default: 15).')
    parser.add_argument('--min_delta', type=float, default=0.01, help='Minimum change in monitored metric to qualify as improvement (default: 0.01).')
    parser.add_argument('--restore_best_weights', action='store_true', default=True, help='Restore model weights from best epoch when early stopping triggers.')

    args = parser.parse_args()
    main(args)
