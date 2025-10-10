"""
Main training script for the Dual-Modal GAN-HTR project.

Version 3 (Final):
- Implements a full tf.data pipeline with a train/validation split.
- Adds a formal evaluation step at configurable intervals to calculate PSNR and SSIM.
- Completes the train_step with all three generator loss components.
- Includes periodic saving of checkpoints and visual samples.
"""

import argparse
import os
import time
import numpy as np
import cv2
import json
from datetime import datetime

# Disable XLA optimization to avoid layout errors - MUST be before TF import
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Also disable JIT compilation at runtime level
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf

# Additional XLA disable at TF config level (runtime)
tf.config.optimizer.set_jit(False)
from tqdm import tqdm

# Fix for ModuleNotFoundError: Add project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import models and utils
from dual_modal_gan.src.models.generator import unet
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
@tf.function
def run_validation_step(val_dataset, generator, psnr_metric, ssim_metric):
    for degraded_images, clean_images, _ in val_dataset:
        generated_images = generator(degraded_images, training=False)
        
        # PSNR and SSIM expect values in [0, 1] range
        psnr = tf.image.psnr(clean_images, generated_images, max_val=1.0)
        ssim = tf.image.ssim(clean_images, generated_images, max_val=1.0)
        
        psnr_metric.update_state(psnr)
        ssim_metric.update_state(ssim)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print(f"--- Configuring to use GPU: {args.gpu_id} ---")
    print("--- Dual-Modal GAN-HTR Training (Completed Script) ---")

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
    
    # --- DEBUG: Save one sample image from dataset after batching ---
    debug_dir = "dual_modal_gan/outputs/debug_samples"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Fetch one batch to inspect
    sample_degraded, sample_clean, _ = next(iter(train_dataset))
    
    # Convert to numpy and save
    cv2.imwrite(os.path.join(debug_dir, "debug_degraded_image_batch0.png"), (sample_degraded[0].numpy()[:,:,0] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, "debug_clean_image_batch0.png"), (sample_clean[0].numpy()[:,:,0] * 255).astype(np.uint8))
    print(f"DEBUG: Saved sample degraded and clean images from first batch to {debug_dir}")
    # --- END DEBUG ---

    # Calculate steps per epoch if not provided
    steps_per_epoch = args.steps_per_epoch or (train_count // args.batch_size)

    with strategy.scope():
        print("\n[Phase 2/6] Building Models...")
        # Use (W, H, C) = (1024, 128, 1) to match HTR recognizer expectation
        generator = unet(input_size=(1024, 128, 1))
        recognizer = load_frozen_recognizer(weights_path=args.recognizer_weights, charset_size=vocab_size - 1)
        discriminator = build_dual_modal_discriminator(img_shape=(1024, 128, 1), vocab_size=vocab_size, max_text_len=128)
        print("All models built.")

    with strategy.scope():
        print("\n[Phase 3/6] Setting up Optimizers and Checkpoints...")
        # Add clipnorm to optimizers for double protection against gradient explosion
        generator_optimizer = tf.keras.optimizers.Adam(args.lr_g, beta_1=0.5, clipnorm=args.gradient_clip_norm)
        # Use SGD for discriminator to save memory (Adam needs 2x memory for momentum states)
        discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_d, momentum=0.9, clipnorm=args.gradient_clip_norm)
        print(f"  Generator optimizer: Adam(lr={args.lr_g}, clipnorm={args.gradient_clip_norm})")
        print(f"  Discriminator optimizer: SGD(lr={args.lr_d}, momentum=0.9, clipnorm={args.gradient_clip_norm})")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print(f"Restored from {ckpt_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

    with strategy.scope():
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
        mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
        val_psnr_metric = tf.keras.metrics.Mean(name='val_psnr')
        val_ssim_metric = tf.keras.metrics.Mean(name='val_ssim')

        @tf.function
        def train_step(degraded_images, clean_images, ground_truth_text):
            real_labels_disc = tf.ones([args.batch_size, 1]) * 0.9
            fake_labels_disc = tf.zeros([args.batch_size, 1])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(degraded_images, training=True)
                clean_logits = recognizer(clean_images, training=False)
                generated_logits = recognizer(generated_images, training=False)

                clean_text_pred = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
                generated_text_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)

                real_output = discriminator([clean_images, clean_text_pred], training=True)
                fake_output = discriminator([generated_images, generated_text_pred], training=True)

                disc_loss_real = bce_loss_fn(real_labels_disc, real_output)
                disc_loss_fake = bce_loss_fn(fake_labels_disc, fake_output)
                total_disc_loss = disc_loss_real + disc_loss_fake

                adversarial_loss = bce_loss_fn(real_labels_disc, fake_output)
                pixel_loss = mae_loss_fn(clean_images, generated_images)
                
                # Only calculate CTC loss if weight > 0 to save computation
                if args.ctc_loss_weight > 0:
                    label_len = tf.math.count_nonzero(ground_truth_text, axis=1, keepdims=True, dtype=tf.int32)
                    label_len = tf.reshape(label_len, [-1])
                    logit_len = tf.fill([args.batch_size], generated_logits.shape[1])
                    
                    ctc_loss_raw = tf.reduce_mean(tf.nn.ctc_loss(labels=tf.cast(ground_truth_text, tf.int32), logits=generated_logits, label_length=label_len, logit_length=logit_len, logits_time_major=False, blank_index=0))
                    # Clip CTC loss to prevent spikes that cause training instability (max observed: 818)
                    ctc_loss = tf.clip_by_value(ctc_loss_raw, 0.0, args.ctc_loss_clip_max)
                else:
                    ctc_loss = 0.0

                total_gen_loss = (args.adv_loss_weight * adversarial_loss) + (args.pixel_loss_weight * pixel_loss) + (args.ctc_loss_weight * ctc_loss)

            generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
            
            # Apply gradient clipping to prevent explosion (especially from CTC loss)
            generator_gradients, gen_grad_norm = tf.clip_by_global_norm(generator_gradients, args.gradient_clip_norm)
            discriminator_gradients, disc_grad_norm = tf.clip_by_global_norm(discriminator_gradients, args.gradient_clip_norm)
            
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            return total_gen_loss, total_disc_loss, adversarial_loss, pixel_loss, ctc_loss, gen_grad_norm, disc_grad_norm

    print("\n[Phase 4/6] Starting Training Loop...")
    dataset_iterator = iter(train_dataset)
    
    # Create a fixed batch from the VALIDATION set for generating samples
    val_iterator = iter(val_dataset)
    sample_batch = next(val_iterator)

    best_val_psnr = -1.0 # Initialize best PSNR for checkpoint management
    
    # Initialize training history for JSON logging
    training_history = {
        "start_time": datetime.now().isoformat(),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "lr_generator": args.lr_g,
            "lr_discriminator": args.lr_d,
            "pixel_loss_weight": args.pixel_loss_weight,
            "ctc_loss_weight": args.ctc_loss_weight,
            "adv_loss_weight": args.adv_loss_weight,
            "gradient_clip_norm": args.gradient_clip_norm,
            "ctc_loss_clip_max": args.ctc_loss_clip_max,
            "optimizer_generator": f"Adam(lr={args.lr_g}, beta_1=0.5, clipnorm={args.gradient_clip_norm})",
            "optimizer_discriminator": f"SGD(lr={args.lr_d}, momentum=0.9, clipnorm={args.gradient_clip_norm})",
            "model_architecture": {
                "generator": "U-Net (30M params, no dropout)",
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

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Track epoch metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "losses": {
                "g_loss": [],
                "d_loss": [],
                "adv_loss": [],
                "l1_loss": [],
                "ctc_loss": [],
                "g_grad_norm": [],
                "d_grad_norm": []
            }
        }
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        for step in pbar:
            degraded_batch, clean_batch, text_batch = next(dataset_iterator)
            g_loss, d_loss, adv_loss, pix_loss, ctc_loss, g_grad_norm, d_grad_norm = train_step(degraded_batch, clean_batch, text_batch)
            
            # Collect metrics
            epoch_metrics["losses"]["g_loss"].append(float(g_loss.numpy()))
            epoch_metrics["losses"]["d_loss"].append(float(d_loss.numpy()))
            epoch_metrics["losses"]["adv_loss"].append(float(adv_loss.numpy()))
            epoch_metrics["losses"]["l1_loss"].append(float(pix_loss.numpy()))
            epoch_metrics["losses"]["ctc_loss"].append(float(ctc_loss.numpy()))
            epoch_metrics["losses"]["g_grad_norm"].append(float(g_grad_norm.numpy()))
            epoch_metrics["losses"]["d_grad_norm"].append(float(d_grad_norm.numpy()))

            if (step + 1) % 50 == 0:
                pbar.set_postfix({
                    'G_Loss': f'{g_loss:.4f}', 
                    'D_Loss': f'{d_loss:.4f}', 
                    'Adv': f'{adv_loss:.4f}', 
                    'L1': f'{pix_loss:.4f}', 
                    'CTC': f'{ctc_loss:.4f}',
                    'G_Grad': f'{g_grad_norm:.2f}',
                    'D_Grad': f'{d_grad_norm:.2f}'
                })

        # Calculate epoch statistics
        epoch_metrics["losses_summary"] = {
            "g_loss_mean": float(np.mean(epoch_metrics["losses"]["g_loss"])),
            "g_loss_std": float(np.std(epoch_metrics["losses"]["g_loss"])),
            "d_loss_mean": float(np.mean(epoch_metrics["losses"]["d_loss"])),
            "d_loss_std": float(np.std(epoch_metrics["losses"]["d_loss"])),
            "ctc_loss_mean": float(np.mean(epoch_metrics["losses"]["ctc_loss"])),
            "ctc_loss_min": float(np.min(epoch_metrics["losses"]["ctc_loss"])),
            "ctc_loss_max": float(np.max(epoch_metrics["losses"]["ctc_loss"])),
            "l1_loss_mean": float(np.mean(epoch_metrics["losses"]["l1_loss"])),
            "g_grad_norm_max": float(np.max(epoch_metrics["losses"]["g_grad_norm"]))
        }
        
        # --- End of Epoch Actions ---
        if (epoch + 1) % args.eval_interval == 0:
            print("  Running validation...")
            run_validation_step(val_dataset, generator, val_psnr_metric, val_ssim_metric)
            psnr_result = val_psnr_metric.result()
            ssim_result = val_ssim_metric.result()
            print(f"  Validation - PSNR: {psnr_result:.2f}, SSIM: {ssim_result:.4f}")
            
            # Add validation metrics to epoch data
            epoch_metrics["validation"] = {
                "psnr": float(psnr_result.numpy()),
                "ssim": float(ssim_result.numpy())
            }
            
            # --- Checkpoint Management: Save only the best model ---
            if psnr_result > best_val_psnr:
                best_val_psnr = psnr_result
                # Save checkpoint to a specific 'best_model' path
                best_ckpt_save_path = os.path.join(args.checkpoint_dir, "best_model")
                checkpoint.save(file_prefix=best_ckpt_save_path)
                print(f"  ✅ New best model saved with PSNR: {best_val_psnr:.2f} at {best_ckpt_save_path}")
                epoch_metrics["best_model_saved"] = True
            else:
                print(f"  Current PSNR {psnr_result:.2f} not better than best {best_val_psnr:.2f}. No checkpoint saved.")
                epoch_metrics["best_model_saved"] = False
            
            val_psnr_metric.reset_state()
            val_ssim_metric.reset_state()
        else:
            epoch_metrics["validation"] = None

        # Always save samples at save_interval, but not necessarily checkpoints
        if (epoch + 1) % args.save_interval == 0:
            generated_samples = generator(sample_batch[0], training=False)
            # Save only the generated images for focused inspection
            img_to_save = (generated_samples * 255).numpy().astype(np.uint8)
            for i in range(args.batch_size):
                img = img_to_save[i]
                if img.shape[-1] == 1:
                    img = np.squeeze(img, axis=-1)
                
                # FIX: Transpose image to correct orientation (H, W) -> (W, H) for horizontal text
                # Model outputs (1024, 128) but text lines should be (128, 1024) horizontal
                if img.shape[0] > img.shape[1]:  # If height > width, transpose
                    img = np.transpose(img)
                
                cv2.imwrite(os.path.join(args.sample_dir, f'epoch_{epoch+1:04d}_sample_{i}.png'), img)
            print(f"  Saved {args.batch_size} sample images to {args.sample_dir}")

        epoch_time = time.time() - epoch_start_time
        epoch_metrics["epoch_time_seconds"] = float(epoch_time)
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        # Append epoch metrics to history
        training_history["epochs"].append(epoch_metrics)
        
        # Save metrics to JSON after each epoch (incremental save)
        metrics_file = os.path.join(metrics_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(training_history, f, indent=2)

    # Finalize training history
    training_history["end_time"] = datetime.now().isoformat()
    training_history["best_val_psnr"] = float(best_val_psnr)
    training_history["total_epochs_completed"] = len(training_history["epochs"])
    
    # Save final metrics
    final_metrics_file = os.path.join(metrics_dir, "training_metrics_final.json")
    with open(final_metrics_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n✅ Training metrics saved to: {final_metrics_file}")

    print("\n[Phase 6/6] Training Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Dual-Modal GAN for HTR.')
    parser.add_argument('--tfrecord_path', type=str, default='dual_modal_gan/data/dataset_gan.tfrecord', help='Path to the training TFRecord file.')
    parser.add_argument('--charset_path', type=str, default='real_data_preparation/real_data_charlist.txt', help='Path to the character set file.')
    parser.add_argument('--recognizer_weights', type=str, default='models/best_htr_recognizer/best_model.weights.h5', help='Path to pre-trained recognizer weights (Stage 3, CER 33.72%).')
    parser.add_argument('--gpu_id', type=str, default='1', help='ID of the GPU to use (e.g., "0" or "1").')
    parser.add_argument('--checkpoint_dir', type=str, default='dual_modal_gan/outputs/checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--sample_dir', type=str, default='dual_modal_gan/outputs/samples', help='Directory to save sample images.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='Number of steps per epoch (if None, calculated from dataset size).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per replica.')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator learning rate.')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator learning rate.')
    parser.add_argument('--pixel_loss_weight', type=float, default=1000.0, help='Weight for the L1 pixel loss (REBALANCED: 100→1000 for better visual quality).')
    parser.add_argument('--ctc_loss_weight', type=float, default=15.0, help='Weight for the HTR CTC loss (REBALANCED: 30→15 to balance visual quality vs readability).')
    parser.add_argument('--adv_loss_weight', type=float, default=1.0, help='Weight for the adversarial loss (REDUCED from 2.0 to balance with CTC).')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm to prevent explosion (TUNED: 5.0→1.0 after observing G_Grad=59796).')
    parser.add_argument('--ctc_loss_clip_max', type=float, default=300.0, help='Maximum value for CTC loss clipping to prevent spikes (observed max: 818).')
    parser.add_argument('--save_interval', type=int, default=5, help='Save a checkpoint and samples every N epochs.')
    parser.add_argument('--eval_interval', type=int, default=5, help='Run evaluation every N epochs.')

    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    main(args)
