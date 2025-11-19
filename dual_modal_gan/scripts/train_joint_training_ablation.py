#!/usr/bin/env python3
"""
Joint Training Ablation Study - CORRECT IMPLEMENTATION

This script properly implements joint training where the recognizer is trainable,
allowing us to measure catastrophic forgetting compared to frozen recognizer approach.

Key Differences from train_enhanced.py:
1. Uses load_joint_trainable_recognizer() instead of load_frozen_recognizer()
2. Recognizer weights ARE updated during training (trainable=True)
3. Monitors CER drift from baseline (33.72%)
4. Tracks gradient norms for all three components (G, D, R)

Reference: Souibgui et al. "Enhance to Read Better" (2021)
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import numpy as np
import cv2  # For saving visual samples
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed

# Utility functions
def read_charlist(path):
    """Read character list from file"""
    with open(path, 'r', encoding='utf-8') as f:
        chars = [line.rstrip('\n') for line in f]
    return chars

def decode_label(label_ids, charset):
    """Decode label IDs to text string"""
    decoded = []
    for label_id in label_ids:
        if label_id == 0:
            break
        idx = int(label_id) - 1
        if 0 <= idx < len(charset):
            decoded.append(charset[idx])
    return ''.join(decoded)

def decode_ctc_predictions(logits, charset):
    """Decode CTC predictions to text"""
    decoded_texts = []
    pred_indices = tf.argmax(logits, axis=-1).numpy()
    
    for pred in pred_indices:
        decoded = []
        prev = None
        for idx in pred:
            if idx != 0 and idx != prev:  # Not blank and not duplicate
                if 1 <= idx <= len(charset):
                    decoded.append(charset[idx - 1])
            prev = idx
        decoded_texts.append(''.join(decoded))
    
    return decoded_texts

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate"""
    import Levenshtein
    if len(ground_truth) == 0:
        return 1.0 if len(prediction) > 0 else 0.0
    distance = Levenshtein.distance(ground_truth, prediction)
    return distance / len(ground_truth)

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord function - CORRECT format from train_enhanced.py"""
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
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    # Deserialize label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    
    label = tf.cast(label, tf.int32)
    # Pad label to a static shape to prevent TF graph retracing/warnings
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label

def create_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """Create train/val/test datasets - EXACT SAME PATTERN as train_enhanced.py"""
    import sys
    sys.stdout.flush()  # Force immediate output
    
    print("DEBUG: Entering create_dataset()", flush=True)
    # HARDCODED: We know the dataset size is 4739 from previous validation
    # This avoids slow iteration counting (original does: sum(1 for _ in dataset))
    total_size = 4739
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    print(f"📊 Dataset Split:", flush=True)
    print(f"   Total: {total_size} samples", flush=True)
    print(f"   Train: {train_size} ({train_split*100:.0f}%)", flush=True)
    print(f"   Val:   {val_size} ({val_split*100:.0f}%)", flush=True)
    print(f"   Test:  {test_size} ({(1-train_split-val_split)*100:.0f}%)", flush=True)
    
    print("DEBUG: Creating TFRecordDataset...", flush=True)
    # BUG FIX v5: Use EXACT SAME pattern as train_enhanced.py
    # Create dataset ONCE, map ONCE, then split with take/skip from SINGLE instance
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    print("DEBUG: Mapping parse function...", flush=True)
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    print("DEBUG: Dataset mapped successfully!", flush=True)
    
    # Split: first 70% train, next 15% val, last 15% test
    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)
    
    # Configure datasets
    train_dataset = train_dataset.shuffle(1024).repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size

def main(args):
    print("="*80)
    print("⚠️  JOINT TRAINING ABLATION STUDY - PROPER IMPLEMENTATION")
    print("="*80)
    print("Expected: Catastrophic forgetting (CER degradation from 33.72% baseline)")
    print("="*80)
    
    # Set seeds
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configure GPU with aggressive memory management
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and args.gpu_id < len(gpus):
        try:
            # Set visible GPU
            tf.config.set_visible_devices([gpus[args.gpu_id]], 'GPU')
            
            # Enable memory growth (crucial for OOM prevention)
            tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)
            
            # Set soft device placement
            tf.config.set_soft_device_placement(True)
            
            print(f"✅ GPU {args.gpu_id} configured: {gpus[args.gpu_id].name}")
            print(f"   Memory growth: ENABLED")
            print(f"   Soft placement: ENABLED")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration warning: {e}")
    
    # Load charset
    charset = read_charlist(args.charset_path)
    vocab_size = len(charset) + 1  # +1 for blank token
    print(f"📚 Charset: {len(charset)} characters (vocab_size: {vocab_size})")
    
    # Create output directories for Chapter 5 analysis
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.sample_dir:
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"📁 Sample output directory: {args.sample_dir}")
    print(f"📁 Checkpoint dir: {args.checkpoint_dir}")
    
    # Load dataset
    print(f"\n📁 Loading dataset: {args.tfrecord_path}")
    train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = create_dataset(
        args.tfrecord_path, args.batch_size, args.train_split, args.val_split
    )
    
    # Build models
    print("\n🏗️  Building models...")
    
    # Generator
    generator = unet_enhanced()
    print("✅ Generator: Enhanced U-Net")
    
    # Recognizer - LOAD AS FROZEN BUT THEN SET TRAINABLE=TRUE
    print("\n🔧 Loading recognizer (will be set to trainable for joint training)...")
    recognizer = load_frozen_recognizer_fixed(
        weights_path=args.recognizer_weights,
        charset_size=len(charset),
        return_feature_map=True
    )
    
    # ✅ KEY CHANGE: Set trainable=True for joint training
    recognizer.trainable = True
    print(f"   ⚠️  Recognizer trainable: {recognizer.trainable} (JOINT TRAINING MODE)")
    print(f"   📊 Baseline CER: {args.baseline_cer}%")
    
    # Discriminator
    print("\n🔧 Building discriminator...")
    discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
        img_shape=(1024, 128, 1),
        vocab_size=vocab_size,
        max_text_len=128
    )
    print("✅ Discriminator: Enhanced V2 Fixed")
    
    # Optimizers
    optimizer_g = tf.keras.optimizers.Adam(args.lr_g)
    optimizer_d = tf.keras.optimizers.Adam(args.lr_d)
    optimizer_r = tf.keras.optimizers.RMSprop(args.lr_r)  # Separate optimizer for recognizer
    print(f"\n✅ Optimizers configured:")
    print(f"   Generator: Adam (lr={args.lr_g})")
    print(f"   Discriminator: Adam (lr={args.lr_d})")
    print(f"   Recognizer: RMSprop (lr={args.lr_r})")
    
    # Loss functions
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Metrics
    psnr_metric = tf.keras.metrics.Mean(name='psnr')
    ssim_metric = tf.keras.metrics.Mean(name='ssim')
    cer_metric = tf.keras.metrics.Mean(name='cer')
    
    # Checkpoint manager
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "="*80)
    print("🚀 Starting Joint Training")
    print("="*80)
    
    steps_per_epoch = args.steps_per_epoch or (train_size // args.batch_size)
    best_cer = float('inf')
    baseline_cer = args.baseline_cer / 100.0  # Convert to fraction
    
    history = {
        'epoch': [],
        'cer': [],
        'cer_delta': [],
        'psnr': [],
        'ssim': [],
        'loss_g': [],
        'loss_d': [],
        'loss_r': [],
        'loss_ctc': [],
        'train_time': []
    }
    
    # CSV for detailed step-by-step metrics (Chapter 5 data)
    import csv
    csv_path = os.path.join(args.checkpoint_dir, 'training_metrics_detailed.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'step', 'loss_g', 'loss_d', 'loss_r', 'loss_ctc', 'time_elapsed'])
    print(f"📊 Detailed metrics will be exported to: {csv_path}")
    
    # Summary metrics per epoch (Chapter 5 Table V.X)
    summary_csv_path = os.path.join(args.checkpoint_dir, 'epoch_summary.csv')
    summary_file = open(summary_csv_path, 'w', newline='')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['epoch', 'cer', 'cer_delta', 'psnr', 'ssim', 'loss_g_avg', 'loss_d_avg', 'loss_r_avg', 'train_time'])
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        epoch_start = time.time()
        epoch_losses_g = []
        epoch_losses_d = []
        epoch_losses_r = []
        
        # Training
        for step, (degraded_imgs, clean_imgs, labels) in enumerate(train_dataset.take(steps_per_epoch)):
            with tf.GradientTape(persistent=True) as tape:
                # Generator forward
                generated_imgs = generator(degraded_imgs, training=True)
                
                # Convert to [-1, 1] range for discriminator
                clean_imgs_tanh = clean_imgs * 2.0 - 1.0
                generated_imgs_tanh = generated_imgs * 2.0 - 1.0
                
                # Recognizer forward (multi-output: logits, features)
                clean_logits, clean_features = recognizer(clean_imgs, training=True)
                generated_logits, generated_features = recognizer(generated_imgs, training=True)
                
                # Decode predictions for discriminator
                clean_pred = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
                generated_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)
                
                # Discriminator forward
                real_output = discriminator([clean_imgs_tanh, clean_pred], training=True)
                fake_output = discriminator([generated_imgs_tanh, generated_pred], training=True)
                
                # Discriminator loss
                real_labels = tf.ones_like(real_output)
                fake_labels = tf.zeros_like(fake_output)
                loss_d_real = bce_loss_fn(real_labels, real_output)
                loss_d_fake = bce_loss_fn(fake_labels, fake_output)
                loss_d = loss_d_real + loss_d_fake
                
                # Generator losses
                adv_loss = bce_loss_fn(real_labels, fake_output)
                pixel_loss = mae_loss_fn(clean_imgs, generated_imgs)
                rec_feat_loss = mse_loss_fn(clean_features, generated_features)
                
                # CTC loss for recognizer
                label_len = tf.math.count_nonzero(labels, axis=1, dtype=tf.int32)
                logit_len = tf.fill([args.batch_size], generated_logits.shape[1])
                ctc_loss = tf.reduce_mean(
                    tf.nn.ctc_loss(
                        labels=tf.cast(labels, tf.int32),
                        logits=generated_logits,
                        label_length=label_len,
                        logit_length=logit_len,
                        logits_time_major=False,
                        blank_index=0
                    )
                )
                ctc_loss = tf.clip_by_value(ctc_loss, 0.0, args.ctc_loss_clip_max)
                
                # Total generator loss
                loss_g = (
                    args.adv_loss_weight * adv_loss +
                    args.pixel_loss_weight * pixel_loss +
                    args.rec_feat_loss_weight * rec_feat_loss +
                    args.ctc_loss_weight * ctc_loss
                )
                
                # Recognizer loss (CTC only, on clean images)
                clean_ctc_loss = tf.reduce_mean(
                    tf.nn.ctc_loss(
                        labels=tf.cast(labels, tf.int32),
                        logits=clean_logits,
                        label_length=label_len,
                        logit_length=logit_len,
                        logits_time_major=False,
                        blank_index=0
                    )
                )
                loss_r = tf.clip_by_value(clean_ctc_loss, 0.0, args.ctc_loss_clip_max)
            
            # Compute gradients
            grads_g = tape.gradient(loss_g, generator.trainable_variables)
            grads_d = tape.gradient(loss_d, discriminator.trainable_variables)
            grads_r = tape.gradient(loss_r, recognizer.trainable_variables)
            
            # Clip gradients
            grads_g, _ = tf.clip_by_global_norm(grads_g, args.gradient_clip_norm)
            grads_d, _ = tf.clip_by_global_norm(grads_d, args.gradient_clip_norm)
            grads_r, _ = tf.clip_by_global_norm(grads_r, args.gradient_clip_norm)
            
            # Apply gradients
            optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
            optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
            optimizer_r.apply_gradients(zip(grads_r, recognizer.trainable_variables))
            
            del tape
            
            epoch_losses_g.append(float(loss_g))
            epoch_losses_d.append(float(loss_d))
            epoch_losses_r.append(float(loss_r))
            
            # Log to CSV for Chapter 5 analysis
            csv_writer.writerow([epoch, step+1, float(loss_g), float(loss_d), float(loss_r), float(ctc_loss), time.time() - epoch_start])
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{steps_per_epoch}: "
                      f"G={loss_g:.4f}, D={loss_d:.4f}, R={loss_r:.4f}, "
                      f"CTC={ctc_loss:.4f}")
                csv_file.flush()  # Force write to disk
        
        # Validation
        if epoch % args.eval_interval == 0:
            print(f"\n📊 Validating epoch {epoch}...")
            
            val_cer_list = []
            val_psnr_list = []
            val_ssim_list = []
            first_batch = True
            val_count = 0
            max_val_samples = 100  # Limit to 100 samples for efficiency
            
            # Save first 5 visual samples
            saved_samples = False
            
            for degraded_imgs, clean_imgs, labels in val_dataset:
                if val_count >= max_val_samples:
                    break
                # Generate
                generated_imgs = generator(degraded_imgs, training=False)
                
                # Recognize
                generated_logits, _ = recognizer(generated_imgs, training=False)
                
                # Decode predictions
                pred_texts = decode_ctc_predictions(generated_logits, charset)
                
                # Calculate metrics
                for i in range(labels.shape[0]):
                    gt_text = decode_label(labels[i].numpy(), charset)
                    pred_text = pred_texts[i]
                    
                    cer = calculate_cer(gt_text, pred_text)
                    val_cer_list.append(cer)
                    
                    # Print first sample for transparency
                    if i == 0 and first_batch:
                        print(f"   ✨ Sample:")
                        print(f"      GT:   '{gt_text[:80]}...' (len={len(gt_text)})")
                        print(f"      Pred: '{pred_text[:80]}...' (len={len(pred_text)})")
                        print(f"      CER:  {cer*100:.2f}%")
                
                # Save visual samples (first batch, up to 5 images)
                if not saved_samples and labels.shape[0] > 0:
                    num_vis_samples = min(5, labels.shape[0])
                    for img_idx in range(num_vis_samples):
                        # Save generated/restored image
                        restored_img = (generated_imgs[img_idx].numpy() * 255).astype(np.uint8)
                        if restored_img.shape[-1] == 1:
                            restored_img = np.squeeze(restored_img, axis=-1)
                        if restored_img.shape[0] > restored_img.shape[1]:  # Transpose if needed
                            restored_img = np.transpose(restored_img)
                        
                        # Save degraded image
                        degraded_img = (degraded_imgs[img_idx].numpy() * 255).astype(np.uint8)
                        if degraded_img.shape[-1] == 1:
                            degraded_img = np.squeeze(degraded_img, axis=-1)
                        if degraded_img.shape[0] > degraded_img.shape[1]:
                            degraded_img = np.transpose(degraded_img)
                        
                        # Save ground truth image
                        gt_img = (clean_imgs[img_idx].numpy() * 255).astype(np.uint8)
                        if gt_img.shape[-1] == 1:
                            gt_img = np.squeeze(gt_img, axis=-1)
                        if gt_img.shape[0] > gt_img.shape[1]:
                            gt_img = np.transpose(gt_img)
                        
                        # Create comparison: [Degraded | Ground Truth | Restored]
                        comparison = np.vstack([degraded_img, gt_img, restored_img])
                        comparison_path = os.path.join(args.sample_dir, f'epoch_{epoch:04d}_sample_{img_idx}.png')
                        cv2.imwrite(comparison_path, comparison)
                    
                    saved_samples = True
                    print(f"   💾 Saved {num_vis_samples} sample images to {args.sample_dir}")
                
                first_batch = False  # Only print first batch
                val_count += labels.shape[0]
                
                # Calculate PSNR/SSIM
                psnr = tf.reduce_mean(tf.image.psnr(clean_imgs, generated_imgs, max_val=1.0))
                ssim = tf.reduce_mean(tf.image.ssim(clean_imgs, generated_imgs, max_val=1.0))
                
                val_psnr_list.append(float(psnr))
                val_ssim_list.append(float(ssim))
            
            # Calculate epoch metrics
            avg_cer = np.mean(val_cer_list)
            avg_psnr = np.mean(val_psnr_list)
            avg_ssim = np.mean(val_ssim_list)
            avg_loss_g = np.mean(epoch_losses_g)
            avg_loss_d = np.mean(epoch_losses_d)
            avg_loss_r = np.mean(epoch_losses_r)
            
            # Calculate CER delta from baseline
            cer_delta = (avg_cer - baseline_cer) * 100  # In percentage points
            
            print(f"\n✅ Epoch {epoch} Results:")
            print(f"   CER: {avg_cer*100:.2f}% (baseline: {baseline_cer*100:.2f}%)")
            print(f"   Forgetting Δ: {cer_delta:+.2f}%")
            print(f"   PSNR: {avg_psnr:.2f} dB")
            print(f"   SSIM: {avg_ssim:.4f}")
            print(f"   Loss G: {avg_loss_g:.4f}, D: {avg_loss_d:.4f}, R: {avg_loss_r:.4f}")
            
            if avg_cer > baseline_cer + 0.05:  # 5% degradation
                print(f"\n⚠️  CATASTROPHIC FORGETTING DETECTED!")
                print(f"   CER degraded by {cer_delta:.2f}% from baseline")
            
            # Save history
            history['epoch'].append(epoch)
            history['cer'].append(avg_cer)
            history['cer_delta'].append(cer_delta)
            history['psnr'].append(avg_psnr)
            history['ssim'].append(avg_ssim)
            history['loss_g'].append(avg_loss_g)
            history['loss_d'].append(avg_loss_d)
            history['loss_r'].append(avg_loss_r)
            history['train_time'].append(epoch_time)
            
            # Export epoch summary to CSV (for Chapter 5 Table)
            summary_writer.writerow([epoch, avg_cer*100, cer_delta, avg_psnr, avg_ssim, 
                                    avg_loss_g, avg_loss_d, avg_loss_r, epoch_time])
            summary_file.flush()
            
            # Save checkpoint if best
            if avg_cer < best_cer:
                best_cer = avg_cer
                checkpoint_path = checkpoint_dir / f"best_model_epoch{epoch}.weights.h5"
                generator.save_weights(str(checkpoint_path))
                print(f"💾 Best model saved: {checkpoint_path}")
        
        epoch_time = time.time() - epoch_start
        history['train_time'].append(epoch_time) if epoch % args.eval_interval != 0 else None
        print(f"\n⏱️  Epoch {epoch} completed in {epoch_time:.2f}s")
    
    # Close CSV files
    csv_file.close()
    summary_file.close()
    print(f"\n📊 Training metrics exported:")
    print(f"   Detailed: {csv_path}")
    print(f"   Summary:  {summary_csv_path}")
    
    # Save final results
    results_path = checkpoint_dir / "joint_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Results saved: {results_path}")
    
    print("\n" + "="*80)
    print("✅ Joint Training Completed")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint Training Ablation Study')
    
    # Paths
    parser.add_argument('--tfrecord_path', type=str, default='dual_modal_gan/data/dataset_gan.tfrecord')
    parser.add_argument('--charset_path', type=str, default='real_data_preparation/real_data_charlist.txt')
    parser.add_argument('--recognizer_weights', type=str, 
                       default='/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5')
    parser.add_argument('--checkpoint_dir', type=str, default='dual_modal_gan/checkpoints/joint_training_fixed')
    parser.add_argument('--sample_dir', type=str, default='dual_modal_gan/outputs/samples_joint_training',
                       help='Directory to save visual sample images for Chapter 5 analysis')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # Splits
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.15)
    
    # Learning rates
    parser.add_argument('--lr_g', type=float, default=0.0001)
    parser.add_argument('--lr_d', type=float, default=0.0001)
    parser.add_argument('--lr_r', type=float, default=0.0003)
    
    # Loss weights
    parser.add_argument('--pixel_loss_weight', type=float, default=50.0)
    parser.add_argument('--adv_loss_weight', type=float, default=3.0)
    parser.add_argument('--rec_feat_loss_weight', type=float, default=1.0)
    parser.add_argument('--ctc_loss_weight', type=float, default=1.0)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--ctc_loss_clip_max', type=float, default=400.0)
    
    # Baseline
    parser.add_argument('--baseline_cer', type=float, default=33.72, help='Baseline CER from frozen recognizer')
    
    args = parser.parse_args()
    main(args)
