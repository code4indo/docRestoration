"""
Joint Training Ablation Study - Simplified Training Script

Purpose: Prove that frozen recognizer prevents catastrophic forgetting
Expected: Joint training shows CER degradation from 33.72% to >40%

This is a MINIMAL implementation focused on the ablation study objective.
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import cv2

# Disable XLA
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Pure FP32
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

# Import models and dataset function
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed  # Use existing working model
from dual_modal_gan.scripts.train_enhanced import create_dataset

def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def compute_ctc_loss(logits, labels, logit_length, label_length):
    """
    Compute CTC loss
    
    CRITICAL FIX: Dataset has token 108 but charset only has 108 chars (0-107)
    Need to use vocab_size = 109 to handle this edge case
    """
    return tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=-1
    )

def decode_ctc(logits, charset):
    """Decode CTC predictions"""
    batch_size = logits.shape[0]
    results = []
    for b in range(batch_size):
        raw_preds = np.argmax(logits[b], axis=-1)
        deduped = []
        prev = -1
        for token in raw_preds:
            if token != prev:
                deduped.append(token)
                prev = token
        chars = [charset[t] for t in deduped if t < len(charset)]
        results.append(''.join(chars))
    return results

def compute_cer(pred_texts, gt_texts):
    """Compute Character Error Rate"""
    import editdistance
    total_dist = 0
    total_len = 0
    for pred, gt in zip(pred_texts, gt_texts):
        total_dist += editdistance.eval(pred, gt)
        total_len += len(gt)
    return total_dist / max(total_len, 1)

def save_samples(generator, val_dataset, sample_dir, epoch, num_samples=5):
    """Save sample images showing degraded, clean, and restored versions"""
    os.makedirs(sample_dir, exist_ok=True)
    
    # Collect samples from validation set
    collected_degraded = []
    collected_clean = []
    
    val_iter = iter(val_dataset)
    samples_collected = 0
    
    while samples_collected < num_samples:
        try:
            batch = next(val_iter)
            degraded_batch, clean_batch, _ = batch
            batch_size_actual = degraded_batch.shape[0]
            samples_needed = min(batch_size_actual, num_samples - samples_collected)
            
            collected_degraded.append(degraded_batch[:samples_needed])
            collected_clean.append(clean_batch[:samples_needed])
            
            samples_collected += samples_needed
        except StopIteration:
            break
    
    if not collected_degraded:
        print("  ⚠️  No samples collected")
        return
    
    # Concatenate collected samples
    degraded_samples = tf.concat(collected_degraded, axis=0)[:num_samples]
    clean_samples = tf.concat(collected_clean, axis=0)[:num_samples]
    
    # Generate restored images (no normalization needed, dataset is [0,1])
    generated_samples = generator(degraded_samples, training=False)
    
    # Save individual samples and comparisons
    for i in range(min(num_samples, degraded_samples.shape[0])):
        # Convert to uint8
        generated_img = (generated_samples[i].numpy() * 255).astype(np.uint8)
        degraded_img = (degraded_samples[i].numpy() * 255).astype(np.uint8)
        clean_img = (clean_samples[i].numpy() * 255).astype(np.uint8)
        
        # Remove channel dimension if grayscale
        if generated_img.shape[-1] == 1:
            generated_img = np.squeeze(generated_img, axis=-1)
            degraded_img = np.squeeze(degraded_img, axis=-1)
            clean_img = np.squeeze(clean_img, axis=-1)
        
        # Transpose if needed (W,H) -> (H,W)
        if generated_img.shape[0] > generated_img.shape[1]:
            generated_img = np.transpose(generated_img)
            degraded_img = np.transpose(degraded_img)
            clean_img = np.transpose(clean_img)
        
        # Save individual restored image
        sample_path = os.path.join(sample_dir, f'epoch_{epoch:04d}_sample_{i}.png')
        cv2.imwrite(sample_path, generated_img)
        
        # Create vertical comparison: [Degraded | Ground Truth | Restored]
        comparison = np.vstack([degraded_img, clean_img, generated_img])
        comparison_path = os.path.join(sample_dir, f'comparison_epoch_{epoch:04d}_sample_{i}.png')
        cv2.imwrite(comparison_path, comparison)
    
    print(f"  ✅ Saved {min(num_samples, degraded_samples.shape[0])} sample images to {sample_dir}")

@tf.function
def train_step_joint(degraded_batch, clean_batch, text_labels,
                     generator, discriminator, recognizer,
                     gen_optimizer, disc_optimizer, rec_optimizer,
                     lambda_pixel, lambda_adv, lambda_ctc):
    """
    Joint training step: G, D, and R all trainable
    Scenario S1: Train recognizer on GT clean images
    
    CRITICAL: Dataset is (W, H, C) but discriminator needs (H, W, C)
    """
    
    batch_size = tf.shape(degraded_batch)[0]
    
    # Transpose for discriminator: (batch, 1024, 128, 1) → (batch, 128, 1024, 1)
    degraded_for_disc = tf.transpose(degraded_batch, perm=[0, 2, 1, 3])
    clean_for_disc = tf.transpose(clean_batch, perm=[0, 2, 1, 3])
    
    # Compute label lengths (non-zero tokens)
    label_lengths = tf.reduce_sum(tf.cast(text_labels > 0, tf.int32), axis=1)
    
    # Logit length depends on recognizer output - get dynamically
    # For now, assume 64 time steps (typical for 1024x128 input after pooling)
    logit_length = tf.fill([batch_size], 64)
    
    # Three gradient tapes for G, D, R
    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape, \
         tf.GradientTape() as rec_tape:
        
        # Generate cleaned images (generator expects W, H, C format)
        generated = generator(degraded_batch, training=True)
        generated_for_disc = tf.transpose(generated, perm=[0, 2, 1, 3])  # Transpose for discriminator
        
        # Get text predictions from recognizer for discriminator
        # Real images: clean
        rec_output_real = recognizer(clean_batch, training=False)  # Don't train R from discriminator
        if isinstance(rec_output_real, list):
            text_pred_real = rec_output_real[0]  # logits shape (batch, 128, 109)
        else:
            text_pred_real = rec_output_real
        text_pred_real = tf.argmax(text_pred_real, axis=-1)  # Shape: (batch, 128)
        
        # Fake images: generated
        rec_output_fake = recognizer(generated, training=False)
        if isinstance(rec_output_fake, list):
            text_pred_fake = rec_output_fake[0]
        else:
            text_pred_fake = rec_output_fake
        text_pred_fake = tf.argmax(text_pred_fake, axis=-1)  # Shape: (batch, 128)
        
        # Discriminator forward (discriminator expects H, W, C format for image + text_pred)
        d_real = discriminator([clean_for_disc, text_pred_real], training=True)
        d_fake = discriminator([generated_for_disc, text_pred_fake], training=True)
        
        # Discriminator loss
        d_loss_real = tf.reduce_mean(tf.square(d_real - 1.0))
        d_loss_fake = tf.reduce_mean(tf.square(d_fake))
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        # Generator adversarial loss
        g_adv_loss = tf.reduce_mean(tf.square(d_fake - 1.0))
        
        # Pixel reconstruction loss
        pixel_loss = tf.reduce_mean(tf.abs(generated - clean_batch))
        
        # CTC loss on generated images (for generator)
        # Already computed above: rec_output_fake
        if isinstance(rec_output_fake, list):
            logits_gen = rec_output_fake[0]
        else:
            logits_gen = rec_output_fake
        
        ctc_loss_gen = compute_ctc_loss(logits_gen, text_labels, logit_length, label_lengths)
        ctc_loss_gen = tf.reduce_mean(ctc_loss_gen)
        ctc_loss_gen = tf.clip_by_value(ctc_loss_gen, 0.0, 400.0)  # Clip extreme values
        
        # Total generator loss
        g_loss = lambda_adv * g_adv_loss + lambda_pixel * pixel_loss + lambda_ctc * ctc_loss_gen
        
        # Recognizer loss on GT clean images (S1 scenario)
        # Already computed above: rec_output_real (but was training=False, need training=True)
        rec_output_train = recognizer(clean_batch, training=True)  # Train R on GT
        if isinstance(rec_output_train, list):
            logits_clean = rec_output_train[0]
        else:
            logits_clean = rec_output_train
        
        ctc_loss_clean = compute_ctc_loss(logits_clean, text_labels, logit_length, label_lengths)
        ctc_loss_clean = tf.reduce_mean(ctc_loss_clean)
        r_loss = tf.clip_by_value(ctc_loss_clean, 0.0, 400.0)
    
    # Apply gradients
    g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    d_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    r_grads = rec_tape.gradient(r_loss, recognizer.trainable_variables)
    
    g_grads = [tf.clip_by_norm(g, 1.0) for g in g_grads]
    d_grads = [tf.clip_by_norm(g, 1.0) for g in d_grads]
    r_grads = [tf.clip_by_norm(g, 1.0) for g in r_grads]
    
    gen_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    rec_optimizer.apply_gradients(zip(r_grads, recognizer.trainable_variables))
    
    return {
        'g_loss': g_loss,
        'd_loss': d_loss,
        'r_loss': r_loss,
        'ctc_gen': ctc_loss_gen,
        'ctc_clean': ctc_loss_clean,
        'pixel_loss': pixel_loss
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("\n" + "="*80)
    print("⚠️  JOINT TRAINING ABLATION STUDY - EXPERIMENTAL MODE")
    print("="*80)
    print("Expected: Catastrophic forgetting (CER degradation from 33.72% to >40%)")
    print("="*80 + "\n")
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get('gpu_id', '0')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU configured: {gpus[0].name}")
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    charset = read_charlist(config['charset_path'])
    
    # CRITICAL FIX: Dataset has token up to 108, but charset only 108 chars (0-107)
    # Use max(len(charset), 109) to handle edge case
    vocab_size = max(len(charset) + 1, 109)  # +1 for blank, min 109 for safety
    
    print(f"   Charset: {len(charset)} characters")
    print(f"   Vocab size (with blank): {vocab_size}")
    
    train_dataset, val_dataset, _, train_count, val_count, _ = create_dataset(
        config['tfrecord_path'],
        config['batch_size'],
        train_split=config.get('train_split', 0.7),
        val_split=config.get('val_split', 0.15)
    )
    print(f"Dataset: {train_count} train, {val_count} val")
    
    # Build models
    print("\n[2/5] Building models...")
    generator = unet_enhanced()
    discriminator = build_dual_modal_discriminator_enhanced_v2_fixed()
    
    # Get joint training config
    joint_config = config.get('joint_training_config', {})
    baseline_cer = joint_config.get('baseline_cer', 33.72)
    
    # Load recognizer using EXISTING working model, but make it trainable
    print("\n⚠️  Loading recognizer in JOINT TRAINING mode (ABLATION STUDY)")
    print("="*80)
    recognizer = load_frozen_recognizer_fixed(
        config['recognizer_weights'],
        vocab_size - 1,  # charset_size (without blank)
        return_feature_map=True
    )
    
    # CRITICAL: Make recognizer trainable (this is the joint training mode)
    recognizer.trainable = True
    print(f"✅ Recognizer loaded: trainable={recognizer.trainable}")
    print(f"   Baseline CER (frozen): {baseline_cer}%")
    print(f"   Expected CER (joint): >40% (catastrophic forgetting)")
    print("="*80)
    
    print(f"\n✅ Generator: {generator.count_params():,} params")
    print(f"✅ Discriminator: {discriminator.count_params():,} params")
    print(f"✅ Recognizer: {recognizer.count_params():,} params (TRAINABLE - ABLATION MODE)")
    
    # Create optimizers
    print("\n[3/5] Creating optimizers...")
    gen_opt = tf.keras.optimizers.Adam(learning_rate=config.get('lr_g', 1e-4))
    disc_opt = tf.keras.optimizers.Adam(learning_rate=config.get('lr_d', 1e-4))
    
    lr_r = joint_config.get('lr_recognizer', 3e-4)
    rec_opt = tf.keras.optimizers.RMSprop(learning_rate=lr_r)
    print(f"✅ Recognizer optimizer: RMSprop (lr={lr_r})")
    
    # Loss weights
    lambda_pixel = config.get('pixel_loss_weight', 50.0)
    lambda_adv = config.get('adv_loss_weight', 3.0)
    lambda_ctc = config.get('ctc_loss_weight', 1.0)
    
    print(f"\nLoss weights: pixel={lambda_pixel}, adv={lambda_adv}, ctc={lambda_ctc}")
    
    # Training loop
    print("\n[4/5] Starting training...")
    epochs = config.get('epochs', 20)
    steps_per_epoch = config.get('steps_per_epoch', 100)
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    log_file = os.path.join(config['checkpoint_dir'], 'training_log.txt')
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*80}")
        
        epoch_losses = []
        step = 0
        
        # Create iterator to ensure fresh iteration each epoch
        train_iter = iter(train_dataset)
        
        for _ in range(steps_per_epoch):
            try:
                batch = next(train_iter)
                degraded, clean, text_labels = batch  # Dataset returns tuple
                
                losses = train_step_joint(
                    degraded, clean, text_labels,
                    generator, discriminator, recognizer,
                    gen_opt, disc_opt, rec_opt,
                    lambda_pixel, lambda_adv, lambda_ctc
                )
                
                epoch_losses.append(losses)
                step += 1
                
                if step % 10 == 0:
                    print(f"  Step {step}/{steps_per_epoch}: "
                          f"G={losses['g_loss']:.4f}, "
                          f"D={losses['d_loss']:.4f}, "
                          f"R={losses['r_loss']:.4f}, "
                          f"CTC_clean={losses['ctc_clean']:.4f}")
            except StopIteration:
                print(f"  ⚠️  Dataset exhausted at step {step}, reinitializing iterator")
                train_iter = iter(train_dataset)
                continue
        
        # Validation
        print(f"\n📊 Validating epoch {epoch}...")
        val_cer_list = []
        val_psnr_list = []
        
        for val_batch in val_dataset.take(10):
            degraded_val, clean_val, label_val = val_batch  # Dataset returns tuple
            
            # Compute CER on GT clean images (monitor forgetting)
            rec_output_val = recognizer(clean_val, training=False)
            # Handle multi-output model
            if isinstance(rec_output_val, list):
                logits_val = rec_output_val[0]  # First output is logits
            else:
                logits_val = rec_output_val
            
            pred_texts = decode_ctc(logits_val.numpy(), charset)
            
            # Decode ground truth from label tokens
            gt_texts = []
            for i in range(label_val.shape[0]):
                tokens = label_val[i].numpy()
                tokens = tokens[tokens > 0]  # Remove padding
                chars = [charset[t] for t in tokens if t < len(charset)]
                gt_texts.append(''.join(chars))
            
            cer = compute_cer(pred_texts, gt_texts)
            val_cer_list.append(cer)
            
            # Compute PSNR on generated
            generated_val = generator(degraded_val, training=False)
            psnr = tf.reduce_mean(tf.image.psnr(generated_val, clean_val, max_val=1.0))
            val_psnr_list.append(psnr.numpy())
        
        avg_cer = np.mean(val_cer_list) * 100
        avg_psnr = np.mean(val_psnr_list)
        forgetting_delta = avg_cer - baseline_cer
        
        print(f"\n✅ Epoch {epoch} Results:")
        print(f"   CER: {avg_cer:.2f}% (baseline: {baseline_cer:.2f}%)")
        print(f"   Forgetting Δ: {forgetting_delta:+.2f}%")
        print(f"   PSNR: {avg_psnr:.2f} dB")
        
        if forgetting_delta > 5.0:
            print(f"\n⚠️  CATASTROPHIC FORGETTING DETECTED!")
            print(f"   CER degraded by {forgetting_delta:.2f}% from baseline")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch}: CER={avg_cer:.2f}%, Delta={forgetting_delta:+.2f}%, PSNR={avg_psnr:.2f}dB\n")
        
        # Save samples every 5 epochs
        if epoch % 5 == 0:
            print(f"\n💾 Saving sample images (epoch {epoch})...")
            save_samples(generator, val_dataset, config['sample_dir'], epoch, num_samples=5)
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(config['checkpoint_dir'], f'ckpt-{epoch}')
            generator.save_weights(f"{ckpt_path}_gen.weights.h5")
            recognizer.save_weights(f"{ckpt_path}_rec.weights.h5")
            print(f"💾 Checkpoint saved: {ckpt_path}")
    
    print("\n" + "="*80)
    print("✅ Training completed!")
    print("="*80)
    print(f"Check results in: {config['checkpoint_dir']}")
    print(f"Training log: {log_file}")

if __name__ == '__main__':
    main()
