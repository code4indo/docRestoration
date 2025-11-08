#!/usr/bin/env python3
"""
Evaluate DIBCO Performance on Best Model from Progressive Finetuning
Purpose: Measure if DIBCO PSNR actually improved despite ANRI catastrophic forgetting
"""

import os
import sys
import json
import numpy as np

# Disable XLA before importing TensorFlow
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Disable XLA at TF config level
tf.config.optimizer.set_jit(False)

# Set FP32
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import same as train_enhanced.py
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed

def calculate_psnr(original, generated):
    """Calculate PSNR between two images"""
    mse = tf.reduce_mean(tf.square(original - generated))
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(10.0)
    return psnr.numpy()

def calculate_ssim(original, generated):
    """Calculate SSIM between two images"""
    return tf.image.ssim(original, generated, max_val=1.0).numpy()

def load_checkpoint(checkpoint_path, generator, discriminator):
    """Load checkpoint weights"""
    print(f"\n🔄 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator
    )
    
    status = checkpoint.restore(checkpoint_path)
    print(f"✅ Checkpoint loaded successfully")
    return status

def evaluate_dibco_dataset(
    checkpoint_path,
    dibco_tfrecord,
    val_split=0.15,
    batch_size=4
):
    """Evaluate DIBCO performance"""
    
    print("="*80)
    print("DIBCO PERFORMANCE EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"DIBCO Dataset: {dibco_tfrecord}")
    print(f"Validation Split: {val_split}")
    print()
    
    # Build models
    print("🏗️  Building models...")
    generator = unet_enhanced(input_size=(1024, 128, 1))
    
    # Discriminator config matching training
    disc_config = {
        'spatial_attention_kernel': 3,
        'cross_modal_common_dim': 128,
        'batchnorm_momentum': 0.9,
        'dropout_rate': 0.3
    }
    
    discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
        img_shape=(128, 1024, 1),
        vocab_size=100,
        max_text_len=128,
        config=disc_config
    )
    
    print("✅ Models built")
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, generator, discriminator)
    
    # Load DIBCO dataset
    print(f"\n📊 Loading DIBCO dataset...")
    
    def parse_tfrecord(example_proto):
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'clean_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'label_length': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        degraded = tf.io.decode_png(parsed['image_raw'], channels=1)
        degraded = tf.cast(degraded, tf.float32) / 255.0
        degraded = tf.image.resize(degraded, [128, 1024])
        
        clean = tf.io.decode_png(parsed['clean_raw'], channels=1)
        clean = tf.cast(clean, tf.float32) / 255.0
        clean = tf.image.resize(clean, [128, 1024])
        
        label_bytes = parsed['label']
        label_length = parsed['label_length']
        
        return degraded, clean, label_bytes, label_length
    
    # Load and split dataset
    full_dataset = tf.data.TFRecordDataset(dibco_tfrecord)
    full_dataset = full_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Count total samples
    total_samples = sum(1 for _ in full_dataset)
    val_samples = int(total_samples * val_split)
    
    print(f"Total DIBCO samples: {total_samples}")
    print(f"Validation samples: {val_samples} ({val_split*100}%)")
    
    # Skip training samples, take validation
    val_dataset = full_dataset.skip(total_samples - val_samples)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Evaluation loop
    print(f"\n🔍 Evaluating on DIBCO validation set...")
    
    psnr_values = []
    ssim_values = []
    batch_count = 0
    
    for batch_idx, (degraded_batch, clean_batch, _, _) in enumerate(val_dataset):
        # Generate restored images
        generated_batch = generator(degraded_batch, training=False)
        
        # Calculate metrics for each sample in batch
        for i in range(degraded_batch.shape[0]):
            clean_img = clean_batch[i:i+1]
            generated_img = generated_batch[i:i+1]
            
            psnr = calculate_psnr(clean_img, generated_img)
            ssim = calculate_ssim(clean_img, generated_img)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches ({batch_count * batch_size} samples)...")
    
    # Calculate statistics
    psnr_array = np.array(psnr_values)
    ssim_array = np.array(ssim_values)
    
    psnr_mean = np.mean(psnr_array)
    psnr_std = np.std(psnr_array)
    psnr_median = np.median(psnr_array)
    psnr_min = np.min(psnr_array)
    psnr_max = np.max(psnr_array)
    
    ssim_mean = np.mean(ssim_array)
    ssim_std = np.std(ssim_array)
    
    # 95% confidence interval
    ci_95 = 1.96 * psnr_std / np.sqrt(len(psnr_array))
    
    print("\n" + "="*80)
    print("DIBCO VALIDATION RESULTS")
    print("="*80)
    print(f"Samples evaluated: {len(psnr_values)}")
    print()
    print(f"📊 PSNR Statistics:")
    print(f"   Mean:   {psnr_mean:.2f} dB")
    print(f"   Std:    {psnr_std:.2f} dB")
    print(f"   Median: {psnr_median:.2f} dB")
    print(f"   Min:    {psnr_min:.2f} dB")
    print(f"   Max:    {psnr_max:.2f} dB")
    print(f"   95% CI: [{psnr_mean - ci_95:.2f}, {psnr_mean + ci_95:.2f}] dB")
    print()
    print(f"📊 SSIM Statistics:")
    print(f"   Mean:   {ssim_mean:.4f}")
    print(f"   Std:    {ssim_std:.4f}")
    print()
    
    # Performance assessment
    print("="*80)
    print("PERFORMANCE ASSESSMENT")
    print("="*80)
    
    if psnr_mean >= 30.0:
        status = "✅ EXCELLENT"
        assessment = "DIBCO performance excellent (≥30 dB)"
    elif psnr_mean >= 28.0:
        status = "✅ GOOD"
        assessment = "DIBCO performance good (28-30 dB)"
    elif psnr_mean >= 26.0:
        status = "⚠️  MARGINAL"
        assessment = "DIBCO performance marginal (26-28 dB)"
    else:
        status = "❌ POOR"
        assessment = "DIBCO performance poor (<26 dB)"
    
    print(f"{status}: {assessment}")
    print()
    
    # Compare with expectations
    expected_min = 29.0
    expected_max = 32.0
    
    print("📊 Comparison with Expectations:")
    print(f"   Expected: {expected_min}-{expected_max} dB")
    print(f"   Actual:   {psnr_mean:.2f} dB")
    
    if psnr_mean >= expected_min:
        print(f"   ✅ MEETS EXPECTATIONS (+{psnr_mean - expected_min:.2f} dB above minimum)")
    else:
        print(f"   ❌ BELOW EXPECTATIONS ({expected_min - psnr_mean:.2f} dB below minimum)")
    
    print()
    
    # Trade-off analysis
    print("="*80)
    print("TRADE-OFF ANALYSIS: ANRI vs DIBCO")
    print("="*80)
    
    anri_start = 33.02
    anri_final = 27.52
    anri_loss = anri_start - anri_final
    
    print(f"ANRI Performance:")
    print(f"   Starting (ckpt-115): {anri_start:.2f} dB")
    print(f"   Final (best model):  {anri_final:.2f} dB")
    print(f"   Loss:                {anri_loss:.2f} dB ({anri_loss/anri_start*100:.1f}%)")
    print()
    print(f"DIBCO Performance:")
    print(f"   Achieved:            {psnr_mean:.2f} dB")
    print()
    
    if psnr_mean >= 28.0 and anri_loss > 5.0:
        print("⚠️  VERDICT: Pyrrhic victory")
        print(f"   DIBCO acceptable but ANRI catastrophic loss too high")
        print(f"   Recommendation: Re-run with ANRI data rehearsal")
    elif psnr_mean >= 30.0:
        print("✅ VERDICT: Trade-off acceptable")
        print(f"   DIBCO excellent, ANRI loss may be acceptable for this use case")
    else:
        print("❌ VERDICT: Unacceptable trade-off")
        print(f"   Both DIBCO and ANRI performance insufficient")
    
    print()
    print("="*80)
    
    # Save results
    results = {
        "checkpoint": checkpoint_path,
        "dataset": dibco_tfrecord,
        "samples_evaluated": len(psnr_values),
        "psnr": {
            "mean": float(psnr_mean),
            "std": float(psnr_std),
            "median": float(psnr_median),
            "min": float(psnr_min),
            "max": float(psnr_max),
            "ci_95_lower": float(psnr_mean - ci_95),
            "ci_95_upper": float(psnr_mean + ci_95)
        },
        "ssim": {
            "mean": float(ssim_mean),
            "std": float(ssim_std)
        },
        "assessment": {
            "status": status,
            "description": assessment,
            "meets_expectations": psnr_mean >= expected_min
        },
        "trade_off_analysis": {
            "anri_start": anri_start,
            "anri_final": anri_final,
            "anri_loss_db": float(anri_loss),
            "anri_loss_percent": float(anri_loss/anri_start*100),
            "dibco_psnr": float(psnr_mean)
        }
    }
    
    output_file = "dibco_performance_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {output_file}")
    print()
    
    return results

if __name__ == "__main__":
    checkpoint_path = "dual_modal_gan/checkpoints/dibco_finetuning_from_anri_v1/best_model/ckpt-125"
    dibco_tfrecord = "dual_modal_gan/data/dibco_tiled_full.tfrecord"
    
    results = evaluate_dibco_dataset(
        checkpoint_path=checkpoint_path,
        dibco_tfrecord=dibco_tfrecord,
        val_split=0.15,
        batch_size=4
    )
