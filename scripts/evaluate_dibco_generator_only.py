#!/usr/bin/env python3
"""
Evaluate DIBCO Performance on Best Model - Generator Only
Purpose: Measure DIBCO PSNR to determine if progressive finetuning achieved its goal
"""

import os
import sys
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.optimizer.set_jit(False)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced

def parse_tfrecord(serialized_example):
    """Parse TFRecord example"""
    feature_description = {
        'degraded_image': tf.io.FixedLenFeature([], tf.string),
        'clean_image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    # Decode images
    degraded = tf.io.decode_png(example['degraded_image'], channels=1)
    clean = tf.io.decode_png(example['clean_image'], channels=1)
    
    # Convert to float32 [0, 1]
    degraded = tf.cast(degraded, tf.float32) / 255.0
    clean = tf.cast(clean, tf.float32) / 255.0
    
    return degraded, clean

def calculate_psnr_batch(clean_images, generated_images):
    """Calculate PSNR for batch"""
    psnr_values = []
    for clean, generated in zip(clean_images, generated_images):
        mse = tf.reduce_mean(tf.square(clean - generated))
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * tf.math.log(1.0 / tf.sqrt(mse)) / tf.math.log(10.0)
        psnr_values.append(psnr.numpy())
    return psnr_values

def evaluate_dibco(checkpoint_path, dibco_tfrecord, val_split=0.15):
    """
    Evaluate DIBCO PSNR using generator checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint (e.g., ckpt-125)
        dibco_tfrecord: Path to DIBCO TFRecord
        val_split: Validation split ratio
    """
    print("="*80)
    print("DIBCO PERFORMANCE EVALUATION (Generator Only)")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"DIBCO TFRecord: {dibco_tfrecord}")
    print(f"Validation Split: {val_split}")
    print()
    
    # Count total samples
    print("[1/5] Counting samples...")
    dataset = tf.data.TFRecordDataset(dibco_tfrecord)
    total_samples = sum(1 for _ in dataset)
    val_samples = int(total_samples * val_split)
    train_samples = total_samples - val_samples
    
    print(f"   Total: {total_samples} samples")
    print(f"   Train: {train_samples} samples ({(1-val_split)*100:.0f}%)")
    print(f"   Val:   {val_samples} samples ({val_split*100:.0f}%)")
    
    # Create validation dataset
    print("\n[2/5] Creating validation dataset...")
    val_dataset = dataset.skip(train_samples).take(val_samples)
    val_dataset = val_dataset.map(parse_tfrecord)
    val_dataset = val_dataset.batch(4)
    
    # Build generator
    print("\n[3/5] Building generator...")
    input_size = (1024, 128, 1)  # Match training config
    generator = unet_enhanced(input_size=input_size)
    
    # Build model with dummy input to create weights
    dummy_input = tf.zeros((1, 1024, 128, 1))
    _ = generator(dummy_input, training=False)
    
    print(f"   Generator built: {generator.count_params():,} parameters")
    
    # Load checkpoint - GENERATOR ONLY
    print("\n[4/5] Loading checkpoint...")
    checkpoint = tf.train.Checkpoint(generator=generator)
    status = checkpoint.restore(checkpoint_path)
    
    # Check if weights loaded
    try:
        status.assert_existing_objects_matched()
        print("   ✅ Generator weights loaded successfully")
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")
        print("   Attempting to continue anyway...")
    
    # Evaluate on validation set
    print(f"\n[5/5] Evaluating on {val_samples} validation samples...")
    all_psnr = []
    all_ssim = []
    
    batch_idx = 0
    for degraded_batch, clean_batch in val_dataset:
        # Generate restored images
        generated_batch = generator(degraded_batch, training=False)
        
        # Calculate metrics
        psnr_values = calculate_psnr_batch(clean_batch, generated_batch)
        all_psnr.extend(psnr_values)
        
        # SSIM
        for clean, generated in zip(clean_batch, generated_batch):
            ssim = tf.image.ssim(clean, generated, max_val=1.0)
            all_ssim.append(ssim.numpy())
        
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"   Progress: {batch_idx * 4}/{val_samples} samples")
    
    # Calculate statistics
    psnr_mean = np.mean(all_psnr)
    psnr_std = np.std(all_psnr)
    psnr_median = np.median(all_psnr)
    psnr_min = np.min(all_psnr)
    psnr_max = np.max(all_psnr)
    
    ssim_mean = np.mean(all_ssim)
    ssim_std = np.std(all_ssim)
    
    # Print results
    print("\n" + "="*80)
    print("DIBCO EVALUATION RESULTS")
    print("="*80)
    print(f"Samples Evaluated: {len(all_psnr)}")
    print()
    print("PSNR Statistics:")
    print(f"   Mean:   {psnr_mean:.2f} dB")
    print(f"   Std:    {psnr_std:.2f} dB")
    print(f"   Median: {psnr_median:.2f} dB")
    print(f"   Min:    {psnr_min:.2f} dB")
    print(f"   Max:    {psnr_max:.2f} dB")
    print()
    print("SSIM Statistics:")
    print(f"   Mean: {ssim_mean:.4f}")
    print(f"   Std:  {ssim_std:.4f}")
    print()
    
    # Assessment
    print("="*80)
    print("ASSESSMENT:")
    print("="*80)
    
    if psnr_mean >= 30.0:
        print("✅ EXCELLENT: DIBCO PSNR ≥30 dB")
        print("   → Progressive finetuning SUCCESSFUL")
        print("   → ANRI sacrifice may be justified")
        assessment = "SUCCESS"
    elif psnr_mean >= 28.0:
        print("⚠️  MARGINAL: DIBCO PSNR 28-30 dB")
        print("   → Moderate improvement")
        print("   → ANRI sacrifice questionable (-5.5 dB loss)")
        assessment = "MARGINAL"
    else:
        print("❌ POOR: DIBCO PSNR <28 dB")
        print("   → Progressive finetuning FAILED")
        print("   → Lost ANRI (-5.5 dB), didn't gain DIBCO")
        assessment = "FAILURE"
    
    print("="*80)
    
    # Save results
    results = {
        'checkpoint': str(checkpoint_path),
        'dibco_tfrecord': str(dibco_tfrecord),
        'validation_samples': len(all_psnr),
        'psnr': {
            'mean': float(psnr_mean),
            'std': float(psnr_std),
            'median': float(psnr_median),
            'min': float(psnr_min),
            'max': float(psnr_max)
        },
        'ssim': {
            'mean': float(ssim_mean),
            'std': float(ssim_std)
        },
        'assessment': assessment
    }
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate DIBCO performance')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (e.g., dual_modal_gan/checkpoints/.../best_model/ckpt-125)')
    parser.add_argument('--dibco_tfrecord', type=str, 
                        default='dual_modal_gan/data/mixed_70base_30dibco_full.tfrecord',
                        help='Path to DIBCO TFRecord')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--output', type=str, default='dibco_evaluation_results.json',
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_dibco(args.checkpoint, args.dibco_tfrecord, args.val_split)
    
    # Save results
    import json
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == '__main__':
    main()
