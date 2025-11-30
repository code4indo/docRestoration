"""
Find Lowest SSIM Sample in Test Set

This script evaluates the LOCKED test set and identifies the sample with the lowest SSIM score.
It saves the degraded, clean, and generated images for this sample, along with a comparison image.

Usage:
    poetry run python dual_modal_gan/scripts/find_lowest_ssim.py \
        --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
        --checkpoint_name ckpt-88 \
        --config configs/production_v3_academic_split_70_15_15.json \
        --output_dir results/lowest_ssim_analysis \
        --gpu_id 1
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

# Disable XLA and configure TF
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)

# Set FP32 precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import models
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced

# --- Utility Functions ---

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord example - EXACT SAME as training."""
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize degraded image
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (H,W,C) → (W,H,C)
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (H,W,C) → (W,H,C)
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    return degraded_image, clean_image

def create_test_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """Create ONLY the test dataset."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    print(f"\n📊 Test Set Configuration:")
    print(f"   Total dataset: {total_size}")
    print(f"   Test samples:  {test_size} (15%) - LOADED ✅")
    
    return test_dataset, test_size

def find_lowest_ssim(args):
    """Find sample with lowest SSIM in test set."""
    
    print("="*80)
    print("🔍 FINDING LOWEST SSIM SAMPLE")
    print("="*80)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    test_dataset, test_size = create_test_dataset(
        config['tfrecord_path'],
        config['batch_size'],
        train_split=config.get('train_split', 0.7),
        val_split=config.get('val_split', 0.15)
    )
    
    print(f"\n🏗️  Building generator...")
    generator = unet_enhanced(input_size=(1024, 128, 1))
    
    print(f"\n📦 Loading checkpoint...")
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    status = checkpoint.restore(checkpoint_path)
    status.expect_partial()
    print(f"   ✅ Loaded: {checkpoint_path}")
    
    min_ssim = float('inf')
    worst_sample = None
    
    print(f"\n🔬 Scanning {test_size} samples...")
    
    batch_count = 0
    sample_global_idx = 0
    
    for batch_idx, (degraded_images, clean_images) in enumerate(tqdm(test_dataset, desc="Scanning")):
        batch_count += 1
        current_batch_size = degraded_images.shape[0]
        
        # Normalize to [-1, 1] for generator
        degraded_images_tanh = degraded_images * 2.0 - 1.0
        clean_images_tanh = clean_images * 2.0 - 1.0
        
        # Generate
        generated_images = generator(degraded_images_tanh, training=False)
        
        # Denormalize to [0, 1]
        generated_images_normalized = (generated_images + 1.0) / 2.0
        clean_images_normalized = (clean_images_tanh + 1.0) / 2.0
        
        # Calculate SSIM
        ssim_values = tf.image.ssim(clean_images_normalized, generated_images_normalized, max_val=1.0)
        psnr_values = tf.image.psnr(clean_images_normalized, generated_images_normalized, max_val=1.0)
        
        ssim_np = ssim_values.numpy()
        psnr_np = psnr_values.numpy()
        
        for i in range(current_batch_size):
            current_ssim = float(ssim_np[i])
            current_psnr = float(psnr_np[i])
            
            if current_ssim < min_ssim:
                min_ssim = current_ssim
                
                # Capture sample data
                worst_sample = {
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'global_idx': sample_global_idx,
                    'ssim': current_ssim,
                    'psnr': current_psnr,
                    'degraded': degraded_images[i].numpy(),
                    'clean': clean_images_normalized[i].numpy(),
                    'generated': generated_images_normalized[i].numpy()
                }
            
            sample_global_idx += 1
            
    print("\n" + "="*80)
    print("📉 LOWEST SSIM RESULT")
    print("="*80)
    
    if worst_sample:
        print(f"Found worst sample at index {worst_sample['global_idx']} (Batch {worst_sample['batch_idx']}, Index {worst_sample['sample_idx']})")
        print(f"SSIM: {worst_sample['ssim']:.6f}")
        print(f"PSNR: {worst_sample['psnr']:.2f} dB")
        
        # Save images
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Prepare images (transpose for correct orientation: 1024x128 -> 128x1024)
        degraded_img = (worst_sample['degraded'] * 255).astype(np.uint8).squeeze().T
        clean_img = (worst_sample['clean'] * 255).astype(np.uint8).squeeze().T
        generated_img = (worst_sample['generated'] * 255).astype(np.uint8).squeeze().T
        
        # Save individual
        cv2.imwrite(os.path.join(args.output_dir, 'lowest_ssim_degraded.png'), degraded_img)
        cv2.imwrite(os.path.join(args.output_dir, 'lowest_ssim_clean.png'), clean_img)
        cv2.imwrite(os.path.join(args.output_dir, 'lowest_ssim_generated.png'), generated_img)
        
        # Create comparison
        h, w = degraded_img.shape
        label_height = 30
        spacing = 10
        total_height = (h + label_height) * 3 + spacing * 2 + 40
        
        comparison = np.ones((total_height, w), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        y = 0
        cv2.putText(comparison, f'Degraded (Input)', (10, y + 20), font, 0.6, 0, 1)
        y += label_height
        comparison[y:y+h, :] = degraded_img
        y += h + spacing
        
        cv2.putText(comparison, f'Clean (Ground Truth)', (10, y + 20), font, 0.6, 0, 1)
        y += label_height
        comparison[y:y+h, :] = clean_img
        y += h + spacing
        
        cv2.putText(comparison, f'Generated (Output)', (10, y + 20), font, 0.6, 0, 1)
        y += label_height
        comparison[y:y+h, :] = generated_img
        y += h + 10
        
        info = f"Lowest SSIM: {worst_sample['ssim']:.6f} | PSNR: {worst_sample['psnr']:.2f} dB | Index: {worst_sample['global_idx']}"
        cv2.putText(comparison, info, (10, y + 20), font, 0.6, 0, 1)
        
        cv2.imwrite(os.path.join(args.output_dir, 'lowest_ssim_comparison.png'), comparison)
        
        print(f"\n✅ Saved images to {args.output_dir}/")
        print(f"   - lowest_ssim_degraded.png")
        print(f"   - lowest_ssim_clean.png")
        print(f"   - lowest_ssim_generated.png")
        print(f"   - lowest_ssim_comparison.png")
        
        # Save metadata
        meta_file = os.path.join(args.output_dir, 'lowest_ssim_info.json')
        with open(meta_file, 'w') as f:
            # Remove numpy arrays for json serialization
            meta = worst_sample.copy()
            del meta['degraded']
            del meta['clean']
            del meta['generated']
            json.dump(meta, f, indent=2)
            
        print(f"   - lowest_ssim_info.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/lowest_ssim')
    parser.add_argument('--gpu_id', type=str, default='1')
    
    args = parser.parse_args()
    find_lowest_ssim(args)
