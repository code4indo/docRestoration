"""
Analyze PSNR vs SSIM Correlation

This script evaluates the relationship between PSNR and SSIM on the test set
to verify if low SSIM always implies low PSNR.

Usage:
    poetry run python dual_modal_gan/scripts/analyze_psnr_ssim_correlation.py \
        --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
        --checkpoint_name ckpt-88 \
        --config configs/production_v3_academic_split_70_15_15.json \
        --output_dir results/correlation_analysis \
        --gpu_id 1
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
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
    """Parse TFRecord example."""
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize degraded image
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    return degraded_image, clean_image

def create_test_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """Create ONLY the test dataset."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return test_dataset, total_size - train_size - val_size

def analyze_correlation(args):
    print("="*80)
    print("📊 ANALYZING PSNR-SSIM CORRELATION")
    print("="*80)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    all_psnr = []
    all_ssim = []
    
    print(f"\n🔬 Collecting metrics from {test_size} samples...")
    
    for degraded_images, clean_images in tqdm(test_dataset, desc="Processing"):
        # Normalize
        degraded_images_tanh = degraded_images * 2.0 - 1.0
        clean_images_tanh = clean_images * 2.0 - 1.0
        
        # Generate
        generated_images = generator(degraded_images_tanh, training=False)
        
        # Denormalize
        generated_images_normalized = (generated_images + 1.0) / 2.0
        clean_images_normalized = (clean_images_tanh + 1.0) / 2.0
        
        # Calculate metrics
        psnr_vals = tf.image.psnr(clean_images_normalized, generated_images_normalized, max_val=1.0)
        ssim_vals = tf.image.ssim(clean_images_normalized, generated_images_normalized, max_val=1.0)
        
        all_psnr.extend(psnr_vals.numpy())
        all_ssim.extend(ssim_vals.numpy())
        
    # Convert to numpy arrays
    psnr_arr = np.array(all_psnr)
    ssim_arr = np.array(all_ssim)
    
    # Calculate correlation
    p_corr, _ = pearsonr(psnr_arr, ssim_arr)
    s_corr, _ = spearmanr(psnr_arr, ssim_arr)
    
    print("\n" + "="*80)
    print("📈 CORRELATION RESULTS")
    print("="*80)
    print(f"Pearson Correlation (Linear):   {p_corr:.4f}")
    print(f"Spearman Correlation (Rank):    {s_corr:.4f}")
    
    # Check for outliers (Low SSIM but High PSNR)
    # Define thresholds (e.g., bottom 10% SSIM, top 50% PSNR)
    ssim_thresh = np.percentile(ssim_arr, 10)
    psnr_median = np.median(psnr_arr)
    
    outliers = np.where((ssim_arr < ssim_thresh) & (psnr_arr > psnr_median))[0]
    
    print(f"\n🔍 Outlier Analysis:")
    print(f"   Thresholds: SSIM < {ssim_thresh:.4f} (Bottom 10%) AND PSNR > {psnr_median:.2f} (Top 50%)")
    print(f"   Number of outliers found: {len(outliers)}")
    
    if len(outliers) > 0:
        print(f"   ⚠️  Found {len(outliers)} samples where SSIM is low but PSNR is relatively high.")
        print("   This proves that low SSIM does NOT guarantee low PSNR in all cases.")
    else:
        print("   ✅ No significant outliers found. The metrics are tightly coupled in the lower range.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(ssim_arr, psnr_arr, alpha=0.5, s=10)
    plt.title(f'PSNR vs SSIM Correlation (n={len(psnr_arr)})\nPearson: {p_corr:.3f}, Spearman: {s_corr:.3f}')
    plt.xlabel('SSIM')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    
    # Highlight the minimum point
    min_idx = np.argmin(ssim_arr)
    plt.scatter(ssim_arr[min_idx], psnr_arr[min_idx], color='red', s=50, label='Lowest SSIM')
    plt.legend()
    
    plot_path = os.path.join(args.output_dir, 'psnr_ssim_correlation.png')
    plt.savefig(plot_path)
    print(f"\n📊 Scatter plot saved to: {plot_path}")
    
    # Save raw data
    data = {'psnr': all_psnr, 'ssim': all_ssim}
    # Convert numpy floats to python floats for json
    data = {k: [float(x) for x in v] for k, v in data.items()}
    with open(os.path.join(args.output_dir, 'metrics_data.json'), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/correlation_analysis')
    parser.add_argument('--gpu_id', type=str, default='1')
    
    args = parser.parse_args()
    analyze_correlation(args)
