#!/usr/bin/env python3
"""
CORRECT MODEL EVALUATION - NO DATA LEAKAGE
Evaluasi HANYA pada validation set (10% terakhir dari TFRecord)
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'dual_modal_gan' / 'src'))

from models.generator_enhanced import unet_enhanced

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def parse_tfrecord_fn(example_proto):
    """Parse TFRecord."""
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    degraded_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    
    clean_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    
    return degraded_image, clean_image


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1, img2, max_val=1.0):
    """Calculate SSIM."""
    img1_hwc = np.transpose(img1, (1, 0, 2))
    img2_hwc = np.transpose(img2, (1, 0, 2))
    img1_batch = np.expand_dims(img1_hwc, axis=0)
    img2_batch = np.expand_dims(img2_hwc, axis=0)
    ssim = tf.image.ssim(img1_batch, img2_batch, max_val=max_val)
    return float(ssim.numpy()[0])


def load_generator(checkpoint_path):
    """Load generator - handles both directory and checkpoint file paths."""
    checkpoint_path = Path(checkpoint_path)
    
    # If directory, find the latest checkpoint
    if checkpoint_path.is_dir():
        print(f"\n📦 Loading from directory: {checkpoint_path}")
        # Look for checkpoint file
        ckpt_file = checkpoint_path / 'checkpoint'
        if ckpt_file.exists():
            # Read checkpoint file to get the actual checkpoint name
            with open(ckpt_file, 'r') as f:
                first_line = f.readline()
                # Extract checkpoint name: model_checkpoint_path: "ckpt-91"
                ckpt_name = first_line.split('"')[1]
                checkpoint_path = checkpoint_path / ckpt_name
                print(f"   Found checkpoint: {ckpt_name}")
        else:
            raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")
    
    print(f"   Loading: {checkpoint_path}")
    
    generator = unet_enhanced(input_size=(1024, 128, 1))
    checkpoint = tf.train.Checkpoint(generator=generator)
    status = checkpoint.restore(str(checkpoint_path))
    
    try:
        status.assert_consumed()
        print("   ✅ All variables loaded")
    except AssertionError:
        print("   ⚠️  Partial load (continuing...)")
    
    # Detect activation type
    dummy = tf.random.normal((1, 1024, 128, 1))
    output = generator(dummy, training=False)
    
    print(f"   Output range: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]")
    
    if output.numpy().min() < -0.5:
        print("   🔍 Detected: TANH activation [-1, 1]")
        return generator, 'tanh'
    else:
        print("   🔍 Detected: SIGMOID activation [0, 1]")
        return generator, 'sigmoid'


def evaluate_on_validation_set(generator, output_type, tfrecord_path, val_split, output_dir):
    """Evaluate ONLY on validation set to avoid data leakage."""
    print(f"\n{'='*80}")
    print(f"CORRECT EVALUATION - VALIDATION SET ONLY")
    print(f"{'='*80}")
    
    # Load and split dataset EXACTLY as in training
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total: {total_size} samples")
    print(f"   Training (first 90%): {train_size} samples (indices 0-{train_size-1})")
    print(f"   Validation (last 10%): {val_size} samples (indices {train_size}-{total_size-1})")
    print(f"\n✅ Evaluating on VALIDATION SET ONLY ({val_size} samples)")
    print(f"   This ensures NO DATA LEAKAGE\n")
    
    # Create validation dataset (SKIP training samples)
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn)
    val_dataset = dataset.skip(train_size)  # Skip training set
    val_dataset = val_dataset.batch(1)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    psnr_values = []
    ssim_values = []
    
    print("🔬 Evaluating...")
    
    for idx, (degraded, clean) in enumerate(tqdm(val_dataset, total=val_size)):
        generated = generator(degraded, training=False)
        
        if output_type == 'tanh':
            generated_norm = (generated + 1.0) / 2.0
        else:
            generated_norm = generated
        
        clean_np = clean.numpy()[0]
        generated_np = generated_norm.numpy()[0]
        
        psnr = calculate_psnr(generated_np, clean_np, max_val=1.0)
        ssim = calculate_ssim(generated_np, clean_np, max_val=1.0)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        
        # Save first 10 samples
        if idx < 10:
            degraded_np = degraded.numpy()[0]
            
            deg_u8 = (degraded_np * 255).astype(np.uint8)
            clean_u8 = (clean_np * 255).astype(np.uint8)
            gen_u8 = (generated_np * 255).astype(np.uint8)
            
            deg_u8 = np.squeeze(np.transpose(deg_u8, (1, 0, 2)), -1)
            clean_u8 = np.squeeze(np.transpose(clean_u8, (1, 0, 2)), -1)
            gen_u8 = np.squeeze(np.transpose(gen_u8, (1, 0, 2)), -1)
            
            comparison = np.vstack([deg_u8, clean_u8, gen_u8])
            cv2.imwrite(str(output_dir / f'val_sample_{idx:04d}.png'), comparison)
    
    psnr_values = np.array(psnr_values)
    ssim_values = np.array(ssim_values)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': total_size,
            'train_samples': train_size,
            'val_samples': val_size,
            'evaluated_on': 'VALIDATION_SET_ONLY',
            'data_leakage': 'NONE - correctly split'
        },
        'metrics': {
            'psnr': {
                'mean': float(np.mean(psnr_values)),
                'std': float(np.std(psnr_values)),
                'median': float(np.median(psnr_values)),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values)),
            },
            'ssim': {
                'mean': float(np.mean(ssim_values)),
                'std': float(np.std(ssim_values)),
                'median': float(np.median(ssim_values)),
                'min': float(np.min(ssim_values)),
                'max': float(np.max(ssim_values)),
            }
        },
        'confidence_95': {
            'psnr': [
                float(np.mean(psnr_values) - 1.96 * np.std(psnr_values) / np.sqrt(val_size)),
                float(np.mean(psnr_values) + 1.96 * np.std(psnr_values) / np.sqrt(val_size))
            ],
            'ssim': [
                float(np.mean(ssim_values) - 1.96 * np.std(ssim_values) / np.sqrt(val_size)),
                float(np.mean(ssim_values) + 1.96 * np.std(ssim_values) / np.sqrt(val_size))
            ]
        }
    }
    
    print(f"\n{'='*80}")
    print(f"VALID EVALUATION RESULTS (NO DATA LEAKAGE)")
    print(f"{'='*80}")
    print(f"\n📊 PSNR: {results['metrics']['psnr']['mean']:.2f} ± {results['metrics']['psnr']['std']:.2f} dB")
    print(f"   95% CI: [{results['confidence_95']['psnr'][0]:.2f}, {results['confidence_95']['psnr'][1]:.2f}]")
    print(f"\n📊 SSIM: {results['metrics']['ssim']['mean']:.4f} ± {results['metrics']['ssim']['std']:.4f}")
    print(f"   95% CI: [{results['confidence_95']['ssim'][0]:.4f}, {results['confidence_95']['ssim'][1]:.4f}]")
    print(f"\n✅ Evaluated on {val_size} validation samples (NO overlap with training)")
    print(f"{'='*80}\n")
    
    with open(output_dir / 'evaluation_valid_no_leakage.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Saved: {output_dir / 'evaluation_valid_no_leakage.json'}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tfrecord', required=True)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--output', default='evaluation_no_leakage')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if args.gpu < len(gpus):
            tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
    
    generator, output_type = load_generator(args.checkpoint)
    evaluate_on_validation_set(generator, output_type, args.tfrecord, args.val_split, args.output)


if __name__ == '__main__':
    main()
