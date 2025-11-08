#!/usr/bin/env python3
"""
Audit TFRecord Preprocessing
=============================
Check exact preprocessing pipeline untuk find mismatch.
"""

import tensorflow as tf
import numpy as np
import cv2
import os

def _parse_tfrecord_fn(example_proto):
    """EXACT same parser dari train_enhanced.py"""
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'degraded_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),
        'label_dtype': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize degraded image
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    
    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    
    return degraded_image, clean_image

def main():
    tfrecord_path = "dual_modal_gan/data/dibco_tiled_no_palm.tfrecord"
    
    if not os.path.exists(tfrecord_path):
        print(f"❌ TFRecord not found: {tfrecord_path}")
        return
    
    print("=" * 80)
    print("AUDIT TFRECORD PREPROCESSING")
    print("=" * 80)
    print(f"TFRecord: {tfrecord_path}\n")
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn)
    
    # Get first 5 samples
    for i, (degraded, clean) in enumerate(dataset.take(5)):
        degraded_np = degraded.numpy()
        clean_np = clean.numpy()
        
        print(f"Sample {i+1}:")
        print(f"  Degraded image:")
        print(f"    - Shape: {degraded_np.shape}")
        print(f"    - Dtype: {degraded_np.dtype}")
        print(f"    - Range: [{degraded_np.min():.3f}, {degraded_np.max():.3f}]")
        print(f"    - Mean: {degraded_np.mean():.3f}, Std: {degraded_np.std():.3f}")
        
        print(f"  Clean image:")
        print(f"    - Shape: {clean_np.shape}")
        print(f"    - Dtype: {clean_np.dtype}")
        print(f"    - Range: [{clean_np.min():.3f}, {clean_np.max():.3f}]")
        print(f"    - Mean: {clean_np.mean():.3f}, Std: {clean_np.std():.3f}")
        
        # Simulate training preprocessing (to [-1, 1])
        degraded_tanh = degraded_np * 2.0 - 1.0
        clean_tanh = clean_np * 2.0 - 1.0
        
        print(f"  After tanh normalization (training):")
        print(f"    - Degraded: [{degraded_tanh.min():.3f}, {degraded_tanh.max():.3f}]")
        print(f"    - Clean: [{clean_tanh.min():.3f}, {clean_tanh.max():.3f}]")
        print()
    
    print("=" * 80)
    print("COMPARISON WITH DIBCO 2012 RAW IMAGES")
    print("=" * 80)
    
    dibco_dir = "dibco_datasets/2012/imgs"
    if os.path.exists(dibco_dir):
        for img_name in sorted(os.listdir(dibco_dir))[:3]:
            img_path = os.path.join(dibco_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            print(f"\n{img_name}:")
            print(f"  Raw from disk:")
            print(f"    - Dtype: {img.dtype}")
            print(f"    - Range: [{img.min()}, {img.max()}]")
            print(f"    - Mean: {img.mean():.1f}, Std: {img.std():.1f}")
            
            # Normalize to [0, 1] (basic)
            img_norm = img.astype(np.float32) / 255.0
            print(f"  After /255.0 normalization:")
            print(f"    - Range: [{img_norm.min():.3f}, {img_norm.max():.3f}]")
            print(f"    - Mean: {img_norm.mean():.3f}, Std: {img_norm.std():.3f}")
            
            # Contrast stretching (ensemble script)
            img_stretched = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_stretched_norm = img_stretched.astype(np.float32) / 255.0
            print(f"  After contrast stretching + /255.0:")
            print(f"    - Range: [{img_stretched_norm.min():.3f}, {img_stretched_norm.max():.3f}]")
            print(f"    - Mean: {img_stretched_norm.mean():.3f}, Std: {img_stretched_norm.std():.3f}")
            
            # Check if contrast stretching changes data significantly
            diff_mean = abs(img_norm.mean() - img_stretched_norm.mean())
            diff_std = abs(img_norm.std() - img_stretched_norm.std())
            
            if diff_mean > 0.05 or diff_std > 0.05:
                print(f"  ⚠️  CONTRAST STRETCHING CHANGES DATA SIGNIFICANTLY!")
                print(f"      Mean diff: {diff_mean:.3f}, Std diff: {diff_std:.3f}")
    
    print("\n" + "=" * 80)
    print("FINDINGS:")
    print("=" * 80)
    print("1. Check if TFRecord data is already in [0, 1] or [0, 255]")
    print("2. Check if DIBCO 2012 images have different intensity distribution")
    print("3. Identify if contrast stretching in inference is helpful or harmful")
    print("=" * 80)

if __name__ == "__main__":
    main()
