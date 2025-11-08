#!/usr/bin/env python3
"""
Convert DIBCO Dataset to TFRecord for GAN Fine-tuning
======================================================
Converts DIBCO binarization datasets (image pairs) to TFRecord format 
compatible with train_enhanced.py for visual-only fine-tuning.

Key Features:
- Handles DIBCO image pairs: degraded (imgs/) + ground truth (gt_imgs/)
- Resizes to 128x1024 with aspect ratio preservation
- Creates dummy labels (visual-only training, CTC loss disabled)
- Compatible with train_enhanced.py TFRecord schema
- Excludes DIBCO 2012 (reserved for test set)

Usage:
    poetry run python scripts/convert_dibco_to_tfrecord.py \
        --dibco_root real_data_preparation/dataset_dibco \
        --output_tfrecord dual_modal_gan/data/dibco_finetuning.tfrecord \
        --exclude_years 2012
        
Author: AI/ML Engineer
Date: 2025-10-26
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import List, Tuple, Dict
import json
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Target dimensions for GAN training
TARGET_HEIGHT = 128
TARGET_WIDTH = 1024
TARGET_CHANNELS = 1

def load_and_preprocess_image(img_path: str, target_h: int = TARGET_HEIGHT, 
                              target_w: int = TARGET_WIDTH) -> np.ndarray:
    """
    Load and preprocess image for GAN training.
    
    Strategy: Resize with padding to maintain aspect ratio (prevents distortion).
    This is crucial for document images where character proportions matter.
    
    Args:
        img_path: Path to image file
        target_h: Target height (128)
        target_w: Target width (1024)
        
    Returns:
        Preprocessed image as float32 numpy array in [0,1] range, shape (H, W, 1)
    """
    # Read image as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # Original dimensions
    orig_h, orig_w = img.shape
    
    # Calculate scaling factor (preserve aspect ratio)
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize image
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded canvas (fill with white background = 255)
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    # Calculate padding to center the image
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    # Place resized image on canvas
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = img_resized
    
    # Normalize to [0, 1] float32
    img_normalized = canvas.astype(np.float32) / 255.0
    
    # Add channel dimension: (H, W) -> (H, W, 1)
    img_normalized = np.expand_dims(img_normalized, axis=-1)
    
    return img_normalized


def create_dummy_label(length: int = 10) -> np.ndarray:
    """
    Create dummy label for visual-only training.
    
    Since DIBCO has no text annotations and we're doing visual-only fine-tuning,
    we create placeholder labels that will be ignored (CTC loss weight = 0).
    
    Args:
        length: Label sequence length (arbitrary, will be padded to 128 anyway)
        
    Returns:
        Dummy label as int64 array
    """
    # Create sequence of zeros (blank tokens)
    # These won't affect training since ctc_loss_weight = 0.0
    return np.zeros(length, dtype=np.int64)


def collect_dibco_image_pairs(dibco_root: str, 
                               exclude_years: List[str] = None) -> List[Tuple[str, str, str]]:
    """
    Collect all DIBCO image pairs from directory structure.
    
    Expected structure:
        dibco_root/
            2009/imgs/*.png + 2009/gt_imgs/*.png
            2010/imgs/*.png + 2010/gt_imgs/*.png
            ...
    
    Args:
        dibco_root: Root directory containing DIBCO datasets
        exclude_years: List of years to exclude (e.g., ['2012'])
        
    Returns:
        List of (degraded_path, clean_path, year) tuples
    """
    if exclude_years is None:
        exclude_years = []
    
    dibco_root = Path(dibco_root)
    image_pairs = []
    
    # Iterate through year folders
    for year_dir in sorted(dibco_root.iterdir()):
        if not year_dir.is_dir():
            continue
            
        year_name = year_dir.name
        
        # Skip excluded years
        if year_name in exclude_years:
            print(f"  ⚠️  Skipping {year_name} (excluded for test set)")
            continue
        
        # Check for imgs and gt_imgs directories
        imgs_dir = year_dir / 'imgs'
        gt_imgs_dir = year_dir / 'gt_imgs'
        
        if not imgs_dir.exists() or not gt_imgs_dir.exists():
            print(f"  ⚠️  Skipping {year_name} (missing imgs/ or gt_imgs/)")
            continue
        
        # Collect image pairs
        degraded_images = sorted(imgs_dir.glob('*.png'))
        
        year_pairs = []
        for degraded_path in degraded_images:
            # Find corresponding ground truth image (same filename)
            clean_path = gt_imgs_dir / degraded_path.name
            
            if clean_path.exists():
                year_pairs.append((str(degraded_path), str(clean_path), year_name))
            else:
                print(f"  ⚠️  Missing GT for {degraded_path.name} in {year_name}")
        
        print(f"  ✓ {year_name}: {len(year_pairs)} image pairs")
        image_pairs.extend(year_pairs)
    
    return image_pairs


def write_tfrecord(image_pairs: List[Tuple[str, str, str]], 
                   output_path: str,
                   max_samples: int = None):
    """
    Write DIBCO image pairs to TFRecord format.
    
    Schema matches train_enhanced.py expectations:
    - degraded_image_raw: serialized float32 tensor
    - degraded_image_shape: [H, W, C] = [128, 1024, 1]
    - degraded_image_dtype: 'float32'
    - clean_image_raw: serialized float32 tensor
    - clean_image_shape: [H, W, C] = [128, 1024, 1]
    - clean_image_dtype: 'float32'
    - label_raw: serialized int64 tensor (dummy)
    - label_shape: [length]
    - label_dtype: 'int64'
    
    Args:
        image_pairs: List of (degraded, clean, year) tuples
        output_path: Output TFRecord file path
        max_samples: Maximum samples to write (for debugging)
    """
    
    def _bytes_feature(value):
        """Convert bytes to Feature."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(value):
        """Convert int64 list to Feature."""
        if not isinstance(value, (list, tuple)):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    # Prepare output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Limit samples if specified
    if max_samples:
        image_pairs = image_pairs[:max_samples]
    
    print(f"\n📝 Writing TFRecord: {output_path}")
    print(f"   Total samples: {len(image_pairs)}")
    
    stats = {
        'total_samples': len(image_pairs),
        'successful': 0,
        'failed': 0,
        'years': {}
    }
    
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for degraded_path, clean_path, year in tqdm(image_pairs, desc="Converting"):
            try:
                # Load and preprocess images
                degraded_img = load_and_preprocess_image(degraded_path)  # (H, W, 1)
                clean_img = load_and_preprocess_image(clean_path)  # (H, W, 1)
                
                # Create dummy label
                dummy_label = create_dummy_label(length=10)
                
                # Validate shapes
                assert degraded_img.shape == (TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS), \
                    f"Degraded shape mismatch: {degraded_img.shape}"
                assert clean_img.shape == (TARGET_HEIGHT, TARGET_WIDTH, TARGET_CHANNELS), \
                    f"Clean shape mismatch: {clean_img.shape}"
                
                # Create TFRecord example (match train_enhanced.py schema EXACTLY)
                feature = {
                    'degraded_image_raw': _bytes_feature(degraded_img.astype(np.float32).tobytes()),
                    'degraded_image_shape': _int64_feature(list(degraded_img.shape)),
                    'degraded_image_dtype': _bytes_feature(b'float32'),
                    'clean_image_raw': _bytes_feature(clean_img.astype(np.float32).tobytes()),
                    'clean_image_shape': _int64_feature(list(clean_img.shape)),
                    'clean_image_dtype': _bytes_feature(b'float32'),
                    'label_raw': _bytes_feature(dummy_label.astype(np.int64).tobytes()),
                    'label_shape': _int64_feature([len(dummy_label)]),
                    'label_dtype': _bytes_feature(b'int64'),
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                
                # Update statistics
                stats['successful'] += 1
                stats['years'][year] = stats['years'].get(year, 0) + 1
                
            except Exception as e:
                print(f"\n  ❌ Error processing {Path(degraded_path).name}: {e}")
                stats['failed'] += 1
                continue
    
    # Print summary
    print(f"\n✅ TFRecord created successfully!")
    print(f"   Output: {output_path}")
    print(f"   Successful: {stats['successful']} samples")
    print(f"   Failed: {stats['failed']} samples")
    print(f"\n   Distribution by year:")
    for year, count in sorted(stats['years'].items()):
        print(f"     {year}: {count} samples")
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'source': 'DIBCO binarization datasets',
        'purpose': 'Fine-tuning GAN for visual quality (CTC-free)',
        'target_dimensions': {'height': TARGET_HEIGHT, 'width': TARGET_WIDTH, 'channels': TARGET_CHANNELS},
        'preprocessing': 'Resize with padding (aspect ratio preserved)',
        'label_type': 'dummy (visual-only training)',
        'statistics': stats
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Metadata: {metadata_path}")
    
    return stats


def verify_tfrecord(tfrecord_path: str, num_samples: int = 3):
    """
    Verify TFRecord can be loaded by train_enhanced.py parser.
    
    Args:
        tfrecord_path: Path to TFRecord file
        num_samples: Number of samples to verify
    """
    print(f"\n🔍 Verifying TFRecord: {tfrecord_path}")
    
    def _parse_tfrecord_fn(example_proto):
        """Parse function from train_enhanced.py"""
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
        degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (H,W,C) -> (W,H,C)
        degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])
        
        # Deserialize clean image
        clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
        clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
        clean_image = tf.reshape(clean_image, clean_image_shape)
        clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (H,W,C) -> (W,H,C)
        clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])
        
        # Deserialize label
        label_shape = tf.cast(example['label_shape'], tf.int32)
        label = tf.io.decode_raw(example['label_raw'], tf.int64)
        label = tf.reshape(label, label_shape)
        label = tf.cast(label, tf.int32)
        
        # Pad label to 128
        padding = [[0, 128 - tf.shape(label)[0]]]
        label = tf.pad(label, padding, "CONSTANT", constant_values=0)
        label.set_shape([128])
        
        return degraded_image, clean_image, label
    
    # Load dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn)
    
    print(f"   Testing with {num_samples} samples...")
    
    for i, (degraded, clean, label) in enumerate(dataset.take(num_samples)):
        print(f"\n   Sample {i+1}:")
        print(f"     Degraded: shape={degraded.shape}, dtype={degraded.dtype}, "
              f"range=[{tf.reduce_min(degraded):.3f}, {tf.reduce_max(degraded):.3f}]")
        print(f"     Clean:    shape={clean.shape}, dtype={clean.dtype}, "
              f"range=[{tf.reduce_min(clean):.3f}, {tf.reduce_max(clean):.3f}]")
        print(f"     Label:    shape={label.shape}, dtype={label.dtype}, "
              f"unique_values={len(tf.unique(label)[0])}")
        
        # Verify shapes
        assert degraded.shape == (1024, 128, 1), f"Degraded shape error: {degraded.shape}"
        assert clean.shape == (1024, 128, 1), f"Clean shape error: {clean.shape}"
        assert label.shape == (128,), f"Label shape error: {label.shape}"
        
    print(f"\n   ✅ All samples parsed successfully!")
    print(f"   ✅ Compatible with train_enhanced.py")


def main():
    parser = argparse.ArgumentParser(
        description='Convert DIBCO datasets to TFRecord for GAN fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all DIBCO years except 2012 (test set)
  python scripts/convert_dibco_to_tfrecord.py \\
      --dibco_root real_data_preparation/dataset_dibco \\
      --output_tfrecord dual_modal_gan/data/dibco_finetuning.tfrecord \\
      --exclude_years 2012
  
  # Quick test with limited samples
  python scripts/convert_dibco_to_tfrecord.py \\
      --dibco_root real_data_preparation/dataset_dibco \\
      --output_tfrecord dual_modal_gan/data/dibco_test.tfrecord \\
      --max_samples 50 \\
      --verify
        """
    )
    
    parser.add_argument('--dibco_root', type=str, 
                        default='real_data_preparation/dataset_dibco',
                        help='Root directory containing DIBCO datasets (default: real_data_preparation/dataset_dibco)')
    parser.add_argument('--output_tfrecord', type=str,
                        default='dual_modal_gan/data/dibco_finetuning.tfrecord',
                        help='Output TFRecord file path (default: dual_modal_gan/data/dibco_finetuning.tfrecord)')
    parser.add_argument('--exclude_years', type=str, nargs='+',
                        default=['2012'],
                        help='DIBCO years to exclude (default: 2012 for test set)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to convert (for debugging)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify TFRecord after creation')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔄 DIBCO to TFRecord Converter")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  DIBCO root:     {args.dibco_root}")
    print(f"  Output:         {args.output_tfrecord}")
    print(f"  Exclude years:  {', '.join(args.exclude_years)}")
    print(f"  Max samples:    {args.max_samples or 'unlimited'}")
    print(f"  Target size:    {TARGET_WIDTH}x{TARGET_HEIGHT}x{TARGET_CHANNELS}")
    
    # Check input directory
    if not Path(args.dibco_root).exists():
        print(f"\n❌ Error: DIBCO root not found: {args.dibco_root}")
        return 1
    
    # Collect image pairs
    print(f"\n📂 Collecting DIBCO image pairs...")
    image_pairs = collect_dibco_image_pairs(args.dibco_root, args.exclude_years)
    
    if not image_pairs:
        print(f"\n❌ Error: No image pairs found in {args.dibco_root}")
        return 1
    
    print(f"\n✅ Found {len(image_pairs)} image pairs total")
    
    # Write TFRecord
    stats = write_tfrecord(image_pairs, args.output_tfrecord, args.max_samples)
    
    # Verify if requested
    if args.verify:
        verify_tfrecord(args.output_tfrecord, num_samples=3)
    
    print(f"\n{'='*80}")
    print(f"✅ Conversion complete!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"  1. Review metadata: {Path(args.output_tfrecord).with_suffix('.json')}")
    print(f"  2. Update config: configs/dibco_finetuning.json")
    print(f"  3. Launch training: ./scripts/universal_train_from_json.sh configs/dibco_finetuning.json")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
