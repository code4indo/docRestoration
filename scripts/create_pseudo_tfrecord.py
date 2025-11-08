#!/usr/bin/env python3
"""
Create TFRecord dataset from filtered pseudo-labeled patches

Strategy:
- Convert patch pairs (degraded, pseudo-GT) to TFRecord format
- Use visual-only mode (empty text labels)
- Compatible with existing training pipeline

Author: Pseudo-Labeling Pipeline
Date: 2025-10-28
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def load_and_normalize_image(image_path: Path, target_size=(1024, 128)) -> np.ndarray:
    """
    Load image and normalize to [-1, 1] range
    
    Args:
        image_path: Path to image file
        target_size: (height, width) - will pad/crop if needed
    
    Returns:
        Normalized image array of shape (height, width, 1)
    """
    with Image.open(image_path) as img:
        if img.mode != 'L':
            img = img.convert('L')
        
        # For patches, we don't resize - we use as-is
        # Model can handle variable sizes during fine-tuning
        arr = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1]
        arr = (arr / 127.5) - 1.0
        
        # Add channel dimension
        arr = arr[:, :, np.newaxis]
        
        return arr


def create_tfrecord_from_patches(
    degraded_dir: Path,
    clean_dir: Path,
    output_path: Path,
    description: str = "dataset"
):
    """
    Create TFRecord from patch directories
    
    Args:
        degraded_dir: Directory with degraded patches
        clean_dir: Directory with pseudo-GT patches
        output_path: Output TFRecord file path
        description: Description for progress bar
    """
    # Get all degraded patches
    degraded_patches = sorted(list(degraded_dir.glob('*.jpg')))
    
    if len(degraded_patches) == 0:
        print(f"⚠️  No patches found in {degraded_dir}")
        return 0
    
    print(f"Creating {description} TFRecord...")
    print(f"  Degraded dir: {degraded_dir}")
    print(f"  Clean dir: {clean_dir}")
    print(f"  Output: {output_path}")
    print(f"  Patches: {len(degraded_patches)}")
    print()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for degraded_path in tqdm(degraded_patches, desc=f"Writing {description}"):
            # Find corresponding clean patch
            clean_path = clean_dir / degraded_path.name
            
            if not clean_path.exists():
                print(f"⚠️  Skipping {degraded_path.name} - no matching clean patch")
                continue
            
            # Load and normalize images
            degraded = load_and_normalize_image(degraded_path)
            clean = load_and_normalize_image(clean_path)
            
            # Ensure same shape
            if degraded.shape != clean.shape:
                print(f"⚠️  Shape mismatch for {degraded_path.name}: {degraded.shape} vs {clean.shape}")
                continue
            
            # For visual-only training, use empty label
            # Label is int32 array of shape (max_label_length,)
            empty_label = np.zeros(128, dtype=np.int32)
            
            # Transpose to (H, W, C) for storage (model expects this format)
            degraded_transposed = degraded  # Already in correct format
            clean_transposed = clean
            
            # Create feature dictionary
            feature = {
                'degraded_image_raw': _bytes_feature(degraded_transposed.astype(np.float32)),
                'degraded_image_shape': _int64_feature(list(degraded_transposed.shape)),
                'degraded_image_dtype': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b'float32'])
                ),
                'clean_image_raw': _bytes_feature(clean_transposed.astype(np.float32)),
                'clean_image_shape': _int64_feature(list(clean_transposed.shape)),
                'clean_image_dtype': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b'float32'])
                ),
                'label_raw': _bytes_feature(empty_label.astype(np.int64)),
                'label_shape': _int64_feature([len(empty_label)]),
                'label_dtype': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b'int64'])
                ),
            }
            
            # Create example and write
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    
    print(f"✅ {description} TFRecord created: {len(degraded_patches)} samples")
    print()
    
    return len(degraded_patches)


def verify_tfrecord(tfrecord_path: Path, n_samples: int = 3):
    """Verify TFRecord can be read correctly"""
    print(f"Verifying TFRecord: {tfrecord_path}")
    
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
    
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    
    count = 0
    for raw_record in dataset.take(n_samples):
        example = tf.io.parse_single_example(raw_record, feature_description)
        
        degraded_shape = example['degraded_image_shape'].numpy()
        clean_shape = example['clean_image_shape'].numpy()
        label_shape = example['label_shape'].numpy()
        
        count += 1
        print(f"  Sample {count}:")
        print(f"    Degraded shape: {degraded_shape}")
        print(f"    Clean shape: {clean_shape}")
        print(f"    Label shape: {label_shape}")
    
    print(f"✅ TFRecord verified ({count} samples checked)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Create TFRecord dataset from filtered pseudo-labeled patches'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='DokumenRusak/anri_patches_filtered',
        help='Directory containing filtered patches (train/val subdirs)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='DokumenRusak/anri_tfrecords',
        help='Output directory for TFRecord files'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='Verify created TFRecords (default: True)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CREATE TFRECORD FROM PSEUDO-LABELED PATCHES")
    print("="*80)
    print()
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create training TFRecord
    train_degraded_dir = input_dir / 'train' / 'degraded'
    train_clean_dir = input_dir / 'train' / 'clean'
    train_output = output_dir / 'train_anri_pseudo.tfrecord'
    
    n_train = create_tfrecord_from_patches(
        train_degraded_dir,
        train_clean_dir,
        train_output,
        description="Training"
    )
    
    # Create validation TFRecord
    val_degraded_dir = input_dir / 'val' / 'degraded'
    val_clean_dir = input_dir / 'val' / 'clean'
    val_output = output_dir / 'val_anri_pseudo.tfrecord'
    
    n_val = create_tfrecord_from_patches(
        val_degraded_dir,
        val_clean_dir,
        val_output,
        description="Validation"
    )
    
    print("="*80)
    print("TFRECORD CREATION SUMMARY")
    print("="*80)
    print()
    print(f"Training TFRecord: {train_output}")
    print(f"  Samples: {n_train}")
    print()
    print(f"Validation TFRecord: {val_output}")
    print(f"  Samples: {n_val}")
    print()
    
    # Verify TFRecords
    if args.verify and n_train > 0:
        verify_tfrecord(train_output)
    
    if args.verify and n_val > 0:
        verify_tfrecord(val_output)
    
    print("="*80)
    print("✅ TFRECORD CREATION COMPLETE")
    print("="*80)
    print()
    print("Next step:")
    print("  Create fine-tuning config:")
    print("  configs/finetune_anri_pseudo_visual_only.json")
    print()
    print("  Then train:")
    print("  nohup ./scripts/universal_train_from_json.sh \\")
    print("    configs/finetune_anri_pseudo_visual_only.json &")
    print()


if __name__ == '__main__':
    main()
