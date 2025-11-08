#!/usr/bin/env python3
"""
Create TFRecord from filtered degraded-pseudoGT pairs

Format compatible with dual_modal_gan training pipeline
"""

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(degraded_path: Path, pseudo_gt_path: Path):
    """Create TFRecord example from image pair matching train_enhanced.py format"""
    
    # Load images
    degraded = cv2.imread(str(degraded_path), cv2.IMREAD_GRAYSCALE)
    pseudo_gt = cv2.imread(str(pseudo_gt_path), cv2.IMREAD_GRAYSCALE)
    
    if degraded is None or pseudo_gt is None:
        return None
    
    # Keep as (H, W) -> (H, W, 1), training script will transpose
    # Images are (128, 1024) grayscale
    degraded = degraded[..., np.newaxis]  # (128, 1024, 1)
    pseudo_gt = pseudo_gt[..., np.newaxis]  # (128, 1024, 1)
    
    # Normalize to [-1, 1]
    degraded = (degraded.astype(np.float32) / 127.5) - 1.0
    pseudo_gt = (pseudo_gt.astype(np.float32) / 127.5) - 1.0
    
    # Serialize to raw bytes
    degraded_bytes = degraded.tobytes()
    pseudo_gt_bytes = pseudo_gt.tobytes()
    
    # Get shapes
    degraded_shape = degraded.shape  # (128, 1024, 1)
    pseudo_gt_shape = pseudo_gt.shape  # (128, 1024, 1)
    
    # Empty label for pseudo-labeling (no HTR)
    label = np.array([], dtype=np.int64)
    label_bytes = label.tobytes()
    label_shape = [len(label)]
    
    # Create features matching train_enhanced.py format
    feature = {
        'degraded_image_raw': _bytes_feature(degraded_bytes),
        'degraded_image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(degraded_shape))),
        'degraded_image_dtype': _bytes_feature(b'float32'),
        'clean_image_raw': _bytes_feature(pseudo_gt_bytes),
        'clean_image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(pseudo_gt_shape))),
        'clean_image_dtype': _bytes_feature(b'float32'),
        'label_raw': _bytes_feature(label_bytes),
        'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=label_shape)),
        'label_dtype': _bytes_feature(b'int64'),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def main():
    parser = argparse.ArgumentParser(description='Create TFRecord from filtered pairs')
    parser.add_argument('--degraded_dir', type=str, required=True,
                        help='Directory with filtered degraded images')
    parser.add_argument('--pseudo_gt_dir', type=str, required=True,
                        help='Directory with filtered pseudo-GT images')
    parser.add_argument('--output_tfrecord', type=str, required=True,
                        help='Output TFRecord file path')
    parser.add_argument('--metadata_json', type=str,
                        help='Optional: Filtered metadata JSON for statistics')
    
    args = parser.parse_args()
    
    degraded_dir = Path(args.degraded_dir)
    pseudo_gt_dir = Path(args.pseudo_gt_dir)
    output_tfrecord = Path(args.output_tfrecord)
    
    output_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all degraded images
    degraded_files = sorted(list(degraded_dir.glob('*.jpg')) + list(degraded_dir.glob('*.png')))
    
    print("="*80)
    print("TFRECORD CREATION FROM PSEUDO-GT PAIRS")
    print("="*80)
    print()
    print(f"Degraded dir: {degraded_dir}")
    print(f"Pseudo-GT dir: {pseudo_gt_dir}")
    print(f"Found {len(degraded_files)} degraded images")
    print(f"Output TFRecord: {output_tfrecord}")
    print()
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(str(output_tfrecord)) as writer:
        written = 0
        skipped = 0
        
        for degraded_path in tqdm(degraded_files, desc="Creating TFRecord"):
            # Find corresponding pseudo-GT
            pseudo_gt_name = degraded_path.stem + '_restored.png'
            pseudo_gt_path = pseudo_gt_dir / pseudo_gt_name
            
            if not pseudo_gt_path.exists():
                # Try without _restored suffix
                pseudo_gt_name = degraded_path.stem + '.png'
                pseudo_gt_path = pseudo_gt_dir / pseudo_gt_name
            
            if not pseudo_gt_path.exists():
                skipped += 1
                continue
            
            # Create TFRecord example
            example = create_example(degraded_path, pseudo_gt_path)
            
            if example is not None:
                writer.write(example.SerializeToString())
                written += 1
            else:
                skipped += 1
    
    print()
    print("="*80)
    print("TFRECORD CREATION COMPLETE")
    print("="*80)
    print()
    print(f"Written: {written} examples")
    print(f"Skipped: {skipped} examples")
    print(f"Output: {output_tfrecord}")
    print(f"Size: {output_tfrecord.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    # Load metadata if provided
    if args.metadata_json:
        metadata_path = Path(args.metadata_json)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print("Filter statistics (from metadata):")
            stats = metadata.get('statistics', {})
            print(f"  Input pairs: {stats.get('input_pairs', 'N/A')}")
            print(f"  Kept pairs: {stats.get('kept_pairs', 'N/A')}")
            print(f"  Keep rate: {(1 - stats.get('rejection_rate', 0)) * 100:.1f}%")
            print()
    
    print("✅ TFRecord ready for fine-tuning!")
    print()
    print("Next steps:")
    print("1. Create fine-tuning config JSON")
    print("2. Launch training with universal_train_from_json.sh")
    print()


if __name__ == '__main__':
    main()
