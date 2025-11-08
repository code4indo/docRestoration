#!/usr/bin/env python3
"""
Create Mixed Dataset for Progressive Finetuning
Combines base synthetic dataset with ANRI dataset in specified ratio
"""
import tensorflow as tf
import argparse
from pathlib import Path
import numpy as np

def parse_tfrecord_example(serialized_example):
    """Parse a single TFRecord example"""
    feature_description = {
        'degraded_image': tf.io.FixedLenFeature([], tf.string),
        'clean_image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(serialized_example, feature_description)

def create_mixed_dataset(base_tfrecord, anri_tfrecord, output_path, base_ratio=0.7, seed=42):
    """
    Create mixed dataset from base and ANRI TFRecords
    
    Args:
        base_tfrecord: Path to base synthetic dataset
        anri_tfrecord: Path to ANRI dataset
        output_path: Output path for mixed dataset
        base_ratio: Ratio of base samples (0.7 = 70% base, 30% ANRI)
        seed: Random seed for reproducibility
    """
    print("="*80)
    print("CREATING MIXED DATASET FOR PROGRESSIVE FINETUNING")
    print("="*80)
    
    # Count samples in each dataset
    print("\n[1/5] Counting samples...")
    base_count = sum(1 for _ in tf.data.TFRecordDataset(base_tfrecord))
    anri_count = sum(1 for _ in tf.data.TFRecordDataset(anri_tfrecord))
    
    print(f"   Base dataset: {base_count} samples")
    print(f"   ANRI dataset: {anri_count} samples")
    
    # Calculate target counts based on ratio
    # We want to match the ANRI count and scale base accordingly
    # e.g., if ANRI=75 and ratio=0.7, we want 70% base, 30% ANRI
    # So: base_samples / (base_samples + anri_samples) = 0.7
    # base_samples = 0.7 * (base_samples + anri_samples)
    # base_samples = 0.7 * base_samples + 0.7 * anri_samples
    # 0.3 * base_samples = 0.7 * anri_samples
    # base_samples = (0.7 / 0.3) * anri_samples
    
    anri_ratio = 1.0 - base_ratio
    target_base_count = int((base_ratio / anri_ratio) * anri_count)
    target_anri_count = anri_count  # Use all ANRI samples
    
    # Cap base samples if we don't have enough
    if target_base_count > base_count:
        print(f"\n⚠️  Warning: Need {target_base_count} base samples but only have {base_count}")
        print(f"   Using all {base_count} base samples and adjusting ratio")
        target_base_count = base_count
        # Recalculate actual ratio
        actual_base_ratio = target_base_count / (target_base_count + target_anri_count)
        actual_anri_ratio = 1.0 - actual_base_ratio
        print(f"   Actual ratio: {actual_base_ratio*100:.1f}% base, {actual_anri_ratio*100:.1f}% ANRI")
    else:
        actual_base_ratio = base_ratio
        actual_anri_ratio = anri_ratio
    
    total_samples = target_base_count + target_anri_count
    
    print(f"\n[2/5] Target composition:")
    print(f"   Base: {target_base_count} samples ({actual_base_ratio*100:.1f}%)")
    print(f"   ANRI: {target_anri_count} samples ({actual_anri_ratio*100:.1f}%)")
    print(f"   Total: {total_samples} samples")
    
    # Load datasets
    print(f"\n[3/5] Loading datasets...")
    base_dataset = tf.data.TFRecordDataset(base_tfrecord)
    anri_dataset = tf.data.TFRecordDataset(anri_tfrecord)
    
    # Shuffle and take required samples
    print(f"   Shuffling base dataset (seed={seed})...")
    base_dataset = base_dataset.shuffle(buffer_size=base_count, seed=seed, reshuffle_each_iteration=False)
    base_dataset = base_dataset.take(target_base_count)
    
    print(f"   Using all ANRI samples...")
    # No need to shuffle ANRI since we're using all samples
    
    # Combine datasets
    print(f"\n[4/5] Combining datasets...")
    mixed_dataset = base_dataset.concatenate(anri_dataset)
    
    # Shuffle combined dataset for better mixing
    print(f"   Shuffling combined dataset...")
    mixed_dataset = mixed_dataset.shuffle(buffer_size=total_samples, seed=seed+1, reshuffle_each_iteration=False)
    
    # Write to output
    print(f"\n[5/5] Writing to {output_path}...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for i, record in enumerate(mixed_dataset):
            writer.write(record.numpy())
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{total_samples} samples written")
    
    print(f"\n✅ Mixed dataset created successfully!")
    print(f"   Output: {output_path}")
    print(f"   Total samples: {total_samples}")
    print(f"   Base: {target_base_count} ({actual_base_ratio*100:.1f}%)")
    print(f"   ANRI: {target_anri_count} ({actual_anri_ratio*100:.1f}%)")
    
    # Verify output
    print(f"\n[Verification] Counting output samples...")
    output_count = sum(1 for _ in tf.data.TFRecordDataset(str(output_path)))
    print(f"   Output file contains: {output_count} samples")
    
    if output_count == total_samples:
        print(f"   ✅ Verification passed!")
    else:
        print(f"   ⚠️  Warning: Expected {total_samples} but got {output_count}")
    
    print("="*80)
    
    return {
        'total_samples': total_samples,
        'base_samples': target_base_count,
        'anri_samples': target_anri_count,
        'base_ratio': actual_base_ratio,
        'anri_ratio': actual_anri_ratio,
        'output_path': str(output_path)
    }

def main():
    parser = argparse.ArgumentParser(description='Create mixed dataset for progressive finetuning')
    parser.add_argument('--base_tfrecord', type=str, required=True,
                        help='Path to base synthetic dataset TFRecord')
    parser.add_argument('--anri_tfrecord', type=str, required=True,
                        help='Path to ANRI dataset TFRecord')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for mixed dataset')
    parser.add_argument('--base_ratio', type=float, default=0.7,
                        help='Ratio of base samples (default: 0.7 for 70%% base, 30%% ANRI)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    result = create_mixed_dataset(
        args.base_tfrecord,
        args.anri_tfrecord,
        args.output,
        args.base_ratio,
        args.seed
    )
    
    print(f"\n📊 Summary:")
    print(f"   Total: {result['total_samples']} samples")
    print(f"   Base: {result['base_samples']} ({result['base_ratio']*100:.1f}%)")
    print(f"   ANRI: {result['anri_samples']} ({result['anri_ratio']*100:.1f}%)")
    print(f"   Output: {result['output_path']}")

if __name__ == '__main__':
    main()
