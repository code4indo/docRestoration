#!/usr/bin/env python3
"""
Create Triple Mixed Dataset for Progressive Finetuning with Data Rehearsal
Combines Base + DIBCO + ANRI datasets in specified ratio (50:30:20)
"""
import tensorflow as tf
import argparse
from pathlib import Path

def create_triple_mixed_dataset(
    base_tfrecord, 
    dibco_tfrecord, 
    anri_tfrecord, 
    output_path, 
    base_ratio=0.50,
    dibco_ratio=0.30,
    anri_ratio=0.20,
    seed=42
):
    """
    Create mixed dataset from Base + DIBCO + ANRI TFRecords
    
    Args:
        base_tfrecord: Path to base synthetic dataset
        dibco_tfrecord: Path to DIBCO dataset  
        anri_tfrecord: Path to ANRI dataset
        output_path: Output path for mixed dataset
        base_ratio: Ratio of base samples (default 0.50 = 50%)
        dibco_ratio: Ratio of DIBCO samples (default 0.30 = 30%)
        anri_ratio: Ratio of ANRI samples (default 0.20 = 20%)
        seed: Random seed for reproducibility
    """
    print("="*80)
    print("CREATING TRIPLE MIXED DATASET (BASE + DIBCO + ANRI)")
    print("="*80)
    
    # Validate ratios
    total_ratio = base_ratio + dibco_ratio + anri_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Count samples in each dataset
    print("\n[1/6] Counting samples...")
    base_count = sum(1 for _ in tf.data.TFRecordDataset(base_tfrecord))
    dibco_count = sum(1 for _ in tf.data.TFRecordDataset(dibco_tfrecord))
    anri_count = sum(1 for _ in tf.data.TFRecordDataset(anri_tfrecord))
    
    print(f"   Base dataset: {base_count} samples")
    print(f"   DIBCO dataset: {dibco_count} samples")
    print(f"   ANRI dataset: {anri_count} samples")
    
    # Calculate target counts
    # We'll base the total size on maintaining reasonable proportions
    # Target ~1000-1500 total samples for manageable training
    target_total = 1074  # Similar to v1 dataset size
    
    target_base_count = int(target_total * base_ratio)
    target_dibco_count = int(target_total * dibco_ratio)
    target_anri_count = int(target_total * anri_ratio)
    
    # Adjust if we don't have enough samples
    if target_base_count > base_count:
        print(f"⚠️  Warning: Need {target_base_count} base samples but only have {base_count}")
        target_base_count = base_count
    if target_dibco_count > dibco_count:
        print(f"⚠️  Warning: Need {target_dibco_count} DIBCO samples but only have {dibco_count}")
        target_dibco_count = dibco_count
    if target_anri_count > anri_count:
        print(f"⚠️  Warning: Need {target_anri_count} ANRI samples but only have {anri_count}")
        target_anri_count = anri_count
    
    actual_total = target_base_count + target_dibco_count + target_anri_count
    actual_base_ratio = target_base_count / actual_total
    actual_dibco_ratio = target_dibco_count / actual_total
    actual_anri_ratio = target_anri_count / actual_total
    
    print(f"\n[2/6] Target composition:")
    print(f"   Base:  {target_base_count:4d} samples ({actual_base_ratio*100:.1f}%)")
    print(f"   DIBCO: {target_dibco_count:4d} samples ({actual_dibco_ratio*100:.1f}%)")
    print(f"   ANRI:  {target_anri_count:4d} samples ({actual_anri_ratio*100:.1f}%)")
    print(f"   Total: {actual_total:4d} samples")
    
    # Load datasets
    print(f"\n[3/6] Loading and sampling datasets...")
    
    print(f"   Loading base dataset...")
    base_dataset = tf.data.TFRecordDataset(base_tfrecord)
    base_dataset = base_dataset.shuffle(buffer_size=base_count, seed=seed, reshuffle_each_iteration=False)
    base_dataset = base_dataset.take(target_base_count)
    
    print(f"   Loading DIBCO dataset...")
    dibco_dataset = tf.data.TFRecordDataset(dibco_tfrecord)
    dibco_dataset = dibco_dataset.shuffle(buffer_size=dibco_count, seed=seed+1, reshuffle_each_iteration=False)
    dibco_dataset = dibco_dataset.take(target_dibco_count)
    
    print(f"   Loading ANRI dataset...")
    anri_dataset = tf.data.TFRecordDataset(anri_tfrecord)
    anri_dataset = anri_dataset.shuffle(buffer_size=anri_count, seed=seed+2, reshuffle_each_iteration=False)
    anri_dataset = anri_dataset.take(target_anri_count)
    
    # Combine datasets
    print(f"\n[4/6] Combining datasets...")
    mixed_dataset = base_dataset.concatenate(dibco_dataset).concatenate(anri_dataset)
    
    # Shuffle combined dataset for better mixing
    print(f"   Shuffling combined dataset (seed={seed+3})...")
    mixed_dataset = mixed_dataset.shuffle(buffer_size=actual_total, seed=seed+3, reshuffle_each_iteration=False)
    
    # Write to output
    print(f"\n[5/6] Writing to {output_path}...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    written_count = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for i, record in enumerate(mixed_dataset):
            writer.write(record.numpy())
            written_count += 1
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{actual_total} samples written")
    
    print(f"\n✅ Triple mixed dataset created successfully!")
    print(f"   Output: {output_path}")
    print(f"   Total samples written: {written_count}")
    
    # Verify output
    print(f"\n[6/6] Verifying output...")
    output_count = sum(1 for _ in tf.data.TFRecordDataset(str(output_path)))
    print(f"   Output file contains: {output_count} samples")
    
    if output_count == actual_total:
        print(f"   ✅ Verification passed!")
    else:
        print(f"   ⚠️  Warning: Expected {actual_total} but got {output_count}")
    
    print("="*80)
    print("📊 FINAL COMPOSITION:")
    print(f"   Base:  {target_base_count:4d} samples ({actual_base_ratio*100:.1f}%)")
    print(f"   DIBCO: {target_dibco_count:4d} samples ({actual_dibco_ratio*100:.1f}%)")
    print(f"   ANRI:  {target_anri_count:4d} samples ({actual_anri_ratio*100:.1f}%)")
    print(f"   Total: {output_count:4d} samples")
    print("="*80)
    
    return {
        'total_samples': output_count,
        'base_samples': target_base_count,
        'dibco_samples': target_dibco_count,
        'anri_samples': target_anri_count,
        'base_ratio': actual_base_ratio,
        'dibco_ratio': actual_dibco_ratio,
        'anri_ratio': actual_anri_ratio,
        'output_path': str(output_path)
    }

def main():
    parser = argparse.ArgumentParser(
        description='Create triple mixed dataset (Base + DIBCO + ANRI) for progressive finetuning with data rehearsal'
    )
    parser.add_argument('--base_tfrecord', type=str, required=True,
                        help='Path to base synthetic dataset TFRecord')
    parser.add_argument('--dibco_tfrecord', type=str, required=True,
                        help='Path to DIBCO dataset TFRecord')
    parser.add_argument('--anri_tfrecord', type=str, required=True,
                        help='Path to ANRI dataset TFRecord')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for mixed dataset')
    parser.add_argument('--base_ratio', type=float, default=0.50,
                        help='Ratio of base samples (default: 0.50 = 50%%)')
    parser.add_argument('--dibco_ratio', type=float, default=0.30,
                        help='Ratio of DIBCO samples (default: 0.30 = 30%%)')
    parser.add_argument('--anri_ratio', type=float, default=0.20,
                        help='Ratio of ANRI samples (default: 0.20 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    result = create_triple_mixed_dataset(
        args.base_tfrecord,
        args.dibco_tfrecord,
        args.anri_tfrecord,
        args.output,
        args.base_ratio,
        args.dibco_ratio,
        args.anri_ratio,
        args.seed
    )
    
    print(f"\n✅ Dataset creation complete!")
    print(f"   Use this dataset for training with data rehearsal to prevent catastrophic forgetting")

if __name__ == '__main__':
    main()
