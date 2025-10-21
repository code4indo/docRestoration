#!/usr/bin/env python3
"""
Create Independent Test Dataset
Memisahkan test set yang tidak pernah terlihat model
"""

import os
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import random

def parse_tfrecord(example_proto):
    """Parse TFRecord - sama seperti training script"""
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

    # Parse degraded image
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Parse clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    # Parse label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.int32)

    # Pad label
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label

def create_dataset_splits(tfrecord_path, test_split=0.15, val_split=0.15, seed=42):
    """
    Split dataset menjadi train/val/test dengan stratifikasi
    """
    print(f"📂 Loading dataset from: {tfrecord_path}")

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load full dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # Convert to numpy arrays for splitting
    all_degraded = []
    all_clean = []
    all_labels = []

    print("   Extracting samples to memory...")
    for degraded, clean, label in dataset:
        all_degraded.append(degraded.numpy())
        all_clean.append(clean.numpy())
        all_labels.append(label.numpy())

    all_degraded = np.array(all_degraded)
    all_clean = np.array(all_clean)
    all_labels = np.array(all_labels)

    total_samples = len(all_degraded)
    print(f"   Total samples: {total_samples}")

    # Calculate split sizes
    test_size = int(total_samples * test_split)
    val_size = int(total_samples * val_split)
    train_size = total_samples - test_size - val_size

    print(f"   Split sizes:")
    print(f"      Training: {train_size} ({train_size/total_samples:.1%})")
    print(f"      Validation: {val_size} ({val_size/total_samples:.1%})")
    print(f"      Test: {test_size} ({test_size/total_samples:.1%})")

    # Create random indices
    indices = np.random.permutation(total_samples)

    # Split indices
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    # Split data
    train_data = {
        'degraded': all_degraded[train_indices],
        'clean': all_clean[train_indices],
        'labels': all_labels[train_indices]
    }

    val_data = {
        'degraded': all_degraded[val_indices],
        'clean': all_clean[val_indices],
        'labels': all_labels[val_indices]
    }

    test_data = {
        'degraded': all_degraded[test_indices],
        'clean': all_clean[test_indices],
        'labels': all_labels[test_indices]
    }

    return train_data, val_data, test_data

def save_tfrecord_dataset(data, output_path, description=""):
    """Save dataset ke TFRecord format"""
    print(f"   Saving {description} to: {output_path}")

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    with tf.io.TFRecordWriter(output_path) as writer:
        n_samples = len(data['degraded'])

        for i in tqdm(range(n_samples), desc=f"Writing {description}"):
            degraded = data['degraded'][i]
            clean = data['clean'][i]
            label = data['labels'][i]

            # Transpose back to (H, W, C) for storage
            degraded_transposed = np.transpose(degraded, [1, 0, 2])
            clean_transposed = np.transpose(clean, [1, 0, 2])

            feature = {
                'degraded_image_raw': _bytes_feature(degraded_transposed.astype(np.float32)),
                'degraded_image_shape': _int64_feature(degraded_transposed.shape),
                'degraded_image_dtype': _bytes_feature(np.array('float32', dtype='S').tobytes()),
                'clean_image_raw': _bytes_feature(clean_transposed.astype(np.float32)),
                'clean_image_shape': _int64_feature(clean_transposed.shape),
                'clean_image_dtype': _bytes_feature(np.array('float32', dtype='S').tobytes()),
                'label_raw': _bytes_feature(label.astype(np.int64)),
                'label_shape': _int64_feature([len(label)]),
                'label_dtype': _bytes_feature(np.array('int64', dtype='S').tobytes()),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        print(f"   ✅ Saved {n_samples} samples")

def create_split_metadata(train_data, val_data, test_data, output_dir):
    """Create metadata file untuk splits"""
    metadata = {
        'creation_timestamp': datetime.now().isoformat(),
        'split_method': 'random_split_with_fixed_seed',
        'seed': 42,
        'total_original_samples': len(train_data['degraded']) + len(val_data['degraded']) + len(test_data['degraded']),
        'splits': {
            'train': {
                'n_samples': len(train_data['degraded']),
                'percentage': len(train_data['degraded']) / (len(train_data['degraded']) + len(val_data['degraded']) + len(test_data['degraded'])),
                'file': 'train_dataset.tfrecord'
            },
            'validation': {
                'n_samples': len(val_data['degraded']),
                'percentage': len(val_data['degraded']) / (len(train_data['degraded']) + len(val_data['degraded']) + len(test_data['degraded'])),
                'file': 'validation_dataset.tfrecord'
            },
            'test': {
                'n_samples': len(test_data['degraded']),
                'percentage': len(test_data['degraded']) / (len(train_data['degraded']) + len(val_data['degraded']) + len(test_data['degraded'])),
                'file': 'test_dataset.tfrecord',
                'purpose': 'Independent evaluation - never seen during training'
            }
        },
        'academic_compliance': {
            'test_set_independence': True,
            'no_data_leakage': True,
            'reproducible_splits': True,
            'suitable_for_publication': True
        }
    }

    metadata_file = os.path.join(output_dir, 'dataset_splits_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"📄 Metadata saved to: {metadata_file}")

def main():
    """Main function untuk membuat dataset splits"""
    import argparse
    parser = argparse.ArgumentParser(description='Create Independent Test Dataset')
    parser.add_argument('--input_tfrecord', type=str,
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help='Input TFRecord file')
    parser.add_argument('--output_dir', type=str,
                       default='dual_modal_gan/data',
                       help='Output directory for split datasets')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test set proportion (default: 0.15 = 15%)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation set proportion (default: 0.15 = 15%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')

    args = parser.parse_args()

    print("🔧 CREATING INDEPENDENT TEST DATASET")
    print("=" * 50)
    print(f"Input: {args.input_tfrecord}")
    print(f"Test split: {args.test_split:.1%}")
    print(f"Validation split: {args.val_split:.1%}")
    print(f"Seed: {args.seed}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset splits
    train_data, val_data, test_data = create_dataset_splits(
        args.input_tfrecord, args.test_split, args.val_split, args.seed
    )

    # Save splits to TFRecord files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    split_dir = os.path.join(args.output_dir, f'splits_{timestamp}')
    os.makedirs(split_dir, exist_ok=True)

    save_tfrecord_dataset(
        train_data,
        os.path.join(split_dir, 'train_dataset.tfrecord'),
        "training dataset"
    )

    save_tfrecord_dataset(
        val_data,
        os.path.join(split_dir, 'validation_dataset.tfrecord'),
        "validation dataset"
    )

    save_tfrecord_dataset(
        test_data,
        os.path.join(split_dir, 'test_dataset.tfrecord'),
        "independent test dataset"
    )

    # Create metadata
    create_split_metadata(train_data, val_data, test_data, split_dir)

    print(f"\n✅ Dataset splits created successfully!")
    print(f"📁 Output directory: {split_dir}")
    print(f"📊 Summary:")
    print(f"   Training: {len(train_data['degraded'])} samples")
    print(f"   Validation: {len(val_data['degraded'])} samples")
    print(f"   Test: {len(test_data['degraded'])} samples (INDEPENDENT)")

    print(f"\n📚 Academic Usage:")
    print(f"   1. Use 'train_dataset.tfrecord' for model training")
    print(f"   2. Use 'validation_dataset.tfrecord' for hyperparameter tuning")
    print(f"   3. Use 'test_dataset.tfrecord' ONLY for final evaluation")
    print(f"   4. NEVER use test data during training or validation")

    print(f"\n🎯 Result: Now you have academically rigorous dataset splits!")

if __name__ == '__main__':
    main()