#!/usr/bin/env python3
"""
H2A Model Evaluation Script - Per-Sample CER Extraction

Extract per-sample CER dari trained model untuk statistical analysis.
Digunakan untuk mendapatkan individual CER measurements (n=710) yang diperlukan
untuk paired t-test dan effect size calculation.

Author: Claude Code (AI/ML Engineer)
Purpose: H2A Hypothesis Validation - CER Data Collection
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import editdistance
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed
from dual_modal_gan.src.models.discriminator_single_modal import build_single_modal_discriminator_enhanced
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed


def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate using edit distance."""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)


def decode_ctc_predictions(predictions, idx_to_char):
    """
    Decode CTC predictions to text.
    
    Args:
        predictions: CTC output from recognizer
        idx_to_char: Dictionary mapping indices to characters
    
    Returns:
        str: Decoded text
    """
    # CTC decode
    input_length = np.array([predictions.shape[1]])
    decoded, _ = tf.keras.backend.ctc_decode(predictions, input_length, greedy=True)
    decoded_indices = decoded[0].numpy()[0]
    
    # Convert indices to text
    decoded_text = ''.join([idx_to_char.get(idx, '') for idx in decoded_indices if idx > 0])
    return decoded_text


def load_charset(charset_path):
    """Load character set from file - EXACT same as train_enhanced.py"""
    with open(charset_path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def load_tfrecord_dataset(tfrecord_path, batch_size=1, split='val'):
    """
    Load validation dataset dari TFRecord - EXACT same format as train_enhanced.py
    
    Args:
        tfrecord_path: Path to TFRecord file
        batch_size: Batch size (default=1 untuk per-sample evaluation)
        split: 'train', 'val', or 'test' - determines which split to use
    
    Returns:
        tf.data.Dataset
    """
    def parse_example(example_proto):
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
        # Transpose from (H, W, C) to (W, H, C)
        degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
        degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])
        
        # Deserialize label
        label_shape = tf.cast(example['label_shape'], tf.int32)
        label = tf.io.decode_raw(example['label_raw'], tf.int64)
        label = tf.reshape(label, label_shape)
        label = tf.cast(label, tf.int32)
        
        # Pad label to static shape
        padding = [[0, 128 - tf.shape(label)[0]]]
        label = tf.pad(label, padding, "CONSTANT", constant_values=0)
        label.set_shape([128])
        
        return degraded_image, label
    
    # Load and parse dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Split dataset (train=70%, val=15%, test=15%)
    total_samples = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    
    if split == 'train':
        dataset = dataset.take(train_size)
    elif split == 'val':
        dataset = dataset.skip(train_size).take(val_size)
    elif split == 'test':
        dataset = dataset.skip(train_size + val_size)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def evaluate_model_per_sample(checkpoint_dir, tfrecord_path, charset_path, 
                              recognizer_weights, discriminator_type='dual_modal',
                              output_path='cer_results.json'):
    """
    Evaluate model dan extract per-sample CER.
    
    Args:
        checkpoint_dir: Path ke model checkpoint
        tfrecord_path: Path ke TFRecord dataset
        charset_path: Path ke charset file
        recognizer_weights: Path ke recognizer weights
        discriminator_type: 'dual_modal' atau 'single_modal'
        output_path: Path untuk save hasil CER
    
    Returns:
        dict: Per-sample CER results
    """
    print(f"\n{'='*80}")
    print(f"H2A MODEL EVALUATION - Per-Sample CER Extraction")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Discriminator Type: {discriminator_type}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    # Load charset
    charset = load_charset(charset_path)
    vocab_size = len(charset) + 1  # +1 for blank token (CTC blank)
    print(f"✅ Charset loaded: {len(charset)} characters, vocab_size={vocab_size}")
    
    # Create char-to-index mapping
    char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
    char_to_idx[''] = 0  # blank token
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    # Load recognizer
    print(f"📥 Loading recognizer from: {recognizer_weights}")
    recognizer = load_frozen_recognizer_fixed(recognizer_weights, vocab_size)
    print(f"✅ Recognizer loaded")
    
    # Build generator
    print(f"🏗️  Building generator (enhanced)...")
    generator = unet_enhanced(input_size=(1024, 128, 1))
    print(f"✅ Generator built")
    
    # Build discriminator
    print(f"🏗️  Building discriminator ({discriminator_type})...")
    if discriminator_type == 'dual_modal':
        discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
            img_shape=(1024, 128, 1),
            vocab_size=vocab_size,
            max_text_len=128
        )
    else:  # single_modal
        discriminator = build_single_modal_discriminator_enhanced(
            img_shape=(1024, 128, 1),
            vocab_size=vocab_size,
            max_text_len=128
        )
    print(f"✅ Discriminator built")
    
    # Load checkpoint
    print(f"📥 Loading checkpoint from: {checkpoint_dir}")
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
    
    # Find latest checkpoint
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"✅ Checkpoint restored: {latest_ckpt}")
    else:
        print(f"⚠️  No checkpoint found, using initialized weights")
    
    # Load validation dataset
    print(f"📂 Loading validation dataset...")
    val_dataset = load_tfrecord_dataset(tfrecord_path, batch_size=1, split='val')
    
    # Count samples
    num_samples = sum(1 for _ in val_dataset)
    print(f"✅ Validation dataset loaded: {num_samples} samples")
    
    # Reload dataset for evaluation
    val_dataset = load_tfrecord_dataset(tfrecord_path, batch_size=1, split='val')
    
    # Evaluate per sample
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL - Per-Sample CER")
    print(f"{'='*80}\n")
    
    all_cer_values = []
    sample_details = []
    
    for idx, (batch_images, batch_labels) in enumerate(val_dataset):
        if idx % 50 == 0:
            print(f"Processing sample {idx}/{num_samples}...")
        
        # Get single sample
        image = batch_images[0]  # (1024, 128, 1)
        label_indices = batch_labels[0].numpy()  # (128,) - padded integer indices
        
        # Decode ground truth label from indices
        true_label = ''.join([idx_to_char.get(int(idx), '') for idx in label_indices if idx > 0])
        
        # Generate enhanced image
        image_batch = tf.expand_dims(image, 0)  # (1, 1024, 128, 1)
        enhanced_image = generator(image_batch, training=False)
        
        # Get predictions from recognizer
        predictions = recognizer.predict(enhanced_image, verbose=0)
        
        # Decode CTC predictions
        predicted_text = decode_ctc_predictions(predictions, idx_to_char)
        
        # Calculate CER untuk sample ini
        cer = calculate_cer(true_label, predicted_text)
        all_cer_values.append(cer)
        
        # Store details
        sample_details.append({
            'sample_idx': idx,
            'true_label': true_label,
            'predicted_label': predicted_text,
            'cer': float(cer)
        })
        
        if idx < 5:  # Show first few examples
            print(f"  Sample {idx}:")
            print(f"    True: {true_label[:50]}...")
            print(f"    Pred: {predicted_text[:50]}...")
            print(f"    CER:  {cer:.4f}")
    
    # Calculate statistics
    cer_mean = np.mean(all_cer_values)
    cer_std = np.std(all_cer_values, ddof=1)
    cer_median = np.median(all_cer_values)
    cer_min = np.min(all_cer_values)
    cer_max = np.max(all_cer_values)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {discriminator_type.upper()}")
    print(f"{'='*80}")
    print(f"Total Samples: {num_samples}")
    print(f"Mean CER:      {cer_mean:.4f} ± {cer_std:.4f}")
    print(f"Median CER:    {cer_median:.4f}")
    print(f"Range:         [{cer_min:.4f}, {cer_max:.4f}]")
    print(f"{'='*80}\n")
    
    # Prepare results
    results = {
        'metadata': {
            'checkpoint_dir': checkpoint_dir,
            'discriminator_type': discriminator_type,
            'num_samples': num_samples,
            'dataset': tfrecord_path,
            'split': 'validation'
        },
        'statistics': {
            'mean': float(cer_mean),
            'std': float(cer_std),
            'median': float(cer_median),
            'min': float(cer_min),
            'max': float(cer_max)
        },
        'per_sample_cer': all_cer_values,
        'sample_details': sample_details[:100]  # Store first 100 details untuk verification
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='H2A Model Evaluation - Per-Sample CER Extraction')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint directory')
    parser.add_argument('--tfrecord', type=str, 
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help='Path to TFRecord dataset')
    parser.add_argument('--charset', type=str,
                       default='real_data_preparation/real_data_charlist.txt',
                       help='Path to charset file')
    parser.add_argument('--recognizer', type=str,
                       default='/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5',
                       help='Path to recognizer weights')
    parser.add_argument('--discriminator_type', type=str, 
                       choices=['dual_modal', 'single_modal'],
                       default='dual_modal',
                       help='Type of discriminator')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for CER results JSON')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Configure TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    
    # Run evaluation
    try:
        results = evaluate_model_per_sample(
            checkpoint_dir=args.checkpoint,
            tfrecord_path=args.tfrecord,
            charset_path=args.charset,
            recognizer_weights=args.recognizer,
            discriminator_type=args.discriminator_type,
            output_path=args.output
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
