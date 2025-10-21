#!/usr/bin/env python3
"""
Test HTR Recognizer on Clean Images
====================================

Skrip standalone untuk menguji performa HTR recognizer pada dataset gambar bersih (clean images).
Output berupa CSV yang membandingkan ground truth vs predicted text dengan metrik CER dan WER.

Author: Belekok & AI Assistant
Date: 2025-10-20
Purpose: Validasi baseline performance recognizer sebelum training lebih lanjut

Usage:
    poetry run python dual_modal_gan/scripts/test_recognizer_on_clean_images.py --num_samples 100
    poetry run python dual_modal_gan/scripts/test_recognizer_on_clean_images.py --num_samples 500 --output results/htr_test_500.csv
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import tensorflow as tf
import numpy as np
import editdistance

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed


def safe_ctc_decode(logits, charset):
    """
    Safe CTC decode with manual fallback.
    EXACT implementation from train_transformer_improved_v2.py
    
    Args:
        logits: Model output logits (1, time_steps, vocab_size)
        charset: List of characters
    
    Returns:
        Decoded string
    """
    charset_size = len(charset)
    raw_preds = np.argmax(logits, axis=-1)[0]
    
    # Manual CTC decode
    deduped = []
    prev = -1
    for token in raw_preds:
        if token != prev:
            deduped.append(token)
            prev = token
    
    result = []
    for token in deduped:
        if token != charset_size:  # Skip blank
            result.append(token)
    
    text = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
    return text


def decode_predictions_batch(predictions, charset):
    """
    Decode batch CTC predictions to text.
    Adopted from evaluate_comprehensive.py
    
    Args:
        predictions: Logits from recognizer (batch, time_steps, vocab_size)
        charset: List of characters
    
    Returns:
        List of decoded strings
    """
    decoded_texts = []
    for i in range(predictions.shape[0]):
        logits = predictions[i:i+1]
        text = safe_ctc_decode(logits, charset)
        decoded_texts.append(text)
    
    return decoded_texts


def calculate_cer(ground_truth: str, predicted: str) -> float:
    """
    Calculate Character Error Rate (CER)
    
    Args:
        ground_truth: Ground truth text
        predicted: Predicted text
        
    Returns:
        CER value (0.0 = perfect, 1.0+ = very poor)
    """
    if len(ground_truth) == 0:
        return 1.0 if len(predicted) > 0 else 0.0
    
    distance = editdistance.eval(ground_truth, predicted)
    return distance / len(ground_truth)


def calculate_wer(ground_truth: str, predicted: str) -> float:
    """
    Calculate Word Error Rate (WER)
    
    Args:
        ground_truth: Ground truth text
        predicted: Predicted text
        
    Returns:
        WER value (0.0 = perfect, 1.0+ = very poor)
    """
    gt_words = ground_truth.split()
    pred_words = predicted.split()
    
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    
    distance = editdistance.eval(gt_words, pred_words)
    return distance / len(gt_words)


def load_tfrecord_dataset(tfrecord_path: str, num_samples: int = 100) -> tf.data.Dataset:
    """
    Load TFRecord dataset containing clean images and labels
    Supports both GAN format and real data format
    
    Args:
        tfrecord_path: Path to TFRecord file
        num_samples: Number of samples to load
        
    Returns:
        TensorFlow dataset yielding (clean_image, label_indices)
    """
    
    def _parse_real_data_format(example_proto):
        """Parse real data TFRecord (train_transformer_improved_v2.py format)"""
        feature_description = {
            'image_degraded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'text': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Parse image (clean)
        def _parse_serialized():
            t = tf.io.parse_tensor(parsed['image'], out_type=tf.float32)
            t = tf.cond(tf.equal(tf.rank(t), 2), lambda: tf.expand_dims(t, -1), lambda: t)
            t.set_shape([128, 1024, 1])  # (H, W, C)
            return t
        
        def _decode_legacy():
            img_legacy = tf.io.decode_png(parsed['image_degraded'], channels=1)
            img_legacy = tf.image.convert_image_dtype(img_legacy, tf.float32)
            return img_legacy
        
        img = tf.cond(tf.not_equal(parsed['image'], tf.constant(b'')), _parse_serialized, _decode_legacy)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [128, 1024])
        img = tf.transpose(img, perm=[1, 0, 2])  # (H, W, C) → (W, H, C) = (1024, 128, 1)
        img = tf.ensure_shape(img, [1024, 128, 1])
        
        # Parse label (numeric format)
        def parse_numeric_label():
            raw = parsed['label']
            seq = tf.io.parse_tensor(raw, out_type=tf.int64)
            seq = tf.cast(seq, tf.int32)
            return seq
        
        def parse_text_label():
            # Fallback for text format (not used in real data)
            return tf.constant([0], dtype=tf.int32)
        
        label = tf.cond(tf.not_equal(parsed['label'], tf.constant(b'')), 
                       parse_numeric_label, parse_text_label)
        
        return img, label
    
    def _parse_gan_format(example_proto):
        """Parse GAN TFRecord (train_enhanced.py format)"""
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
        
        # Deserialize clean image
        clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
        clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
        clean_image = tf.reshape(clean_image, clean_image_shape)
        # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
        clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
        clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])
        
        # Deserialize label
        label_shape = tf.cast(example['label_shape'], tf.int32)
        label = tf.io.decode_raw(example['label_raw'], tf.int64)
        label = tf.reshape(label, label_shape)
        label = tf.cast(label, tf.int32)
        
        return clean_image, label
    
    # Try to auto-detect format by checking filename
    if 'real_data' in tfrecord_path:
        parse_fn = _parse_real_data_format
    else:
        parse_fn = _parse_gan_format
    
    # Load dataset
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.take(num_samples)
    
    return dataset


def prepare_image_for_recognizer(image: tf.Tensor, target_height: int = 64) -> tf.Tensor:
    """
    Prepare image for recognizer input
    
    Args:
        image: Input image tensor
        target_height: Target height for recognizer
        
    Returns:
        Prepared image tensor
    """
    # Get current shape
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    
    # Calculate new width maintaining aspect ratio
    aspect_ratio = tf.cast(width, tf.float32) / tf.cast(height, tf.float32)
    new_width = tf.cast(tf.cast(target_height, tf.float32) * aspect_ratio, tf.int32)
    
    # Resize image
    image = tf.image.resize(image, [target_height, new_width], method='bilinear')
    
    # Normalize to [0, 1] if needed
    # Assuming input is already in appropriate range based on training data
    
    return image


def run_htr_test(
    recognizer_weights_path: str,
    tfrecord_path: str,
    num_samples: int = 100,
    output_csv: str = "results/recognizer_test_clean_images.csv",
    char_list_path: str = "dual_modal_gan/dataset/char_list.txt"
) -> Dict:
    """
    Run HTR test on clean images
    
    Args:
        recognizer_weights_path: Path to recognizer weights
        tfrecord_path: Path to TFRecord file
        num_samples: Number of samples to test
        output_csv: Output CSV file path
        char_list_path: Path to character list file
        
    Returns:
        Dictionary with summary statistics
    """
    print("=" * 80)
    print("HTR RECOGNIZER TEST ON CLEAN IMAGES")
    print("=" * 80)
    print(f"Recognizer weights: {recognizer_weights_path}")
    print(f"TFRecord dataset: {tfrecord_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Output CSV: {output_csv}")
    print("=" * 80)
    
    # Load character list
    if os.path.exists(char_list_path):
        with open(char_list_path, 'r', encoding='utf-8') as f:
            characters = [line.rstrip('\n') for line in f]  # Keep all lines including empty
        print(f"Loaded {len(characters)} characters from {char_list_path}")
    else:
        print(f"WARNING: Character list file not found: {char_list_path}")
        print("Trying alternative paths...")
        alt_paths = [
            'real_data_preparation/real_data_charlist.txt',
            'dual_modal_gan/data/char_list.txt'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                with open(alt_path, 'r', encoding='utf-8') as f:
                    characters = [line.rstrip('\n') for line in f]
                print(f"Loaded {len(characters)} characters from {alt_path}")
                char_list_path = alt_path
                break
        else:
            # Fallback to ASCII if nothing found
            characters = [chr(i) for i in range(32, 127)]
            print(f"Using fallback ASCII character list: {len(characters)} characters")
    
    # Build recognizer model using load_frozen_recognizer_fixed
    print("\nBuilding recognizer model...")
    print(f"Loading weights from {recognizer_weights_path}...")
    recognizer = load_frozen_recognizer_fixed(
        weights_path=recognizer_weights_path,
        charset_size=len(characters),
        num_transformer_layers=6,
        dropout_rate=0.20,
        return_feature_map=False  # Only return logits for CTC decode
    )
    print("✓ Model loaded successfully")
    
    # Load dataset
    print(f"\nLoading dataset from {tfrecord_path}...")
    dataset = load_tfrecord_dataset(tfrecord_path, num_samples)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'sample_id', 
            'ground_truth', 
            'predicted_text', 
            'cer', 
            'wer', 
            'confidence',
            'gt_length',
            'pred_length',
            'match'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Statistics
        total_cer = 0.0
        total_wer = 0.0
        perfect_matches = 0
        total_samples = 0
        
        print("\nProcessing samples...")
        print("-" * 80)
        
        # Process each sample
        for idx, (clean_image, label_indices) in enumerate(dataset):
            sample_id = idx + 1
            
            # Convert label indices to ground truth text
            # NOTE: Labels are 1-indexed (0 = padding), so we need to subtract 1
            label_indices_np = label_indices.numpy()
            # Remove padding (zeros)
            label_indices_np = label_indices_np[label_indices_np > 0]
            # Convert indices to characters (subtract 1 because labels are 1-indexed)
            ground_truth = ''.join([characters[i - 1] if 1 <= i <= len(characters) else '<?>' for i in label_indices_np])
            
            # Image already in correct format from TFRecord: (1024, 128, 1)
            # DEBUG: Check image stats
            if idx == 0:
                img_min = tf.reduce_min(clean_image).numpy()
                img_max = tf.reduce_max(clean_image).numpy()
                img_mean = tf.reduce_mean(clean_image).numpy()
                print(f"\n[DEBUG IMAGE] Shape: {clean_image.shape}")
                print(f"[DEBUG IMAGE] Range: [{img_min:.4f}, {img_max:.4f}], Mean: {img_mean:.4f}")
            
            # Just add batch dimension
            image_batch = tf.expand_dims(clean_image, axis=0)
            
            # Run inference
            predictions = recognizer.predict(image_batch, verbose=0)
            
            # DEBUG: Check raw logits
            if idx == 0:
                print(f"\n[DEBUG] Predictions shape: {predictions.shape}")
                print(f"[DEBUG] First 5 time steps logits (max values):")
                for t in range(min(5, predictions.shape[1])):
                    max_idx = np.argmax(predictions[0, t, :])
                    max_val = predictions[0, t, max_idx]
                    print(f"  t={t}: max_idx={max_idx}, max_val={max_val:.4f}, char='{characters[max_idx] if max_idx < len(characters) else '?'}'")
            
            # Decode predictions using our decode function
            predicted_texts = decode_predictions_batch(predictions, characters)
            predicted_text = predicted_texts[0] if predicted_texts else ""
            
            # Calculate metrics
            cer = calculate_cer(ground_truth, predicted_text)
            wer = calculate_wer(ground_truth, predicted_text)
            
            # Calculate confidence (average of max probabilities)
            # predictions shape: (batch, time_steps, num_classes)
            confidence = float(np.mean(np.max(predictions[0], axis=-1)))
            
            # Check if perfect match
            is_match = (ground_truth == predicted_text)
            if is_match:
                perfect_matches += 1
            
            # Update statistics
            total_cer += cer
            total_wer += wer
            total_samples += 1
            
            # Write to CSV
            writer.writerow({
                'sample_id': sample_id,
                'ground_truth': ground_truth,
                'predicted_text': predicted_text,
                'cer': f"{cer:.4f}",
                'wer': f"{wer:.4f}",
                'confidence': f"{confidence:.4f}",
                'gt_length': len(ground_truth),
                'pred_length': len(predicted_text),
                'match': 'YES' if is_match else 'NO'
            })
            
            # Print progress
            if (sample_id) % 10 == 0 or sample_id == 1:
                print(f"Sample {sample_id:4d}/{num_samples}: "
                      f"CER={cer:.4f}, WER={wer:.4f}, "
                      f"Match={'✓' if is_match else '✗'}")
                print(f"  GT:   '{ground_truth}'")
                print(f"  Pred: '{predicted_text}'")
                print()
    
    # Calculate summary statistics
    avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
    avg_wer = total_wer / total_samples if total_samples > 0 else 0.0
    match_rate = (perfect_matches / total_samples * 100) if total_samples > 0 else 0.0
    
    summary = {
        'total_samples': total_samples,
        'avg_cer': avg_cer,
        'avg_wer': avg_wer,
        'perfect_matches': perfect_matches,
        'match_rate': match_rate
    }
    
    # Print summary
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total samples processed: {total_samples}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Perfect matches: {perfect_matches}/{total_samples} ({match_rate:.2f}%)")
    print("=" * 80)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if avg_cer < 0.10:
        interpretation = "EXCELLENT - Recognizer performing very well on clean images"
    elif avg_cer < 0.25:
        interpretation = "GOOD - Recognizer performance acceptable"
    elif avg_cer < 0.40:
        interpretation = "ACCEPTABLE - Some recognition issues but usable"
    else:
        interpretation = "POOR - Significant recognition issues, needs investigation"
    
    print(f"  {interpretation}")
    print(f"\nResults saved to: {output_csv}")
    print("=" * 80)
    
    return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test HTR Recognizer on Clean Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test 100 samples (default)
    poetry run python dual_modal_gan/scripts/test_recognizer_on_clean_images.py
    
    # Test 500 samples with custom output
    poetry run python dual_modal_gan/scripts/test_recognizer_on_clean_images.py \\
        --num_samples 500 \\
        --output results/htr_test_500.csv
    
    # Use custom TFRecord file
    poetry run python dual_modal_gan/scripts/test_recognizer_on_clean_images.py \\
        --tfrecord dual_modal_gan/dataset/validation.tfrecord \\
        --num_samples 200
        """
    )
    
    parser.add_argument(
        '--recognizer_weights',
        type=str,
        default='dual_modal_gan/models/best_model.weights.h5',
        help='Path to recognizer weights file (default: dual_modal_gan/models/best_model.weights.h5)'
    )
    
    parser.add_argument(
        '--tfrecord',
        type=str,
        default='dual_modal_gan/dataset/train.tfrecord',
        help='Path to TFRecord file (default: dual_modal_gan/dataset/train.tfrecord)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to test (default: 100)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/recognizer_test_clean_images.csv',
        help='Output CSV file path (default: results/recognizer_test_clean_images.csv)'
    )
    
    parser.add_argument(
        '--char_list',
        type=str,
        default='real_data_preparation/real_data_charlist.txt',
        help='Path to character list file (default: real_data_preparation/real_data_charlist.txt)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.recognizer_weights):
        print(f"ERROR: Recognizer weights not found: {args.recognizer_weights}")
        sys.exit(1)
    
    if not os.path.exists(args.tfrecord):
        print(f"ERROR: TFRecord file not found: {args.tfrecord}")
        sys.exit(1)
    
    # Run test
    try:
        summary = run_htr_test(
            recognizer_weights_path=args.recognizer_weights,
            tfrecord_path=args.tfrecord,
            num_samples=args.num_samples,
            output_csv=args.output,
            char_list_path=args.char_list
        )
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
