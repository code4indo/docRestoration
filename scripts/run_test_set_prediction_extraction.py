#!/usr/bin/env python3
"""
Run Full Test Set Evaluation and Save Predictions

This will:
1. Load test set (712 samples)
2. Run inference on degraded images
3. Get HTR predictions
4. Save GT texts and predictions for character-level analysis

Output: predictions.json with GT and pred texts for all 712 samples
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def read_charlist(path):
    """Load character list"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def decode_ctc_predictions(logits, charset):
    """Manual CTC decode"""
    charset_size = len(charset)
    batch_size = logits.shape[0]
    results = []
    
    for b in range(batch_size):
        raw_preds = np.argmax(logits[b], axis=-1)
        
        # CTC decode with deduplication
        deduped = []
        prev = -1
        for token in raw_preds:
            if token != prev:
                deduped.append(token)
                prev = token
        
        # Remove blank tokens
        result = []
        for token in deduped:
            if token != charset_size:
                result.append(token)
        
        # Convert to string
        decoded = ''.join([charset[i] if 0 <= i < len(charset) else '<UNK>' for i in result])
        results.append(decoded)
    
    return results

def decode_label(label_ids, charset):
    """Decode label IDs to text string"""
    decoded_chars = []
    prev_id = -1
    for label_id in label_ids:
        label_id = int(label_id)
        if label_id > 0 and label_id != prev_id:
            if label_id - 1 < len(charset):
                decoded_chars.append(charset[label_id - 1])
        prev_id = label_id
    return ''.join(decoded_chars)

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord"""
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
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    # Deserialize label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.int32)
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label

def create_test_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """Create test dataset (last 15% of data)"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return test_dataset, test_size

def extract_predictions(
    tfrecord_path: str,
    charset_path: str,
    recognizer_weights: str,
    batch_size: int = 8,
    output_path: str = 'test_predictions.json'
):
    """
    Extract GT and predictions for full test set
    """
    
    print("="*80)
    print("EXTRACTING PREDICTIONS FROM TEST SET")
    print("="*80)
    
    # Load charset
    charset = read_charlist(charset_path)
    vocab_size = len(charset) + 1
    print(f"\nCharset: {vocab_size} characters (including blank)")
    
    # Load test dataset
    print(f"\nLoading test dataset: {tfrecord_path}")
    test_dataset, test_size = create_test_dataset(tfrecord_path, batch_size)
    print(f"Test set size: {test_size} samples")
    
    # Load recognizer
    print(f"\nLoading recognizer: {recognizer_weights}")
    recognizer = load_frozen_recognizer(
        weights_path=recognizer_weights,
        charset_size=vocab_size - 1,
        return_feature_map=False
    )
    print("Recognizer loaded ✅")
    
    # Process test set
    all_predictions = []
    sample_id = 0
    
    print(f"\n{'='*80}")
    print("PROCESSING TEST SET...")
    print(f"{'='*80}\n")
    
    for batch_idx, (degraded_images, clean_images, labels) in enumerate(tqdm(test_dataset, desc="Extracting", total=test_size//batch_size+1)):
        current_batch_size = degraded_images.shape[0]
        
        # Get predictions on degraded images
        recognizer_output = recognizer(degraded_images, training=False)
        
        # Handle output format (may be tuple or single tensor)
        if isinstance(recognizer_output, (list, tuple)):
            logits_degraded = recognizer_output[0]
        else:
            logits_degraded = recognizer_output
        
        preds_degraded = decode_ctc_predictions(logits_degraded.numpy(), charset)
        
        # Decode GT labels
        for i in range(current_batch_size):
            gt_text = decode_label(labels[i].numpy(), charset)
            pred_text = preds_degraded[i]
            
            all_predictions.append({
                'sample_id': sample_id,
                'batch_idx': batch_idx,
                'gt_text': gt_text,
                'pred_text_degraded': pred_text,
                'gt_length': len(gt_text),
                'pred_length': len(pred_text)
            })
            
            sample_id += 1
    
    # Save results
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total samples: {len(all_predictions)}")
    print(f"Saving to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_samples': len(all_predictions),
                'tfrecord_path': tfrecord_path,
                'charset_path': charset_path,
                'recognizer_weights': recognizer_weights,
                'batch_size': batch_size
            },
            'predictions': all_predictions
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved: {output_path}")
    
    # Quick stats
    total_gt_chars = sum(p['gt_length'] for p in all_predictions)
    total_pred_chars = sum(p['pred_length'] for p in all_predictions)
    
    print(f"\nQuick Stats:")
    print(f"  Total GT characters: {total_gt_chars:,}")
    print(f"  Total pred characters: {total_pred_chars:,}")
    print(f"  Avg GT length: {total_gt_chars/len(all_predictions):.1f}")
    print(f"  Avg pred length: {total_pred_chars/len(all_predictions):.1f}")
    
    return all_predictions

if __name__ == '__main__':
    # Configuration
    TFRECORD_PATH = 'dual_modal_gan/data/dataset_gan.tfrecord'
    CHARSET_PATH = 'real_data_preparation/real_data_charlist.txt'
    RECOGNIZER_WEIGHTS = '/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5'
    BATCH_SIZE = 8
    OUTPUT_PATH = 'dual_modal_gan/analysis/test_predictions.json'
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Extract predictions
    predictions = extract_predictions(
        tfrecord_path=TFRECORD_PATH,
        charset_path=CHARSET_PATH,
        recognizer_weights=RECOGNIZER_WEIGHTS,
        batch_size=BATCH_SIZE,
        output_path=OUTPUT_PATH
    )
    
    print(f"\n✅ DONE! Now run: poetry run python scripts/extract_character_errors_from_predictions.py")
