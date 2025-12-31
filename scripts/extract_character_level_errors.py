#!/usr/bin/env python3
"""
Phase 1: Character-Level Error Extraction

Extract character-level errors from HTR predictions on:
1. Degraded images (baseline)
2. Restored images (our method)
3. Clean GT images (upper bound)

This enables diagnostic analysis of:
- Which characters are most confused
- Position-dependent error patterns
- Degradation-specific impacts

Author: Character-Level Diagnostic Analysis
Date: 2025-11-30
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import editdistance

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

@dataclass
class CharacterError:
    """Single character-level error instance"""
    sample_id: int
    batch_idx: int
    position_in_word: int
    position_in_sequence: int
    word_position: str  # 'start', 'middle', 'end'
    
    gt_char: str
    pred_char_degraded: str
    pred_char_restored: str
    pred_char_clean: str
    
    context_before: str
    context_after: str
    full_word: str
    full_gt_text: str
    
    error_type_degraded: str  # 'correct', 'substitution', 'insertion', 'deletion'
    error_type_restored: str
    error_type_clean: str
    
    is_ligature: bool
    is_capital: bool
    is_punctuation: bool
    
    # Will be added if degradation metadata available
    degradation_type: str = 'unknown'
    degradation_severity: float = 0.0

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
            if token != charset_size:  # charset_size is blank token
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

def align_strings(gt: str, pred: str) -> List[Tuple[str, str, str]]:
    """
    Align two strings using edit distance and return aligned pairs
    
    Returns:
        List of (gt_char, pred_char, error_type)
        error_type: 'correct', 'substitution', 'insertion', 'deletion'
    """
    # Use dynamic programming to get alignment
    n, m = len(gt), len(pred)
    
    # DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    # Backtrack to get alignment
    alignments = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i-1] == pred[j-1]:
            # Match
            alignments.append((gt[i-1], pred[j-1], 'correct'))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            alignments.append((gt[i-1], pred[j-1], 'substitution'))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            # Insertion (extra char in prediction)
            alignments.append(('', pred[j-1], 'insertion'))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion (missing char in prediction)
            alignments.append((gt[i-1], '', 'deletion'))
            i -= 1
    
    alignments.reverse()
    return alignments

def detect_word_position(char_idx: int, text: str) -> str:
    """Detect if character is at start/middle/end of word"""
    if char_idx == 0 or text[char_idx-1] == ' ':
        return 'start'
    
    if char_idx == len(text) - 1 or (char_idx + 1 < len(text) and text[char_idx+1] == ' '):
        return 'end'
    
    return 'middle'

def is_ligature(char: str, context: str) -> bool:
    """
    Detect if character is part of common Dutch paleographic ligature
    
    Common ligatures: st, ct, ck, ae, oe, ij, ff, ll
    """
    ligatures = ['st', 'ct', 'ck', 'ae', 'oe', 'ij', 'ff', 'll', 'ss']
    
    for lig in ligatures:
        if lig in context.lower():
            return True
    
    return False

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
    
    # Map parsing
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Skip train and val to get test set
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return test_dataset, test_size

def extract_character_errors(
    tfrecord_path: str,
    charset_path: str,
    recognizer_weights: str,
    generator_checkpoint: str,
    batch_size: int = 2,
    output_path: str = 'character_errors.json'
):
    """
    Main extraction function
    
    Process:
    1. Load test dataset
    2. For each sample:
       a. Get GT text
       b. Predict on degraded image
       c. Restore image with generator
       d. Predict on restored image
       e. Predict on clean GT image (upper bound)
       f. Align all predictions with GT
       g. Extract character-level errors
    3. Save to JSON
    """
    
    print("="*80)
    print("PHASE 1: CHARACTER-LEVEL ERROR EXTRACTION")
    print("="*80)
    
    # Load charset
    charset = read_charlist(charset_path)
    vocab_size = len(charset) + 1  # +1 for blank
    print(f"\nCharset loaded: {vocab_size} characters (including blank)")
    
    # Load test dataset
    print(f"\nLoading test dataset from: {tfrecord_path}")
    test_dataset, test_size = create_test_dataset(tfrecord_path, batch_size)
    print(f"Test set size: {test_size} samples")
    
    # Load recognizer
    print(f"\nLoading frozen recognizer from: {recognizer_weights}")
    recognizer = load_frozen_recognizer(
        weights_path=recognizer_weights,
        charset_size=vocab_size - 1,
        return_feature_map=False
    )
    print("Recognizer loaded successfully")
    
    # Load generator
    print(f"\nLoading generator from: {generator_checkpoint}")
    # Implementation depends on your generator architecture
    # For now, placeholder - will be added based on your actual model
    generator = tf.keras.models.load_model(generator_checkpoint, compile=False)
    print("Generator loaded successfully")
    
    # Extract errors
    all_errors = []
    sample_id = 0
    
    print(f"\n{'='*80}")
    print("EXTRACTING CHARACTER-LEVEL ERRORS...")
    print(f"{'='*80}\n")
    
    for batch_idx, (degraded_images, clean_images, labels) in enumerate(tqdm(test_dataset, desc="Processing")):
        current_batch_size = degraded_images.shape[0]
        
        # Restore images
        restored_images = generator(degraded_images, training=False)
        
        # Get predictions on all three versions
        logits_degraded = recognizer(degraded_images, training=False)
        logits_restored = recognizer(restored_images, training=False)
        logits_clean = recognizer(clean_images, training=False)
        
        # Decode predictions
        preds_degraded = decode_ctc_predictions(logits_degraded[0].numpy(), charset)
        preds_restored = decode_ctc_predictions(logits_restored[0].numpy(), charset)
        preds_clean = decode_ctc_predictions(logits_clean[0].numpy(), charset)
        
        # Process each sample in batch
        for i in range(current_batch_size):
            gt_text = decode_label(labels[i].numpy(), charset)
            
            # Align predictions with GT
            align_degraded = align_strings(gt_text, preds_degraded[i])
            align_restored = align_strings(gt_text, preds_restored[i])
            align_clean = align_strings(gt_text, preds_clean[i])
            
            # Extract character-level errors
            for char_idx, (align_deg, align_res, align_cln) in enumerate(zip(align_degraded, align_restored, align_clean)):
                gt_char, pred_deg, err_type_deg = align_deg
                _, pred_res, err_type_res = align_res
                _, pred_cln, err_type_cln = align_clean
                
                # Skip if all correct (no diagnostic value)
                if err_type_deg == 'correct' and err_type_res == 'correct' and err_type_cln == 'correct':
                    continue
                
                # Determine word position
                word_pos = detect_word_position(char_idx, gt_text) if gt_char else 'unknown'
                
                # Extract context
                context_before = gt_text[max(0, char_idx-2):char_idx]
                context_after = gt_text[char_idx+1:min(len(gt_text), char_idx+3)]
                
                # Find current word
                words = gt_text.split()
                current_word = ''
                char_count = 0
                for word in words:
                    if char_count <= char_idx < char_count + len(word):
                        current_word = word
                        break
                    char_count += len(word) + 1  # +1 for space
                
                # Create error instance
                error = CharacterError(
                    sample_id=sample_id,
                    batch_idx=batch_idx,
                    position_in_word=char_idx - char_count if current_word else 0,
                    position_in_sequence=char_idx,
                    word_position=word_pos,
                    
                    gt_char=gt_char,
                    pred_char_degraded=pred_deg,
                    pred_char_restored=pred_res,
                    pred_char_clean=pred_cln,
                    
                    context_before=context_before,
                    context_after=context_after,
                    full_word=current_word,
                    full_gt_text=gt_text,
                    
                    error_type_degraded=err_type_deg,
                    error_type_restored=err_type_res,
                    error_type_clean=err_type_cln,
                    
                    is_ligature=is_ligature(gt_char, current_word),
                    is_capital=gt_char.isupper() if gt_char else False,
                    is_punctuation=not gt_char.isalnum() if gt_char else False
                )
                
                all_errors.append(asdict(error))
            
            sample_id += 1
    
    # Save results
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total samples processed: {sample_id}")
    print(f"Total character-level errors extracted: {len(all_errors)}")
    print(f"Saving to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_samples': sample_id,
                'total_errors': len(all_errors),
                'tfrecord_path': tfrecord_path,
                'charset_path': charset_path,
                'recognizer_weights': recognizer_weights,
                'generator_checkpoint': generator_checkpoint
            },
            'errors': all_errors
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Character-level error extraction complete!")
    print(f"   Output: {output_path}")
    
    return all_errors

if __name__ == '__main__':
    # Configuration
    TFRECORD_PATH = 'dual_modal_gan/data/dataset_gan.tfrecord'
    CHARSET_PATH = 'real_data_preparation/real_data_charlist.txt'
    RECOGNIZER_WEIGHTS = '/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5'
    GENERATOR_CHECKPOINT = 'dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/ckpt-50'
    BATCH_SIZE = 4
    OUTPUT_PATH = 'dual_modal_gan/analysis/character_errors.json'
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Extract errors
    errors = extract_character_errors(
        tfrecord_path=TFRECORD_PATH,
        charset_path=CHARSET_PATH,
        recognizer_weights=RECOGNIZER_WEIGHTS,
        generator_checkpoint=GENERATOR_CHECKPOINT,
        batch_size=BATCH_SIZE,
        output_path=OUTPUT_PATH
    )
