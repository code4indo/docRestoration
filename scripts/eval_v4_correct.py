#!/usr/bin/env python3
"""
Correct V4 Model Evaluation
Fixes:
1. Proper label decoding (1-indexed to charset)
2. Proper CTC prediction decoding
3. Matches V4 training data format
"""
import sys
import os
import numpy as np
import tensorflow as tf
import jiwer

# Add scripts to path
sys.path.insert(0, 'scripts')
from train_transformer_v4_conservative import (
    read_charlist, build_model, parse_tfrecord,
    IMG_WIDTH, IMG_HEIGHT, MAX_LABEL_LENGTH, BATCH_SIZE
)

CHARSET_PATH = '/home/lambda_one/tesis/GAN-HTR-ORI/real_data_preparation/real_data_charlist.txt'
TFRECORD_PATH = '/home/lambda_one/tesis/GAN-HTR-ORI/real_data_final_fixed_v2.tfrecord'
MODEL_PATH = 'HTR/v4_conservative_20251106_072619/best_model.weights.h5'

def decode_label_to_text(label_indices, charset):
    """
    Decode label indices to text
    Label format: 1-indexed (1=first char in charset, 0=padding)
    """
    text_chars = []
    for idx in label_indices:
        idx_val = int(idx)
        if idx_val > 0 and idx_val <= len(charset):
            text_chars.append(charset[idx_val - 1])  # Convert 1-indexed to 0-indexed
    return ''.join(text_chars)

def ctc_decode_predictions(preds, charset):
    """
    CTC decode predictions
    Model output: logits with shape (batch, time, num_classes)
    num_classes = len(charset) + 1 (charset + blank)
    blank_index = len(charset)
    """
    results = []
    blank_index = len(charset)
    
    for pred in preds:
        # Get predicted indices (argmax)
        indices = np.argmax(pred, axis=-1)
        
        # CTC decode: remove blanks and consecutive duplicates
        decoded = []
        prev = None
        for idx in indices:
            # Skip blank token
            if idx == blank_index:
                prev = idx
                continue
            
            # Skip consecutive duplicates
            if idx == prev:
                continue
            
            # Valid character (0-indexed in charset)
            if 0 <= idx < len(charset):
                decoded.append(charset[idx])
            
            prev = idx
        
        results.append(''.join(decoded))
    return results

def main():
    print("="*80)
    print("🔍 V4 Model Evaluation - CORRECT VERSION")
    print("="*80)
    
    # Load charset
    print(f"\nLoading charset...")
    charset = read_charlist(CHARSET_PATH)
    print(f"✓ Loaded {len(charset)} characters")
    
    # Create char mapping for parse_tfrecord
    char_to_num = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=list(charset), values=list(range(len(charset))), 
            key_dtype=tf.string, value_dtype=tf.int64
        ), default_value=-1
    )
    
    # Load and parse dataset
    print(f"\nLoading dataset...")
    dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    dataset = dataset.map(lambda x: parse_tfrecord(x, char_to_num), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # Count total samples
    total_samples = sum(1 for _ in tf.data.TFRecordDataset(TFRECORD_PATH))
    train_size = int(total_samples * 0.8)
    val_size = total_samples - train_size
    print(f"✓ Total: {total_samples}, Train: {train_size}, Val: {val_size}")
    
    # Get validation dataset
    val_dataset = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Load model
    print(f"\nLoading model...")
    model = build_model(len(charset), num_transformer_layers=6)
    model.load_weights(MODEL_PATH)
    print('✓ Model loaded')
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Output shape: (batch, time, {len(charset)+1})")
    print(f"   Blank index: {len(charset)}")
    
    # Evaluate
    print(f"\n{'='*80}")
    print(f"📊 Evaluating on validation set ({val_size} samples)...")
    print(f"{'='*80}\n")
    
    all_predictions = []
    all_ground_truths = []
    batch_count = 0
    
    for batch_images, batch_labels in val_dataset:
        batch_count += 1
        if batch_count % 5 == 0:
            print(f"Progress: Batch {batch_count} ({len(all_predictions)} samples)", end='\r')
        
        # Predict (model outputs logits)
        preds = model.predict(batch_images, verbose=0)
        pred_texts = ctc_decode_predictions(preds, charset)
        
        # Ground truth (labels are 1-indexed)
        for label in batch_labels:
            gt_text = decode_label_to_text(label.numpy(), charset)
            all_ground_truths.append(gt_text)
        
        all_predictions.extend(pred_texts)
    
    print(f"\nProgress: Complete! Evaluated {len(all_predictions)} samples")
    
    # Calculate metrics
    cer = jiwer.cer(all_ground_truths, all_predictions)
    wer = jiwer.wer(all_ground_truths, all_predictions)
    
    print(f"\n{'='*80}")
    print(f"📈 FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Character Error Rate (CER): {cer*100:.2f}%")
    print(f"Word Error Rate (WER): {wer*100:.2f}%")
    print(f"{'='*80}")
    
    # Show examples
    print(f"\n📝 Sample Predictions (first 10):")
    print("-"*80)
    for i in range(min(10, len(all_predictions))):
        print(f"\n{i+1}.")
        print(f"  GT: {all_ground_truths[i][:80]}")
        print(f"  PD: {all_predictions[i][:80]}")
        if all_ground_truths[i]:
            sample_cer = jiwer.cer([all_ground_truths[i]], [all_predictions[i]])
            print(f"  CER: {sample_cer*100:.2f}%")
    
    # Save results
    import json
    results = {
        "model_path": MODEL_PATH,
        "validation_samples": val_size,
        "cer": float(cer),
        "wer": float(wer),
        "cer_percentage": float(cer * 100),
        "wer_percentage": float(wer * 100),
        "charset_size": len(charset),
        "model_parameters": int(model.count_params())
    }
    
    results_path = "HTR/v4_conservative_20251106_072619/evaluation_results_correct.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Check target
    print(f"\n{'='*80}")
    print(f"🎯 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline CER (V2): 33.72%")
    print(f"Target CER: 28-30%")
    print(f"V4 CER: {cer*100:.2f}%")
    print(f"{'='*80}")
    
    if cer * 100 < 31:
        improvement = 33.72 - (cer * 100)
        print(f"✅ SUCCESS! CER < 31%")
        print(f"   Improvement: {improvement:.2f} percentage points")
    elif cer * 100 < 33.72:
        improvement = 33.72 - (cer * 100)
        print(f"⚠️  PARTIAL SUCCESS - Better than baseline but missed target")
        print(f"   Improvement: {improvement:.2f} percentage points")
    else:
        regression = (cer * 100) - 33.72
        print(f"❌ REGRESSION - Worse than baseline")
        print(f"   Regression: +{regression:.2f} percentage points")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
