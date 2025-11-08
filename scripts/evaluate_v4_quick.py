#!/usr/bin/env python3
"""Quick evaluation of V4 model to get CER"""
import sys
import os
import json
import numpy as np
import tensorflow as tf
import jiwer

# Paths
MODEL_DIR = "HTR/v4_conservative_20251106_072619"
MODEL_PATH = f"{MODEL_DIR}/best_model.weights.h5"
CHARSET_PATH = "/home/lambda_one/tesis/GAN-HTR-ORI/real_data_preparation/real_data_charlist.txt"
TFRECORD_PATH = "/home/lambda_one/tesis/GAN-HTR-ORI/real_data_final_fixed_v2.tfrecord"

# Constants from V4
IMG_WIDTH = 1024
IMG_HEIGHT = 128
MAX_LABEL_LENGTH = 128
BATCH_SIZE = 32

def read_charlist(filepath):
    """Read charset from file - match V4 training behavior"""
    with open(filepath, 'r', encoding='utf-8') as f:
        chars = [line.rstrip('\n') for line in f]  # Keep empty lines, only strip newline
    return chars

def parse_tfrecord(example_proto, charset, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, max_label_length=MAX_LABEL_LENGTH):
    """Parse TFRecord - support both legacy and new format"""
    # Try new format first
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
    }
    
    try:
        features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_png(features['image'], channels=1)
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.cast(image, tf.float32) / 255.0
        
        label_text = features['label'].numpy().decode('utf-8') if hasattr(features['label'], 'numpy') else features['label']
        label_indices = [charset.index(char) + 1 if char in charset else 0 for char in label_text]
        label_indices = label_indices[:max_label_length]
        label_indices += [0] * (max_label_length - len(label_indices))
        label = tf.constant(label_indices, dtype=tf.float32)
        
        return image, label, label_text
    except:
        # Legacy format
        feature_description_legacy = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/label': tf.io.FixedLenFeature([], tf.string),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
        }
        features = tf.io.parse_single_example(example_proto, feature_description_legacy)
        image = tf.io.decode_png(features['image/encoded'], channels=1)
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.cast(image, tf.float32) / 255.0
        
        label_text = features['image/label'].numpy().decode('utf-8') if hasattr(features['image/label'], 'numpy') else features['image/label']
        label_indices = [charset.index(char) + 1 if char in charset else 0 for char in label_text]
        label_indices = label_indices[:max_label_length]
        label_indices += [0] * (max_label_length - len(label_indices))
        label = tf.constant(label_indices, dtype=tf.float32)
        
        return image, label, label_text

def decode_prediction(pred, charset):
    """Decode CTC prediction to text"""
    # Argmax to get predicted indices
    pred_indices = np.argmax(pred, axis=-1)[0]
    
    # CTC decode: remove blank (108) and consecutive duplicates
    decoded = []
    prev = None
    for idx in pred_indices:
        if idx != len(charset) and idx != prev:  # Not blank and not duplicate
            if idx > 0:  # Valid character index
                decoded.append(charset[idx - 1])
        prev = idx
    
    return ''.join(decoded)

def main():
    print("="*80)
    print("🔍 V4 Model Evaluation - Quick CER Check")
    print("="*80)
    
    # Load charset
    print(f"\nLoading charset from {CHARSET_PATH}...")
    charset = read_charlist(CHARSET_PATH)
    print(f"✓ Loaded {len(charset)} characters")
    
    # Load model (need to rebuild architecture first)
    print(f"\nRebuilding model architecture...")
    from train_transformer_v4_conservative import build_model
    model = build_model(len(charset), num_transformer_layers=6)
    
    print(f"Loading weights from {MODEL_PATH}...")
    model.load_weights(MODEL_PATH)
    print("✓ Model loaded")
    
    # Load validation dataset
    print(f"\nLoading dataset from {TFRECORD_PATH}...")
    dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    
    # Parse and split
    all_data = []
    for record in dataset:
        try:
            img, label, text = parse_tfrecord(record, charset)
            all_data.append((img.numpy(), label.numpy(), text))
        except Exception as e:
            continue
    
    print(f"✓ Loaded {len(all_data)} samples")
    
    # Split 80/20
    split_idx = int(len(all_data) * 0.8)
    val_data = all_data[split_idx:]
    print(f"✓ Validation samples: {len(val_data)}")
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("📊 Evaluating on validation set...")
    print("="*80)
    
    predictions = []
    ground_truths = []
    
    for i, (img, label, text) in enumerate(val_data):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(val_data)}", end='\r')
        
        # Predict
        img_batch = np.expand_dims(img, axis=0)
        pred = model.predict(img_batch, verbose=0)
        pred_text = decode_prediction(pred, charset)
        
        predictions.append(pred_text)
        ground_truths.append(text)
    
    print(f"Progress: {len(val_data)}/{len(val_data)}")
    
    # Calculate CER
    cer = jiwer.cer(ground_truths, predictions)
    wer = jiwer.wer(ground_truths, predictions)
    
    print("\n" + "="*80)
    print("📈 RESULTS")
    print("="*80)
    print(f"Character Error Rate (CER): {cer*100:.2f}%")
    print(f"Word Error Rate (WER): {wer*100:.2f}%")
    print("="*80)
    
    # Show some examples
    print("\n📝 Sample Predictions (first 10):")
    print("-"*80)
    for i in range(min(10, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"  Ground Truth: {ground_truths[i]}")
        print(f"  Prediction:   {predictions[i]}")
        sample_cer = jiwer.cer([ground_truths[i]], [predictions[i]])
        print(f"  CER: {sample_cer*100:.2f}%")
    
    # Save results
    results = {
        "model_path": MODEL_PATH,
        "validation_samples": len(val_data),
        "cer": float(cer),
        "wer": float(wer),
        "cer_percentage": float(cer * 100),
        "wer_percentage": float(wer * 100)
    }
    
    results_path = f"{MODEL_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Check if target met
    print("\n" + "="*80)
    if cer * 100 < 31:
        print("✅ SUCCESS! CER < 31% - Target achieved!")
    else:
        print(f"❌ Target not met. CER {cer*100:.2f}% >= 31%")
    print("="*80)

if __name__ == '__main__':
    main()
