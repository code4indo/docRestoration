#!/bin/bash
# Simple evaluation using V4 training script's built-in functions

cd /home/lambda_one/tesis/GAN-HTR-ORI/docRestoration

CUDA_VISIBLE_DEVICES=1 poetry run python -c "
import sys
import numpy as np
import tensorflow as tf
import jiwer

# Import from V4 script
sys.path.insert(0, 'scripts')
from train_transformer_v4_conservative import (
    read_charlist, build_model, parse_tfrecord,
    IMG_WIDTH, IMG_HEIGHT, MAX_LABEL_LENGTH, BATCH_SIZE
)

CHARSET_PATH = '/home/lambda_one/tesis/GAN-HTR-ORI/real_data_preparation/real_data_charlist.txt'
TFRECORD_PATH = '/home/lambda_one/tesis/GAN-HTR-ORI/real_data_final_fixed_v2.tfrecord'
MODEL_PATH = 'HTR/v4_conservative_20251106_072619/best_model.weights.h5'

print('='*80)
print('🔍 V4 Model Evaluation')
print('='*80)

# Load charset
print(f'\nLoading charset...')
charset = read_charlist(CHARSET_PATH)
print(f'✓ Loaded {len(charset)} characters')

# Create char mapping
char_to_num = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=list(charset), values=list(range(len(charset))), 
        key_dtype=tf.string, value_dtype=tf.int64
    ), default_value=-1
)

# Load and parse dataset
print(f'\nLoading dataset...')
dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
dataset = dataset.map(lambda x: parse_tfrecord(x, char_to_num), num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Count samples
total_samples = sum(1 for _ in tf.data.TFRecordDataset(TFRECORD_PATH))
train_size = int(total_samples * 0.8)
val_size = total_samples - train_size
print(f'✓ Total: {total_samples}, Train: {train_size}, Val: {val_size}')

# Skip to validation set
val_dataset = dataset.skip(train_size // BATCH_SIZE)

# Load model
print(f'\nLoading model...')
model = build_model(len(charset), num_transformer_layers=6)
model.load_weights(MODEL_PATH)
print('✓ Model loaded')

# CTC decode function
def ctc_decode_predictions(preds, charset):
    results = []
    for pred in preds:
        indices = np.argmax(pred, axis=-1)
        decoded = []
        prev = None
        for idx in indices:
            if idx != len(charset) and idx != prev and idx > 0:
                decoded.append(charset[idx - 1])
            prev = idx
        results.append(''.join(decoded))
    return results

# Evaluate
print(f'\n📊 Evaluating on validation set ({val_size} samples)...')
all_predictions = []
all_ground_truths = []

for batch_images, batch_labels in val_dataset:
    # Predict
    preds = model.predict(batch_images, verbose=0)
    pred_texts = ctc_decode_predictions(preds, charset)
    
    # Ground truth
    for label in batch_labels:
        label_indices = [int(idx) - 1 for idx in label.numpy() if idx > 0]
        gt = ''.join([charset[idx] for idx in label_indices if 0 <= idx < len(charset)])
        all_ground_truths.append(gt)
    
    all_predictions.extend(pred_texts)

# Calculate CER
cer = jiwer.cer(all_ground_truths, all_predictions)
wer = jiwer.wer(all_ground_truths, all_predictions)

print(f'\n' + '='*80)
print(f'�� RESULTS')
print(f'='*80)
print(f'Character Error Rate (CER): {cer*100:.2f}%')
print(f'Word Error Rate (WER): {wer*100:.2f}%')
print(f'='*80)

# Show examples
print(f'\n📝 Sample Predictions (first 5):')
for i in range(min(5, len(all_predictions))):
    print(f'\n{i+1}. GT: {all_ground_truths[i][:80]}')
    print(f'   PD: {all_predictions[i][:80]}')
    sample_cer = jiwer.cer([all_ground_truths[i]], [all_predictions[i]])
    print(f'   CER: {sample_cer*100:.2f}%')

# Check target
print(f'\n' + '='*80)
if cer * 100 < 31:
    print(f'✅ SUCCESS! CER {cer*100:.2f}% < 31% - Target achieved!')
else:
    print(f'❌ CER {cer*100:.2f}% >= 31% - Target not met')
    print(f'   (Baseline: 33.72%, Target: 28-30%)')
print(f'='*80)
"
