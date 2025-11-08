"""
Baseline Evaluation: No Restoration (Direct Degraded Input)

Evaluasi degraded images langsung tanpa restorasi untuk mendapatkan:
- PSNR (degraded vs clean)
- SSIM (degraded vs clean)
- CER (degraded → recognizer)
- WER (degraded → recognizer)

Author: Experiment for Table Baseline
Date: 2025-11-04
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import editdistance

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

# Disable GPU for evaluation (optional, use CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # atau '0' sesuai GPU yang tersedia

def read_charlist(path):
    """Load character list"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def decode_ctc_predictions(logits, charset):
    """Manual CTC decode - consistent with training script"""
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
        decoded = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
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

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate"""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)

def calculate_wer(ground_truth, prediction):
    """Calculate Word Error Rate"""
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = editdistance.eval(gt_words, pred_words)
    return distance / len(gt_words)

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord - same as training script"""
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
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (H, W, C) → (W, H, C)
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

def evaluate_baseline_no_restoration(
    tfrecord_path,
    charset_path,
    recognizer_weights,
    batch_size=2,
    output_dir='dual_modal_gan/checkpoints/baseline_no_restoration'
):
    """
    Evaluate baseline: degraded images without restoration
    
    Returns:
        dict: metrics (PSNR, SSIM, CER, WER)
    """
    
    print("="*80)
    print("BASELINE EVALUATION: NO RESTORATION (DEGRADED INPUT ONLY)")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    
    # Load charset
    charset = read_charlist(charset_path)
    vocab_size = len(charset) + 1
    print(f"\nCharset loaded: {vocab_size} characters (including blank)")
    
    # Load test dataset
    print(f"\nLoading test dataset from: {tfrecord_path}")
    test_dataset, test_size = create_test_dataset(tfrecord_path, batch_size)
    print(f"Test set size: {test_size} samples")
    
    # Load frozen recognizer
    print(f"\nLoading frozen recognizer from: {recognizer_weights}")
    recognizer = load_frozen_recognizer(
        weights_path=recognizer_weights,
        charset_size=vocab_size - 1,  # vocab_size includes blank, charset_size doesn't
        return_feature_map=False
    )
    print("Recognizer loaded successfully")
    
    # Metrics accumulators
    all_psnr = []
    all_ssim = []
    all_cer = []
    all_wer = []
    
    print(f"\n{'='*80}")
    print("EVALUATING TEST SET...")
    print(f"{'='*80}\n")
    
    batch_count = 0
    sample_count = 0
    
    for degraded_images, clean_images, labels in tqdm(test_dataset, desc="Evaluating"):
        batch_count += 1
        current_batch_size = degraded_images.shape[0]
        sample_count += current_batch_size
        
        # =====================================================
        # 1. VISUAL METRICS: PSNR & SSIM (degraded vs clean)
        # =====================================================
        # Images are in [0, 1] range (from TFRecord)
        psnr_batch = tf.image.psnr(clean_images, degraded_images, max_val=1.0).numpy()
        ssim_batch = tf.image.ssim(clean_images, degraded_images, max_val=1.0).numpy()
        
        all_psnr.extend(psnr_batch.tolist())
        all_ssim.extend(ssim_batch.tolist())
        
        # =====================================================
        # 2. TEXT METRICS: CER & WER (degraded → recognizer)
        # =====================================================
        # Run recognizer on degraded images (NO restoration)
        recognizer_output = recognizer(degraded_images, training=False)
        
        # Extract logits
        if isinstance(recognizer_output, (list, tuple)):
            degraded_logits = recognizer_output[0]
        else:
            degraded_logits = recognizer_output
        
        # Decode predictions
        degraded_predictions = decode_ctc_predictions(degraded_logits.numpy(), charset)
        
        # Convert labels to text
        labels_np = labels.numpy()
        batch_cer = []
        batch_wer = []
        
        for i in range(current_batch_size):
            # Decode ground truth label
            gt_text = decode_label(labels_np[i], charset)
            pred_text = degraded_predictions[i]
            
            # Calculate CER and WER
            cer = calculate_cer(gt_text, pred_text)
            wer = calculate_wer(gt_text, pred_text)
            
            batch_cer.append(cer)
            batch_wer.append(wer)
        
        all_cer.extend(batch_cer)
        all_wer.extend(batch_wer)
    
    # =====================================================
    # 3. COMPUTE STATISTICS
    # =====================================================
    psnr_mean = np.mean(all_psnr)
    psnr_std = np.std(all_psnr, ddof=1)
    psnr_ci = 1.96 * psnr_std / np.sqrt(len(all_psnr))
    
    ssim_mean = np.mean(all_ssim)
    ssim_std = np.std(all_ssim, ddof=1)
    ssim_ci = 1.96 * ssim_std / np.sqrt(len(all_ssim))
    
    cer_mean = np.mean(all_cer)
    cer_std = np.std(all_cer, ddof=1)
    
    wer_mean = np.mean(all_wer)
    wer_std = np.std(all_wer, ddof=1)
    
    # =====================================================
    # 4. PRINT RESULTS
    # =====================================================
    print(f"\n{'='*80}")
    print("BASELINE EVALUATION RESULTS (NO RESTORATION)")
    print(f"{'='*80}")
    print(f"Test set size: {sample_count} samples")
    print(f"\nVISUAL QUALITY (Degraded vs Clean):")
    print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB (95% CI: [{psnr_mean-psnr_ci:.2f}, {psnr_mean+psnr_ci:.2f}])")
    print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f} (95% CI: [{ssim_mean-ssim_ci:.4f}, {ssim_mean+ssim_ci:.4f}])")
    print(f"\nTEXT RECOGNITION (Degraded → HTR):")
    print(f"  CER:  {cer_mean*100:.2f}% ± {cer_std*100:.2f}%")
    print(f"  WER:  {wer_mean*100:.2f}% ± {wer_std*100:.2f}%")
    print(f"{'='*80}\n")
    
    # =====================================================
    # 5. SAVE RESULTS
    # =====================================================
    results = {
        "experiment_name": "baseline_no_restoration",
        "description": "Baseline evaluation: degraded images without restoration",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "tfrecord_path": tfrecord_path,
            "test_size": test_size,
            "batch_size": batch_size
        },
        "metrics": {
            "psnr": {
                "mean": float(psnr_mean),
                "std": float(psnr_std),
                "ci_95": float(psnr_ci),
                "n": len(all_psnr)
            },
            "ssim": {
                "mean": float(ssim_mean),
                "std": float(ssim_std),
                "ci_95": float(ssim_ci),
                "n": len(all_ssim)
            },
            "cer": {
                "mean": float(cer_mean),
                "std": float(cer_std),
                "n": len(all_cer)
            },
            "wer": {
                "mean": float(wer_mean),
                "std": float(wer_std),
                "n": len(all_wer)
            }
        },
        "for_table": {
            "PSNR_dB": round(psnr_mean, 2),
            "SSIM": round(ssim_mean, 3),
            "CER_percent": round(cer_mean * 100, 1),
            "WER_percent": round(wer_mean * 100, 1)
        }
    }
    
    # Save to JSON
    results_path = os.path.join(output_dir, 'metrics', 'baseline_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {results_path}")
    
    # Print for table
    print(f"\n{'='*80}")
    print("FOR TABLE (copy-paste ready):")
    print(f"{'='*80}")
    print(f"Tanpa Perbaikan (Terdegradasi) & {results['for_table']['PSNR_dB']} & {results['for_table']['SSIM']} & - & {results['for_table']['CER_percent']} & {results['for_table']['WER_percent']} \\\\")
    print(f"{'='*80}\n")
    
    return results

if __name__ == '__main__':
    # Configuration
    TFRECORD_PATH = 'dual_modal_gan/data/dataset_gan.tfrecord'
    CHARSET_PATH = 'real_data_preparation/real_data_charlist.txt'
    RECOGNIZER_WEIGHTS = '/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5'
    BATCH_SIZE = 2
    
    # Run evaluation
    results = evaluate_baseline_no_restoration(
        tfrecord_path=TFRECORD_PATH,
        charset_path=CHARSET_PATH,
        recognizer_weights=RECOGNIZER_WEIGHTS,
        batch_size=BATCH_SIZE
    )
    
    print("\n✅ Baseline evaluation completed successfully!")
