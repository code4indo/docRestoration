#!/usr/bin/env python3
"""
Evaluate CER (Character Error Rate) for Baseline Models

Evaluates all 3 baseline models on HTR task to compare with main GAN-HTR model:
1. Plain U-Net (no GAN, no HTR)
2. Standard GAN (with adversarial, no HTR)
3. CTC-Only (with HTR loss, no adversarial)

Compares with main GAN-HTR CER improvement (58.47%)
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.recognizer import load_frozen_recognizer


def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord example - matches format in dual_modal_gan/data/dataset_gan.tfrecord"""
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
    
    # Degraded image
    degraded_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_shape)
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (H,W,C) → (W,H,C)
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])
    
    # Clean image
    clean_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (H,W,C) → (W,H,C)
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])
    
    # Label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int32)
    label = tf.reshape(label, label_shape)
    label_length = label_shape[0]
    
    return degraded_image, clean_image, label, label_length


def load_dataset(tfrecord_path, batch_size=4):
    """Load dataset from TFRecord"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_charlist(charlist_path):
    """Load character list"""
    with open(charlist_path, 'r', encoding='utf-8') as f:
        charlist = [line.strip() for line in f if line.strip()]
    return charlist


def decode_label(label_indices, charlist):
    """Decode label indices to text"""
    # Remove padding and blank tokens
    filtered = [idx for idx in label_indices if 0 <= idx < len(charlist)]
    return ''.join([charlist[idx] for idx in filtered])


def calculate_cer(pred_text, true_text):
    """Calculate Character Error Rate using Levenshtein distance"""
    if len(true_text) == 0:
        return 1.0 if len(pred_text) > 0 else 0.0
    
    # Initialize distance matrix
    d = np.zeros((len(pred_text) + 1, len(true_text) + 1), dtype=np.int32)
    
    for i in range(len(pred_text) + 1):
        d[i][0] = i
    for j in range(len(true_text) + 1):
        d[0][j] = j
    
    # Calculate Levenshtein distance
    for i in range(1, len(pred_text) + 1):
        for j in range(1, len(true_text) + 1):
            if pred_text[i-1] == true_text[j-1]:
                cost = 0
            else:
                cost = 1
            
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    
    return d[len(pred_text)][len(true_text)] / len(true_text)


def load_baseline_generator(checkpoint_dir, baseline_type):
    """Load generator from baseline checkpoint"""
    print(f"\n{'='*80}")
    print(f"Loading {baseline_type} generator...")
    print(f"{'='*80}")
    
    # Create generator
    generator = unet()
    
    # Find checkpoint
    if baseline_type == "unet":
        weight_pattern = f"{checkpoint_dir}/generator.weights.h5"
    elif baseline_type == "standard_gan":
        weight_pattern = f"{checkpoint_dir}/generator.weights.h5"
    elif baseline_type == "ctc_only":
        weight_pattern = f"{checkpoint_dir}/generator.weights.h5"
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    # Check if checkpoint exists
    if not os.path.exists(weight_pattern):
        raise FileNotFoundError(f"Checkpoint not found: {weight_pattern}")
    
    # Load weights
    print(f"Loading weights from: {weight_pattern}")
    generator.load_weights(weight_pattern)
    print("✅ Generator loaded successfully!")
    
    return generator


def find_latest_baseline_checkpoint(baseline_type):
    """Find latest baseline checkpoint directory"""
    patterns = {
        "unet": "dual_modal_gan/outputs/baseline_unet_*",
        "standard_gan": "dual_modal_gan/outputs/baseline_standard_gan_*",
        "ctc_only": "dual_modal_gan/outputs/baseline_ctc_only_*"
    }
    
    pattern = patterns.get(baseline_type)
    if not pattern:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"No checkpoint found for {baseline_type}")
    
    # Get latest by modification time
    latest_dir = max(dirs, key=os.path.getmtime)
    checkpoint_dir = os.path.join(latest_dir, "checkpoints", "best_model")
    
    return checkpoint_dir


def evaluate_baseline_cer(baseline_type, dataset_path, charlist_path, recognizer_weights, vocab_size):
    """Evaluate CER for a baseline model"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {baseline_type.upper()}")
    print(f"{'='*80}\n")
    
    # Find checkpoint
    checkpoint_dir = find_latest_baseline_checkpoint(baseline_type)
    print(f"Checkpoint: {checkpoint_dir}\n")
    
    # Load models
    generator = load_baseline_generator(checkpoint_dir, baseline_type)
    recognizer = load_frozen_recognizer(recognizer_weights, charset_size=vocab_size - 1)
    charlist = load_charlist(charlist_path)
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path, batch_size=1)  # Batch size 1 for easier processing
    
    # Take first 100 samples for evaluation (same as main model)
    dataset = dataset.take(100)
    
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80 + "\n")
    
    # Evaluation metrics
    degraded_cers = []
    generated_cers = []
    sample_count = 0
    
    for batch in tqdm(dataset, desc="Evaluating"):
        degraded_batch, clean_batch, labels, label_lengths = batch
        
        # Generate enhanced images
        generated_batch = generator(degraded_batch, training=False)
        
        # Get HTR predictions
        degraded_pred = recognizer(degraded_batch, training=False)
        generated_pred = recognizer(generated_batch, training=False)
        
        # Decode predictions (greedy decoding)
        degraded_pred_indices = tf.argmax(degraded_pred, axis=-1).numpy()
        generated_pred_indices = tf.argmax(generated_pred, axis=-1).numpy()
        
        # Process each sample in batch
        for i in range(len(labels)):
            # Get true label
            label = labels[i].numpy()
            label_length = label_lengths[i].numpy()
            true_label = label[:label_length]
            true_text = decode_label(true_label, charlist)
            
            # Decode predictions
            degraded_text = decode_label(degraded_pred_indices[i], charlist)
            generated_text = decode_label(generated_pred_indices[i], charlist)
            
            # Calculate CER
            degraded_cer = calculate_cer(degraded_text, true_text)
            generated_cer = calculate_cer(generated_text, true_text)
            
            degraded_cers.append(degraded_cer)
            generated_cers.append(generated_cer)
            
            sample_count += 1
            
            # Print first 5 samples
            if sample_count <= 5:
                print(f"\nSample {sample_count}:")
                print(f"  True:      '{true_text}'")
                print(f"  Degraded:  '{degraded_text}' (CER: {degraded_cer:.4f})")
                print(f"  Generated: '{generated_text}' (CER: {generated_cer:.4f})")
    
    # Calculate average CER
    avg_degraded_cer = np.mean(degraded_cers) * 100  # Convert to percentage
    avg_generated_cer = np.mean(generated_cers) * 100
    cer_improvement = avg_degraded_cer - avg_generated_cer
    improvement_percentage = (cer_improvement / avg_degraded_cer) * 100
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Samples evaluated: {sample_count}")
    print(f"\nDegraded CER:  {avg_degraded_cer:.2f}%")
    print(f"Generated CER: {avg_generated_cer:.2f}%")
    print(f"CER Reduction: {cer_improvement:.2f} percentage points")
    print(f"CER Improvement: {improvement_percentage:.2f}%")
    print("="*80 + "\n")
    
    return {
        "baseline_type": baseline_type,
        "samples_evaluated": sample_count,
        "degraded_cer": float(avg_degraded_cer),
        "generated_cer": float(avg_generated_cer),
        "cer_reduction": float(cer_improvement),
        "cer_improvement_percentage": float(improvement_percentage),
        "checkpoint_dir": checkpoint_dir
    }


def create_comparison_table(results, main_model_result):
    """Create comparison table with all baselines and main model"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CER COMPARISON")
    print("="*80 + "\n")
    
    # Header
    print(f"{'Method':<25} {'Degraded CER':<15} {'Generated CER':<15} {'Improvement':<15}")
    print("-"*80)
    
    # Main model
    print(f"{'GAN-HTR (Main Model)':<25} "
          f"{main_model_result['degraded_cer']:<15.2f} "
          f"{main_model_result['generated_cer']:<15.2f} "
          f"{main_model_result['improvement']:<15.2f}%")
    print("-"*80)
    
    # Baselines
    baseline_names = {
        "unet": "Baseline 1: Plain U-Net",
        "standard_gan": "Baseline 2: Standard GAN",
        "ctc_only": "Baseline 3: CTC-Only"
    }
    
    for result in results:
        name = baseline_names.get(result['baseline_type'], result['baseline_type'])
        print(f"{name:<25} "
              f"{result['degraded_cer']:<15.2f} "
              f"{result['generated_cer']:<15.2f} "
              f"{result['cer_improvement_percentage']:<15.2f}%")
    
    print("-"*80 + "\n")
    
    # Gap analysis
    print("="*80)
    print("GAP ANALYSIS (vs Main Model)")
    print("="*80 + "\n")
    
    print(f"{'Method':<25} {'Improvement Gap':<20} {'Status':<20}")
    print("-"*80)
    
    for result in results:
        name = baseline_names.get(result['baseline_type'], result['baseline_type'])
        gap = result['cer_improvement_percentage'] - main_model_result['improvement']
        
        if gap >= 0:
            status = f"✅ {gap:.2f}% BETTER"
        else:
            status = f"❌ {abs(gap):.2f}% WORSE"
        
        print(f"{name:<25} {gap:>18.2f}% {status:<20}")
    
    print("-"*80 + "\n")


def save_results(results, main_model_result, output_path):
    """Save evaluation results to JSON"""
    
    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "main_model": main_model_result,
        "baselines": results,
        "analysis": {
            "best_baseline": max(results, key=lambda x: x['cer_improvement_percentage'])['baseline_type'],
            "worst_baseline": min(results, key=lambda x: x['cer_improvement_percentage'])['baseline_type'],
            "main_model_rank": None  # Will be calculated
        }
    }
    
    # Calculate main model rank
    all_improvements = [main_model_result['improvement']] + \
                      [r['cer_improvement_percentage'] for r in results]
    all_improvements_sorted = sorted(all_improvements, reverse=True)
    main_model_rank = all_improvements_sorted.index(main_model_result['improvement']) + 1
    summary['analysis']['main_model_rank'] = main_model_rank
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models on CER")
    parser.add_argument('--baseline', type=str, choices=['unet', 'standard_gan', 'ctc_only', 'all'],
                       default='all', help="Baseline to evaluate")
    parser.add_argument('--dataset', type=str, 
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help="Path to TFRecord dataset")
    parser.add_argument('--charlist', type=str,
                       default='real_data_preparation/real_data_charlist.txt',
                       help="Path to character list")
    parser.add_argument('--recognizer_weights', type=str,
                       default='models/best_htr_recognizer/best_model.weights.h5',
                       help="Path to HTR recognizer weights")
    parser.add_argument('--vocab_size', type=int, default=109,
                       help="Vocabulary size (108 chars + CTC blank)")
    parser.add_argument('--output', type=str,
                       default='dual_modal_gan/outputs/baseline_analysis/baseline_cer_evaluation.json',
                       help="Output JSON path")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BASELINE CER EVALUATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Charlist: {args.charlist}")
    print(f"Recognizer: {args.recognizer_weights}")
    print(f"Vocab size: {args.vocab_size}")
    print("="*80 + "\n")
    
    # Determine which baselines to evaluate
    if args.baseline == 'all':
        baselines_to_eval = ['unet', 'standard_gan', 'ctc_only']
    else:
        baselines_to_eval = [args.baseline]
    
    # Evaluate each baseline
    results = []
    for baseline_type in baselines_to_eval:
        try:
            result = evaluate_baseline_cer(
                baseline_type=baseline_type,
                dataset_path=args.dataset,
                charlist_path=args.charlist,
                recognizer_weights=args.recognizer_weights,
                vocab_size=args.vocab_size
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error evaluating {baseline_type}: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Main model result (from previous evaluation)
    main_model_result = {
        "name": "GAN-HTR (Main Model)",
        "degraded_cer": 80.96,
        "generated_cer": 33.62,
        "improvement": 58.47
    }
    
    # Create comparison table
    if results:
        create_comparison_table(results, main_model_result)
        save_results(results, main_model_result, args.output)
    else:
        print("\n❌ No baseline evaluations succeeded.\n")


if __name__ == "__main__":
    main()
