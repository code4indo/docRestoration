#!/usr/bin/env python3
"""
Standalone CER/WER Evaluation Script
Evaluates text recognition quality of enhanced images without retraining.
"""

import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import editdistance

# Add project root to path
script_dir = Path(__file__).parent
dual_modal_root = script_dir.parent
sys.path.insert(0, str(dual_modal_root / 'src'))

from models.generator import build_generator
from models.recognizer import build_recognizer
from data.data_loader import create_dataset


def decode_label(label_ids, charset):
    """
    Decode label IDs to text string.
    Removes blank token (0) and handles CTC duplicates.
    """
    decoded_chars = []
    prev_id = -1
    
    for label_id in label_ids:
        if label_id == 0:  # Blank token
            prev_id = -1
            continue
        if label_id == prev_id:  # CTC duplicate
            continue
        if label_id > 0 and label_id <= len(charset):
            decoded_chars.append(charset[label_id - 1])
            prev_id = label_id
    
    return ''.join(decoded_chars)


def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate using Levenshtein distance."""
    if len(ground_truth) == 0 and len(prediction) == 0:
        return 0.0
    if len(ground_truth) == 0:
        return 1.0
    
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)


def calculate_wer(ground_truth, prediction):
    """Calculate Word Error Rate using Levenshtein distance."""
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    
    if len(gt_words) == 0 and len(pred_words) == 0:
        return 0.0
    if len(gt_words) == 0:
        return 1.0
    
    distance = editdistance.eval(gt_words, pred_words)
    return distance / len(gt_words)


def load_charset(charlist_path):
    """Load character set from file."""
    with open(charlist_path, 'r', encoding='utf-8') as f:
        charset = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded charset: {len(charset)} characters")
    return charset


def evaluate_model(args):
    """Main evaluation function."""
    print("\n" + "="*80)
    print("🔍 CER/WER Evaluation Script")
    print("="*80)
    
    # Check checkpoint exists
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find latest checkpoint
    checkpoints = sorted(checkpoint_dir.glob("generator_epoch_*.weights.h5"))
    if not checkpoints:
        print(f"❌ No generator checkpoints found in {checkpoint_dir}")
        return
    
    latest_checkpoint = checkpoints[-1]
    epoch_num = int(latest_checkpoint.stem.split('_')[-1])
    print(f"📂 Latest checkpoint: {latest_checkpoint.name} (Epoch {epoch_num})")
    
    # Load charset
    charset = load_charset(args.charlist_path)
    
    # Load dataset
    print(f"\n📊 Loading validation dataset: {args.tfrecord_path}")
    dataset = create_dataset(
        tfrecord_path=args.tfrecord_path,
        batch_size=args.batch_size,
        is_training=False,
        buffer_size=100
    )
    
    # Build models
    print("\n🏗️  Building models...")
    generator = build_generator(
        input_shape=(64, 1024, 1),
        num_residual_blocks=6
    )
    print(f"   Generator: {generator.count_params():,} parameters")
    
    recognizer = build_recognizer(
        input_shape=(64, 1024, 1),
        num_classes=len(charset) + 1,
        model_name="htr_recognizer"
    )
    print(f"   Recognizer: {recognizer.count_params():,} parameters")
    
    # Load generator weights
    print(f"\n⚙️  Loading generator weights: {latest_checkpoint.name}")
    try:
        generator.load_weights(str(latest_checkpoint))
        print("   ✅ Generator weights loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load generator weights: {e}")
        return
    
    # Load recognizer weights
    recognizer_checkpoint = Path(args.recognizer_checkpoint)
    if not recognizer_checkpoint.exists():
        print(f"❌ Recognizer checkpoint not found: {recognizer_checkpoint}")
        return
    
    print(f"\n⚙️  Loading recognizer weights: {recognizer_checkpoint.name}")
    try:
        recognizer.load_weights(str(recognizer_checkpoint))
        print("   ✅ Recognizer weights loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load recognizer weights: {e}")
        return
    
    # Run evaluation
    print(f"\n🚀 Running evaluation on {args.num_batches} batches...")
    print("-" * 80)
    
    all_cers = []
    all_wers = []
    all_cers_clean = []  # CER of clean images (baseline)
    all_wers_clean = []  # WER of clean images (baseline)
    
    sample_results = []  # Store sample comparisons
    
    batch_count = 0
    for batch_data in dataset:
        if batch_count >= args.num_batches:
            break
        
        noisy_images, clean_images, labels = batch_data
        
        # Generate enhanced images
        generated_images = generator(noisy_images, training=False)
        
        # Get predictions from recognizer
        clean_logits = recognizer(clean_images, training=False)
        generated_logits = recognizer(generated_images, training=False)
        
        # Greedy decoding
        clean_predictions = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
        generated_predictions = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)
        
        # Convert to numpy for processing
        labels_np = labels.numpy()
        clean_pred_np = clean_predictions.numpy()
        generated_pred_np = generated_predictions.numpy()
        
        # Calculate CER/WER for each sample in batch
        for i in range(labels_np.shape[0]):
            gt_text = decode_label(labels_np[i], charset)
            clean_text = decode_label(clean_pred_np[i], charset)
            generated_text = decode_label(generated_pred_np[i], charset)
            
            # CER/WER: ground truth vs clean
            cer_clean = calculate_cer(gt_text, clean_text)
            wer_clean = calculate_wer(gt_text, clean_text)
            
            # CER/WER: ground truth vs generated
            cer_generated = calculate_cer(gt_text, generated_text)
            wer_generated = calculate_wer(gt_text, generated_text)
            
            all_cers_clean.append(cer_clean)
            all_wers_clean.append(wer_clean)
            all_cers.append(cer_generated)
            all_wers.append(wer_generated)
            
            # Store sample for detailed report
            if len(sample_results) < args.num_samples:
                sample_results.append({
                    'batch': batch_count,
                    'sample': i,
                    'ground_truth': gt_text,
                    'clean_prediction': clean_text,
                    'generated_prediction': generated_text,
                    'cer_clean': cer_clean,
                    'wer_clean': wer_clean,
                    'cer_generated': cer_generated,
                    'wer_generated': wer_generated,
                    'improvement_cer': cer_clean - cer_generated,
                    'improvement_wer': wer_clean - wer_generated
                })
        
        batch_count += 1
        if batch_count % 5 == 0:
            print(f"   Processed {batch_count}/{args.num_batches} batches...")
    
    # Calculate overall statistics
    mean_cer_clean = np.mean(all_cers_clean)
    mean_wer_clean = np.mean(all_wers_clean)
    mean_cer_generated = np.mean(all_cers)
    mean_wer_generated = np.mean(all_wers)
    
    std_cer_clean = np.std(all_cers_clean)
    std_wer_clean = np.std(all_wers_clean)
    std_cer_generated = np.std(all_cers)
    std_wer_generated = np.std(all_wers)
    
    improvement_cer = mean_cer_clean - mean_cer_generated
    improvement_wer = mean_wer_clean - mean_wer_generated
    
    # Print results
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print(f"Checkpoint: {latest_checkpoint.name} (Epoch {epoch_num})")
    print(f"Samples evaluated: {len(all_cers)}")
    print("-" * 80)
    
    print("\n🔹 Clean Images (Baseline):")
    print(f"   CER: {mean_cer_clean:.4f} ± {std_cer_clean:.4f}")
    print(f"   WER: {mean_wer_clean:.4f} ± {std_wer_clean:.4f}")
    
    print("\n🔹 Generated Images (Enhanced):")
    print(f"   CER: {mean_cer_generated:.4f} ± {std_cer_generated:.4f}")
    print(f"   WER: {mean_wer_generated:.4f} ± {std_wer_generated:.4f}")
    
    print("\n🔹 Improvement (Clean → Generated):")
    cer_improvement_pct = (improvement_cer / mean_cer_clean * 100) if mean_cer_clean > 0 else 0
    wer_improvement_pct = (improvement_wer / mean_wer_clean * 100) if mean_wer_clean > 0 else 0
    
    if improvement_cer > 0:
        print(f"   ✅ CER: {improvement_cer:+.4f} ({cer_improvement_pct:+.1f}%)")
    elif improvement_cer < 0:
        print(f"   ❌ CER: {improvement_cer:+.4f} ({cer_improvement_pct:+.1f}%) - DEGRADATION")
    else:
        print(f"   ➖ CER: {improvement_cer:+.4f} (no change)")
    
    if improvement_wer > 0:
        print(f"   ✅ WER: {improvement_wer:+.4f} ({wer_improvement_pct:+.1f}%)")
    elif improvement_wer < 0:
        print(f"   ❌ WER: {improvement_wer:+.4f} ({wer_improvement_pct:+.1f}%) - DEGRADATION")
    else:
        print(f"   ➖ WER: {improvement_wer:+.4f} (no change)")
    
    # Sample comparisons
    print("\n" + "="*80)
    print("📝 SAMPLE COMPARISONS")
    print("="*80)
    
    for idx, sample in enumerate(sample_results[:args.num_samples], 1):
        print(f"\nSample #{idx} (Batch {sample['batch']}, Index {sample['sample']}):")
        print(f"  Ground Truth: {sample['ground_truth'][:80]}")
        print(f"  Clean Pred:   {sample['clean_prediction'][:80]}")
        print(f"  Generated:    {sample['generated_prediction'][:80]}")
        print(f"  CER: Clean={sample['cer_clean']:.4f}, Generated={sample['cer_generated']:.4f}, Δ={sample['improvement_cer']:+.4f}")
        print(f"  WER: Clean={sample['wer_clean']:.4f}, Generated={sample['wer_generated']:.4f}, Δ={sample['improvement_wer']:+.4f}")
    
    # Save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_epoch{epoch_num}_{timestamp}.json"
    
    results = {
        'checkpoint': str(latest_checkpoint),
        'epoch': epoch_num,
        'timestamp': timestamp,
        'num_samples': len(all_cers),
        'metrics': {
            'clean': {
                'cer_mean': float(mean_cer_clean),
                'cer_std': float(std_cer_clean),
                'wer_mean': float(mean_wer_clean),
                'wer_std': float(std_wer_clean)
            },
            'generated': {
                'cer_mean': float(mean_cer_generated),
                'cer_std': float(std_cer_generated),
                'wer_mean': float(mean_wer_generated),
                'wer_std': float(std_wer_generated)
            },
            'improvement': {
                'cer_absolute': float(improvement_cer),
                'cer_percentage': float(cer_improvement_pct),
                'wer_absolute': float(improvement_wer),
                'wer_percentage': float(wer_improvement_pct)
            }
        },
        'samples': sample_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print(f"✅ Results saved to: {output_file}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CER/WER of GAN-HTR model")
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='dual_modal_gan/outputs/checkpoints_final_100',
        help='Directory containing generator checkpoints'
    )
    
    parser.add_argument(
        '--recognizer_checkpoint',
        type=str,
        default='models/best_htr_recognizer/best_model.weights.h5',
        help='Path to recognizer checkpoint'
    )
    
    parser.add_argument(
        '--tfrecord_path',
        type=str,
        default='dual_modal_gan/data/dataset_gan.tfrecord',
        help='Path to validation TFRecord'
    )
    
    parser.add_argument(
        '--charlist_path',
        type=str,
        default='real_data_preparation/real_data_charlist.txt',
        help='Path to character list file'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )
    
    parser.add_argument(
        '--num_batches',
        type=int,
        default=50,
        help='Number of batches to evaluate (default: 50)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of sample comparisons to print (default: 10)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='logbook/cer_evaluations',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(args)


if __name__ == '__main__':
    main()
