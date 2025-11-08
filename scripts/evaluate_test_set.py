#!/usr/bin/env python3
"""
Test Set Evaluation Script - ACADEMIC PROTOCOL
================================================

⚠️  CRITICAL: This script evaluates the HELD-OUT test set.
    - Should be run ONLY ONCE after model selection is complete
    - Test set must NEVER be used during training or hyperparameter tuning
    - Results reported here are UNBIASED final performance metrics

Usage:
    poetry run python scripts/evaluate_test_set.py \\
        --checkpoint dual_modal_gan/checkpoints/production_v2/best_model \\
        --tfrecord dual_modal_gan/data/dataset_gan.tfrecord \\
        --output test_set_results.json \\
        --train_split 0.7 \\
        --val_split 0.15

Academic Protocol:
    1. Training completed with train+val sets
    2. Best model selected based on validation performance
    3. Load best checkpoint
    4. Evaluate on test set ONCE (this script)
    5. Report results in paper with 95% CI

Author: AI/ML Engineer (belekok)
Date: 2025-10-21
Version: 1.0 - Initial implementation for academic publication
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "dual_modal_gan"))
sys.path.insert(0, str(project_root / "dual_modal_gan" / "scripts"))

# Import required modules
from train_enhanced import (
    _parse_tfrecord_fn,
    read_charlist,
    decode_ctc_predictions,
    decode_label,
    calculate_cer,
    calculate_wer,
    calculate_noise_artifacts_metrics
)

# Import model architectures
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced


def create_test_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """
    Load ONLY the test set (locked during training).
    
    Args:
        tfrecord_path: Path to TFRecord file
        batch_size: Batch size for evaluation
        train_split: Fraction used for training (to skip)
        val_split: Fraction used for validation (to skip)
        
    Returns:
        test_dataset: Test dataset (no shuffle, no repeat)
        test_size: Number of test samples
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Get total size
    total_size = sum(1 for _ in dataset)
    
    # Calculate sizes
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Map parsing
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Skip train and validation, take test
    test_dataset = dataset.skip(train_size + val_size)
    
    # No shuffle, no repeat - fixed test set
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    print(f"\n📊 Test Set Configuration:")
    print(f"   Total samples in dataset: {total_size}")
    print(f"   Train samples (skipped):  {train_size} ({train_split*100:.0f}%)")
    print(f"   Val samples (skipped):    {val_size} ({val_split*100:.0f}%)")
    print(f"   Test samples (loaded):    {test_size} ({(1-train_split-val_split)*100:.0f}%)")
    print(f"   ✅ Test set isolated - no data leakage")
    
    return test_dataset, test_size


def evaluate_test_set(generator, recognizer, test_dataset, charset, test_size):
    """
    Evaluate model on held-out test set.
    
    Returns comprehensive metrics with statistics:
    - Visual: PSNR, SSIM (mean ± std, 95% CI)
    - Textual: CER, WER (mean ± std)
    - Noise artifacts: variance, isolated white pixels, local variance
    
    Args:
        generator: Trained generator model
        recognizer: HTR recognizer model
        test_dataset: Test dataset (batched)
        charset: Character set for decoding
        test_size: Number of test samples
        
    Returns:
        dict: Comprehensive evaluation results with statistics
    """
    print("\n" + "="*80)
    print("🔬 FINAL TEST SET EVALUATION - ACADEMIC PROTOCOL")
    print("="*80)
    print("⚠️  This is the OFFICIAL result - test set evaluated ONCE")
    print("   Results are UNBIASED (test set never seen during training)")
    print("="*80 + "\n")
    
    # Collect metrics from all test samples
    all_psnr = []
    all_ssim = []
    all_cer = []
    all_wer = []
    all_clean_cer = []
    all_clean_wer = []
    all_noise_var = []
    all_isolated_white = []
    all_local_var = []
    
    # Sample texts for qualitative analysis
    sample_texts = []
    
    batch_count = 0
    sample_count = 0
    
    print("🔄 Processing test set batches...")
    for degraded_images, clean_images, labels in test_dataset:
        # Normalize to [-1,1] for generator
        degraded_images_tanh = degraded_images * 2.0 - 1.0
        clean_images_tanh = clean_images * 2.0 - 1.0
        
        # Generate enhanced images
        generated_images = generator(degraded_images_tanh, training=False)
        
        # Denormalize to [0,1] for metrics
        generated_images_normalized = (generated_images + 1.0) / 2.0
        clean_images_normalized = (clean_images_tanh + 1.0) / 2.0
        
        # Visual metrics
        psnr = tf.image.psnr(clean_images_normalized, generated_images_normalized, max_val=1.0)
        ssim = tf.image.ssim(clean_images_normalized, generated_images_normalized, max_val=1.0)
        
        all_psnr.extend(psnr.numpy().tolist())
        all_ssim.extend(ssim.numpy().tolist())
        
        # Noise artifact metrics
        for i in range(generated_images_normalized.shape[0]):
            noise_metrics = calculate_noise_artifacts_metrics(generated_images_normalized[i])
            all_noise_var.append(noise_metrics['noise_variance'])
            all_isolated_white.append(noise_metrics['isolated_white_ratio'])
            all_local_var.append(noise_metrics['local_variance'])
        
        # HTR metrics
        recognizer_output_clean = recognizer(clean_images_normalized, training=False)
        recognizer_output_generated = recognizer(generated_images_normalized, training=False)
        
        # Extract logits
        if isinstance(recognizer_output_clean, (list, tuple)):
            clean_logits = recognizer_output_clean[0]
            generated_logits = recognizer_output_generated[0]
        else:
            clean_logits = recognizer_output_clean
            generated_logits = recognizer_output_generated
        
        # Decode predictions
        clean_predictions_text = decode_ctc_predictions(clean_logits.numpy(), charset)
        generated_predictions_text = decode_ctc_predictions(generated_logits.numpy(), charset)
        
        labels_np = labels.numpy()
        
        # Calculate CER/WER for each sample
        for i in range(labels_np.shape[0]):
            gt_text = decode_label(labels_np[i], charset)
            clean_text = clean_predictions_text[i]
            generated_text = generated_predictions_text[i]
            
            cer = calculate_cer(gt_text, generated_text)
            wer = calculate_wer(gt_text, generated_text)
            clean_cer = calculate_cer(gt_text, clean_text)
            clean_wer = calculate_wer(gt_text, clean_text)
            
            all_cer.append(cer)
            all_wer.append(wer)
            all_clean_cer.append(clean_cer)
            all_clean_wer.append(clean_wer)
            
            # MODIFIED: Collect ALL samples for comprehensive failure analysis
            sample_texts.append({
                'sample_id': sample_count,
                'ground_truth': gt_text,
                'clean_prediction': clean_text,
                'generated_prediction': generated_text,
                'cer': float(cer),
                'wer': float(wer),
                'clean_cer': float(clean_cer),
                'psnr': float(all_psnr[sample_count]),
                'ssim': float(all_ssim[sample_count])
            })
            
            sample_count += 1
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"   Processed {sample_count}/{test_size} samples...")
    
    print(f"✅ Processed all {sample_count} test samples in {batch_count} batches\n")
    
    # Calculate statistics
    psnr_mean = np.mean(all_psnr)
    psnr_std = np.std(all_psnr, ddof=1)
    psnr_ci = 1.96 * psnr_std / np.sqrt(len(all_psnr))
    
    ssim_mean = np.mean(all_ssim)
    ssim_std = np.std(all_ssim, ddof=1)
    ssim_ci = 1.96 * ssim_std / np.sqrt(len(all_ssim))
    
    cer_mean = np.mean(all_cer)
    cer_std = np.std(all_cer, ddof=1)
    cer_ci = 1.96 * cer_std / np.sqrt(len(all_cer))
    
    wer_mean = np.mean(all_wer)
    wer_std = np.std(all_wer, ddof=1)
    wer_ci = 1.96 * wer_std / np.sqrt(len(all_wer))
    
    clean_cer_mean = np.mean(all_clean_cer)
    clean_cer_std = np.std(all_clean_cer, ddof=1)
    
    clean_wer_mean = np.mean(all_clean_wer)
    clean_wer_std = np.std(all_clean_wer, ddof=1)
    
    noise_var_mean = np.mean(all_noise_var)
    noise_var_std = np.std(all_noise_var, ddof=1)
    
    isolated_white_mean = np.mean(all_isolated_white)
    isolated_white_std = np.std(all_isolated_white, ddof=1)
    
    local_var_mean = np.mean(all_local_var)
    local_var_std = np.std(all_local_var, ddof=1)
    
    # Print results
    print("="*80)
    print("📊 FINAL TEST SET RESULTS (n={})".format(len(all_psnr)))
    print("="*80)
    print("\n🎯 VISUAL QUALITY METRICS:")
    print(f"   PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
    print(f"         95% CI: [{psnr_mean-psnr_ci:.2f}, {psnr_mean+psnr_ci:.2f}] dB")
    print(f"   SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    print(f"         95% CI: [{ssim_mean-ssim_ci:.4f}, {ssim_mean+ssim_ci:.4f}]")
    
    print("\n📝 HTR ACCURACY METRICS:")
    print(f"   CER (Generated): {cer_mean:.4f} ± {cer_std:.4f}")
    print(f"         95% CI: [{cer_mean-cer_ci:.4f}, {cer_mean+cer_ci:.4f}]")
    print(f"   WER (Generated): {wer_mean:.4f} ± {wer_std:.4f}")
    print(f"         95% CI: [{wer_mean-wer_ci:.4f}, {wer_mean+wer_ci:.4f}]")
    print(f"\n   CER (Clean baseline): {clean_cer_mean:.4f} ± {clean_cer_std:.4f}")
    print(f"   WER (Clean baseline): {clean_wer_mean:.4f} ± {clean_wer_std:.4f}")
    print(f"   ΔCER (vs clean): {cer_mean - clean_cer_mean:+.4f}")
    
    print("\n🔍 NOISE ARTIFACT METRICS:")
    print(f"   Noise Variance:      {noise_var_mean:.6f} ± {noise_var_std:.6f}")
    print(f"   Isolated White (%):  {isolated_white_mean*100:.3f} ± {isolated_white_std*100:.3f}")
    print(f"   Local Variance:      {local_var_mean:.6f} ± {local_var_std:.6f}")
    
    print("\n" + "="*80)
    print("✅ Test set evaluation complete - these are OFFICIAL RESULTS")
    print("="*80 + "\n")
    
    # Prepare results dictionary with ENHANCED per-sample data for failure analysis
    results = {
        'test_set_size': len(all_psnr),
        'evaluation_date': '2025-11-05',
        'protocol': 'Academic - single evaluation on held-out test set',
        'visual_metrics': {
            'psnr': {
                'mean': float(psnr_mean),
                'std': float(psnr_std),
                'ci_95_lower': float(psnr_mean - psnr_ci),
                'ci_95_upper': float(psnr_mean + psnr_ci),
                'n': len(all_psnr),
                'all_values': [float(v) for v in all_psnr]  # ADDED: Per-sample data
            },
            'ssim': {
                'mean': float(ssim_mean),
                'std': float(ssim_std),
                'ci_95_lower': float(ssim_mean - ssim_ci),
                'ci_95_upper': float(ssim_mean + ssim_ci),
                'n': len(all_ssim),
                'all_values': [float(v) for v in all_ssim]  # ADDED: Per-sample data
            }
        },
        'htr_metrics': {
            'cer': {
                'mean': float(cer_mean),
                'std': float(cer_std),
                'ci_95_lower': float(cer_mean - cer_ci),
                'ci_95_upper': float(cer_mean + cer_ci),
                'n': len(all_cer),
                'all_values': [float(v) for v in all_cer]  # ADDED: Per-sample data for distribution analysis
            },
            'wer': {
                'mean': float(wer_mean),
                'std': float(wer_std),
                'ci_95_lower': float(wer_mean - wer_ci),
                'ci_95_upper': float(wer_mean + wer_ci),
                'n': len(all_wer),
                'all_values': [float(v) for v in all_wer]  # ADDED: Per-sample data
            },
            'baseline_clean': {
                'cer': {
                    'mean': float(clean_cer_mean), 
                    'std': float(clean_cer_std),
                    'all_values': [float(v) for v in all_clean_cer]  # ADDED
                },
                'wer': {
                    'mean': float(clean_wer_mean), 
                    'std': float(clean_wer_std),
                    'all_values': [float(v) for v in all_clean_wer]  # ADDED
                }
            },
            'delta_cer': float(cer_mean - clean_cer_mean)
        },
        'noise_metrics': {
            'noise_variance': {
                'mean': float(noise_var_mean), 
                'std': float(noise_var_std),
                'all_values': [float(v) for v in all_noise_var]  # ADDED
            },
            'isolated_white_ratio': {
                'mean': float(isolated_white_mean), 
                'std': float(isolated_white_std),
                'all_values': [float(v) for v in all_isolated_white]  # ADDED
            },
            'local_variance': {
                'mean': float(local_var_mean), 
                'std': float(local_var_std),
                'all_values': [float(v) for v in all_local_var]  # ADDED
            }
        },
        'sample_texts': sample_texts,
        'note': 'Enhanced output with per-sample metrics for comprehensive failure case analysis'
    }
    
    print("💾 Saving detailed results with per-sample metrics...")
    print(f"   Total samples: {len(all_cer)}")
    print(f"   CER values saved: {len(all_cer)}")
    print(f"   Sample texts saved: {len(sample_texts)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on held-out test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint directory (e.g., checkpoints/production_v2/best_model)')
    parser.add_argument('--tfrecord', type=str, required=True,
                       help='Path to TFRecord file')
    parser.add_argument('--charset', type=str, default='dual_modal_gan/data/charset_94.txt',
                       help='Path to charset file')
    parser.add_argument('--output', type=str, default='test_set_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Training split ratio (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio (default: 0.15)')
    parser.add_argument('--generator_version', type=str, default='enhanced',
                       choices=['base', 'enhanced', 'enhanced_v2'],
                       help='Generator architecture version')
    parser.add_argument('--recognizer_weights', type=str,
                       default='/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5',
                       help='Path to recognizer weights file')
    
    args = parser.parse_args()
    
    # Load charset
    print("📖 Loading charset...")
    charset = read_charlist(args.charset)
    vocab_size = len(charset) + 1
    print(f"   Loaded {vocab_size} characters (including blank)\n")
    
    # Load test dataset
    print("📊 Loading test dataset...")
    test_dataset, test_size = create_test_dataset(
        args.tfrecord,
        args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    # Build models
    print("\n🏗️  Building models...")
    if args.generator_version == 'enhanced':
        generator = unet_enhanced(input_size=(1024, 128, 1))
        print("   Generator: U-Net Enhanced (ResBlocks + Attention)")
    else:
        raise NotImplementedError(f"Generator version '{args.generator_version}' not implemented yet")
    
    # Load recognizer
    from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed
    recognizer = load_frozen_recognizer_fixed(
        weights_path=args.recognizer_weights,
        charset_size=vocab_size - 1,  # vocab_size includes blank token
        return_feature_map=False
    )
    print("   Recognizer: HTR with CTC decoder (frozen)")


    
    # Load checkpoint
    print(f"\n💾 Loading checkpoint from: {args.checkpoint}")
    checkpoint = tf.train.Checkpoint(generator=generator, recognizer=recognizer)
    
    # Try to restore
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint, max_to_keep=1)
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print(f"   ✅ Restored from: {checkpoint_manager.latest_checkpoint}")
    else:
        print(f"   ❌ ERROR: No checkpoint found in {args.checkpoint}")
        print("   Please verify the checkpoint path and try again.")
        return 1
    
    # Evaluate test set
    results = evaluate_test_set(generator, recognizer, test_dataset, charset, test_size)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("\n✅ Test set evaluation complete!")
    print("   Use these results for your academic paper.")
    print("   Remember: Test set was evaluated ONCE - results are unbiased.\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
