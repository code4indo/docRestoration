#!/usr/bin/env python3
"""
Post-hoc CER/WER Evaluation for Single-Modal Ablation Study
===========================================================

Evaluates the actual HTR performance (CER/WER) of single-modal checkpoint
to compare fairly with dual-modal results.

Background:
-----------
Single-modal training used [VISUAL-ONLY MODE] - HTR recognizer was NOT 
loaded during validation, so CER=1.0 in metrics is a PLACEHOLDER value.

This script:
1. Loads single-modal generator checkpoint (ckpt-14)
2. Loads HTR recognizer with weights
3. Runs inference on validation set
4. Calculates REAL CER/WER metrics
5. Compares with dual-modal results

Usage:
------
poetry run python dual_modal_gan/scripts/evaluate_single_modal_cer.py \
    --checkpoint_dir dual_modal_gan/checkpoints/ablation_single_modal_image_only/best_model \
    --output_json dual_modal_gan/checkpoints/ablation_single_modal_image_only/posthoc_cer_evaluation.json

Expected Results:
-----------------
If visual quality is good (PSNR 20.82 dB confirms), CER should be comparable
to dual-modal (~0.40-0.45 range), proving visual quality ≠ HTR performance gap.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import editdistance

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import generator architectures from dual_modal_gan (NOT network/model_enhanced.py!)
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.generator_enhanced_v2 import unet_enhanced_v2
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer


# === Utility Functions (copied from train_enhanced.py for self-containment) ===

def read_charlist(path):
    """Load character list from file"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord - same as train_enhanced.py"""
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
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (H,W,C) → (W,H,C)
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
    
    # Pad label to static shape
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label


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


def decode_ctc_predictions(logits, charset):
    """
    Manual CTC decode - EXACT implementation from training script.
    This is the CORRECT way to decode HTR predictions.
    
    Args:
        logits: (batch, time_steps, vocab_size+1) - raw logits from recognizer
        charset: list of characters
    
    Returns:
        list of decoded strings
    """
    charset_size = len(charset)
    batch_size = logits.shape[0]
    results = []
    
    for b in range(batch_size):
        # Get predictions for this sample
        raw_preds = np.argmax(logits[b], axis=-1)
        
        # Manual CTC decode with deduplication
        deduped = []
        prev = -1
        for token in raw_preds:
            if token != prev:
                deduped.append(token)
                prev = token
        
        # Remove blank tokens (blank_token = charset_size)
        result = []
        for token in deduped:
            if token != charset_size:  # Skip blank
                result.append(token)
        
        # Convert to string
        decoded = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
        results.append(decoded)
    
    return results


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


def create_htr_recognizer(charset_size, proj_dim=512, num_layers=6, num_heads=8, 
                          ff_dim=2048, dropout_rate=0.20):
    """
    Create HTR recognizer - same architecture as train_enhanced.py
    
    Architecture: Projection → TransformerEncoder → Dense → CTC
    Input: (batch, W, H, C) where W=1024, H=128, C=1
    Output: (batch, time_steps, vocab_size) logits for CTC
    """
    # Input layer: (W, H, C) = (1024, 128, 1)
    inputs = tf.keras.layers.Input(shape=(1024, 128, 1), name='recognizer_input')
    
    # Reshape from (W, H, C) to (W, H*C) untuk sequence processing
    # Output: (batch, 1024, 128)
    x = tf.keras.layers.Reshape((1024, 128))(inputs)
    
    # Projection layer: Map 128 channels to proj_dim
    x = tf.keras.layers.Dense(proj_dim, activation='relu', name='projection')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Transformer Encoder Layers
    for i in range(num_layers):
        # Multi-Head Self-Attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=proj_dim // num_heads,
            dropout=dropout_rate,
            name=f'mha_{i}'
        )(x, x)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, attn_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-Forward Network
        ffn_output = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(proj_dim)
        ], name=f'ffn_{i}')(x)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, ffn_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Output layer: Map to vocabulary size for CTC
    # Output shape: (batch, time_steps=1024, vocab_size)
    outputs = tf.keras.layers.Dense(
        charset_size + 1,  # +1 for CTC blank token
        activation=None,
        name='ctc_output'
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='htr_recognizer')
    return model


def load_generator_checkpoint(checkpoint_path, generator_version='enhanced'):
    """Load generator from TensorFlow checkpoint"""
    print(f"\n[2/6] Loading Generator...")
    print(f"  Architecture: {generator_version}")
    
    # Create generator based on version (matching train_enhanced.py)
    input_size = (1024, 128, 1)  # (W, H, C)
    
    if generator_version == 'enhanced':
        generator = unet_enhanced(input_size=input_size)
        generator_name = "U-Net Enhanced (ResBlocks+Attention)"
    elif generator_version == 'enhanced_v2':
        generator = unet_enhanced_v2(input_size=input_size)
        generator_name = "U-Net Enhanced V2 (CBAM+RDB+MSFP)"
    else:
        raise ValueError(f"Unknown generator version: {generator_version}. Use 'enhanced' or 'enhanced_v2'.")
    
    print(f"  Architecture: {generator_name}")
    print(f"  Parameters: {generator.count_params():,}")
    
    # Create checkpoint object
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Restore checkpoint
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_dir():
        # Directory format - find checkpoint file
        ckpt_file = checkpoint_dir / 'checkpoint'
        if ckpt_file.exists():
            status = checkpoint.restore(tf.train.latest_checkpoint(str(checkpoint_dir)))
        else:
            raise FileNotFoundError(f"No checkpoint file in {checkpoint_dir}")
    else:
        # Direct file format
        status = checkpoint.restore(checkpoint_path)
    
    status.expect_partial()
    print(f"  ✅ Generator loaded from: {checkpoint_path}")
    
    return generator


def evaluate_single_modal_cer(args):
    """Main evaluation function"""
    print("\n" + "="*80)
    print("🔬 POST-HOC CER/WER EVALUATION - SINGLE-MODAL ABLATION")
    print("="*80)
    print(f"\nPurpose: Get REAL HTR metrics (not placeholder CER=1.0)")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Expected: CER ~0.40-0.50 if visual quality matches PSNR 20.82 dB")
    
    # [1/6] Load charset
    print(f"\n[1/6] Loading Character Set...")
    charset = read_charlist(args.charset_path)
    charset_size = len(charset)
    print(f"  ✅ Charset loaded: {charset_size} characters")
    
    # [2/6] Load generator
    generator = load_generator_checkpoint(args.checkpoint_dir, args.generator_version)
    
    # [3/6] Load HTR recognizer
    print(f"\n[3/6] Loading HTR Recognizer...")
    
    # Use load_frozen_recognizer like train_enhanced.py
    recognizer = load_frozen_recognizer(
        weights_path=args.recognizer_weights,
        charset_size=charset_size,
        return_feature_map=False  # We only need logits for CER/WER
    )
    print(f"  Parameters: {recognizer.count_params():,}")
    print(f"  ✅ Recognizer loaded from: {args.recognizer_weights}")
    
    # [4/6] Load validation dataset
    print(f"\n[4/6] Loading Validation Dataset...")
    print(f"  TFRecord: {args.tfrecord_path}")
    
    # Parse TFRecord
    raw_dataset = tf.data.TFRecordDataset([args.tfrecord_path])
    dataset = raw_dataset.map(
        _parse_tfrecord_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Count total samples
    total_batches = sum(1 for _ in dataset)
    dataset = dataset.take(args.num_batches) if args.num_batches > 0 else dataset
    eval_batches = min(total_batches, args.num_batches) if args.num_batches > 0 else total_batches
    
    print(f"  ✅ Dataset ready")
    print(f"     Total batches: {total_batches}")
    print(f"     Evaluating: {eval_batches} batches ({eval_batches * args.batch_size} samples)")
    
    # [5/6] Run evaluation
    print(f"\n[5/6] Running HTR Evaluation...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Progress:")
    
    all_cer_degraded = []
    all_cer_generated = []
    all_wer_degraded = []
    all_wer_generated = []
    
    all_psnr = []
    all_ssim = []
    
    sample_results = []
    
    for batch_idx, (degraded_images, clean_images, labels) in enumerate(tqdm(dataset, total=eval_batches)):
        # ✅ CRITICAL FIX: Normalize to [-1,1] for generator (same as training!)
        # TFRecord data is [0,1], but generator expects [-1,1] input
        degraded_images_tanh = degraded_images * 2.0 - 1.0
        
        # Generate enhanced images
        generated_images = generator(degraded_images_tanh, training=False)
        
        # ✅ FIX: Denormalize generator output [-1,1] → [0,1] for PSNR/SSIM
        generated_images_normalized = (generated_images + 1.0) / 2.0
        
        # Calculate PSNR & SSIM (both images must be in [0,1] range)
        psnr_batch = tf.image.psnr(clean_images, generated_images_normalized, max_val=1.0)
        ssim_batch = tf.image.ssim(clean_images, generated_images_normalized, max_val=1.0)
        
        all_psnr.extend(psnr_batch.numpy())
        all_ssim.extend(ssim_batch.numpy())
        
        # HTR inference (recognizer expects [0,1] range images)
        degraded_logits = recognizer(degraded_images, training=False)  # Degraded already [0,1]
        generated_logits = recognizer(generated_images_normalized, training=False)  # Use normalized
        
        # ✅ CRITICAL FIX: Use CTC decode (same as training!), not simple argmax
        # Simple argmax doesn't handle CTC blank tokens and duplicates properly
        degraded_texts = decode_ctc_predictions(degraded_logits.numpy(), charset)
        generated_texts = decode_ctc_predictions(generated_logits.numpy(), charset)
        
        # Convert labels to numpy
        labels_np = labels.numpy()
        
        # Calculate CER/WER for each sample
        for i in range(labels_np.shape[0]):
            gt_text = decode_label(labels_np[i], charset)
            degraded_text = degraded_texts[i]
            generated_text = generated_texts[i]
            
            # CER/WER: ground truth vs predictions
            cer_deg = calculate_cer(gt_text, degraded_text)
            cer_gen = calculate_cer(gt_text, generated_text)
            wer_deg = calculate_wer(gt_text, degraded_text)
            wer_gen = calculate_wer(gt_text, generated_text)
            
            all_cer_degraded.append(cer_deg)
            all_cer_generated.append(cer_gen)
            all_wer_degraded.append(wer_deg)
            all_wer_generated.append(wer_gen)
            
            # Store first N samples for detailed report
            if len(sample_results) < args.num_samples:
                sample_results.append({
                    'batch': batch_idx,
                    'sample': i,
                    'ground_truth': gt_text,
                    'degraded_prediction': degraded_text,
                    'generated_prediction': generated_text,
                    'cer_degraded': float(cer_deg),
                    'cer_generated': float(cer_gen),
                    'wer_degraded': float(wer_deg),
                    'wer_generated': float(wer_gen),
                    'improvement_cer': float(cer_deg - cer_gen),
                    'improvement_wer': float(wer_deg - wer_gen),
                    'psnr': float(psnr_batch[i]),
                    'ssim': float(ssim_batch[i])
                })
    
    # [6/6] Calculate statistics
    print(f"\n[6/6] Calculating Statistics...")
    
    mean_psnr = np.mean(all_psnr)
    std_psnr = np.std(all_psnr)
    mean_ssim = np.mean(all_ssim)
    std_ssim = np.std(all_ssim)
    
    mean_cer_deg = np.mean(all_cer_degraded)
    std_cer_deg = np.std(all_cer_degraded)
    mean_cer_gen = np.mean(all_cer_generated)
    std_cer_gen = np.std(all_cer_generated)
    
    mean_wer_deg = np.mean(all_wer_degraded)
    std_wer_deg = np.std(all_wer_degraded)
    mean_wer_gen = np.mean(all_wer_generated)
    std_wer_gen = np.std(all_wer_generated)
    
    improvement_cer = mean_cer_deg - mean_cer_gen
    improvement_wer = mean_wer_deg - mean_wer_gen
    
    cer_improvement_pct = (improvement_cer / mean_cer_deg) * 100 if mean_cer_deg > 0 else 0
    wer_improvement_pct = (improvement_wer / mean_wer_deg) * 100 if mean_wer_deg > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("📊 POST-HOC EVALUATION RESULTS - SINGLE-MODAL")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Samples evaluated: {len(all_cer_generated)}")
    print("-" * 80)
    
    print("\n🔹 Visual Quality:")
    print(f"   PSNR: {mean_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"   SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
    
    print("\n🔹 HTR Performance - Degraded Input (Baseline):")
    print(f"   CER: {mean_cer_deg:.4f} ± {std_cer_deg:.4f}")
    print(f"   WER: {mean_wer_deg:.4f} ± {std_wer_deg:.4f}")
    
    print("\n🔹 HTR Performance - Enhanced Output (Single-Modal):")
    print(f"   CER: {mean_cer_gen:.4f} ± {std_cer_gen:.4f}")
    print(f"   WER: {mean_wer_gen:.4f} ± {std_wer_gen:.4f}")
    
    print("\n🔹 Improvement (Degraded → Enhanced):")
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
    
    for idx, sample in enumerate(sample_results[:10], 1):
        print(f"\nSample #{idx} (Batch {sample['batch']}, Index {sample['sample']}):")
        print(f"  GT:         {sample['ground_truth'][:80]}")
        print(f"  Degraded:   {sample['degraded_prediction'][:80]}")
        print(f"  Enhanced:   {sample['generated_prediction'][:80]}")
        print(f"  CER: Deg={sample['cer_degraded']:.4f}, Enh={sample['cer_generated']:.4f}, Δ={sample['improvement_cer']:+.4f}")
        print(f"  WER: Deg={sample['wer_degraded']:.4f}, Enh={sample['wer_generated']:.4f}, Δ={sample['improvement_wer']:+.4f}")
        print(f"  Visual: PSNR={sample['psnr']:.2f} dB, SSIM={sample['ssim']:.4f}")
    
    # Save results to JSON
    output_dir = Path(args.output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'checkpoint': str(args.checkpoint_dir),
        'generator_version': args.generator_version,
        'timestamp': timestamp,
        'num_samples': len(all_cer_generated),
        'metrics': {
            'visual_quality': {
                'psnr_mean': float(mean_psnr),
                'psnr_std': float(std_psnr),
                'ssim_mean': float(mean_ssim),
                'ssim_std': float(std_ssim)
            },
            'degraded': {
                'cer_mean': float(mean_cer_deg),
                'cer_std': float(std_cer_deg),
                'wer_mean': float(mean_wer_deg),
                'wer_std': float(std_wer_deg)
            },
            'generated': {
                'cer_mean': float(mean_cer_gen),
                'cer_std': float(std_cer_gen),
                'wer_mean': float(mean_wer_gen),
                'wer_std': float(std_wer_gen)
            },
            'improvement': {
                'cer_absolute': float(improvement_cer),
                'cer_percentage': float(cer_improvement_pct),
                'wer_absolute': float(improvement_wer),
                'wer_percentage': float(wer_improvement_pct)
            }
        },
        'samples': sample_results,
        'comparison_with_dual_modal': {
            '_note': 'Compare with dual_modal_gan/checkpoints/exp_proof_gt_v2_balanced/metrics/training_metrics_fp32_final.json',
            '_dual_modal_gt_cer': 0.4092,
            '_dual_modal_pred_cer': 0.4276,
            '_single_modal_cer': float(mean_cer_gen),
            '_cer_gap_vs_gt': float(mean_cer_gen - 0.4092),
            '_cer_gap_vs_pred': float(mean_cer_gen - 0.4276),
            '_psnr_gap': {
                '_single_modal': float(mean_psnr),
                '_dual_modal_gt': 20.23,
                '_dual_modal_pred': 20.07,
                '_gap_vs_gt': float(mean_psnr - 20.23),
                '_gap_vs_pred': float(mean_psnr - 20.07)
            }
        }
    }
    
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print(f"✅ Results saved to: {args.output_json}")
    print("="*80 + "\n")
    
    # Print comparison analysis
    print("\n" + "="*80)
    print("🔍 COMPARATIVE ANALYSIS: SINGLE-MODAL vs DUAL-MODAL")
    print("="*80)
    
    print(f"\n📊 Visual Quality (PSNR):")
    print(f"   Single-Modal:    {mean_psnr:.4f} dB")
    print(f"   Dual-Modal GT:   20.23 dB")
    print(f"   Dual-Modal Pred: 20.07 dB")
    print(f"   Gap vs GT:       {mean_psnr - 20.23:+.4f} dB")
    print(f"   Gap vs Pred:     {mean_psnr - 20.07:+.4f} dB")
    
    print(f"\n📊 HTR Performance (CER):")
    print(f"   Single-Modal:    {mean_cer_gen:.4f} (NOW MEASURED!)")
    print(f"   Dual-Modal GT:   0.4092")
    print(f"   Dual-Modal Pred: 0.4276")
    print(f"   Gap vs GT:       {mean_cer_gen - 0.4092:+.4f}")
    print(f"   Gap vs Pred:     {mean_cer_gen - 0.4276:+.4f}")
    
    # Interpret results
    print(f"\n💡 Interpretation:")
    if abs(mean_cer_gen - 0.4092) < 0.05:
        print(f"   ✅ Single-modal CER ≈ Dual-modal CER (within 5% margin)")
        print(f"   ⚠️  Visual quality (PSNR) does NOT correlate with HTR performance!")
        print(f"   ⚠️  Dual-modal trades {20.23 - mean_psnr:.2f} dB PSNR for text supervision")
        print(f"   ⚠️  BUT text supervision does NOT improve CER significantly")
        print(f"   ❌ Novelty claim WEAKENED: Dual-modal benefit is MINIMAL")
    elif mean_cer_gen > 0.45:
        print(f"   ✅ Dual-modal CER significantly better than single-modal")
        print(f"   ✅ Text supervision provides meaningful HTR guidance")
        print(f"   ✅ Novelty claim VALIDATED: Dual-modal improves HTR readability")
        print(f"   ⚠️  Trade-off: {20.23 - mean_psnr:.2f} dB PSNR for {mean_cer_gen - 0.4092:.4f} CER improvement")
    else:
        print(f"   ⚠️  Mixed results: Dual-modal shows MINOR CER improvement")
        print(f"   ⚠️  Trade-off questionable: {20.23 - mean_psnr:.2f} dB PSNR for {mean_cer_gen - 0.4092:.4f} CER gain")
        print(f"   💭 Novelty claim needs refinement")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Post-hoc CER/WER evaluation for single-modal ablation study"
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='dual_modal_gan/checkpoints/ablation_single_modal_image_only/best_model',
        help='Directory containing single-modal checkpoint (ckpt-*)'
    )
    
    parser.add_argument(
        '--generator_version',
        type=str,
        default='enhanced',
        choices=['enhanced', 'enhanced_v2'],
        help='Generator architecture version (enhanced or enhanced_v2)'
    )
    
    parser.add_argument(
        '--recognizer_weights',
        type=str,
        default='/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5',
        help='Path to HTR recognizer weights'
    )
    
    parser.add_argument(
        '--tfrecord_path',
        type=str,
        default='dual_modal_gan/data/dataset_gan.tfrecord',
        help='Path to validation TFRecord'
    )
    
    parser.add_argument(
        '--charset_path',
        type=str,
        default='real_data_preparation/real_data_charlist.txt',
        help='Path to character list file'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--num_batches',
        type=int,
        default=50,
        help='Number of batches to evaluate (0 = all)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=20,
        help='Number of sample comparisons to save'
    )
    
    parser.add_argument(
        '--output_json',
        type=str,
        default='dual_modal_gan/checkpoints/ablation_single_modal_image_only/posthoc_cer_evaluation.json',
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    evaluate_single_modal_cer(args)
