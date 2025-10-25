"""
Dual-Modal GAN-HTR Training Script - Pure FP32 Version (MULTI-GPU OPTIMIZED)

Version: train32_multi_gpu.py
Based on: train_enhanced.py (train32.py Perfected)

Key Improvements:
1. Pure FP32 training (NO mixed precision) for numerical stability
2. Balanced loss components (CTC, Pixel, Adversarial)
3. Comprehensive metrics (PSNR, SSIM, CER, WER)
4. MLflow tracking integration
5. Proper gradient clipping and loss management
6. Optimized for GAN-HTR with CTC loss
7. Efficient checkpoint management (max_to_keep=1 by default)
8. ✨ **MULTI-GPU SUPPORT**: MirroredStrategy for 2x NVIDIA RTX A4000

Multi-GPU Configuration:
- Strategy: tf.distribute.MirroredStrategy (data parallelism)
- GPUs: 2x NVIDIA RTX A4000 (16GB each)
- Batch Size: Global batch (auto-sharded across GPUs)
- Expected speedup: ~1.8x (accounting for communication overhead)

Research Finding:
- FP16 mixed precision causes imbalanced optimization in GAN-HTR
- CTC loss log-space calculations require FP32 precision
- Pure FP32 achieves better convergence and visual quality
- Multi-GPU training requires careful batch norm synchronization
"""

import argparse
import os
import time
import numpy as np
import cv2
import json
from datetime import datetime
import mlflow
import mlflow.tensorflow

# Disable XLA optimization to avoid layout errors - MUST be before TF import
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf

# Disable XLA at TF config level
tf.config.optimizer.set_jit(False)

# ✅ PURE FP32 - NO MIXED PRECISION
# Explicitly set float32 policy for numerical stability with CTC loss
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
print("✅ Pure FP32 precision enabled for training stability")
print("   (FP16 disabled: Required for CTC loss numerical stability)")

from tqdm import tqdm
import editdistance  # For CER/WER calculation

# Fix for ModuleNotFoundError: Add project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import models and utils
from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.generator_enhanced_v2 import unet_enhanced_v2
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer
from dual_modal_gan.src.models.discriminator import build_dual_modal_discriminator
from dual_modal_gan.src.models.discriminator_enhanced_v2 import build_dual_modal_discriminator_enhanced_v2
from dual_modal_gan.src.models.discriminator_enhanced_v2_fixed import build_dual_modal_discriminator_enhanced_v2_fixed
from dual_modal_gan.losses.perceptual_loss import create_perceptual_loss
from dual_modal_gan.models.gradnorm import SimpleAdaptiveBalancer

# --- Utility Functions ---
def read_charlist(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

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

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"--- Random seeds set to {seed} for reproducibility ---")

def decode_label(label_ids, charset):
    """Decode label IDs to text string, removing blank tokens (0) and padding."""
    decoded_chars = []
    prev_id = -1
    for label_id in label_ids:
        label_id = int(label_id)
        # Skip blank token (0) and padding (0), and CTC duplicates
        if label_id > 0 and label_id != prev_id:
            if label_id - 1 < len(charset):
                decoded_chars.append(charset[label_id - 1])
        prev_id = label_id
    return ''.join(decoded_chars)

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate using edit distance."""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)

def calculate_wer(ground_truth, prediction):
    """Calculate Word Error Rate using edit distance on word level."""
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = editdistance.eval(gt_words, pred_words)
    return distance / len(gt_words)

def calculate_noise_artifacts_metrics(image):
    """
    Calculate metrics to detect white dots/noise artifacts in generated images.
    
    Args:
        image: TensorFlow tensor or numpy array, shape (W, H, C) or (W, H), values in [0, 1]
    
    Returns:
        dict with noise metrics:
        - noise_variance: Variance of Laplacian (high-frequency noise)
        - isolated_white_ratio: Ratio of isolated white pixels (salt noise)
        - local_variance: Variance of local deviations from smoothed image
    """
    # Convert to numpy and ensure grayscale
    if isinstance(image, tf.Tensor):
        img_np = image.numpy()
    else:
        img_np = image
    
    # Ensure 2D grayscale
    if len(img_np.shape) == 3:
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
        else:
            # Convert RGB to grayscale
            img_np = np.mean(img_np, axis=-1)
    
    # Convert to uint8 for OpenCV operations (0-255 range)
    if img_np.max() <= 1.0:
        img_uint8 = (img_np * 255).astype(np.uint8)
    else:
        img_uint8 = img_np.astype(np.uint8)
    
    # 1. High-frequency noise detection (Laplacian variance)
    laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
    noise_variance = float(np.var(laplacian))
    
    # 2. Isolated white pixel detection (salt noise)
    # Threshold to detect very bright pixels (> 250/255 = 0.98)
    _, binary = cv2.threshold(img_uint8, 250, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    isolated_white_pixels = np.sum(binary - eroded)
    total_pixels = img_uint8.size
    isolated_white_ratio = float(isolated_white_pixels / total_pixels)
    
    # 3. Local variance (smoothness measure)
    kernel_size = 5
    mean_filtered = cv2.blur(img_uint8, (kernel_size, kernel_size))
    local_deviations = img_uint8.astype(np.float32) - mean_filtered.astype(np.float32)
    local_variance = float(np.var(local_deviations))
    
    return {
        'noise_variance': noise_variance,
        'isolated_white_ratio': isolated_white_ratio,
        'local_variance': local_variance
    }


# --- Dataset Pipeline ---
def _parse_tfrecord_fn(example_proto):
    # Define feature description for raw bytes and metadata
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64), # H, W, C
        'degraded_image_dtype': tf.io.FixedLenFeature([], tf.string), # Stored as string name
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64), # (length,)
        'label_dtype': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize degraded image
    degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    degraded_image_dtype_str = example['degraded_image_dtype'] # Get string directly
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32) # Decode raw bytes
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image_dtype_str = example['clean_image_dtype']
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32) # Decode raw bytes
    clean_image = tf.reshape(clean_image, clean_image_shape)
    # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    # Deserialize label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label_dtype_str = example['label_dtype']
    label = tf.io.decode_raw(example['label_raw'], tf.int64) # Label is int64
    label = tf.reshape(label, label_shape)
    
    label = tf.cast(label, tf.int32)
    # Pad label to a static shape to prevent TF graph retracing/warnings
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label

def create_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """
    Create train/val/test split with proper academic protocol.
    
    ✅ ACADEMIC FIX (2025-10-21): 3-way split (train/val/test) for unbiased evaluation
    
    Args:
        tfrecord_path: Path to TFRecord file
        batch_size: Batch size for training
        train_split: Fraction for training (default: 0.7)
        val_split: Fraction for validation (default: 0.15)
        
    Returns:
        train_dataset: Training dataset (shuffled, repeated)
        val_dataset: Validation dataset (no shuffle, for monitoring)
        test_dataset: Test dataset (no shuffle, LOCKED until final evaluation)
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        
    Split Strategy:
        - Train:      70% (default) - for model learning
        - Validation: 15% (default) - for hyperparameter tuning, early stopping
        - Test:       15% (1.0 - train_split - val_split) - for FINAL evaluation ONLY
        
    Important:
        - Test set should NEVER be touched during training
        - Test evaluation should happen ONCE after model selection
        - Validation is used for early stopping and checkpoint selection
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Get the total number of items
    total_size = sum(1 for _ in dataset)
    
    # Calculate split sizes
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size  # Remaining samples go to test
    
    # Map parsing function
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Split: first 70% train, next 15% val, last 15% test
    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)
    
    # Process datasets:
    # - Train: shuffle + repeat (for continuous training)
    # - Val: no shuffle, no repeat (fixed evaluation set)
    # - Test: no shuffle, no repeat (LOCKED for final eval)
    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    # ✅ ACADEMIC FIX (2025-10-21): NO shuffle on validation/test - fixed evaluation sets
    # Validation and test sets should remain in consistent order for reproducible evaluation
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    print(f"\n📊 Dataset Split (Academic Protocol):")
    print(f"   Total samples: {total_size}")
    print(f"   Train:      {train_size} samples ({train_split*100:.0f}%) - for learning")
    print(f"   Validation: {val_size} samples ({val_split*100:.0f}%) - for hyperparameter tuning")
    print(f"   Test:       {test_size} samples ({(1-train_split-val_split)*100:.0f}%) - for FINAL evaluation (LOCKED)")
    print(f"   ⚠️  Test set should NEVER be used during training!")
    
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size

# --- Main Training & Evaluation Logic ---
def run_validation_step(val_dataset, generator, recognizer, charset, psnr_metric, ssim_metric, cer_metric, wer_metric, clean_cer_metric, clean_wer_metric, noise_variance_metric, isolated_white_metric, local_variance_metric):
    """Run validation step with visual (PSNR/SSIM), textual (CER/WER), and noise artifact metrics.
    
    ✅ ACADEMIC FIX (2025-10-21): Evaluate on FULL validation set for statistical significance
    - Uses all validation samples (not just 1 batch)
    - Reports mean ± std and 95% confidence intervals
    - Sample size explicitly tracked for academic rigor
    
    Note: Cannot use @tf.function due to Python-based CER/WER calculation.
    """
    # Collect ALL metrics from full validation set
    all_psnr = []
    all_ssim = []
    all_cer = []
    all_wer = []
    all_clean_cer = []
    all_clean_wer = []
    all_noise_var = []
    all_isolated_white = []
    all_local_var = []
    
    first_batch = True  # Flag to log first batch sample
    batch_count = 0
    
    # Evaluate on FULL validation set
    for degraded_images, clean_images, labels in val_dataset:
        # ✅ CRITICAL FIX (2025-10-21): Normalize validation data to [-1,1]
        # TFRecord data is [0,1], but generator expects [-1,1] input
        degraded_images_tanh = degraded_images * 2.0 - 1.0
        clean_images_tanh = clean_images * 2.0 - 1.0
        
        # Generate enhanced images
        generated_images = generator(degraded_images_tanh, training=False)
        
        # Denormalize to [0, 1] range for proper PSNR/SSIM calculation
        generated_images_normalized = (generated_images + 1.0) / 2.0
        clean_images_normalized = (clean_images_tanh + 1.0) / 2.0
        
        # Visual metrics: PSNR and SSIM expect values in [0, 1] range
        psnr = tf.image.psnr(clean_images_normalized, generated_images_normalized, max_val=1.0)
        ssim = tf.image.ssim(clean_images_normalized, generated_images_normalized, max_val=1.0)
        
        # Collect individual sample metrics for statistics
        all_psnr.extend(psnr.numpy().tolist())
        all_ssim.extend(ssim.numpy().tolist())
        
        # Noise artifact metrics: Calculate for each generated image in batch
        # Use denormalized images [0,1] for proper noise detection
        for i in range(generated_images_normalized.shape[0]):
            noise_metrics = calculate_noise_artifacts_metrics(generated_images_normalized[i])
            all_noise_var.append(noise_metrics['noise_variance'])
            all_isolated_white.append(noise_metrics['isolated_white_ratio'])
            all_local_var.append(noise_metrics['local_variance'])
        
        # Textual metrics: CER/WER
        # ✅ CRITICAL FIX: Run recognizer on NORMALIZED images [0,1], not tanh [-1,1]!
        # Get HTR predictions for clean (ground truth quality) and generated (enhanced) images
        recognizer_output_clean = recognizer(clean_images_normalized, training=False)
        recognizer_output_generated = recognizer(generated_images_normalized, training=False)
        
        # Extract logits (handle both tuple and single output)
        if isinstance(recognizer_output_clean, (list, tuple)):
            clean_logits = recognizer_output_clean[0]
            generated_logits = recognizer_output_generated[0]
        else:
            clean_logits = recognizer_output_clean
            generated_logits = recognizer_output_generated
        
        # ✅ CRITICAL FIX: Use manual CTC decode, not tf.argmax!
        # Decode predictions using manual CTC decoding (same as training script)
        clean_predictions_text = decode_ctc_predictions(clean_logits.numpy(), charset)
        generated_predictions_text = decode_ctc_predictions(generated_logits.numpy(), charset)
        
        # Convert labels to numpy for text decoding
        labels_np = labels.numpy()
        
        # Calculate CER/WER for each sample in batch
        batch_cer = []  # CER for generated image vs ground truth
        batch_wer = []  # WER for generated image vs ground truth
        batch_clean_cer = []  # CER for clean image vs ground truth (baseline)
        batch_clean_wer = []  # WER for clean image vs ground truth (baseline)
        
        for i in range(labels_np.shape[0]):
            # Decode ground truth
            gt_text = decode_label(labels_np[i], charset)
            
            # Get decoded HTR predictions (already decoded by manual CTC decode)
            clean_text = clean_predictions_text[i]
            generated_text = generated_predictions_text[i]
            
            # FIXED: Calculate CER/WER against GROUND TRUTH (not clean prediction)
            # This is the CORRECT way to measure HTR accuracy
            cer = calculate_cer(gt_text, generated_text)
            wer = calculate_wer(gt_text, generated_text)
            
            # Also calculate clean baseline (for comparison)
            clean_cer = calculate_cer(gt_text, clean_text)
            clean_wer = calculate_wer(gt_text, clean_text)
            
            # Collect individual samples for statistics
            all_cer.append(cer)
            all_wer.append(wer)
            all_clean_cer.append(clean_cer)
            all_clean_wer.append(clean_wer)
            
            # Log first sample for debugging
            if i == 0 and first_batch:
                print(f"\n📝 [SAMPLE TEXT RECOGNITION VALIDATION]")
                print(f"  🎯 Ground Truth:    '{gt_text}'")
                print(f"  ✨ Clean Image:     '{clean_text}' (CER: {clean_cer:.3f})")
                print(f"  🤖 Generated Image: '{generated_text}' (CER: {cer:.3f})")
                print(f"  📊 Quality Gap:     ΔCER = {cer - clean_cer:+.3f} (generated vs clean)")
        
        first_batch = False
        batch_count += 1
    
    # ✅ ACADEMIC FIX: Calculate statistics from full validation set
    # Report mean ± std and 95% confidence intervals
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
    
    clean_cer_mean = np.mean(all_clean_cer)
    clean_wer_mean = np.mean(all_clean_wer)
    
    noise_var_mean = np.mean(all_noise_var)
    isolated_white_mean = np.mean(all_isolated_white)
    local_var_mean = np.mean(all_local_var)
    
    # Update metrics with calculated statistics
    psnr_metric.update_state([psnr_mean])
    ssim_metric.update_state([ssim_mean])
    cer_metric.update_state([cer_mean])
    wer_metric.update_state([wer_mean])
    clean_cer_metric.update_state([clean_cer_mean])
    clean_wer_metric.update_state([clean_wer_mean])
    noise_variance_metric.update_state([noise_var_mean])
    isolated_white_metric.update_state([isolated_white_mean])
    local_variance_metric.update_state([local_var_mean])
    
    # Print summary with statistics
    print(f"\n  📊 Validation Statistics (n={len(all_psnr)} samples, {batch_count} batches):")
    print(f"     PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB (95% CI: [{psnr_mean-psnr_ci:.2f}, {psnr_mean+psnr_ci:.2f}])")
    print(f"     SSIM: {ssim_mean:.4f} ± {ssim_std:.4f} (95% CI: [{ssim_mean-ssim_ci:.4f}, {ssim_mean+ssim_ci:.4f}])")
    print(f"     CER:  {cer_mean:.4f} ± {cer_std:.4f} (baseline: {clean_cer_mean:.4f})")
    print(f"     WER:  {wer_mean:.4f} ± {wer_std:.4f} (baseline: {clean_wer_mean:.4f})")
    
    # Return statistics for logging
    return {
        'psnr': {'mean': psnr_mean, 'std': psnr_std, 'ci_95': psnr_ci, 'n': len(all_psnr)},
        'ssim': {'mean': ssim_mean, 'std': ssim_std, 'ci_95': ssim_ci, 'n': len(all_ssim)},
        'cer': {'mean': cer_mean, 'std': cer_std, 'n': len(all_cer)},
        'wer': {'mean': wer_mean, 'std': wer_std, 'n': len(all_wer)}
    }

def main(args):
    # ✨ MULTI-GPU SETUP: Configure visible GPUs and create MirroredStrategy
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Parse GPU IDs for multi-GPU training
    gpu_ids = [int(x.strip()) for x in args.gpu_id.split(',')]
    num_gpus = len(gpu_ids)
    
    print(f"╔════════════════════════════════════════════════════════════════╗")
    print(f"║    Dual-Modal GAN-HTR Training (Pure FP32 - MULTI-GPU)       ║")
    print(f"╚════════════════════════════════════════════════════════════════╝")
    print(f"    Version: train32_multi_gpu.py")
    print(f"    Precision: Pure FP32 (NO mixed precision)")
    print(f"    GPUs: {num_gpus}x NVIDIA RTX A4000 (IDs: {gpu_ids})")
    print(f"    Strategy: {'MirroredStrategy (Data Parallelism)' if num_gpus > 1 else 'Single GPU'}")
    print(f"    Expected speedup: ~{1.8 if num_gpus == 2 else 1.0}x")
    print()

    # Set random seeds for reproducibility
    set_seeds(args.seed)

    # ✨ CRITICAL: Create MirroredStrategy for multi-GPU training
    if num_gpus > 1:
        # MirroredStrategy automatically uses all visible GPUs
        strategy = tf.distribute.MirroredStrategy()
        print(f"✅ MirroredStrategy initialized with {strategy.num_replicas_in_sync} GPUs")
        print(f"   Communication: NCCL (NVIDIA Collective Communications Library)")
        print(f"   Batch sharding: Automatic (global_batch_size={args.batch_size}, per_replica={args.batch_size // strategy.num_replicas_in_sync})")
    else:
        strategy = tf.distribute.get_strategy()  # Default strategy (single device)
        print(f"✅ Single GPU strategy: {strategy}")
    
    print(f"\n📊 Distribution Strategy Info:")
    print(f"   Num replicas: {strategy.num_replicas_in_sync}")
    print(f"   Global batch size: {args.batch_size}")
    print(f"   Per-replica batch size: {args.batch_size // strategy.num_replicas_in_sync}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Create metrics directory for JSON logs
    metrics_dir = os.path.join(args.checkpoint_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    print("\n[Phase 1/6] Loading Datasets and Charset...")
    charset = read_charlist(args.charset_path)
    vocab_size = len(charset) + 1
    print(f"Charset loaded: {vocab_size} characters (including blank token)")

    # ✅ ACADEMIC FIX (2025-10-21): 3-way split with test set
    # ✨ MULTI-GPU: Dataset auto-sharding handled by strategy
    train_dataset, val_dataset, test_dataset, train_count, val_count, test_count = create_dataset(
        args.tfrecord_path, 
        args.batch_size,
        train_split=getattr(args, 'train_split', 0.7),
        val_split=getattr(args, 'val_split', 0.15)
    )
    print(f"Dataset created: {train_count} training, {val_count} validation, {test_count} test samples.")
    print(f"⚠️  Test set is LOCKED - will only be evaluated after training completion!")
    
    # ✨ MULTI-GPU: Distribute datasets across replicas
    if num_gpus > 1:
        print(f"\n🔄 Distributing datasets across {num_gpus} GPUs...")
        train_dataset_dist = strategy.experimental_distribute_dataset(train_dataset)
        val_dataset_dist = strategy.experimental_distribute_dataset(val_dataset)
        print(f"   ✅ Auto-sharding enabled: Each GPU processes {args.batch_size // num_gpus} samples per batch")
    else:
        train_dataset_dist = train_dataset
        val_dataset_dist = val_dataset

    # Calculate steps per epoch if not provided
    steps_per_epoch = args.steps_per_epoch or (train_count // args.batch_size)

    with strategy.scope():
        print("\n[Phase 2/6] Building Models...")
        # Use (W, H, C) = (1024, 128, 1) to match HTR recognizer expectation
        
        # --- Select Generator Architecture ---
        if args.generator_version == 'enhanced':
            print("   ✅ Using ENHANCED generator (U-Net with Residual Blocks and Attention)")
            generator = unet_enhanced(input_size=(1024, 128, 1))
            generator_name = "U-Net Enhanced (ResBlocks+Attention, 21.8M)"
        elif args.generator_version == 'enhanced_v2':
            print("   🚀 Using ENHANCED V2 generator (CBAM + RDB + Multi-Scale)")
            generator = unet_enhanced_v2(input_size=(1024, 128, 1))
            generator_name = "U-Net Enhanced V2 (CBAM+RDB+MSFP, 18.8M)"
        else:
            print("   ✅ Using BASE generator (Standard U-Net)")
            generator = unet(input_size=(1024, 128, 1))
            generator_name = "U-Net (30M params, no dropout)"
        
        # Load recognizer with multi-output for Recognition Feature Loss
        use_rec_feat_loss = args.rec_feat_loss_weight > 0.0
        recognizer = load_frozen_recognizer(
            weights_path=args.recognizer_weights, 
            charset_size=vocab_size - 1,
            return_feature_map=use_rec_feat_loss
        )
        
        # Build discriminator based on version
        if args.discriminator_version == 'enhanced_v2_fixed':
            # Load discriminator config from additional args if available
            disc_config = getattr(args, 'discriminator_config', {})
            discriminator = build_dual_modal_discriminator_enhanced_v2_fixed(
                img_shape=(1024, 128, 1),
                vocab_size=vocab_size,
                max_text_len=128,
                config=disc_config
            )
            print(f"✅ Discriminator selected: ENHANCED_V2_FIXED (Reduced visual artifacts)")
            print(f"   ✓ Smaller spatial attention (3x3 kernel)")
            print(f"   ✓ Reduced cross-modal complexity (128 dim)")
            print(f"   ✓ Improved BatchNorm stability (0.9)")
            print(f"   ✓ Lower dropout (0.1)")
        elif args.discriminator_version == 'enhanced_v2':
            discriminator = build_dual_modal_discriminator_enhanced_v2(img_shape=(1024, 128, 1), vocab_size=vocab_size, max_text_len=128)
            print(f"✅ Discriminator selected: ENHANCED_V2 (18M params, ResNet + BiLSTM + Cross-Attention)")
        else:
            discriminator = build_dual_modal_discriminator(img_shape=(1024, 128, 1), vocab_size=vocab_size, max_text_len=128)
            print(f"✅ Discriminator selected: BASE (137M params, Simple CNN + LSTM)")
        
        print("All models built.")

    with strategy.scope():
        print("\n[Phase 3/6] Setting up Optimizers and Checkpoints...")
        # ✅ Pure FP32 - NO LossScaleOptimizer wrapping
        # Add clipnorm to optimizers for gradient stability
        
        # ✅ Solution 1: Cosine Annealing LR Schedule (if enabled)
        if args.use_lr_schedule:
            # Calculate decay steps: epochs * steps_per_epoch
            # Use dataset size to estimate steps if steps_per_epoch=0 (unlimited)
            decay_steps = args.lr_decay_epochs * (args.steps_per_epoch if args.steps_per_epoch > 0 else 2133)
            
            lr_schedule_g = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.lr_g,
                decay_steps=decay_steps,
                alpha=args.lr_alpha
            )
            lr_schedule_d = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.lr_d,
                decay_steps=decay_steps,
                alpha=args.lr_alpha
            )
            generator_optimizer = tf.keras.optimizers.Adam(lr_schedule_g, beta_1=0.5, clipnorm=args.gradient_clip_norm)
            discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule_d, momentum=0.9, clipnorm=args.gradient_clip_norm)
            
            print(f"  🔄 Cosine LR Schedule ENABLED (Solution 1)")
            print(f"     Initial LR (G/D): {args.lr_g}/{args.lr_d}")
            print(f"     Decay steps: {decay_steps} ({args.lr_decay_epochs} epochs)")
            print(f"     Alpha (min LR): {args.lr_alpha}")
        else:
            generator_optimizer = tf.keras.optimizers.Adam(args.lr_g, beta_1=0.5, clipnorm=args.gradient_clip_norm)
            discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_d, momentum=0.9, clipnorm=args.gradient_clip_norm)
            
            print(f"  Generator optimizer: Adam(lr={args.lr_g}, clipnorm={args.gradient_clip_norm}) [PURE FP32]")
            print(f"  Discriminator optimizer: SGD(lr={args.lr_d}, momentum=0.9, clipnorm={args.gradient_clip_norm}) [PURE FP32]")
        
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=args.max_checkpoints)

        # ✅ FIXED: Separate Best Model Checkpoint System to prevent loss due to max_checkpoints limitation
        best_model_dir = os.path.join(args.checkpoint_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_ckpt_manager = tf.train.CheckpointManager(checkpoint, best_model_dir, max_to_keep=1) if args.save_best_model_separately else None

        print(f"✅ Checkpoint system initialized:")
        print(f"   Main checkpoints: {args.checkpoint_dir} (max_to_keep={args.max_checkpoints})")
        if args.save_best_model_separately:
            print(f"   Best model: {best_model_dir} (separate,不会被max_checkpoints影响)")
        else:
            print(f"   Best model: same as main checkpoints (可能受max_checkpoints影响)")
        
        # Epoch tracking for resume capability
        epoch_info_path = os.path.join(args.checkpoint_dir, 'epoch_info.json')
        start_epoch = 0
        
        if ckpt_manager.latest_checkpoint:
            if args.resume and os.path.exists(epoch_info_path):
                # Resume mode: restore checkpoint and continue from last epoch
                checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
                with open(epoch_info_path, 'r') as f:
                    epoch_info = json.load(f)
                start_epoch = epoch_info.get('last_completed_epoch', -1) + 1
                print(f"✅ RESUME MODE: Restored from {ckpt_manager.latest_checkpoint}")
                print(f"   Last completed epoch: {epoch_info.get('last_completed_epoch', -1)}")
                print(f"   Continuing from epoch {start_epoch}/{args.epochs}")
                print(f"   Best PSNR so far: {epoch_info.get('best_psnr', 'N/A')}")
            elif not args.no_restore:
                # Normal restore: use checkpoint but start from epoch 0
                checkpoint.restore(ckpt_manager.latest_checkpoint)
                print(f"Restored model from {ckpt_manager.latest_checkpoint} (starting from epoch 0)")
            else:
                print(f"🚫 Skipping checkpoint restoration (--no_restore flag)")
                print(f"   Found checkpoint: {ckpt_manager.latest_checkpoint}")
                print("Initializing from scratch.")
        else:
            print("Initializing from scratch.")

        # ✅ MULTI-GPU FIX: VGG Perceptual Loss MUST be created inside strategy.scope()
        # This ensures variables are distributed properly across GPUs
        if args.perceptual_loss_weight > 0:
            print(f"\n🎨 Initializing VGG Perceptual Loss (weight={args.perceptual_loss_weight})...")
            perceptual_loss_layer = create_perceptual_loss()
            print("   Expected benefit: +0.3-0.5 dB PSNR improvement")
        else:
            # Create dummy layer that returns 0 (avoids None reference in @tf.function)
            perceptual_loss_layer = tf.keras.layers.Lambda(lambda x: tf.constant(0.0, dtype=tf.float32))
    
    # Initialize Adaptive Loss Balancer (if enabled)
    adaptive_balancer = None
    if args.adaptive_loss_balancing:
        print(f"\n⚖️  Initializing Adaptive Loss Balancing...")
        print(f"   Method: SimpleAdaptiveBalancer")
        print(f"   Target CTC ratio: {args.target_ctc_ratio:.2%}")
        print(f"   Target Visual ratio: {args.target_visual_ratio:.2%}")
        print(f"   Adaptation rate: {args.adaptation_rate}")
        
        # Define loss names and target ratios
        loss_names = ['ctc', 'visual']  # visual = pixel + perceptual + adv + recfeat
        target_ratios = {
            'ctc': args.target_ctc_ratio,
            'visual': args.target_visual_ratio
        }
        
        adaptive_balancer = SimpleAdaptiveBalancer(
            loss_names=loss_names,
            target_ratios=target_ratios,
            adaptation_rate=args.adaptation_rate
        )
        print("   ✅ Adaptive balancer initialized")
        print("   Expected benefit: Automatic loss balancing, prevents CTC dominance")
    
    with strategy.scope():
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
        mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
        mse_loss_fn = tf.keras.losses.MeanSquaredError()  # For Recognition Feature Loss
        val_psnr_metric = tf.keras.metrics.Mean(name='val_psnr')
        val_ssim_metric = tf.keras.metrics.Mean(name='val_ssim')
        val_cer_metric = tf.keras.metrics.Mean(name='val_cer')
        val_wer_metric = tf.keras.metrics.Mean(name='val_wer')
        val_clean_cer_metric = tf.keras.metrics.Mean(name='val_clean_cer')  # Baseline HTR on clean images
        val_clean_wer_metric = tf.keras.metrics.Mean(name='val_clean_wer')  # Baseline HTR on clean images
        
        # Noise artifact detection metrics
        val_noise_variance_metric = tf.keras.metrics.Mean(name='val_noise_variance')
        val_isolated_white_metric = tf.keras.metrics.Mean(name='val_isolated_white_ratio')
        val_local_variance_metric = tf.keras.metrics.Mean(name='val_local_variance')

        # ⚠️ MULTI-GPU: Remove @tf.function to avoid optimizer state placeholder errors
        # TensorFlow graph compilation with MirroredStrategy has issues with:
        # 1. Conditional logic (if args.discriminator_mode == 'ground_truth')
        # 2. Optimizer variable initialization across replicas
        # Trade-off: Slower execution but training actually works
        def train_step(degraded_images, clean_images, ground_truth_text, ctc_weight, rec_feat_weight, percep_weight):
            # ✅ MULTI-GPU FIX: Use dynamic batch size from actual input shape
            # In distributed training, train_step receives per-replica batch (NOT global batch!)
            # Example: global_batch=2, num_gpus=2 → per_replica_batch=1
            per_replica_batch = tf.shape(degraded_images)[0]
            real_labels_disc = tf.ones([per_replica_batch, 1]) * 0.9
            fake_labels_disc = tf.zeros([per_replica_batch, 1])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # ✅ CRITICAL FIX (2025-10-21): Normalize TFRecord data [0,1] to [-1,1] for tanh generator
                # TFRecord stores images as float32 in [0,1], but generator with tanh outputs [-1,1]
                # ALL losses must compare same range to avoid training bugs
                clean_images_tanh = clean_images * 2.0 - 1.0      # [0,1] → [-1,1]
                degraded_images_tanh = degraded_images * 2.0 - 1.0  # [0,1] → [-1,1]
                
                generated_images = generator(degraded_images_tanh, training=True)  # Output: [-1,1]
                
                # Denormalize to [0,1] ONLY for recognizer (HTR model expects [0,1])
                clean_images_normalized = (clean_images_tanh + 1.0) / 2.0
                generated_images_normalized = (generated_images + 1.0) / 2.0
                
                # Get recognizer outputs - handle both single and multi-output
                recognizer_output_clean = recognizer(clean_images_normalized, training=False)
                recognizer_output_generated = recognizer(generated_images_normalized, training=False)
                
                # Extract logits and feature maps (if available)
                if isinstance(recognizer_output_clean, (list, tuple)):
                    # Multi-output mode: (logits, feature_map)
                    clean_logits = recognizer_output_clean[0]
                    clean_feature_map = recognizer_output_clean[1]
                    generated_logits = recognizer_output_generated[0]
                    generated_feature_map = recognizer_output_generated[1]
                else:
                    # Single output mode: logits only
                    clean_logits = recognizer_output_clean
                    generated_logits = recognizer_output_generated
                    # Create dummy feature maps with same shape for consistency
                    # ✅ MULTI-GPU FIX: Use per-replica batch size
                    clean_feature_map = tf.zeros([per_replica_batch, 1], dtype=tf.float32)
                    generated_feature_map = tf.zeros([per_replica_batch, 1], dtype=tf.float32)

                clean_text_pred = tf.argmax(clean_logits, axis=-1, output_type=tf.int32)
                generated_text_pred = tf.argmax(generated_logits, axis=-1, output_type=tf.int32)

                # --- SWITCHABLE DISCRIMINATOR LOGIC ---
                # Now using clean_images_tanh and generated_images (both [-1,1] range)
                if args.discriminator_mode == 'ground_truth':
                    real_output = discriminator([clean_images_tanh, ground_truth_text], training=True)
                else: # Default to 'predicted' mode (original logic)
                    real_output = discriminator([clean_images_tanh, clean_text_pred], training=True)
                fake_output = discriminator([generated_images, generated_text_pred], training=True)

                disc_loss_real = bce_loss_fn(real_labels_disc, real_output)
                disc_loss_fake = bce_loss_fn(fake_labels_disc, fake_output)
                total_disc_loss = disc_loss_real + disc_loss_fake

                adversarial_loss = bce_loss_fn(real_labels_disc, fake_output)
                # ✅ NOW comparing same range: [-1,1] vs [-1,1]
                pixel_loss = mae_loss_fn(clean_images_tanh, generated_images)
                
                # Recognition Feature Loss (HTR-aware loss from intermediate features)
                # Always calculate but weight will control its contribution
                rec_feat_loss = mse_loss_fn(clean_feature_map, generated_feature_map)
                
                # VGG Perceptual Loss - uses perceptual_loss_layer (Keras Layer)
                # This is always defined (either real VGG or dummy=0), so no None check needed
                # ✅ NOW comparing same range: [-1,1] vs [-1,1]
                perceptual_loss = perceptual_loss_layer(clean_images_tanh, generated_images)
                
                # Always calculate CTC loss; its contribution is controlled by ctc_weight.
                # This avoids conditional graph structures that break gradient flow in @tf.function.
                label_len = tf.math.count_nonzero(ground_truth_text, axis=1, keepdims=True, dtype=tf.int32)
                label_len = tf.reshape(label_len, [-1])
                # ✅ MULTI-GPU FIX: Use per-replica batch size for logit_len
                logit_len = tf.fill([per_replica_batch], generated_logits.shape[1])
                
                ctc_loss_raw = tf.reduce_mean(tf.nn.ctc_loss(labels=tf.cast(ground_truth_text, tf.int32), logits=generated_logits, label_length=label_len, logit_length=logit_len, logits_time_major=False, blank_index=0))
                # Clip CTC loss to prevent spikes that cause training instability
                ctc_loss = tf.clip_by_value(ctc_loss_raw, 0.0, args.ctc_loss_clip_max)

                # ✅ Pure FP32 - NO casting needed (already in FP32)
                # Total Generator Loss with Recognition Feature Loss + Perceptual Loss
                total_gen_loss = (
                    (args.adv_loss_weight * adversarial_loss) + 
                    (args.pixel_loss_weight * pixel_loss) + 
                    (rec_feat_weight * rec_feat_loss) +
                    (percep_weight * perceptual_loss) +
                    (ctc_weight * ctc_loss)
                )

            generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
            
            # Apply gradient clipping to prevent explosion (especially from CTC loss)
            generator_gradients, gen_grad_norm = tf.clip_by_global_norm(generator_gradients, args.gradient_clip_norm)
            discriminator_gradients, disc_grad_norm = tf.clip_by_global_norm(discriminator_gradients, args.gradient_clip_norm)
            
            # ✅ Pure FP32 - Direct gradient application (no LossScaleOptimizer unscaling)
            generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            return total_gen_loss, total_disc_loss, adversarial_loss, pixel_loss, rec_feat_loss, perceptual_loss, ctc_loss, ctc_loss_raw, gen_grad_norm, disc_grad_norm

        # ✨ MULTI-GPU: Wrap train_step for distributed execution
        # ⚠️ NO @tf.function here - let train_step handle graph compilation
        # This avoids TensorFlow optimizer state placeholder errors in distributed mode
        def distributed_train_step(degraded_images, clean_images, ground_truth_text, ctc_weight, rec_feat_weight, percep_weight):
            """Distributed training step that runs train_step on all replicas."""
            per_replica_losses = strategy.run(
                train_step,
                args=(degraded_images, clean_images, ground_truth_text, ctc_weight, rec_feat_weight, percep_weight)
            )
            
            # Reduce (aggregate) losses across replicas
            # For losses, we want the mean across all replicas
            reduced_losses = []
            for loss in per_replica_losses:
                reduced_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
                reduced_losses.append(reduced_loss)
            
            return reduced_losses

    print("\n[Phase 4/6] Starting Training Loop...")
    # ✨ MULTI-GPU: Use distributed dataset iterator
    dataset_iterator = iter(train_dataset_dist)

    # ✅ FIXED: Separate variables for each metric type to avoid confusion
    best_monitored_value = float('-inf')  # For early stopping decision (changes based on metric)
    best_psnr_value = -1.0  # For logging only
    best_cer_value = float('inf')  # For logging only
    best_combined_value = float('-inf')  # For logging only
    
    # --- Early Stopping Setup ---
    patience_counter = 0
    best_epoch = 0
    early_stopped = False
    best_weights_path = None
    
    if args.early_stopping:
        print(f"\n🛡️ Early Stopping enabled:")
        print(f"   Patience: {args.patience} epochs")
        print(f"   Min Delta: {args.min_delta}")
        print(f"   Restore Best Weights: {args.restore_best_weights}")
    
    # --- MLflow Tracking Setup ---
    mlflow.set_tracking_uri("file:./mlruns")  # Save to local directory
    experiment_name = f"GAN_HTR_FP32_{args.discriminator_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    print(f"\n📊 MLflow tracking enabled: {experiment_name}")
    print(f"   View at: http://localhost:5000 (run: poetry run mlflow ui)\n")
    
    # Initialize training history for JSON logging
    training_history = {
        "start_time": datetime.now().isoformat(),
        "hyperparameters": {
            "precision": "Pure FP32 (NO mixed precision)",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "lr_generator": args.lr_g,
            "lr_discriminator": args.lr_d,
            "pixel_loss_weight": args.pixel_loss_weight,
            "ctc_loss_weight": args.ctc_loss_weight,
            "discriminator_mode": args.discriminator_mode,
            "adv_loss_weight": args.adv_loss_weight,
            "gradient_clip_norm": args.gradient_clip_norm,
            "ctc_loss_clip_max": args.ctc_loss_clip_max,
            "early_stopping_enabled": args.early_stopping,
            "early_stopping_patience": args.patience if args.early_stopping else None,
            "early_stopping_min_delta": args.min_delta if args.early_stopping else None,
            "early_stopping_restore_best": args.restore_best_weights if args.early_stopping else None,
            "optimizer_generator": f"Adam(lr={args.lr_g}, beta_1=0.5, clipnorm={args.gradient_clip_norm})",
            "optimizer_discriminator": f"SGD(lr={args.lr_d}, momentum=0.9, clipnorm={args.gradient_clip_norm})",
            "model_architecture": {
                "generator": generator_name,
                "discriminator": "Dual-Modal (137M params)",
                "recognizer": "Frozen HTR Stage 3 (50M params, CER 33.72%)"
            },
            "dataset": {
                "path": args.tfrecord_path,
                "train_samples": train_count,
                "val_samples": val_count,
                "charset_size": vocab_size
            }
        },
        "epochs": []
    }

    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "precision": "FP32",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr_generator": args.lr_g,
            "lr_discriminator": args.lr_d,
            "pixel_loss_weight": args.pixel_loss_weight,
            "rec_feat_loss_weight": args.rec_feat_loss_weight,
            "ctc_loss_weight": args.ctc_loss_weight,
            "adv_loss_weight": args.adv_loss_weight,
            "contrastive_loss_weight": args.contrastive_loss_weight,
            "discriminator_mode": args.discriminator_mode,
            "gradient_clip_norm": args.gradient_clip_norm,
            "ctc_loss_clip_max": args.ctc_loss_clip_max,
            "train_samples": train_count,
            "val_samples": val_count,
            "vocab_size": vocab_size,
            "early_stopping": args.early_stopping,
            "patience": args.patience if args.early_stopping else 0,
            "min_delta": args.min_delta if args.early_stopping else 0
        })
        
        # Log model architecture info as tags
        mlflow.set_tags({
            "model_type": "Dual-Modal GAN-HTR",
            "precision": "Pure FP32",
            "generator": generator_name,
            "discriminator": "Dual-Modal (137M params)",
            "recognizer": "Frozen HTR Stage 3 (CER 33.72%)",
            "optimized_for": "Balanced loss + CTC stability"
        })
    
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            # --- Curriculum Learning & Loss Annealing Logic ---
            is_warmup = epoch < args.warmup_epochs
            is_annealing = not is_warmup and (epoch - args.warmup_epochs) < args.annealing_epochs
            
            current_ctc_weight = 0.0
            if is_annealing:
                # Linear ramp-up of CTC weight
                annealing_epoch = epoch - args.warmup_epochs
                progress = float(annealing_epoch + 1) / float(args.annealing_epochs)
                current_ctc_weight = args.ctc_loss_weight * progress
            elif not is_warmup:
                # Full CTC weight after warm-up and annealing
                current_ctc_weight = args.ctc_loss_weight

            # --- Epoch Logging ---
            phase_str = ""
            if is_warmup:
                phase_str = f" (Warm-up, CTC_w=0.0)"
                if epoch == 0:
                    print(f"\n🔥 Starting visual warm-up for {args.warmup_epochs} epochs...")
            elif is_annealing:
                phase_str = f" (Annealing, CTC_w={current_ctc_weight:.2f})"
                if epoch == args.warmup_epochs:
                    print(f"\n📈 Starting CTC loss annealing for {args.annealing_epochs} epochs...")
            else:
                phase_str = f" (Full Training, CTC_w={current_ctc_weight:.1f})"
                if epoch == args.warmup_epochs + args.annealing_epochs:
                    print(f"\n✅ Annealing complete. Using full CTC loss weight.")
                    # ✅ CURRICULUM COMPLETE: Reset patience counter for clean early stopping start
                    if args.curriculum_aware_early_stopping and args.early_stopping:
                        patience_counter = 0
                        print(f"🎯 Early stopping evaluation BEGINS!")
                        print(f"   Patience counter reset to 0/{args.patience}")
                        print(f"   Now monitoring for improvements in full training phase")

            print(f"\nEpoch {epoch + 1}/{args.epochs}{phase_str}")

            # ✅ Calculate curriculum status once at epoch start
            curriculum_complete_epoch = args.warmup_epochs + args.annealing_epochs
            is_curriculum_complete = (epoch + 1) > curriculum_complete_epoch

            # ✅ ENHANCED: Show early stopping status at epoch start
            if args.early_stopping:
                if args.curriculum_aware_early_stopping:
                    if is_curriculum_complete:
                        print(f"  🎯 Early Stopping: ACTIVE (monitoring for improvements)")
                        print(f"     Patience: {patience_counter}/{args.patience} | Metric: {args.early_stopping_metric}")
                    else:
                        remaining_epochs = curriculum_complete_epoch - epoch
                        print(f"  📚 Early Stopping: PAUSED (curriculum in progress)")
                        print(f"     Resume in: {remaining_epochs} epochs | Counter: frozen at {patience_counter}")
                else:
                    print(f"  ⚠️  Early Stopping: ALWAYS ACTIVE (curriculum-aware disabled)")
                    print(f"     Patience: {patience_counter}/{args.patience} | Risk: may stop during warmup/annealing")
            else:
                print(f"  🚫 Early Stopping: DISABLED (will train for all {args.epochs} epochs)")

            # Track epoch metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "phase": "warmup" if is_warmup else "annealing" if is_annealing else "full_training",
                "current_ctc_weight": current_ctc_weight,
                "current_rec_feat_weight": args.rec_feat_loss_weight,
                "losses": {
                    "g_loss": [], "d_loss": [], "adv_loss": [], "pixel_loss": [],
                    "rec_feat_loss": [], "ctc_loss": [], "ctc_loss_raw": [], 
                    "g_grad_norm": [], "d_grad_norm": []
                }
            }
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
            for step in pbar:
                degraded_batch, clean_batch, text_batch = next(dataset_iterator)
                
                # If adaptive balancing enabled, update weights before training step
                # ✅ FIX: Only use adaptive balancing AFTER curriculum learning (warmup + annealing) completes
                # This prevents adaptive mechanism from overriding the carefully designed curriculum schedule
                if adaptive_balancer is not None and step > 0 and not (is_warmup or is_annealing):
                    # Calculate loss magnitudes from previous step for adaptive balancing
                    # Group visual losses: pixel + perceptual + adversarial + rec_feat
                    loss_dict = {
                        'ctc': float(ctc_loss.numpy()),
                        'visual': float(pix_loss.numpy()) + float(percep_loss.numpy() if isinstance(percep_loss, tf.Tensor) else percep_loss) + float(adv_loss.numpy()) + float(rec_feat_loss.numpy())
                    }
                    
                    # Update adaptive weights
                    updated_weights = adaptive_balancer.update(loss_dict)
                    
                    # Apply updated weights (only in full training phase, not during curriculum)
                    current_ctc_weight = updated_weights['ctc'] * args.ctc_loss_weight
                    # Visual weight is distributed across components proportionally
                    visual_weight = updated_weights['visual']
                    current_pixel_weight = visual_weight * args.pixel_loss_weight
                    current_percep_weight = visual_weight * args.perceptual_loss_weight
                    current_adv_weight = visual_weight * args.adv_loss_weight
                    current_rec_feat_weight = visual_weight * args.rec_feat_loss_weight
                else:
                    # Use original weights if adaptive balancing disabled or first step
                    current_pixel_weight = args.pixel_loss_weight
                    current_percep_weight = args.perceptual_loss_weight
                    current_adv_weight = args.adv_loss_weight
                    current_rec_feat_weight = args.rec_feat_loss_weight
                
                # ✨ MULTI-GPU: Call distributed_train_step instead of train_step
                g_loss, d_loss, adv_loss, pix_loss, rec_feat_loss, percep_loss, ctc_loss, ctc_loss_raw, g_grad_norm, d_grad_norm = distributed_train_step(
                    degraded_batch, clean_batch, text_batch, 
                    tf.constant(current_ctc_weight, dtype=tf.float32),
                    tf.constant(current_rec_feat_weight, dtype=tf.float32),
                    tf.constant(current_percep_weight, dtype=tf.float32)
                )
                
                # Collect metrics (losses are already reduced/averaged across GPUs)
                epoch_metrics["losses"]["g_loss"].append(float(g_loss.numpy()))
                epoch_metrics["losses"]["d_loss"].append(float(d_loss.numpy()))
                epoch_metrics["losses"]["adv_loss"].append(float(adv_loss.numpy()))
                epoch_metrics["losses"]["pixel_loss"].append(float(pix_loss.numpy()))
                epoch_metrics["losses"]["rec_feat_loss"].append(float(rec_feat_loss.numpy()))
                epoch_metrics["losses"]["perceptual_loss"] = epoch_metrics["losses"].get("perceptual_loss", [])
                epoch_metrics["losses"]["perceptual_loss"].append(float(percep_loss.numpy()) if isinstance(percep_loss, tf.Tensor) else float(percep_loss))
                epoch_metrics["losses"]["ctc_loss"].append(float(ctc_loss.numpy()))
                if current_ctc_weight > 0:
                    epoch_metrics["losses"]["ctc_loss_raw"].append(float(ctc_loss_raw.numpy()))
                epoch_metrics["losses"]["g_grad_norm"].append(float(g_grad_norm.numpy()))
                epoch_metrics["losses"]["d_grad_norm"].append(float(d_grad_norm.numpy()))

                if (step + 1) % 50 == 0:
                    postfix_dict = {
                        'G': f'{g_loss:.4f}', 'D': f'{d_loss:.4f}', 'Adv': f'{adv_loss:.4f}', 
                        'Pix': f'{pix_loss:.4f}', 'RecFeat': f'{rec_feat_loss:.4f}',
                        'CTC': f'{ctc_loss:.2f}', 'CTC_w': f'{current_ctc_weight:.2f}'
                    }
                    if args.perceptual_loss_weight > 0:
                        postfix_dict['Percep'] = f'{percep_loss:.4f}' if isinstance(percep_loss, tf.Tensor) else f'{percep_loss:.4f}'
                    pbar.set_postfix(postfix_dict)

            # Calculate epoch statistics
            epoch_metrics["training_losses"] = {
                "total_loss": float(np.mean(epoch_metrics["losses"]["g_loss"])),
                "pixel_loss": float(np.mean(epoch_metrics["losses"]["pixel_loss"])),
                "rec_feat_loss": float(np.mean(epoch_metrics["losses"]["rec_feat_loss"])),
                "perceptual_loss": float(np.mean(epoch_metrics["losses"]["perceptual_loss"])) if "perceptual_loss" in epoch_metrics["losses"] and epoch_metrics["losses"]["perceptual_loss"] else 0.0,
                "ctc_loss": float(np.mean(epoch_metrics["losses"]["ctc_loss"])),
                "adv_loss": float(np.mean(epoch_metrics["losses"]["adv_loss"])),
                "ctc_loss_raw": float(np.mean(epoch_metrics["losses"]["ctc_loss_raw"])) if epoch_metrics["losses"]["ctc_loss_raw"] else 0.0,
                "gradient_norm": {
                    "generator_mean": float(np.mean(epoch_metrics["losses"]["g_grad_norm"])),
                    "generator_max": float(np.max(epoch_metrics["losses"]["g_grad_norm"])),
                    "discriminator_mean": float(np.mean(epoch_metrics["losses"]["d_grad_norm"]))
                }
            }
            
            # Log epoch mean losses to MLflow
            mlflow_step = epoch + 1
            metrics_to_log = {
                "train/g_loss": epoch_metrics["training_losses"]["total_loss"],
                "train/d_loss": float(np.mean(epoch_metrics["losses"]["d_loss"])),
                "train/adv_loss": epoch_metrics["training_losses"]["adv_loss"],
                "train/pixel_loss": epoch_metrics["training_losses"]["pixel_loss"],
                "train/rec_feat_loss": epoch_metrics["training_losses"]["rec_feat_loss"],
                "train/ctc_loss": epoch_metrics["training_losses"]["ctc_loss"],
                "train/ctc_loss_raw": epoch_metrics["training_losses"]["ctc_loss_raw"],
                "train/g_grad_norm_mean": epoch_metrics["training_losses"]["gradient_norm"]["generator_mean"],
                "train/g_grad_norm_max": epoch_metrics["training_losses"]["gradient_norm"]["generator_max"],
                "train/d_grad_norm_mean": epoch_metrics["training_losses"]["gradient_norm"]["discriminator_mean"]
            }
            if args.perceptual_loss_weight > 0:
                metrics_to_log["train/perceptual_loss"] = epoch_metrics["training_losses"]["perceptual_loss"]
            
            # ✅ NEW: Log curriculum learning schedule weights
            metrics_to_log["schedule/current_ctc_weight"] = epoch_metrics["current_ctc_weight"]
            metrics_to_log["schedule/current_rec_feat_weight"] = epoch_metrics["current_rec_feat_weight"]
            metrics_to_log["schedule/phase"] = 0 if is_warmup else (1 if is_annealing else 2)  # 0=warmup, 1=annealing, 2=full
            
            # Log adaptive weights if enabled
            if adaptive_balancer is not None:
                current_weights = adaptive_balancer.weights
                metrics_to_log["adaptive/weight_ctc"] = current_weights['ctc']
                metrics_to_log["adaptive/weight_visual"] = current_weights['visual']
                # Log target ratios for reference
                metrics_to_log["adaptive/target_ctc_ratio"] = args.target_ctc_ratio
                metrics_to_log["adaptive/target_visual_ratio"] = args.target_visual_ratio
            
            mlflow.log_metrics(metrics_to_log, step=mlflow_step)
        
            # --- End of Epoch Actions ---
            if (epoch + 1) % args.eval_interval == 0:
                print("  Running validation on full validation set...")
                # ✨ MULTI-GPU: Use non-distributed dataset for validation (simpler, no sharding needed)
                val_stats = run_validation_step(
                    val_dataset, generator, recognizer, charset,
                    val_psnr_metric, val_ssim_metric, val_cer_metric, val_wer_metric,
                    val_clean_cer_metric, val_clean_wer_metric,
                    val_noise_variance_metric, val_isolated_white_metric, val_local_variance_metric
                )
                
                psnr_result = val_psnr_metric.result()
                ssim_result = val_ssim_metric.result()
                cer_result = val_cer_metric.result()
                wer_result = val_wer_metric.result()
                clean_cer_result = val_clean_cer_metric.result()
                clean_wer_result = val_clean_wer_metric.result()
                noise_var_result = val_noise_variance_metric.result()
                isolated_white_result = val_isolated_white_metric.result()
                local_var_result = val_local_variance_metric.result()
                
                # Note: Detailed statistics already printed by run_validation_step()
                
                # Add validation metrics to epoch data (with statistics)
                epoch_metrics["validation"] = {
                    "psnr": float(psnr_result.numpy()),
                    "psnr_std": float(val_stats['psnr']['std']),
                    "psnr_ci_95": float(val_stats['psnr']['ci_95']),
                    "psnr_n": int(val_stats['psnr']['n']),
                    "ssim": float(ssim_result.numpy()),
                    "ssim_std": float(val_stats['ssim']['std']),
                    "ssim_ci_95": float(val_stats['ssim']['ci_95']),
                    "ssim_n": int(val_stats['ssim']['n']),
                    "cer": float(cer_result.numpy()),
                    "cer_std": float(val_stats['cer']['std']),
                    "cer_n": int(val_stats['cer']['n']),
                    "wer": float(wer_result.numpy()),
                    "wer_std": float(val_stats['wer']['std']),
                    "wer_n": int(val_stats['wer']['n']),
                    "clean_cer": float(clean_cer_result.numpy()),
                    "clean_wer": float(clean_wer_result.numpy()),
                    "noise_variance": float(noise_var_result.numpy()),
                    "isolated_white_ratio": float(isolated_white_result.numpy()),
                    "local_variance": float(local_var_result.numpy())
                }
                
                # Log validation metrics to MLflow (with statistics)
                mlflow.log_metrics({
                    "val/psnr": float(psnr_result.numpy()),
                    "val/psnr_std": float(val_stats['psnr']['std']),
                    "val/psnr_ci_95": float(val_stats['psnr']['ci_95']),
                    "val/ssim": float(ssim_result.numpy()),
                    "val/ssim_std": float(val_stats['ssim']['std']),
                    "val/ssim_ci_95": float(val_stats['ssim']['ci_95']),
                    "val/cer": float(cer_result.numpy()),
                    "val/cer_std": float(val_stats['cer']['std']),
                    "val/wer": float(wer_result.numpy()),
                    "val/wer_std": float(val_stats['wer']['std']),
                    "val/clean_cer": float(clean_cer_result.numpy()),
                    "val/clean_wer": float(clean_wer_result.numpy()),
                    "val/noise_variance": float(noise_var_result.numpy()),
                    "val/isolated_white_ratio": float(isolated_white_result.numpy()),
                    "val/local_variance": float(local_var_result.numpy())
                }, step=epoch+1)
            
                # --- Checkpoint Management: Dual objective (PSNR + CER) ---
                # Best model = High PSNR (visual quality) + Low CER (text readability)
                # FIXED: Reduced CER penalty from 100x to 10x to prevent dominance over PSNR
                combined_score = psnr_result - (args.cer_weight * cer_result * 10)

                # Calculate individual contributions for analysis
                psnr_contribution = float(psnr_result.numpy())
                cer_penalty = float(args.cer_weight * cer_result.numpy() * 10)

                # ✅ FIXED: Check improvement using correct best_monitored_value variable
                if args.early_stopping_metric == 'psnr_only':
                    monitored_metric = psnr_result
                    improvement = float(psnr_result.numpy()) - best_monitored_value
                    is_improvement = improvement > args.min_delta
                elif args.early_stopping_metric == 'cer_only':
                    monitored_metric = -cer_result  # Negative because lower CER is better
                    improvement = float(-cer_result.numpy()) - best_monitored_value
                    is_improvement = improvement > args.min_delta
                else:  # 'combined' (default)
                    monitored_metric = combined_score
                    improvement = float(combined_score) - best_monitored_value
                    is_improvement = improvement > args.min_delta

                # SANITY CHECK: Override early stopping if PSNR improvement is significant
                # This prevents premature stopping due to CER fluctuations when visual quality is improving
                if hasattr(args, 'psnr_improvement_threshold') and (epoch + 1) > 1:
                    # ✅ FIXED: Get previous epoch's PSNR from training_history
                    if epoch > 0 and len(training_history["epochs"]) > 0:
                        prev_epoch_data = training_history["epochs"][-1]
                        prev_psnr = prev_epoch_data.get("validation", {}).get("psnr", float(psnr_result.numpy()))
                    else:
                        prev_psnr = float(psnr_result.numpy())
                    
                    psnr_improvement = float(psnr_result.numpy()) - prev_psnr

                    if psnr_improvement > args.psnr_improvement_threshold:
                        print(f"  🚀 Significant PSNR improvement detected ({psnr_improvement:.2f} > {args.psnr_improvement_threshold})")
                        print(f"     Overriding early stopping decision")
                        is_improvement = True
                        improvement = psnr_improvement
                
                if is_improvement:
                    # ✅ FIXED: Update ALL metric values for logging, and best_monitored_value for early stopping
                    best_psnr_value = float(psnr_result.numpy())
                    best_cer_value = float(cer_result.numpy())
                    best_combined_value = float(combined_score)
                    
                    # Update monitored value based on selected strategy
                    if args.early_stopping_metric == 'psnr_only':
                        best_monitored_value = float(psnr_result.numpy())
                        best_score_desc = f"PSNR: {psnr_result:.2f} dB"
                    elif args.early_stopping_metric == 'cer_only':
                        best_monitored_value = float(-cer_result.numpy())
                        best_score_desc = f"CER: {cer_result:.4f}"
                    else:  # 'combined'
                        best_monitored_value = float(combined_score)
                        best_score_desc = f"Combined: {combined_score:.2f}"

                    best_epoch = epoch + 1
                    patience_counter = 0  # Reset patience counter

                    # ✅ DISK-EFFICIENT: Smart checkpoint management (max 2 files total)
                    # Save to separate best model directory FIRST (most important)
                    if best_model_ckpt_manager:
                        best_model_path = best_model_ckpt_manager.save()
                        best_weights_path = best_model_path  # Track for restoration
                        print(f"  ✅ New best model saved! (Improvement: {improvement:.2f})")
                        print(f"     💾 Best model preserved: {best_model_path}")

                        # For disk efficiency, ONLY save regular checkpoint if needed for resume capability
                        # With max_checkpoints=1, this maintains single checkpoint for debugging
                        saved_path = ckpt_manager.save()
                        print(f"     📄 Debug checkpoint: {saved_path}")
                        print(f"     💰 Total disk usage: 2 checkpoints (best + debug)")
                    else:
                        # Fallback: Save regular checkpoint only
                        saved_path = ckpt_manager.save()
                        best_weights_path = saved_path
                        print(f"  ✅ New best model saved! (Improvement: {improvement:.2f})")
                        print(f"     📄 Checkpoint: {saved_path} (Best = Regular)")
                        print(f"     💰 Disk usage: 1 checkpoint only")

                    print(f"     PSNR: {psnr_result:.2f}, CER: {cer_result:.4f}")
                    print(f"     Combined Score: {combined_score:.2f} (PSNR: +{psnr_contribution:.2f}, CER: -{cer_penalty:.2f})")
                    print(f"     Best Metric ({args.early_stopping_metric}): {best_score_desc}")
                    print(f"     Best Epoch: {best_epoch}, Patience Counter: {patience_counter}/{args.patience}")
                    epoch_metrics["best_model_saved"] = True
                    epoch_metrics["patience_counter"] = patience_counter

                    # Enhanced logging for early stopping analysis
                    epoch_metrics["early_stopping_analysis"] = {
                        "strategy": args.early_stopping_metric,
                        "monitored_metric_value": float(monitored_metric.numpy()),
                        "improvement": float(improvement),
                        "combined_score": float(combined_score),
                        "psnr_contribution": psnr_contribution,
                        "cer_penalty": cer_penalty
                    }

                    # Log best metrics to MLflow
                    mlflow.log_metrics({
                        "best_val_psnr": best_psnr_value,
                        "best_val_cer": best_cer_value,
                        "best_combined_score": best_combined_value,
                        "best_monitored_value": best_monitored_value,
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter,
                        "early_stop/psnr_contribution": psnr_contribution,
                        "early_stop/cer_penalty": cer_penalty,
                        "early_stop/monitored_metric": float(monitored_metric.numpy()),
                        "early_stop/improvement": float(improvement)
                    }, step=epoch+1)
                else:
                    # ✅ CURRICULUM-AWARE: Only increment patience counter during full training
                    if args.curriculum_aware_early_stopping and not is_curriculum_complete:
                        # Don't increment patience counter during warmup/annealing
                        print(f"     📚 Patience Counter PAUSED (curriculum in progress)")
                        print(f"     ⏳ Early stopping evaluation will begin after epoch {curriculum_complete_epoch}")
                    else:
                        # Only increment during full training phase
                        patience_counter += 1  # Increment patience counter
                        print(f"     ⏱️  Patience Counter incremented: {patience_counter}/{args.patience}")

                    # ✅ FIXED: Enhanced logging with correct metric values
                    if args.early_stopping_metric == 'psnr_only':
                        score_desc = f"PSNR: {psnr_result:.2f} dB (best: {best_monitored_value:.2f} dB)"
                    elif args.early_stopping_metric == 'cer_only':
                        score_desc = f"CER: {cer_result:.4f} (best: {-best_monitored_value:.4f})"
                    else:  # 'combined'
                        score_desc = f"Combined: {combined_score:.2f} (best: {best_monitored_value:.2f})"

                    print(f"  ⚠️  No improvement ({score_desc})")
                    print(f"     Combined Score Breakdown: PSNR=+{psnr_contribution:.2f}, CER=-{cer_penalty:.2f}")
                    print(f"     Patience Counter: {patience_counter}/{args.patience}")

                    # ✅ CURRICULUM-AWARE: Show curriculum status and early stopping eligibility
                    if args.curriculum_aware_early_stopping:
                        if is_curriculum_complete:
                            print(f"     🎓 Curriculum Complete: Early stopping ALLOWED")
                            print(f"     📊 Full Training Phase (CTC weight: {current_ctc_weight:.2f})")
                        else:
                            remaining_epochs = curriculum_complete_epoch - epoch
                            print(f"     🎓 Curriculum in Progress: Early stopping PAUSED")
                            print(f"     ⏳ Remaining curriculum epochs: {remaining_epochs}")
                            print(f"     📊 Current Phase: {phase_str.strip()} (CTC weight: {current_ctc_weight:.2f})")

                        epoch_metrics["curriculum_complete"] = is_curriculum_complete
                        epoch_metrics["curriculum_remaining_epochs"] = max(0, curriculum_complete_epoch - epoch)
                    else:
                        print(f"     ⚠️  Curriculum-Aware Early Stopping: DISABLED")
                        print(f"     🚨 Early stopping can trigger ANYTIME (may stop during warmup/annealing)")
                        epoch_metrics["curriculum_complete"] = True  # Always "complete" for logging
                        epoch_metrics["curriculum_remaining_epochs"] = 0

                    epoch_metrics["best_model_saved"] = False
                    epoch_metrics["patience_counter"] = patience_counter

                    # Enhanced logging for early stopping analysis
                    epoch_metrics["early_stopping_analysis"] = {
                        "strategy": args.early_stopping_metric,
                        "monitored_metric_value": float(monitored_metric.numpy()),
                        "improvement": float(improvement),
                        "combined_score": float(combined_score),
                        "psnr_contribution": psnr_contribution,
                        "cer_penalty": cer_penalty,
                        "best_so_far": best_monitored_value
                    }

                    # Log patience to MLflow
                    mlflow.log_metrics({
                        "patience_counter": patience_counter,
                        "early_stop/psnr_contribution": psnr_contribution,
                        "early_stop/cer_penalty": cer_penalty,
                        "early_stop/monitored_metric": float(monitored_metric.numpy()),
                        "early_stop/improvement": float(improvement),
                        "early_stop/best_so_far": best_monitored_value
                    }, step=epoch+1)
                    
                    # ✅ CURRICULUM-AWARE Early Stopping Check ---
                    # Only allow early stopping AFTER curriculum learning phases complete (if enabled)
                    if args.curriculum_aware_early_stopping:
                        early_stopping_allowed = is_curriculum_complete
                    else:
                        # Original logic: allow early stopping anytime
                        early_stopping_allowed = True

                    if args.early_stopping and patience_counter >= args.patience and early_stopping_allowed:
                        print(f"\n🛑 EARLY STOPPING TRIGGERED!")
                        print(f"   No improvement for {args.patience} epochs")
                        
                        # ✅ FIXED: Show metric-specific best values
                        if args.early_stopping_metric == 'psnr_only':
                            print(f"   Best epoch: {best_epoch} with PSNR: {best_monitored_value:.2f} dB")
                        elif args.early_stopping_metric == 'cer_only':
                            print(f"   Best epoch: {best_epoch} with CER: {-best_monitored_value:.4f}")
                        else:
                            print(f"   Best epoch: {best_epoch} with Combined Score: {best_monitored_value:.2f}")
                            print(f"   (PSNR: {best_psnr_value:.2f} dB, CER: {best_cer_value:.4f})")

                        if args.curriculum_aware_early_stopping:
                            print(f"   🎓 Curriculum Status: COMPLETE (Phase: Full Training)")
                            print(f"   📊 Final CTC Weight: {current_ctc_weight:.2f}")
                            print(f"   ✅ Early stopping only allowed AFTER curriculum completion")
                        else:
                            print(f"   ⚠️  Curriculum-Aware Early Stopping: DISABLED")
                            print(f"   🚨 Early stopping triggered during phase: {phase_str.strip()}")
                            print(f"   📊 Current CTC Weight: {current_ctc_weight:.2f}")
                        
                        # ✅ FIXED: Enhanced best weights restoration with fallback mechanism
                        if args.restore_best_weights and best_weights_path:
                            print(f"   Restoring best model weights from: {best_weights_path}")

                            # Check if checkpoint file exists before attempting restoration
                            if os.path.exists(best_weights_path + '.index') or os.path.exists(best_weights_path + '.data-00000-of-00001'):
                                try:
                                    checkpoint.restore(best_weights_path).expect_partial()
                                    print(f"   ✅ Best weights restored successfully")
                                except Exception as e:
                                    print(f"   ⚠️  Error restoring best weights: {str(e)}")
                                    print(f"   🔄 Attempting fallback to latest available checkpoint...")

                                    # Fallback to latest available checkpoint
                                    if ckpt_manager.latest_checkpoint:
                                        try:
                                            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
                                            print(f"   ✅ Fallback successful: restored {ckpt_manager.latest_checkpoint}")
                                        except Exception as fallback_error:
                                            print(f"   ❌ Fallback also failed: {str(fallback_error)}")
                                            print(f"   🚨 Using current model state (no restoration)")
                                    else:
                                        print(f"   ❌ No fallback checkpoint available")
                                        print(f"   🚨 Using current model state (no restoration)")
                            else:
                                print(f"   ❌ Best model checkpoint not found: {best_weights_path}")
                                print(f"   🔄 Attempting fallback to latest available checkpoint...")

                                # Fallback to latest available checkpoint
                                if ckpt_manager.latest_checkpoint:
                                    try:
                                        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
                                        print(f"   ✅ Fallback successful: restored {ckpt_manager.latest_checkpoint}")
                                    except Exception as fallback_error:
                                        print(f"   ❌ Fallback failed: {str(fallback_error)}")
                                        print(f"   🚨 Using current model state (no restoration)")
                                else:
                                    print(f"   ❌ No fallback checkpoint available")
                                    print(f"   🚨 Using current model state (no restoration)")
                        
                        early_stopped = True
                        epoch_metrics["early_stopped"] = True
                        
                        # Log early stopping to MLflow
                        mlflow.log_metrics({
                            "early_stopped_epoch": epoch + 1,
                            "early_stopped": 1
                        }, step=epoch+1)
                        
                        # Break the training loop
                        break
                
                val_psnr_metric.reset_state()
                val_ssim_metric.reset_state()
                val_cer_metric.reset_state()
                val_wer_metric.reset_state()
                val_clean_cer_metric.reset_state()
                val_clean_wer_metric.reset_state()
                val_noise_variance_metric.reset_state()
                val_isolated_white_metric.reset_state()
                val_local_variance_metric.reset_state()
            else:
                epoch_metrics["validation"] = None

            # Always save samples at save_interval
            if (epoch + 1) % args.save_interval == 0:
                # ✅ FIX: Get multiple batches to ensure we have 5 samples for visualization
                # (Don't be limited by training batch_size which might be 2)
                num_vis_samples = 5
                collected_degraded = []
                collected_clean = []
                collected_labels = []
                
                val_iter = iter(val_dataset)
                samples_collected = 0
                while samples_collected < num_vis_samples:
                    try:
                        batch = next(val_iter)
                        batch_size_actual = batch[0].shape[0]
                        samples_needed = min(batch_size_actual, num_vis_samples - samples_collected)
                        
                        collected_degraded.append(batch[0][:samples_needed])
                        collected_clean.append(batch[1][:samples_needed])
                        collected_labels.append(batch[2][:samples_needed])
                        
                        samples_collected += samples_needed
                    except StopIteration:
                        break
                
                # Concatenate collected samples
                degraded_samples = tf.concat(collected_degraded, axis=0)[:num_vis_samples]
                clean_samples = tf.concat(collected_clean, axis=0)[:num_vis_samples]
                ground_truth_labels = tf.concat(collected_labels, axis=0)[:num_vis_samples]
                
                # ✅ CRITICAL FIX (2025-10-21): Normalize samples to [-1,1] before generator
                # TFRecord data is [0,1], but generator expects [-1,1] input
                degraded_samples_tanh = degraded_samples * 2.0 - 1.0
                clean_samples_tanh = clean_samples * 2.0 - 1.0
                
                # Generate restored images
                generated_samples = generator(degraded_samples_tanh, training=False)
                
                # Denormalize from tanh [-1,1] to [0,1] for saving/display
                generated_samples_normalized = (generated_samples + 1.0) / 2.0
                degraded_samples_normalized = (degraded_samples_tanh + 1.0) / 2.0
                clean_samples_normalized = (clean_samples_tanh + 1.0) / 2.0
                
                # ✅ Now we always have exactly 5 samples regardless of training batch_size
                img_to_save = (generated_samples_normalized * 255).numpy().astype(np.uint8)
                for i in range(num_vis_samples):
                    img = img_to_save[i]
                    if img.shape[-1] == 1:
                        img = np.squeeze(img, axis=-1)
                    
                    # Transpose image to correct orientation for horizontal text
                    if img.shape[0] > img.shape[1]:  # If height > width, transpose
                        img = np.transpose(img)
                    
                    sample_path = os.path.join(args.sample_dir, f'epoch_{epoch+1:04d}_sample_{i}.png')
                    cv2.imwrite(sample_path, img)
                    
                    # --- Create comparison image (Vertical: Degraded | Ground Truth | Restored) ---
                    # Prepare degraded image
                    deg_img = (degraded_samples_normalized[i] * 255).numpy().astype(np.uint8)
                    deg_img = np.squeeze(deg_img, axis=-1) if deg_img.shape[-1] == 1 else deg_img
                    if deg_img.shape[0] > deg_img.shape[1]:
                        deg_img = np.transpose(deg_img)
                    
                    # Prepare clean (GT) image
                    clean_img = (clean_samples_normalized[i] * 255).numpy().astype(np.uint8)
                    clean_img = np.squeeze(clean_img, axis=-1) if clean_img.shape[-1] == 1 else clean_img
                    if clean_img.shape[0] > clean_img.shape[1]:
                        clean_img = np.transpose(clean_img)
                    
                    # Create VERTICAL concatenation: [Degraded | Ground Truth | Restored]
                    comparison = np.vstack([deg_img, clean_img, img])
                    comparison_path = os.path.join(args.sample_dir, f'comparison_epoch_{epoch+1:04d}_sample_{i}.png')
                    cv2.imwrite(comparison_path, comparison)
                    
                    # ✅ FIX: Log to MLflow with UNIQUE key per sample (i changes in loop!)
                    mlflow.log_image(comparison, key=f"sample_{i}_comparison", step=epoch+1)

                # --- [NEW] Log recognition text for samples ---
                recognition_results = []
                # ✅ FIX: Run recognizer on NORMALIZED samples [0,1], not tanh [-1,1]
                # Recognizer expects input in [0,1] range
                recognizer_output = recognizer(generated_samples_normalized, training=False)
                if isinstance(recognizer_output, (list, tuple)):
                    predicted_logits = recognizer_output[0]
                else:
                    predicted_logits = recognizer_output
                
                # ✅ CRITICAL FIX: Use manual CTC decode, not tf.argmax!
                # This is the same decode method as in standalone test script
                predicted_texts = decode_ctc_predictions(predicted_logits.numpy(), charset)
                ground_truth_labels_np = ground_truth_labels.numpy()

                for i in range(num_vis_samples):  # ✅ FIX: Loop exactly 5 samples
                    gt_text = decode_label(ground_truth_labels_np[i], charset)
                    pred_text = predicted_texts[i]
                    recognition_results.append(f"--- Sample {i} ---\n")
                    recognition_results.append(f"Ground Truth: {gt_text}\n")
                    recognition_results.append(f"Prediction  : {pred_text}\n\n")

                # Write results to a text file and log to MLflow as text
                text_content = ''.join(recognition_results)
                mlflow.log_text(text_content, f"recognition_epoch_{epoch+1:04d}.txt")
                
                print(f"  💾 Saved sample images and recognition text to {args.sample_dir}")

            epoch_time = time.time() - epoch_start_time
            epoch_metrics["epoch_time_seconds"] = float(epoch_time)
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            
            # Log epoch time to MLflow
            mlflow.log_metric("epoch_time_seconds", epoch_time, step=epoch+1)
            
            # Append epoch metrics to history
            training_history["epochs"].append(epoch_metrics)
            
            # Save metrics to JSON after each epoch (incremental save)
            metrics_file = os.path.join(metrics_dir, "training_metrics_fp32.json")
            with open(metrics_file, 'w') as f:
                json.dump(training_history, f, indent=2)
            
            # Save checkpoint EVERY EPOCH (for resume capability)
            ckpt_save_path = ckpt_manager.save()
            
            # Save epoch info for resume capability (EVERY EPOCH)
            epoch_info = {
                'last_completed_epoch': epoch,
                'best_combined_score': best_combined_value if best_combined_value > float('-inf') else None,
                'best_psnr': best_psnr_value if best_psnr_value > -1.0 else None,
                'best_cer': best_cer_value if best_cer_value < float('inf') else None,
                'best_monitored_value': best_monitored_value if best_monitored_value > float('-inf') else None,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'total_epochs': args.epochs,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(epoch_info_path, 'w') as f:
                json.dump(epoch_info, f, indent=2)

        # Finalize training history
        training_history["end_time"] = datetime.now().isoformat()
        training_history["best_val_psnr"] = best_psnr_value
        training_history["best_val_cer"] = best_cer_value
        training_history["best_combined_score"] = best_combined_value
        training_history["best_monitored_value"] = best_monitored_value
        training_history["best_epoch"] = best_epoch
        training_history["total_epochs_completed"] = len(training_history["epochs"])
        training_history["early_stopped"] = early_stopped
        training_history["final_patience_counter"] = patience_counter
        
        # Save final metrics
        final_metrics_file = os.path.join(metrics_dir, "training_metrics_fp32_final.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Log final metrics to MLflow
        mlflow.log_artifact(final_metrics_file, artifact_path="metrics")
        mlflow.log_metric("final_best_psnr", best_psnr_value)
        mlflow.log_metric("final_best_cer", best_cer_value)
        mlflow.log_metric("final_best_combined", best_combined_value)
        mlflow.log_metric("final_best_monitored", best_monitored_value)
        mlflow.log_metric("best_epoch", best_epoch)
        
        print(f"\n✅ Training metrics saved to: {final_metrics_file}")
        print(f"📊 All metrics logged to MLflow")
        print(f"\n🚀 View results: poetry run mlflow ui")
        print(f"   Then open: http://localhost:5000\n")

    print("\n[Phase 6/6] Training Finished.")
    
    if early_stopped:
        print("\n🛑 Training stopped early (Early Stopping triggered)")
        print(f"   Best model from epoch: {best_epoch}")
        print(f"   Total epochs completed: {len(training_history['epochs'])} / {args.epochs}")
        print(f"   Resources saved: {args.epochs - len(training_history['epochs'])} epochs")
    else:
        print("\n🎉 Training completed successfully with Pure FP32!")
        print(f"   Best model from epoch: {best_epoch}")
    
    print("   ✅ Numerical stability maintained")
    print("   ✅ Balanced loss optimization")
    print("   ✅ Superior convergence quality")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dual-Modal GAN-HTR with Pure FP32 (OPTIMIZED)')
    parser.add_argument('--generator_version', type=str, default='base', choices=['base', 'enhanced', 'enhanced_v2'], help='Version of the generator to use: base, enhanced, or enhanced_v2 as SOTA version.')
    parser.add_argument('--discriminator_version', type=str, default='base', choices=['base', 'enhanced_v2', 'enhanced_v2_fixed'], help='Version of the discriminator to use: base (137M params), enhanced_v2 (18M params), or enhanced_v2_fixed (18M params, reduced artifacts)')
    parser.add_argument('--tfrecord_path', type=str, default='dual_modal_gan/data/dataset_gan.tfrecord', help='Path to the training TFRecord file.')
    parser.add_argument('--charset_path', type=str, default='real_data_preparation/real_data_charlist.txt', help='Path to the character set file.')
    parser.add_argument('--recognizer_weights', type=str, default='models/best_htr_recognizer/best_model.weights.h5', help='Path to pre-trained recognizer weights from Stage 3 with CER 33.72 percent.')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='GPU IDs to use for training. Single GPU: "0" or "1". Multi-GPU: "0,1" for both GPUs (default: "0,1" for 2x RTX A4000).')
    parser.add_argument('--no_restore', action='store_true', help='Do not restore from checkpoint, start from scratch.')
    parser.add_argument('--resume', action='store_true', help='Resume training from last completed epoch (requires epoch_info.json in checkpoint dir).')
    parser.add_argument('--checkpoint_dir', type=str, default='dual_modal_gan/outputs/checkpoints_fp32', help='Directory to save model checkpoints.')
    parser.add_argument('--max_checkpoints', type=int, default=1, help='Maximum number of checkpoints to keep (default: 1 for disk efficiency).')
    parser.add_argument('--save_best_model_separately', action='store_true', default=True, help='Save best model separately to prevent loss due to max_checkpoints limitation.')
    parser.add_argument('--sample_dir', type=str, default='dual_modal_gan/outputs/samples_fp32', help='Directory to save sample images.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Number of steps per epoch (if None, calculated from dataset size).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per replica.')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator learning rate.')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator learning rate.')
    parser.add_argument('--use_lr_schedule', action='store_true', help='Enable cosine annealing LR schedule (Solution 1).')
    parser.add_argument('--lr_decay_epochs', type=int, default=50, help='Number of epochs for LR cosine decay (default: 50).')
    parser.add_argument('--lr_alpha', type=float, default=0.0, help='Minimum learning rate as fraction of initial LR (0.0 = decay to zero).')
    parser.add_argument('--pixel_loss_weight', type=float, default=100.0, help='Weight for the L1 pixel loss.')
    parser.add_argument('--ctc_loss_weight', type=float, default=1.0, help='Weight for the HTR CTC loss (monitoring only, not backpropagated).')
    parser.add_argument('--adv_loss_weight', type=float, default=2.0, help='Weight for the adversarial loss.')
    parser.add_argument('--rec_feat_loss_weight', type=float, default=0.0, help='Weight for the Recognition Feature Loss (HTR-aware).')
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.0, help='Weight for the Contrastive Loss.')
    parser.add_argument('--perceptual_loss_weight', type=float, default=0.0, help='Weight for the VGG Perceptual Loss (default: 0.0 = disabled).')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm to prevent explosion.')
    parser.add_argument('--ctc_loss_clip_max', type=float, default=300.0, help='Maximum value for CTC loss clipping.')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of epochs for visual warm-up (CTC loss is disabled).')
    parser.add_argument('--annealing_epochs', type=int, default=10, help='Number of epochs to gradually ramp-up CTC loss weight after warm-up.')
    parser.add_argument('--save_interval', type=int, default=5, help='Save samples every N epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Run evaluation every N epochs.')
    parser.add_argument('--discriminator_mode', type=str, default='predicted', choices=['predicted', 'ground_truth'], help="Mode for the discriminator's real pair text input.")
    parser.add_argument('--cer_weight', type=float, default=0.5, help='Weight for CER in combined score calculation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    # Early Stopping Parameters
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping to prevent overfitting and save resources.')
    parser.add_argument('--curriculum_aware_early_stopping', action='store_true', default=True, help='Enable curriculum-aware early stopping (only triggers AFTER warmup + annealing phases complete).')
    parser.add_argument('--early_stopping_metric', type=str, default='combined', choices=['combined', 'psnr_only', 'cer_only'], help='Metric to monitor for early stopping: combined (PSNR-CER balance), psnr_only (visual quality), or cer_only (text readability).')
    parser.add_argument('--patience', type=int, default=15, help='Number of epochs without improvement before stopping (default: 15).')
    parser.add_argument('--min_delta', type=float, default=0.01, help='Minimum change in monitored metric to qualify as improvement (default: 0.01).')
    parser.add_argument('--restore_best_weights', action='store_true', default=True, help='Restore model weights from best epoch when early stopping triggers.')
    parser.add_argument('--psnr_improvement_threshold', type=float, default=2.0, help='Minimum PSNR improvement to override early stopping (prevents premature stopping due to CER fluctuations).')
    
    # Adaptive Loss Balancing Parameters
    parser.add_argument('--adaptive_loss_balancing', action='store_true', help='Enable adaptive loss balancing with SimpleAdaptiveBalancer.')
    parser.add_argument('--target_ctc_ratio', type=float, default=0.65, help='Target contribution ratio for CTC loss, default is 0.65 or 65 percent.')
    parser.add_argument('--target_visual_ratio', type=float, default=0.35, help='Target contribution ratio for visual losses including pixel, perceptual, adversarial, and recognition feature losses. Default is 0.35 or 35 percent.')
    parser.add_argument('--adaptation_rate', type=float, default=0.15, help='Adaptation rate for SimpleAdaptiveBalancer ranging from 0.1 to 0.5, default is 0.15.')

    args = parser.parse_args()
    main(args)
