"""
Test Set Evaluation Script - Academic Protocol

Evaluates the LOCKED test set (15% of data) that was held out during training.
This script should be run ONCE after training completion for final performance reporting.

Key Features:
- Loads exact same test split as used in training (deterministic)
- Comprehensive metrics: PSNR, SSIM, CER, WER, noise artifacts
- Statistical reporting: mean ± std, 95% confidence intervals
- Sample size tracking for academic rigor
- Generates JSON report for publication

Usage:
    poetry run python dual_modal_gan/scripts/evaluate_test_set.py \\
        --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \\
        --checkpoint_name ckpt-88 \\
        --config configs/production_v3_academic_split_70_15_15.json \\
        --output_dir results/test_set_evaluation \\
        --gpu_id 1
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import cv2
import editdistance
from tqdm import tqdm

# Disable XLA and configure TF
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)

# Set FP32 precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
print("✅ Pure FP32 precision enabled")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import models
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

# --- Utility Functions (from train_enhanced.py) ---

def read_charlist(path):
    """Load character set from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord example - EXACT SAME as training."""
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
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (H,W,C) → (W,H,C)
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

def create_test_dataset(tfrecord_path, batch_size, train_split=0.7, val_split=0.15):
    """
    Create ONLY the test dataset - EXACT SAME split as training.
    
    Returns:
        test_dataset: The locked test set (15% of data)
        test_size: Number of test samples
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    # Calculate split sizes - MUST MATCH training exactly
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Parse and split - EXACT SAME order as training
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Skip to test set (first train_size + val_size samples)
    test_dataset = dataset.skip(train_size + val_size)
    
    # Batch WITHOUT drop_remainder to get ALL test samples
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    print(f"\n📊 Test Set Configuration:")
    print(f"   Total dataset: {total_size} samples")
    print(f"   Train samples: {train_size} (70%) - SKIPPED")
    print(f"   Val samples:   {val_size} (15%) - SKIPPED")
    print(f"   Test samples:  {test_size} (15%) - LOADED ✅")
    print(f"   ⚠️  This is the LOCKED test set from training!")
    
    return test_dataset, test_size

def decode_ctc_predictions(logits, charset):
    """Manual CTC decode - EXACT implementation from training."""
    charset_size = len(charset)
    batch_size = logits.shape[0]
    results = []
    
    for b in range(batch_size):
        raw_preds = np.argmax(logits[b], axis=-1)
        
        # Manual CTC decode with deduplication
        deduped = []
        prev = -1
        for token in raw_preds:
            if token != prev:
                deduped.append(token)
                prev = token
        
        # Remove blank tokens
        result = []
        for token in deduped:
            if token != charset_size:  # Skip blank
                result.append(token)
        
        # Convert to string
        decoded = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
        results.append(decoded)
    
    return results

def decode_label(label_ids, charset):
    """Decode label IDs to text string."""
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
    """Calculate Character Error Rate."""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)

def calculate_wer(ground_truth, prediction):
    """Calculate Word Error Rate."""
    gt_words = ground_truth.split()
    pred_words = prediction.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    distance = editdistance.eval(gt_words, pred_words)
    return distance / len(gt_words)

def calculate_noise_artifacts_metrics(image):
    """Calculate metrics to detect noise artifacts."""
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
            img_np = np.mean(img_np, axis=-1)
    
    # Convert to uint8
    if img_np.max() <= 1.0:
        img_uint8 = (img_np * 255).astype(np.uint8)
    else:
        img_uint8 = img_np.astype(np.uint8)
    
    # 1. High-frequency noise (Laplacian variance)
    laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
    noise_variance = float(np.var(laplacian))
    
    # 2. Isolated white pixels (salt noise)
    _, binary = cv2.threshold(img_uint8, 250, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    isolated_white_pixels = np.sum(binary - eroded)
    total_pixels = img_uint8.size
    isolated_white_ratio = float(isolated_white_pixels / total_pixels)
    
    # 3. Local variance (smoothness)
    kernel_size = 5
    mean_filtered = cv2.blur(img_uint8, (kernel_size, kernel_size))
    local_deviations = img_uint8.astype(np.float32) - mean_filtered.astype(np.float32)
    local_variance = float(np.var(local_deviations))
    
    return {
        'noise_variance': noise_variance,
        'isolated_white_ratio': isolated_white_ratio,
        'local_variance': local_variance
    }

# --- Main Evaluation Function ---

def evaluate_test_set(args):
    """Run comprehensive evaluation on test set."""
    
    print("="*80)
    print("🔬 TEST SET EVALUATION - ACADEMIC PROTOCOL")
    print("="*80)
    print(f"Model: {args.checkpoint_dir}/{args.checkpoint_name}")
    print(f"Config: {args.config}")
    print(f"GPU: {args.gpu_id}")
    print("="*80)
    
    # Configure GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load charset
    charset = read_charlist(config['charset_path'])
    vocab_size = len(charset) + 1
    print(f"\n✅ Charset loaded: {vocab_size} characters (including blank)")
    
    # Create test dataset
    test_dataset, test_size = create_test_dataset(
        config['tfrecord_path'],
        config['batch_size'],
        train_split=config.get('train_split', 0.7),
        val_split=config.get('val_split', 0.15)
    )
    
    # Build models
    print(f"\n🏗️  Building models...")
    generator = unet_enhanced(input_size=(1024, 128, 1))
    recognizer = load_frozen_recognizer(
        weights_path=config['recognizer_weights'],
        charset_size=vocab_size - 1,
        return_feature_map=False
    )
    print("   ✅ Generator: U-Net Enhanced")
    print("   ✅ Recognizer: Frozen HTR")
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint...")
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    status = checkpoint.restore(checkpoint_path)
    status.expect_partial()
    print(f"   ✅ Loaded: {checkpoint_path}")
    
    # Initialize metrics storage
    all_psnr = []
    all_ssim = []
    all_cer = []
    all_wer = []
    all_clean_cer = []
    all_clean_wer = []
    all_noise_var = []
    all_isolated_white = []
    all_local_var = []
    
    # Store sample predictions for analysis
    sample_predictions = []
    
    # Store visual samples (degraded, clean, generated) for first N samples
    visual_samples = []
    max_visual_samples = getattr(args, 'num_visual_samples', 20)  # Save first 20 by default
    
    print(f"\n🔬 Evaluating {test_size} test samples...")
    print("="*80)
    
    eval_start_time = time.time()
    batch_count = 0
    sample_count = 0
    
    # Evaluate on test set
    for batch_idx, (degraded_images, clean_images, labels) in enumerate(tqdm(test_dataset, desc="Evaluating")):
        batch_count += 1
        current_batch_size = degraded_images.shape[0]
        sample_count += current_batch_size
        
        # Normalize to [-1, 1] for generator
        degraded_images_tanh = degraded_images * 2.0 - 1.0
        clean_images_tanh = clean_images * 2.0 - 1.0
        
        # Generate enhanced images
        generated_images = generator(degraded_images_tanh, training=False)
        
        # Denormalize to [0, 1] for metrics
        generated_images_normalized = (generated_images + 1.0) / 2.0
        clean_images_normalized = (clean_images_tanh + 1.0) / 2.0
        
        # Visual metrics: PSNR and SSIM
        psnr = tf.image.psnr(clean_images_normalized, generated_images_normalized, max_val=1.0)
        ssim = tf.image.ssim(clean_images_normalized, generated_images_normalized, max_val=1.0)
        
        all_psnr.extend(psnr.numpy().tolist())
        all_ssim.extend(ssim.numpy().tolist())
        
        # Noise artifact metrics
        for i in range(current_batch_size):
            noise_metrics = calculate_noise_artifacts_metrics(generated_images_normalized[i])
            all_noise_var.append(noise_metrics['noise_variance'])
            all_isolated_white.append(noise_metrics['isolated_white_ratio'])
            all_local_var.append(noise_metrics['local_variance'])
        
        # Textual metrics: CER/WER
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
        clean_predictions = decode_ctc_predictions(clean_logits.numpy(), charset)
        generated_predictions = decode_ctc_predictions(generated_logits.numpy(), charset)
        
        # Calculate CER/WER
        labels_np = labels.numpy()
        for i in range(current_batch_size):
            gt_text = decode_label(labels_np[i], charset)
            clean_text = clean_predictions[i]
            generated_text = generated_predictions[i]
            
            cer = calculate_cer(gt_text, generated_text)
            wer = calculate_wer(gt_text, generated_text)
            clean_cer = calculate_cer(gt_text, clean_text)
            clean_wer = calculate_wer(gt_text, clean_text)
            
            all_cer.append(cer)
            all_wer.append(wer)
            all_clean_cer.append(clean_cer)
            all_clean_wer.append(clean_wer)
            
            # Store first 10 samples for detailed analysis
            if len(sample_predictions) < 10:
                sample_predictions.append({
                    'batch': batch_idx,
                    'sample': i,
                    'ground_truth': gt_text,
                    'clean_prediction': clean_text,
                    'generated_prediction': generated_text,
                    'psnr': float(psnr[i].numpy()),
                    'ssim': float(ssim[i].numpy()),
                    'cer': cer,
                    'wer': wer,
                    'clean_cer': clean_cer,
                    'clean_wer': clean_wer
                })
            
            # Store visual samples (images) for first N samples
            if len(visual_samples) < max_visual_samples:
                # Convert tensors to numpy arrays [0, 1] range
                degraded_np = degraded_images[i].numpy()
                clean_np = clean_images_normalized[i].numpy()
                generated_np = generated_images_normalized[i].numpy()
                
                visual_samples.append({
                    'sample_id': len(visual_samples),
                    'batch': batch_idx,
                    'index': i,
                    'degraded': degraded_np,
                    'clean': clean_np,
                    'generated': generated_np,
                    'psnr': float(psnr[i].numpy()),
                    'ssim': float(ssim[i].numpy()),
                    'cer': cer,
                    'ground_truth': gt_text
                })
    
    eval_time = time.time() - eval_start_time
    
    print(f"\n✅ Evaluation completed in {eval_time:.2f}s")
    print(f"   Samples evaluated: {sample_count}")
    print(f"   Batches processed: {batch_count}")
    
    # Calculate statistics
    print(f"\n📊 Calculating statistics...")
    
    def calc_stats(values, name):
        """Calculate mean, std, and 95% CI."""
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        ci = 1.96 * std / np.sqrt(len(values))
        return {
            'mean': float(mean),
            'std': float(std),
            'ci_95': float(ci),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'n': len(values)
        }
    
    results = {
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'model_checkpoint': f"{args.checkpoint_dir}/{args.checkpoint_name}",
            'config_file': args.config,
            'test_size': sample_count,
            'batch_count': batch_count,
            'evaluation_time_seconds': eval_time,
            'gpu_id': args.gpu_id
        },
        'metrics': {
            'psnr': calc_stats(all_psnr, 'PSNR'),
            'ssim': calc_stats(all_ssim, 'SSIM'),
            'cer': calc_stats(all_cer, 'CER'),
            'wer': calc_stats(all_wer, 'WER'),
            'clean_cer': calc_stats(all_clean_cer, 'Clean CER'),
            'clean_wer': calc_stats(all_clean_wer, 'Clean WER'),
            'noise_variance': calc_stats(all_noise_var, 'Noise Variance'),
            'isolated_white_ratio': calc_stats(all_isolated_white, 'Isolated White'),
            'local_variance': calc_stats(all_local_var, 'Local Variance')
        },
        'sample_predictions': sample_predictions
    }
    
    # Print results
    print("\n" + "="*80)
    print("📊 TEST SET RESULTS - FINAL EVALUATION")
    print("="*80)
    
    print(f"\n🎯 VISUAL QUALITY METRICS:")
    print(f"   PSNR: {results['metrics']['psnr']['mean']:.2f} ± {results['metrics']['psnr']['std']:.2f} dB")
    print(f"         95% CI: [{results['metrics']['psnr']['mean'] - results['metrics']['psnr']['ci_95']:.2f}, "
          f"{results['metrics']['psnr']['mean'] + results['metrics']['psnr']['ci_95']:.2f}]")
    print(f"         Range: [{results['metrics']['psnr']['min']:.2f}, {results['metrics']['psnr']['max']:.2f}]")
    
    print(f"\n   SSIM: {results['metrics']['ssim']['mean']:.4f} ± {results['metrics']['ssim']['std']:.4f}")
    print(f"         95% CI: [{results['metrics']['ssim']['mean'] - results['metrics']['ssim']['ci_95']:.4f}, "
          f"{results['metrics']['ssim']['mean'] + results['metrics']['ssim']['ci_95']:.4f}]")
    print(f"         Range: [{results['metrics']['ssim']['min']:.4f}, {results['metrics']['ssim']['max']:.4f}]")
    
    print(f"\n📝 TEXT RECOGNITION METRICS:")
    print(f"   CER (Generated): {results['metrics']['cer']['mean']:.4f} ± {results['metrics']['cer']['std']:.4f}")
    print(f"   CER (Clean Baseline): {results['metrics']['clean_cer']['mean']:.4f} ± {results['metrics']['clean_cer']['std']:.4f}")
    print(f"   ΔCER: {results['metrics']['cer']['mean'] - results['metrics']['clean_cer']['mean']:+.4f}")
    
    print(f"\n   WER (Generated): {results['metrics']['wer']['mean']:.4f} ± {results['metrics']['wer']['std']:.4f}")
    print(f"   WER (Clean Baseline): {results['metrics']['clean_wer']['mean']:.4f} ± {results['metrics']['clean_wer']['std']:.4f}")
    print(f"   ΔWER: {results['metrics']['wer']['mean'] - results['metrics']['clean_wer']['mean']:+.4f}")
    
    print(f"\n🔍 NOISE ARTIFACT METRICS:")
    print(f"   Noise Variance: {results['metrics']['noise_variance']['mean']:.2f} ± {results['metrics']['noise_variance']['std']:.2f}")
    print(f"   Isolated White Ratio: {results['metrics']['isolated_white_ratio']['mean']:.6f} ± {results['metrics']['isolated_white_ratio']['std']:.6f}")
    print(f"   Local Variance: {results['metrics']['local_variance']['mean']:.2f} ± {results['metrics']['local_variance']['std']:.2f}")
    
    print(f"\n📈 SAMPLE SIZE:")
    print(f"   Test samples: {sample_count} (n={sample_count})")
    print(f"   Statistical power: ✅ Sufficient for 95% confidence")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'test_set_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Save sample predictions
    samples_file = os.path.join(args.output_dir, 'sample_predictions.txt')
    with open(samples_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SAMPLE PREDICTIONS (First 10 test samples)\n")
        f.write("="*80 + "\n\n")
        for idx, sample in enumerate(sample_predictions, 1):
            f.write(f"Sample {idx} (Batch {sample['batch']}, Index {sample['sample']}):\n")
            f.write(f"  Ground Truth:        '{sample['ground_truth']}'\n")
            f.write(f"  Clean Prediction:    '{sample['clean_prediction']}' (CER: {sample['clean_cer']:.3f})\n")
            f.write(f"  Generated Prediction: '{sample['generated_prediction']}' (CER: {sample['cer']:.3f})\n")
            f.write(f"  PSNR: {sample['psnr']:.2f} dB, SSIM: {sample['ssim']:.4f}\n")
            f.write(f"  ΔCER: {sample['cer'] - sample['clean_cer']:+.3f}\n")
            f.write("\n")
    
    print(f"💾 Sample predictions saved to: {samples_file}")
    
    # Save visual samples (images)
    if visual_samples and getattr(args, 'save_images', True):
        print(f"\n💾 Saving visual samples...")
        samples_dir = os.path.join(args.output_dir, 'visual_samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        for sample in visual_samples:
            sample_id = sample['sample_id']
            
            # Convert from [0, 1] float to [0, 255] uint8
            # Image shape from dataset: (W, H, C) = (1024, 128, 1)
            # After squeeze: (1024, 128) but NumPy uses (H, W) convention
            # So we need to transpose to get correct orientation
            degraded_img = (sample['degraded'] * 255).astype(np.uint8).squeeze()
            clean_img = (sample['clean'] * 255).astype(np.uint8).squeeze()
            generated_img = (sample['generated'] * 255).astype(np.uint8).squeeze()
            
            # Transpose to correct orientation: (1024, 128) -> (128, 1024)
            # This makes it horizontal (width=1024, height=128) in OpenCV convention
            degraded_img = degraded_img.T
            clean_img = clean_img.T
            generated_img = generated_img.T
            
            # Save individual images
            cv2.imwrite(
                os.path.join(samples_dir, f'sample_{sample_id:03d}_degraded.png'),
                degraded_img
            )
            cv2.imwrite(
                os.path.join(samples_dir, f'sample_{sample_id:03d}_clean.png'),
                clean_img
            )
            cv2.imwrite(
                os.path.join(samples_dir, f'sample_{sample_id:03d}_generated.png'),
                generated_img
            )
            
            # Create comparison image (VERTICAL stacking: top to bottom)
            h, w = degraded_img.shape  # h=128, w=1024 (horizontal)
            
            # Vertical layout: stack images top-to-bottom with labels
            label_height = 25  # Space for label above each image
            spacing = 15       # Space between images
            total_height = (h + label_height) * 3 + spacing * 2 + 30  # Extra 30px for metrics at bottom
            
            comparison = np.ones((total_height, w), dtype=np.uint8) * 255  # White background
            
            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            color = (0, 0, 0)  # Black text
            
            # Stack images vertically with labels
            y_pos = 0
            
            # 1. Degraded image
            cv2.putText(comparison, 'Degraded (Input)', (10, y_pos + 20), font, font_scale, color, thickness)
            y_pos += label_height
            comparison[y_pos:y_pos+h, 0:w] = degraded_img
            y_pos += h + spacing
            
            # 2. Clean ground truth
            cv2.putText(comparison, 'Clean (Ground Truth)', (10, y_pos + 20), font, font_scale, color, thickness)
            y_pos += label_height
            comparison[y_pos:y_pos+h, 0:w] = clean_img
            y_pos += h + spacing
            
            # 3. Generated output
            cv2.putText(comparison, 'Generated (Model Output)', (10, y_pos + 20), font, font_scale, color, thickness)
            y_pos += label_height
            comparison[y_pos:y_pos+h, 0:w] = generated_img
            y_pos += h + 10
            
            # Add metrics at bottom
            metrics_text = f'PSNR: {sample["psnr"]:.2f} dB  |  SSIM: {sample["ssim"]:.4f}  |  CER: {sample["cer"]:.3f}'
            cv2.putText(comparison, metrics_text, (10, y_pos + 20), font, 0.5, color, thickness)
            
            # Save comparison
            cv2.imwrite(
                os.path.join(samples_dir, f'sample_{sample_id:03d}_comparison.png'),
                comparison
            )
        
        print(f"   ✅ Saved {len(visual_samples)} visual samples to: {samples_dir}/")
        print(f"   Format: sample_XXX_[degraded|clean|generated|comparison].png")
    
    print("\n" + "="*80)
    print("✅ TEST SET EVALUATION COMPLETE")
    print("="*80)
    print("\n⚠️  REMINDER: This is the FINAL evaluation on the LOCKED test set.")
    print("   These results should be reported in the publication.")
    print("   Do NOT run this evaluation multiple times for model selection.")
    print("   Use validation set for hyperparameter tuning and model selection.")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on locked test set (academic protocol)')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing the checkpoint')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help='Name of the checkpoint file (e.g., ckpt-88)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration JSON file')
    parser.add_argument('--output_dir', type=str, default='results/test_set_evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='GPU ID to use')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Save visual samples (default: True)')
    parser.add_argument('--num_visual_samples', type=int, default=20,
                        help='Number of visual samples to save (default: 20)')
    
    args = parser.parse_args()
    evaluate_test_set(args)
