"""
Find Top K Highest PSNR Samples in Test Set with Transcription

This script evaluates the LOCKED test set and identifies the top K samples with the HIGHEST PSNR scores.
It performs HTR on these samples to generate transcriptions and calculates CER.
It saves the degraded, clean, and generated images for these samples, along with comparison images annotated with text.

Usage:
    poetry run python dual_modal_gan/scripts/find_top_k_psnr.py \
        --checkpoint_dir dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model \
        --checkpoint_name ckpt-88 \
        --config configs/production_v3_academic_split_70_15_15.json \
        --output_dir results/top_5_psnr_analysis \
        --gpu_id 1 \
        --top_k 5
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
import cv2
import editdistance
from tqdm import tqdm
import heapq

# Disable XLA and configure TF
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)

# Set FP32 precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import models
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

# --- Utility Functions ---

def read_charlist(path):
    """Load character set from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

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

def decode_ctc_predictions(logits, charset):
    """Manual CTC decode."""
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

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate."""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    distance = editdistance.eval(ground_truth, prediction)
    return distance / len(ground_truth)

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord example - EXACT SAME as training."""
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),
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
    """Create ONLY the test dataset."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    print(f"\n📊 Test Set Configuration:")
    print(f"   Total dataset: {total_size}")
    print(f"   Test samples:  {test_size} (15%) - LOADED ✅")
    
    return test_dataset, test_size

def find_top_k_psnr(args):
    """Find top K samples with HIGHEST PSNR in test set."""
    
    print("="*80)
    print(f"🔍 FINDING TOP {args.top_k} HIGHEST PSNR SAMPLES WITH TRANSCRIPTION")
    print("="*80)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # Load charset
    charset = read_charlist(config['charset_path'])
    vocab_size = len(charset) + 1
    print(f"✅ Charset loaded: {vocab_size} characters")
    
    test_dataset, test_size = create_test_dataset(
        config['tfrecord_path'],
        config['batch_size'],
        train_split=config.get('train_split', 0.7),
        val_split=config.get('val_split', 0.15)
    )
    
    print(f"\n🏗️  Building models...")
    generator = unet_enhanced(input_size=(1024, 128, 1))
    recognizer = load_frozen_recognizer(
        weights_path=config['recognizer_weights'],
        charset_size=vocab_size - 1,
        return_feature_map=False
    )
    
    print(f"\n📦 Loading checkpoint...")
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    status = checkpoint.restore(checkpoint_path)
    status.expect_partial()
    print(f"   ✅ Loaded: {checkpoint_path}")
    
    # Use a min-heap to keep track of top K samples (based on PSNR)
    # Heap elements: (psnr, sample_data)
    top_k_heap = []
    
    print(f"\n🔬 Scanning {test_size} samples...")
    
    batch_count = 0
    sample_global_idx = 0
    
    for batch_idx, (degraded_images, clean_images, labels) in enumerate(tqdm(test_dataset, desc="Scanning")):
        batch_count += 1
        current_batch_size = degraded_images.shape[0]
        
        # Normalize to [-1, 1] for generator
        degraded_images_tanh = degraded_images * 2.0 - 1.0
        clean_images_tanh = clean_images * 2.0 - 1.0
        
        # Generate
        generated_images = generator(degraded_images_tanh, training=False)
        
        # Denormalize to [0, 1]
        generated_images_normalized = (generated_images + 1.0) / 2.0
        clean_images_normalized = (clean_images_tanh + 1.0) / 2.0
        
        # Calculate PSNR
        psnr_values = tf.image.psnr(clean_images_normalized, generated_images_normalized, max_val=1.0)
        ssim_values = tf.image.ssim(clean_images_normalized, generated_images_normalized, max_val=1.0)
        
        psnr_np = psnr_values.numpy()
        ssim_np = ssim_values.numpy()
        
        for i in range(current_batch_size):
            current_psnr = float(psnr_np[i])
            current_ssim = float(ssim_np[i])
            
            sample_data = {
                'batch_idx': batch_idx,
                'sample_idx': i,
                'global_idx': sample_global_idx,
                'psnr': current_psnr,
                'ssim': current_ssim,
                'degraded_tensor': degraded_images[i],
                'clean_tensor': clean_images_normalized[i],
                'generated_tensor': generated_images_normalized[i],
                'label': labels[i].numpy()
            }
            
            # Maintain top K heap
            if len(top_k_heap) < args.top_k:
                heapq.heappush(top_k_heap, (current_psnr, sample_data))
            else:
                if current_psnr > top_k_heap[0][0]:
                    heapq.heappop(top_k_heap)
                    heapq.heappush(top_k_heap, (current_psnr, sample_data))
            
            sample_global_idx += 1
            
    # Sort top K samples by PSNR descending
    top_k_samples = sorted([item[1] for item in top_k_heap], key=lambda x: x['psnr'], reverse=True)
    
    print("\n" + "="*80)
    print(f"📈 TOP {args.top_k} HIGHEST PSNR RESULTS")
    print("="*80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_summary = []
    
    for rank, sample in enumerate(top_k_samples, 1):
        print(f"\n🏆 Rank {rank}: Index {sample['global_idx']} (PSNR: {sample['psnr']:.2f} dB, SSIM: {sample['ssim']:.6f})")
        
        # --- Perform Recognition ---
        # Prepare inputs for recognizer (add batch dim)
        gen_input = tf.expand_dims(sample['generated_tensor'], 0)
        
        # Run recognizer
        gen_logits = recognizer(gen_input, training=False)
        
        # Decode
        if isinstance(gen_logits, (list, tuple)):
            gen_logits = gen_logits[0]
            
        gen_text = decode_ctc_predictions(gen_logits.numpy(), charset)[0]
        gt_text = decode_label(sample['label'], charset)
        
        # Calculate CER
        cer = calculate_cer(gt_text, gen_text)
        
        print(f"   Ground Truth: '{gt_text}'")
        print(f"   Prediction:   '{gen_text}'")
        print(f"   CER:          {cer:.4f}")
        
        # --- Save Images ---
        # Prepare images (transpose for correct orientation: 1024x128 -> 128x1024)
        degraded_img = (sample['degraded_tensor'].numpy() * 255).astype(np.uint8).squeeze().T
        clean_img = (sample['clean_tensor'].numpy() * 255).astype(np.uint8).squeeze().T
        generated_img = (sample['generated_tensor'].numpy() * 255).astype(np.uint8).squeeze().T
        
        filename_base = f"rank_{rank}_idx_{sample['global_idx']}"
        
        # Save individual
        cv2.imwrite(os.path.join(args.output_dir, f'{filename_base}_degraded.png'), degraded_img)
        cv2.imwrite(os.path.join(args.output_dir, f'{filename_base}_clean.png'), clean_img)
        cv2.imwrite(os.path.join(args.output_dir, f'{filename_base}_generated.png'), generated_img)
        
        # Create comparison
        h, w = degraded_img.shape
        label_height = 40
        text_height = 60 # Extra space for transcription
        spacing = 15
        
        # Total height: 3 images + labels + text for GT and Pred
        total_height = (h + label_height) + (h + label_height + text_height) + (h + label_height + text_height) + spacing * 4
        
        comparison = np.ones((total_height, w), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_scale = 0.6
        thickness = 1
        
        y = spacing
        
        # 1. Degraded
        cv2.putText(comparison, f'Degraded (Input)', (10, y + 25), font, font_scale, 0, 2)
        y += label_height
        comparison[y:y+h, :] = degraded_img
        y += h + spacing
        
        # 2. Clean (GT)
        cv2.putText(comparison, f'Clean (Ground Truth)', (10, y + 25), font, font_scale, 0, 2)
        y += label_height
        comparison[y:y+h, :] = clean_img
        y += h + 10
        # Add GT Text
        cv2.putText(comparison, f"GT Text: {gt_text}", (10, y + 25), font, text_scale, 0, 1)
        y += text_height + spacing
        
        # 3. Generated (Pred)
        cv2.putText(comparison, f'Generated (Output) | PSNR: {sample["psnr"]:.2f} dB | SSIM: {sample["ssim"]:.4f}', (10, y + 25), font, font_scale, 0, 2)
        y += label_height
        comparison[y:y+h, :] = generated_img
        y += h + 10
        # Add Predicted Text and CER
        cv2.putText(comparison, f"Pred Text: {gen_text}", (10, y + 25), font, text_scale, 0, 1)
        cv2.putText(comparison, f"CER: {cer:.4f} ({cer*100:.1f}%)", (10, y + 50), font, text_scale, 0, 1)
        
        cv2.imwrite(os.path.join(args.output_dir, f'{filename_base}_comparison.png'), comparison)
        
        results_summary.append({
            'rank': rank,
            'global_idx': sample['global_idx'],
            'psnr': sample['psnr'],
            'ssim': sample['ssim'],
            'gt_text': gt_text,
            'pred_text': gen_text,
            'cer': cer
        })

    print(f"\n✅ Saved images for top {args.top_k} samples to {args.output_dir}/")
    
    # Save metadata
    meta_file = os.path.join(args.output_dir, 'top_k_psnr_info.json')
    with open(meta_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"   - top_k_psnr_info.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/top_k_psnr')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--top_k', type=int, default=5)
    
    args = parser.parse_args()
    find_top_k_psnr(args)
