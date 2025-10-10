#!/usr/bin/env python3
"""
Comprehensive Evaluation Script untuk GAN-HTR
Menghitung: PSNR, SSIM, CER, WER pada generated images

Usage:
    poetry run python dual_modal_gan/scripts/evaluate_comprehensive.py \
        --checkpoint_path outputs/checkpoints_final_100/best_model \
        --output_json outputs/checkpoints_final_100/comprehensive_metrics.json
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import Mean
import editdistance
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Constants
IMG_WIDTH = 1024
IMG_HEIGHT = 128
MAX_LABEL_LENGTH = 128


def read_charlist(file_path):
    """Read character list from file - same as train_transformer_improved_v2.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        charset = []
        for line in f:
            content = line.rstrip('\n')
            if content == ' ':
                charset.append(' ')
            elif content:
                charset.append(content)
    return charset


def parse_tfrecord_eval(example):
    """Parse TFRecord - exact same format as train.py for dataset_gan.tfrecord"""
    # Define feature description for raw bytes and metadata
    feature_description = {
        'degraded_image_raw': tf.io.FixedLenFeature([], tf.string),
        'degraded_image_shape': tf.io.FixedLenFeature([3], tf.int64),  # H, W, C
        'degraded_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'clean_image_raw': tf.io.FixedLenFeature([], tf.string),
        'clean_image_shape': tf.io.FixedLenFeature([3], tf.int64),
        'clean_image_dtype': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([1], tf.int64),  # (length,)
        'label_dtype': tf.io.FixedLenFeature([], tf.string),
    }
    example_parsed = tf.io.parse_single_example(example, feature_description)
    
    # Deserialize degraded image
    degraded_image_shape = tf.cast(example_parsed['degraded_image_shape'], tf.int32)
    degraded_image = tf.io.decode_raw(example_parsed['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_image_shape)
    # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Deserialize clean image
    clean_image_shape = tf.cast(example_parsed['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example_parsed['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    # Transpose from (H, W, C) to (W, H, C) to match recognizer expectation
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (128, 1024, 1) → (1024, 128, 1)
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    # Deserialize label
    label_shape = tf.cast(example_parsed['label_shape'], tf.int32)
    label = tf.io.decode_raw(example_parsed['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.int32)
    
    # Pad label to static shape (same as train.py)
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])
    
    return degraded_image, clean_image, label


def safe_ctc_decode(logits, charset):
    """Safe CTC decoding - same as train_transformer_improved_v2.py"""
    charset_size = len(charset)
    raw_preds = np.argmax(logits, axis=-1)[0]
    
    # Manual CTC decode
    deduped = []
    prev = -1
    for token in raw_preds:
        if token != prev:
            deduped.append(token)
            prev = token
    
    result = []
    for token in deduped:
        if token != charset_size:  # Skip blank
            result.append(token)
    
    text = ''.join([charset[i] if 0 <= i < len(charset) else '<?>' for i in result])
    return text


def decode_predictions_batch(predictions, charset):
    """
    Decode batch CTC predictions ke text.
    
    Args:
        predictions: Logits dari recognizer (batch, time_steps, vocab_size)
        charset: List karakter
    
    Returns:
        List of decoded strings
    """
    decoded_texts = []
    for i in range(predictions.shape[0]):
        logits = predictions[i:i+1]
        text = safe_ctc_decode(logits, charset)
        decoded_texts.append(text)
    
    return decoded_texts


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


def create_improved_recognizer(charset_size, proj_dim=512, num_layers=6, num_heads=8, ff_dim=2048, dropout_rate=0.20):
    """Create recognizer dengan struktur PERSIS sama dengan train_transformer_improved_v2.py"""
    inputs = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name='image_input')
    x = inputs
    
    # CNN Backbone (EXACT same as train_transformer_improved_v2.py)
    def conv_block(inp, filters, k=3, s=(1,1), name_prefix='cb', dropout=0.0):
        """Proper conv block with BatchNorm and correct dropout"""
        y = layers.Conv2D(filters, k, strides=s, padding='same', 
                         use_bias=False, name=f'{name_prefix}_conv')(inp)
        y = layers.BatchNormalization(name=f'{name_prefix}_bn')(y)
        y = layers.Activation('gelu', name=f'{name_prefix}_gelu')(y)
        if dropout > 0:
            y = layers.Dropout(dropout, name=f'{name_prefix}_drop')(y)
        return y
    
    # Progressive feature extraction
    x = conv_block(x, 64, k=7, s=(1,2), name_prefix='s1_1', dropout=dropout_rate*0.5)
    x = conv_block(x, 64, k=3, s=(1,1), name_prefix='s1_2', dropout=dropout_rate*0.5)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool1')(x)
    
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_1', dropout=dropout_rate*0.7)
    x = conv_block(x, 128, k=3, s=(1,1), name_prefix='s2_2', dropout=dropout_rate*0.7)
    x = layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)
    
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_1', dropout=dropout_rate)
    x = conv_block(x, 256, k=3, s=(1,1), name_prefix='s3_2', dropout=dropout_rate)
    x = layers.MaxPooling2D(pool_size=(2,1), name='pool3')(x)
    
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_1', dropout=dropout_rate)
    x = conv_block(x, 512, k=3, s=(1,1), name_prefix='s4_2', dropout=dropout_rate)
    
    # Sequence Projection
    x = layers.Lambda(
        lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2]*tf.shape(t)[3])),
        name='flatten_height'
    )(x)
    
    x = layers.Dense(proj_dim, name='proj_dense')(x)
    x = layers.LayerNormalization(name='proj_ln')(x)
    x = layers.Dropout(dropout_rate, name='proj_drop')(x)
    
    # Positional Encoding
    seq_len = 128  # target time steps
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding_layer = layers.Embedding(
        input_dim=seq_len,
        output_dim=proj_dim,
        name='positional_embedding'
    )
    x = x + pos_embedding_layer(positions)
    
    # Transformer Encoder (EXACT same structure)
    for i in range(num_layers):
        # Multi-head attention
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=proj_dim // num_heads,
            dropout=dropout_rate,
            name=f'trn_attn_{i}'
        )(x, x)
        x = layers.LayerNormalization(name=f'trn_ln1_{i}')(x + attn)
        
        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation='gelu', name=f'trn_ffn1_{i}')(x)
        ffn = layers.Dropout(dropout_rate, name=f'trn_ffn_drop_{i}')(ffn)
        ffn = layers.Dense(proj_dim, name=f'trn_ffn2_{i}')(ffn)
        x = layers.LayerNormalization(name=f'trn_ln2_{i}')(x + ffn)
        x = layers.Dropout(dropout_rate, name=f'trn_drop_{i}')(x)
    
    # CTC Output
    outputs = layers.Dense(charset_size + 1, activation=None, name='logits')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='HTR_Transformer_Improved_V2')
    
    print(f"✅ Recognizer created: {num_layers} transformer layers, "
          f"{num_heads} heads, FFN dim={ff_dim}")
    print(f"   Output units: {charset_size + 1} (charset={charset_size}, blank={charset_size})")
    
    return model


def evaluate_comprehensive(args):
    """Run comprehensive evaluation."""
    
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION - GAN-HTR")
    print("=" * 70)
    
    # Load charset
    print(f"\n[1/6] Loading charset from {args.charset_path}...")
    charset = read_charlist(args.charset_path)
    charset_size = len(charset)
    print(f"  ✅ Charset loaded: {charset_size} characters")
    
    # Create char_to_num lookup
    keys = tf.constant(charset, dtype=tf.string)
    values = tf.range(1, len(charset) + 1, dtype=tf.int64)
    char_to_num = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=tf.constant(0, dtype=tf.int64)
    )
    
    # Load dataset
    print(f"\n[2/6] Loading validation dataset from {args.tfrecord_path}...")
    dataset = tf.data.TFRecordDataset(args.tfrecord_path)
    dataset_size = sum(1 for _ in dataset)
    train_size = int(dataset_size * 0.9)
    val_size = dataset_size - train_size
    
    # Recreate val dataset - same as train.py
    dataset = tf.data.TFRecordDataset(args.tfrecord_path)
    val_dataset = dataset.skip(train_size).take(val_size) \
        .map(parse_tfrecord_eval, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(args.batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)
    
    print(f"  ✅ Validation dataset: {val_size} samples, batch_size={args.batch_size}")
    
    # Load models
    print(f"\n[3/6] Loading Generator and Recognizer...")
    
    # Import generator from train.py
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    from dual_modal_gan.src.models.generator import unet
    
    generator = unet(input_size=(1024, 128, 1))
    
    # Create recognizer dengan struktur PERSIS sama dengan train_transformer_improved_v2.py
    recognizer = create_improved_recognizer(
        charset_size=charset_size,
        proj_dim=512,
        num_layers=args.num_transformer_layers,
        num_heads=8,
        ff_dim=2048,
        dropout_rate=0.20
    )
    
    print(f"  ✅ Generator: {generator.count_params():,} parameters")
    print(f"  ✅ Recognizer: {recognizer.count_params():,} parameters")
    
    # Load recognizer weights from standalone training
    if os.path.exists(args.recognizer_weights):
        print(f"  📥 Loading recognizer weights from {args.recognizer_weights}...")
        recognizer.load_weights(args.recognizer_weights)
        print(f"  ✅ Recognizer weights loaded successfully")
    else:
        print(f"  ⚠️  Recognizer weights not found: {args.recognizer_weights}")
        print(f"  ⚠️  Using random weights (evaluation will be meaningless)")
    
    # Load from checkpoint
    checkpoint = tf.train.Checkpoint(generator=generator, recognizer=recognizer)
    checkpoint_path = tf.train.latest_checkpoint(os.path.dirname(args.checkpoint_path))
    
    if checkpoint_path:
        status = checkpoint.restore(checkpoint_path)
        # Gunakan expect_partial() karena ada optimizer states yang tidak kita load
        status.expect_partial()
        print(f"  ✅ Models loaded from {checkpoint_path}")
    else:
        print(f"  ⚠️  No checkpoint found, trying direct path...")
        # Try direct path
        if os.path.exists(args.checkpoint_path + '.index'):
            checkpoint.restore(args.checkpoint_path).expect_partial()
            print(f"  ✅ Models loaded from {args.checkpoint_path}")
        else:
            print(f"  ❌ No checkpoint found at {args.checkpoint_path}")
            print(f"  ⚠️  Continuing with random weights (evaluation will be meaningless)")
    
    # Initialize metrics
    print(f"\n[4/6] Initializing metrics...")
    psnr_metric = Mean(name='psnr')
    ssim_metric = Mean(name='ssim')
    cer_degraded_metric = Mean(name='cer_degraded')
    cer_generated_metric = Mean(name='cer_generated')
    wer_degraded_metric = Mean(name='wer_degraded')
    wer_generated_metric = Mean(name='wer_generated')
    
    print(f"  ✅ Metrics ready: PSNR, SSIM, CER (degraded/generated), WER (degraded/generated)")
    
    # Evaluation loop
    print(f"\n[5/6] Running evaluation on validation set...")
    start_time = time.time()
    
    total_samples = 0
    all_results = []
    
    for batch_idx, (degraded_images, clean_images, text_labels) in enumerate(val_dataset):
        batch_size = degraded_images.shape[0]
        
        # Generate enhanced images
        generated_images = generator(degraded_images, training=False)
        
        # Calculate PSNR and SSIM
        psnr_batch = tf.image.psnr(clean_images, generated_images, max_val=1.0)
        ssim_batch = tf.image.ssim(clean_images, generated_images, max_val=1.0)
        
        psnr_metric.update_state(psnr_batch)
        ssim_metric.update_state(ssim_batch)
        
        # Recognizer inference - sudah dalam format (W, H, C)
        degraded_logits = recognizer(degraded_images, training=False)
        generated_logits = recognizer(generated_images, training=False)
        
        # Decode predictions
        degraded_texts = decode_predictions_batch(degraded_logits.numpy(), charset)
        generated_texts = decode_predictions_batch(generated_logits.numpy(), charset)
        
        # Ground truth texts - decode dari numeric labels
        gt_texts = []
        for label in text_labels:
            # Label dalam dataset GAN adalah 1-indexed (0 adalah padding)
            # Convert ke charset index (0-indexed)
            label_indices = [int(idx) - 1 for idx in label.numpy() if idx > 0]
            gt_text = ''.join([charset[idx] for idx in label_indices if 0 <= idx < charset_size])
            gt_texts.append(gt_text)
        
        # Calculate CER and WER for each sample
        for i in range(batch_size):
            gt = gt_texts[i]
            deg = degraded_texts[i]
            gen = generated_texts[i]
            
            cer_deg = calculate_cer(gt, deg)
            cer_gen = calculate_cer(gt, gen)
            wer_deg = calculate_wer(gt, deg)
            wer_gen = calculate_wer(gt, gen)
            
            cer_degraded_metric.update_state(cer_deg)
            cer_generated_metric.update_state(cer_gen)
            wer_degraded_metric.update_state(wer_deg)
            wer_generated_metric.update_state(wer_gen)
            
            # Store sample results (first 10 samples only for inspection)
            if len(all_results) < 10:
                all_results.append({
                    "batch": batch_idx,
                    "sample": i,
                    "ground_truth": gt,
                    "degraded_text": deg,
                    "generated_text": gen,
                    "psnr": float(psnr_batch[i].numpy()),
                    "ssim": float(ssim_batch[i].numpy()),
                    "cer_degraded": float(cer_deg),
                    "cer_generated": float(cer_gen),
                    "wer_degraded": float(wer_deg),
                    "wer_generated": float(wer_gen),
                    "cer_improvement": float(cer_deg - cer_gen)
                })
        
        total_samples += batch_size
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {total_samples} samples...")
    
    eval_time = time.time() - start_time
    
    # Aggregate results
    print(f"\n[6/6] Aggregating results...")
    
    results = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint_path,
            "dataset": args.tfrecord_path,
            "total_samples": total_samples,
            "batch_size": args.batch_size,
            "evaluation_time_seconds": float(eval_time)
        },
        "metrics": {
            "image_quality": {
                "psnr_mean": float(psnr_metric.result().numpy()),
                "ssim_mean": float(ssim_metric.result().numpy())
            },
            "readability_degraded": {
                "cer_mean": float(cer_degraded_metric.result().numpy()),
                "wer_mean": float(wer_degraded_metric.result().numpy())
            },
            "readability_generated": {
                "cer_mean": float(cer_generated_metric.result().numpy()),
                "wer_mean": float(wer_generated_metric.result().numpy())
            },
            "improvements": {
                "cer_improvement": float(cer_degraded_metric.result().numpy() - 
                                       cer_generated_metric.result().numpy()),
                "wer_improvement": float(wer_degraded_metric.result().numpy() - 
                                       wer_generated_metric.result().numpy()),
                "cer_improvement_percent": float((cer_degraded_metric.result().numpy() - 
                                                 cer_generated_metric.result().numpy()) / 
                                                cer_degraded_metric.result().numpy() * 100),
                "wer_improvement_percent": float((wer_degraded_metric.result().numpy() - 
                                                 wer_generated_metric.result().numpy()) / 
                                                wer_degraded_metric.result().numpy() * 100)
            }
        },
        "sample_results": all_results
    }
    
    # Save to JSON
    print(f"\nSaving results to {args.output_json}...")
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n📊 Image Quality Metrics:")
    print(f"  PSNR:  {results['metrics']['image_quality']['psnr_mean']:.2f} dB")
    print(f"  SSIM:  {results['metrics']['image_quality']['ssim_mean']:.4f}")
    
    print(f"\n📖 Readability Metrics (Degraded Input):")
    print(f"  CER:   {results['metrics']['readability_degraded']['cer_mean']:.4f} ({results['metrics']['readability_degraded']['cer_mean']*100:.2f}%)")
    print(f"  WER:   {results['metrics']['readability_degraded']['wer_mean']:.4f} ({results['metrics']['readability_degraded']['wer_mean']*100:.2f}%)")
    
    print(f"\n✨ Readability Metrics (Generated/Enhanced):")
    print(f"  CER:   {results['metrics']['readability_generated']['cer_mean']:.4f} ({results['metrics']['readability_generated']['cer_mean']*100:.2f}%)")
    print(f"  WER:   {results['metrics']['readability_generated']['wer_mean']:.4f} ({results['metrics']['readability_generated']['wer_mean']*100:.2f}%)")
    
    print(f"\n🚀 Improvements:")
    print(f"  CER:   {results['metrics']['improvements']['cer_improvement']:.4f} ({results['metrics']['improvements']['cer_improvement_percent']:.2f}% improvement)")
    print(f"  WER:   {results['metrics']['improvements']['wer_improvement']:.4f} ({results['metrics']['improvements']['wer_improvement_percent']:.2f}% improvement)")
    
    print(f"\n⏱️  Evaluation completed in {eval_time:.2f}s ({total_samples/eval_time:.1f} samples/sec)")
    print(f"\n✅ Full results saved to: {args.output_json}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of GAN-HTR model')
    parser.add_argument('--tfrecord_path', type=str, 
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help='Path to the TFRecord dataset')
    parser.add_argument('--charset_path', type=str, 
                       default='real_data_preparation/real_data_charlist.txt',
                       help='Path to the character set file')
    parser.add_argument('--checkpoint_path', type=str, 
                       default='dual_modal_gan/outputs/checkpoints_final_100/best_model',
                       help='Path to the checkpoint directory')
    parser.add_argument('--recognizer_weights', type=str,
                       default='models/best_htr_recognizer/best_model.weights.h5',
                       help='Path to standalone recognizer weights')
    parser.add_argument('--output_json', type=str, 
                       default='dual_modal_gan/outputs/checkpoints_final_100/metrics/comprehensive_metrics.json',
                       help='Path to save comprehensive metrics JSON')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for evaluation')
    parser.add_argument('--num_transformer_layers', type=int, default=6,
                       help='Number of transformer layers in recognizer (must match training, default: 6)')
    parser.add_argument('--gpu_id', type=str, default='1', 
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Run evaluation
    evaluate_comprehensive(args)
