"""
Generate Sample Visual Comparison: Degraded → Restored → Clean

Creates visual examples showing the restoration quality
with CER/WER metrics for each sample

Author: Visual Samples for Paper
Date: 2025-11-04
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import editdistance

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced as create_generator
from dual_modal_gan.src.models.recognizer_fixed import load_frozen_recognizer_fixed as load_frozen_recognizer

# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def read_charlist(path):
    """Load character list"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def decode_ctc_predictions(logits, charset):
    """Decode CTC predictions"""
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
    """Decode label IDs to text"""
    decoded_chars = []
    prev_id = -1
    for label_id in label_ids:
        label_id = int(label_id)
        if label_id > 0 and label_id != prev_id:
            if label_id - 1 < len(charset):
                decoded_chars.append(charset[label_id - 1])
        prev_id = label_id
    return ''.join(decoded_chars)

def calculate_cer(gt, pred):
    """Calculate CER"""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(gt, pred) / len(gt)

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord"""
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
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])

    # Clean image
    clean_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])

    # Label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.int32)
    padding = [[0, 128 - tf.shape(label)[0]]]
    label = tf.pad(label, padding, "CONSTANT", constant_values=0)
    label.set_shape([128])

    return degraded_image, clean_image, label

def create_test_dataset(tfrecord_path, batch_size=1, train_split=0.7, val_split=0.15):
    """Create test dataset"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return test_dataset

def generate_sample_visualizations(
    num_samples=10,
    output_dir='Paper/visualizations'
):
    """Generate sample visual comparisons"""
    
    print("="*80)
    print("SAMPLE VISUAL COMPARISON GENERATOR")
    print("="*80)
    
    # Configuration
    TFRECORD_PATH = 'dual_modal_gan/data/dataset_gan.tfrecord'
    CHARSET_PATH = 'real_data_preparation/real_data_charlist.txt'
    GENERATOR_WEIGHTS = 'dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_generator.weights.h5'
    RECOGNIZER_WEIGHTS = '/home/lambda_one/tesis/GAN-HTR-ORI/htr_improved_v2_20251001_221138/best_model.weights.h5'
    
    # Load charset
    charset = read_charlist(CHARSET_PATH)
    vocab_size = len(charset) + 1
    print(f"\nCharset: {vocab_size} characters")
    
    # Load models
    print("\n🔧 Loading models...")
    
    # Generator
    generator = create_generator(input_size=(1024, 128, 1))
    generator.load_weights(GENERATOR_WEIGHTS)
    print(f"✅ Generator loaded: {GENERATOR_WEIGHTS}")
    
    # Recognizer
    recognizer = load_frozen_recognizer(
        weights_path=RECOGNIZER_WEIGHTS,
        charset_size=vocab_size - 1,
        return_feature_map=False
    )
    print(f"✅ Recognizer loaded: {RECOGNIZER_WEIGHTS}")
    
    # Load test dataset
    print(f"\n📂 Loading test dataset...")
    test_dataset = create_test_dataset(TFRECORD_PATH, batch_size=1)
    
    # Collect samples
    print(f"\n🎨 Generating {num_samples} sample visualizations...")
    
    samples = []
    count = 0
    
    for degraded, clean, label in test_dataset:
        if count >= num_samples:
            break
        
        # Generate restored image
        restored = generator(degraded, training=False)
        
        # Get predictions
        degraded_logits = recognizer(degraded, training=False)
        if isinstance(degraded_logits, (list, tuple)):
            degraded_logits = degraded_logits[0]
        
        restored_logits = recognizer(restored, training=False)
        if isinstance(restored_logits, (list, tuple)):
            restored_logits = restored_logits[0]
        
        clean_logits = recognizer(clean, training=False)
        if isinstance(clean_logits, (list, tuple)):
            clean_logits = clean_logits[0]
        
        # Decode texts
        gt_text = decode_label(label[0].numpy(), charset)
        degraded_text = decode_ctc_predictions(degraded_logits.numpy(), charset)[0]
        restored_text = decode_ctc_predictions(restored_logits.numpy(), charset)[0]
        clean_text = decode_ctc_predictions(clean_logits.numpy(), charset)[0]
        
        # Calculate CER
        cer_degraded = calculate_cer(gt_text, degraded_text)
        cer_restored = calculate_cer(gt_text, restored_text)
        cer_clean = calculate_cer(gt_text, clean_text)
        
        # Calculate PSNR
        psnr_degraded = tf.image.psnr(clean, degraded, max_val=1.0).numpy()[0]
        psnr_restored = tf.image.psnr(clean, restored, max_val=1.0).numpy()[0]
        
        samples.append({
            'degraded': degraded[0].numpy(),
            'restored': restored[0].numpy(),
            'clean': clean[0].numpy(),
            'gt_text': gt_text,
            'degraded_text': degraded_text,
            'restored_text': restored_text,
            'clean_text': clean_text,
            'cer_degraded': cer_degraded,
            'cer_restored': cer_restored,
            'cer_clean': cer_clean,
            'psnr_degraded': psnr_degraded,
            'psnr_restored': psnr_restored
        })
        
        count += 1
        print(f"  Sample {count}/{num_samples}: CER {cer_degraded*100:.1f}% → {cer_restored*100:.1f}% (GT: {cer_clean*100:.1f}%)")
    
    # Create visualization
    print(f"\n📊 Creating visualization grid...")
    
    fig = plt.figure(figsize=(14, 2.5 * num_samples))
    gs = GridSpec(num_samples, 4, figure=fig, wspace=0.05, hspace=0.3,
                  width_ratios=[1, 1, 1, 0.5])
    
    for i, sample in enumerate(samples):
        # Degraded
        ax_deg = fig.add_subplot(gs[i, 0])
        ax_deg.imshow(sample['degraded'].squeeze(), cmap='gray')
        ax_deg.set_title(f'Terdegradasi\nPSNR: {sample["psnr_degraded"]:.1f} dB\nCER: {sample["cer_degraded"]*100:.1f}%',
                        fontsize=8, color='red', fontweight='bold')
        ax_deg.axis('off')
        
        # Restored
        ax_res = fig.add_subplot(gs[i, 1])
        ax_res.imshow(sample['restored'].squeeze(), cmap='gray')
        ax_res.set_title(f'Terestorasi (Usulan)\nPSNR: {sample["psnr_restored"]:.1f} dB\nCER: {sample["cer_restored"]*100:.1f}%',
                        fontsize=8, color='green', fontweight='bold')
        ax_res.axis('off')
        
        # Clean (GT)
        ax_clean = fig.add_subplot(gs[i, 2])
        ax_clean.imshow(sample['clean'].squeeze(), cmap='gray')
        ax_clean.set_title(f'Bersih (GT)\nCER: {sample["cer_clean"]*100:.1f}%',
                          fontsize=8, color='blue', fontweight='bold')
        ax_clean.axis('off')
        
        # Text info
        ax_text = fig.add_subplot(gs[i, 3])
        ax_text.axis('off')
        
        info_text = f'GT:\n{sample["gt_text"][:30]}\n\n'
        info_text += f'Deg:\n{sample["degraded_text"][:30]}\n\n'
        info_text += f'Res:\n{sample["restored_text"][:30]}\n\n'
        info_text += f'Clean:\n{sample["clean_text"][:30]}'
        
        ax_text.text(0.05, 0.5, info_text, fontsize=6, verticalalignment='center',
                    family='monospace', wrap=True)
    
    plt.suptitle('Perbandingan Visual: Terdegradasi → Terestorasi → Bersih',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    output_path = os.path.join(output_dir, 'fig_sample_comparisons.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()
    
    # Create summary statistics
    avg_cer_degraded = np.mean([s['cer_degraded'] for s in samples]) * 100
    avg_cer_restored = np.mean([s['cer_restored'] for s in samples]) * 100
    avg_cer_clean = np.mean([s['cer_clean'] for s in samples]) * 100
    avg_psnr_degraded = np.mean([s['psnr_degraded'] for s in samples])
    avg_psnr_restored = np.mean([s['psnr_restored'] for s in samples])
    
    print(f"\n{'='*80}")
    print("SAMPLE STATISTICS:")
    print(f"{'='*80}")
    print(f"Average PSNR (Degraded): {avg_psnr_degraded:.2f} dB")
    print(f"Average PSNR (Restored): {avg_psnr_restored:.2f} dB")
    print(f"Average CER (Degraded):  {avg_cer_degraded:.1f}%")
    print(f"Average CER (Restored):  {avg_cer_restored:.1f}%")
    print(f"Average CER (Clean):     {avg_cer_clean:.1f}%")
    print(f"{'='*80}\n")
    
    return samples

if __name__ == '__main__':
    os.makedirs('Paper/visualizations', exist_ok=True)
    generate_sample_visualizations(num_samples=8)
    print("✅ Sample visualizations completed!")
