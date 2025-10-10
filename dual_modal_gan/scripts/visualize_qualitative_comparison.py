#!/usr/bin/env python3
"""
Qualitative Comparison: Visual Results dari 4 Methods

Generates side-by-side comparison images showing:
- Degraded Input
- Plain U-Net Output
- Standard GAN Output  
- CTC-Only Output
- GAN-HTR Output
- Ground Truth (Clean)

Plus HTR predictions untuk masing-masing method.

Usage:
    poetry run python dual_modal_gan/scripts/visualize_qualitative_comparison.py
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dual_modal_gan.src.models.generator import unet
from dual_modal_gan.src.models.recognizer import load_frozen_recognizer

# Styling
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8


def load_charlist(charlist_path):
    """Load character list"""
    with open(charlist_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def decode_label(label_indices, charlist):
    """Decode label indices to text"""
    filtered = [idx for idx in label_indices if 0 <= idx < len(charlist)]
    return ''.join([charlist[idx] for idx in filtered])


def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord example"""
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
    
    return degraded_image, clean_image, label


def load_dataset(tfrecord_path, num_samples=10):
    """Load samples from TFRecord"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.take(num_samples)
    return dataset


def load_generator(checkpoint_path):
    """Load generator from checkpoint"""
    generator = unet()
    generator.load_weights(checkpoint_path)
    return generator


def denormalize_image(image):
    """Convert from [-1, 1] to [0, 255] for display"""
    # Clip to valid range
    image = tf.clip_by_value(image, -1.0, 1.0)
    # Denormalize
    image = (image + 1.0) * 127.5
    image = tf.cast(image, tf.uint8)
    return image.numpy()


def create_comparison_grid(sample_idx, degraded, clean, label_text,
                          generated_unet, generated_gan, generated_ctc, generated_gan_htr,
                          pred_degraded, pred_unet, pred_gan, pred_ctc, pred_gan_htr,
                          output_path):
    """
    Create comprehensive comparison grid showing:
    Row 1: All images side-by-side
    Row 2: HTR predictions below each image
    """
    
    # Fix orientation: images are (W=1024, H=128, C=1) - need correct orientation
    def fix_orientation(img):
        denorm = denormalize_image(img)
        # Image shape is (1024, 128, 1) - already landscape, just need proper rotation
        # Rotate 90 degrees clockwise to get text horizontal: (1024, 128) -> (128, 1024)
        return np.rot90(denorm.squeeze(), k=-1)[:, :, np.newaxis]  # k=-1 = clockwise
    
    # Denormalize all images with correct orientation
    imgs = {
        'Degraded\nInput': fix_orientation(degraded),
        'Plain U-Net': fix_orientation(generated_unet),
        'Standard GAN': fix_orientation(generated_gan),
        'CTC-Only': fix_orientation(generated_ctc),
        'GAN-HTR\n(Ours)': fix_orientation(generated_gan_htr),
        'Ground Truth': fix_orientation(clean)
    }
    
    predictions = {
        'Degraded\nInput': pred_degraded,
        'Plain U-Net': pred_unet,
        'Standard GAN': pred_gan,
        'CTC-Only': pred_ctc,
        'GAN-HTR\n(Ours)': pred_gan_htr,
        'Ground Truth': label_text
    }
    
    # Create figure with VERTICAL layout (1 column, 6 rows)
    # Each image is 128x1024 (HxW) after rotation - landscape orientation
    # Layout: 6 rows (one per method), 2 columns (image | prediction text)
    fig = plt.figure(figsize=(20, 24))  # Tall for 6 rows
    gs = GridSpec(6, 2, width_ratios=[3, 1], hspace=0.15, wspace=0.1)
    
    # Title
    fig.suptitle(f'Qualitative Comparison - Sample {sample_idx + 1}',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Plot images and predictions - VERTICAL LAYOUT (6 rows)
    methods_list = list(imgs.keys())
    predictions_list = list(predictions.values())
    
    for row, (method, img) in enumerate(imgs.items()):
        # Column 0: Image
        ax_img = fig.add_subplot(gs[row, 0])
        img_data = img.squeeze()
        ax_img.imshow(img_data, cmap='gray', aspect='auto', interpolation='bilinear')
        ax_img.axis('off')
        
        # Method label on left side
        color = 'darkgreen' if 'GAN-HTR' in method else 'black'
        weight = 'bold' if 'GAN-HTR' in method else 'normal'
        ax_img.set_ylabel(method, fontsize=12, fontweight=weight, color=color, 
                         rotation=0, ha='right', va='center', labelpad=10)
        
        # Highlight GAN-HTR with green border
        if 'GAN-HTR' in method:
            for spine in ax_img.spines.values():
                spine.set_edgecolor('darkgreen')
                spine.set_linewidth(4)
                spine.set_visible(True)
        
        # Column 1: Prediction text
        ax_text = fig.add_subplot(gs[row, 1])
        ax_text.axis('off')
        
        pred_text = predictions[method]
        
        # Color coding based on correctness
        if method == 'Ground Truth':
            text_color = 'black'
            bg_color = 'lightgray'
        elif pred_text == label_text:
            text_color = 'darkgreen'
            bg_color = 'lightgreen'
        else:
            text_color = 'darkred'
            bg_color = 'mistyrose'
        
        # Display prediction with word wrap
        display_text = pred_text if pred_text else '[empty]'
        if len(display_text) > 80:
            display_text = display_text[:80] + '...'
        
        ax_text.text(0.05, 0.5, display_text,
                    ha='left', va='center',
                    fontsize=9, wrap=True,
                    color=text_color,
                    bbox=dict(boxstyle='round,pad=0.8', 
                            facecolor=bg_color, 
                            edgecolor=text_color,
                            linewidth=2))
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
    
    # Add legend at bottom
    correct_patch = mpatches.Patch(color='lightgreen', label='Correct Prediction')
    wrong_patch = mpatches.Patch(color='mistyrose', label='Wrong Prediction')
    gt_patch = mpatches.Patch(color='lightgray', label='Ground Truth')
    fig.legend(handles=[correct_patch, wrong_patch, gt_patch],
              loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01),
              frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def create_summary_figure(all_results, output_path):
    """
    Create summary figure showing success rate per method
    """
    methods = ['Degraded\nInput', 'Plain U-Net', 'Standard GAN', 
               'CTC-Only', 'GAN-HTR\n(Ours)']
    
    correct_counts = {method: 0 for method in methods}
    total = len(all_results)
    
    # Count correct predictions
    for result in all_results:
        gt = result['ground_truth']
        for method in methods:
            if result['predictions'][method] == gt:
                correct_counts[method] += 1
    
    # Calculate accuracy
    accuracies = [correct_counts[m] / total * 100 for m in methods]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#2ca02c']
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8,
                 edgecolor='black', linewidth=1.0)
    
    # Styling
    ax.set_ylabel('Character Accuracy (%)', fontweight='bold', fontsize=11)
    ax.set_title('Character-Level Accuracy Comparison Across Methods',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{acc:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight best method
    best_idx = accuracies.index(max(accuracies))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Qualitative comparison of all methods')
    parser.add_argument('--dataset', type=str,
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help='Path to TFRecord dataset')
    parser.add_argument('--charlist', type=str,
                       default='real_data_preparation/real_data_charlist.txt',
                       help='Path to character list')
    parser.add_argument('--recognizer_weights', type=str,
                       default='models/best_htr_recognizer/best_model.weights.h5',
                       help='Path to HTR recognizer weights')
    parser.add_argument('--checkpoint_unet', type=str,
                       default='dual_modal_gan/outputs/baseline_unet_20251002_104041/checkpoints/best_model/generator.weights.h5',
                       help='U-Net checkpoint')
    parser.add_argument('--checkpoint_gan', type=str,
                       default='dual_modal_gan/outputs/baseline_standard_gan_20251002_144040/checkpoints/best_model/generator.weights.h5',
                       help='Standard GAN checkpoint')
    parser.add_argument('--checkpoint_ctc', type=str,
                       default='dual_modal_gan/outputs/baseline_ctc_only_20251002_142720/checkpoints/best_model/generator.weights.h5',
                       help='CTC-Only checkpoint')
    parser.add_argument('--checkpoint_gan_htr', type=str,
                       default='dual_modal_gan/outputs/checkpoints_final_100/generator.weights.h5',
                       help='GAN-HTR checkpoint')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str,
                       default='dual_modal_gan/outputs/baseline_analysis/qualitative_comparison',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("QUALITATIVE COMPARISON - 4 METHODS")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load charlist
    print("\n📚 Loading character list...")
    charlist = load_charlist(args.charlist)
    print(f"  ✅ Loaded {len(charlist)} characters")
    
    # Load recognizer
    print("\n🔍 Loading HTR recognizer...")
    # vocab_size = charlist size + 1 (CTC blank token)
    recognizer = load_frozen_recognizer(args.recognizer_weights, charset_size=len(charlist) + 1)
    print("  ✅ Recognizer loaded")
    
    # Load generators
    print("\n🎨 Loading generators...")
    print("  Loading Plain U-Net...")
    gen_unet = load_generator(args.checkpoint_unet)
    print("  Loading Standard GAN...")
    gen_gan = load_generator(args.checkpoint_gan)
    print("  Loading CTC-Only...")
    gen_ctc = load_generator(args.checkpoint_ctc)
    print("  Loading GAN-HTR...")
    gen_gan_htr = load_generator(args.checkpoint_gan_htr)
    print("  ✅ All generators loaded")
    
    # Load dataset
    print(f"\n📁 Loading {args.num_samples} samples...")
    dataset = load_dataset(args.dataset, args.num_samples)
    
    # Process samples
    print("\n" + "="*80)
    print("PROCESSING SAMPLES")
    print("="*80)
    
    all_results = []
    
    for idx, (degraded, clean, label) in enumerate(dataset):
        print(f"\n📊 Processing sample {idx + 1}/{args.num_samples}...")
        
        # Expand dims for batch
        degraded_batch = tf.expand_dims(degraded, 0)
        
        # Generate with all methods
        generated_unet = gen_unet(degraded_batch, training=False)[0]
        generated_gan = gen_gan(degraded_batch, training=False)[0]
        generated_ctc = gen_ctc(degraded_batch, training=False)[0]
        generated_gan_htr = gen_gan_htr(degraded_batch, training=False)[0]
        
        # Get HTR predictions for all versions
        pred_degraded_logits = recognizer(degraded_batch, training=False)
        pred_unet_logits = recognizer(tf.expand_dims(generated_unet, 0), training=False)
        pred_gan_logits = recognizer(tf.expand_dims(generated_gan, 0), training=False)
        pred_ctc_logits = recognizer(tf.expand_dims(generated_ctc, 0), training=False)
        pred_gan_htr_logits = recognizer(tf.expand_dims(generated_gan_htr, 0), training=False)
        
        # Decode predictions (greedy)
        pred_degraded = decode_label(tf.argmax(pred_degraded_logits[0], axis=-1).numpy(), charlist)
        pred_unet = decode_label(tf.argmax(pred_unet_logits[0], axis=-1).numpy(), charlist)
        pred_gan = decode_label(tf.argmax(pred_gan_logits[0], axis=-1).numpy(), charlist)
        pred_ctc = decode_label(tf.argmax(pred_ctc_logits[0], axis=-1).numpy(), charlist)
        pred_gan_htr = decode_label(tf.argmax(pred_gan_htr_logits[0], axis=-1).numpy(), charlist)
        
        # Ground truth
        label_text = decode_label(label.numpy(), charlist)
        
        print(f"  Ground Truth: '{label_text}'")
        print(f"  Degraded:     '{pred_degraded}'")
        print(f"  U-Net:        '{pred_unet}'")
        print(f"  Standard GAN: '{pred_gan}'")
        print(f"  CTC-Only:     '{pred_ctc}'")
        print(f"  GAN-HTR:      '{pred_gan_htr}'")
        
        # Save results
        all_results.append({
            'ground_truth': label_text,
            'predictions': {
                'Degraded\nInput': pred_degraded,
                'Plain U-Net': pred_unet,
                'Standard GAN': pred_gan,
                'CTC-Only': pred_ctc,
                'GAN-HTR\n(Ours)': pred_gan_htr
            }
        })
        
        # Create comparison grid
        output_path = os.path.join(args.output_dir, f'comparison_sample_{idx+1:02d}.png')
        create_comparison_grid(
            idx, degraded, clean, label_text,
            generated_unet, generated_gan, generated_ctc, generated_gan_htr,
            pred_degraded, pred_unet, pred_gan, pred_ctc, pred_gan_htr,
            output_path
        )
    
    # Create summary figure
    print("\n📊 Creating summary figure...")
    summary_path = os.path.join(args.output_dir, 'summary_accuracy_comparison.png')
    create_summary_figure(all_results, summary_path)
    
    print("\n" + "="*80)
    print("✅ QUALITATIVE COMPARISON COMPLETE!")
    print("="*80)
    print(f"\n📁 Output directory: {args.output_dir}")
    print(f"   - {args.num_samples} individual comparison images")
    print(f"   - 1 summary accuracy chart")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
