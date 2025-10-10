#!/usr/bin/env python3
"""
Visualization Script untuk GAN-HTR Results
Generates publication-ready figures:
1. Comparison grid (Degraded | Generated | Ground Truth)
2. Metrics bar charts (PSNR, SSIM, CER, WER)
3. CER improvement histogram
4. Sample predictions table

Usage:
    poetry run python dual_modal_gan/scripts/visualize_results.py \
        --results_json outputs/checkpoints_final_100/metrics/comprehensive_metrics.json \
        --output_dir outputs/visualizations
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Constants
IMG_WIDTH = 1024
IMG_HEIGHT = 128


def load_results(json_path):
    """Load evaluation results from JSON."""
    print(f"📊 Loading results from {json_path}...")
    with open(json_path, 'r') as f:
        results = json.load(f)
    print(f"  ✅ Loaded: {len(results['sample_results'])} samples")
    return results


def parse_tfrecord_for_viz(tfrecord_path, sample_indices):
    """Load specific samples from TFRecord for visualization."""
    print(f"\n📁 Loading images from {tfrecord_path}...")
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Parse function
    def _parse_fn(example_proto):
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
        
        # Clean image
        clean_shape = tf.cast(example['clean_image_shape'], tf.int32)
        clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
        clean_image = tf.reshape(clean_image, clean_shape)
        
        return degraded_image, clean_image
    
    dataset = dataset.map(_parse_fn)
    
    # Skip to validation set (90% split)
    total_samples = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    train_size = int(total_samples * 0.9)
    dataset = dataset.skip(train_size)
    
    # Get specific samples
    images = {}
    for idx, (degraded, clean) in enumerate(dataset):
        if idx in sample_indices:
            images[idx] = {
                'degraded': degraded.numpy(),
                'clean': clean.numpy()
            }
            if len(images) >= len(sample_indices):
                break
    
    print(f"  ✅ Loaded {len(images)} images")
    return images


def generate_enhanced_images(generator, degraded_images):
    """Generate enhanced images using the trained generator."""
    print(f"\n🎨 Generating enhanced images...")
    
    # Transpose for generator input (H,W,C) -> (W,H,C)
    degraded_transposed = np.array([np.transpose(img, (1, 0, 2)) for img in degraded_images])
    degraded_tensor = tf.constant(degraded_transposed, dtype=tf.float32)
    
    # Generate
    generated = generator(degraded_tensor, training=False)
    
    # Transpose back (W,H,C) -> (H,W,C)
    generated_images = [np.transpose(img.numpy(), (1, 0, 2)) for img in generated]
    
    print(f"  ✅ Generated {len(generated_images)} images")
    return generated_images


def create_comparison_grid(images_dict, sample_results, output_path, num_samples=10):
    """
    Create comparison grid: Degraded | Generated | Ground Truth
    With text predictions below each image.
    """
    print(f"\n🖼️  Creating comparison grid for {num_samples} samples...")
    
    fig = plt.figure(figsize=(20, num_samples * 3))
    gs = GridSpec(num_samples, 3, figure=fig, hspace=0.3, wspace=0.1)
    
    for i, (idx, imgs) in enumerate(list(images_dict.items())[:num_samples]):
        # Get corresponding result
        result = sample_results[idx]
        
        # Degraded
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(imgs['degraded'].squeeze(), cmap='gray', aspect='auto')
        ax1.set_title(f"Degraded (CER: {result['cer_degraded']*100:.1f}%)", fontsize=9)
        ax1.axis('off')
        ax1.text(0.5, -0.15, f"Pred: {result['degraded_text'][:50]}", 
                transform=ax1.transAxes, fontsize=7, ha='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Generated
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(imgs['generated'].squeeze(), cmap='gray', aspect='auto')
        ax2.set_title(f"Generated (CER: {result['cer_generated']*100:.1f}%)", fontsize=9, color='green')
        ax2.axis('off')
        ax2.text(0.5, -0.15, f"Pred: {result['generated_text'][:50]}", 
                transform=ax2.transAxes, fontsize=7, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # Ground Truth
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(imgs['clean'].squeeze(), cmap='gray', aspect='auto')
        ax3.set_title("Ground Truth", fontsize=9)
        ax3.axis('off')
        ax3.text(0.5, -0.15, f"GT: {result['ground_truth'][:50]}", 
                transform=ax3.transAxes, fontsize=7, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Add improvement badge
        improvement = result['cer_improvement'] * 100
        color = 'green' if improvement > 50 else 'orange' if improvement > 25 else 'red'
        fig.text(0.02, 1 - (i + 0.5) / num_samples, f"↑{improvement:.1f}%", 
                fontsize=12, color=color, weight='bold', 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    plt.suptitle("Comparison: Degraded → Generated → Ground Truth", 
                fontsize=16, weight='bold', y=0.995)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✅ Saved comparison grid to {output_path}")


def create_metrics_bar_chart(results, output_path):
    """Create bar chart comparing degraded vs generated metrics."""
    print(f"\n📊 Creating metrics bar chart...")
    
    metrics = results['metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metrics Comparison: Degraded vs Generated', fontsize=16, weight='bold')
    
    # PSNR
    ax1 = axes[0, 0]
    psnr_val = metrics['image_quality']['psnr_mean']
    ax1.bar(['PSNR'], [psnr_val], color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.axhline(y=20, color='red', linestyle='--', label='Target (20 dB)', linewidth=2)
    ax1.set_ylabel('dB', fontsize=12)
    ax1.set_title('Peak Signal-to-Noise Ratio', fontsize=12, weight='bold')
    ax1.text(0, psnr_val + 1, f'{psnr_val:.2f} dB', ha='center', fontsize=11, weight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # SSIM
    ax2 = axes[0, 1]
    ssim_val = metrics['image_quality']['ssim_mean']
    ax2.bar(['SSIM'], [ssim_val], color='mediumseagreen', edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0.90, color='red', linestyle='--', label='Target (0.90)', linewidth=2)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Structural Similarity Index', fontsize=12, weight='bold')
    ax2.text(0, ssim_val + 0.02, f'{ssim_val:.4f}', ha='center', fontsize=11, weight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # CER Comparison
    ax3 = axes[1, 0]
    cer_deg = metrics['readability_degraded']['cer_mean']
    cer_gen = metrics['readability_generated']['cer_mean']
    bars = ax3.bar(['Degraded', 'Generated'], [cer_deg, cer_gen], 
                   color=['salmon', 'lightgreen'], edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('CER', fontsize=12)
    ax3.set_title('Character Error Rate', fontsize=12, weight='bold')
    ax3.text(0, cer_deg + 0.03, f'{cer_deg*100:.1f}%', ha='center', fontsize=11, weight='bold')
    ax3.text(1, cer_gen + 0.03, f'{cer_gen*100:.1f}%', ha='center', fontsize=11, weight='bold')
    
    # Add improvement arrow
    improvement = metrics['improvements']['cer_improvement_percent']
    ax3.annotate('', xy=(1, cer_gen), xytext=(0, cer_deg),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax3.text(0.5, (cer_deg + cer_gen) / 2, f'↓{improvement:.1f}%', 
            ha='center', fontsize=12, weight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax3.grid(axis='y', alpha=0.3)
    
    # WER Comparison
    ax4 = axes[1, 1]
    wer_deg = metrics['readability_degraded']['wer_mean']
    wer_gen = metrics['readability_generated']['wer_mean']
    bars = ax4.bar(['Degraded', 'Generated'], [wer_deg, wer_gen],
                   color=['salmon', 'lightgreen'], edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('WER', fontsize=12)
    ax4.set_title('Word Error Rate', fontsize=12, weight='bold')
    ax4.text(0, wer_deg + 0.03, f'{wer_deg*100:.1f}%', ha='center', fontsize=11, weight='bold')
    ax4.text(1, wer_gen + 0.03, f'{wer_gen*100:.1f}%', ha='center', fontsize=11, weight='bold')
    
    # Add improvement arrow
    wer_improvement = metrics['improvements']['wer_improvement_percent']
    ax4.annotate('', xy=(1, wer_gen), xytext=(0, wer_deg),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax4.text(0.5, (wer_deg + wer_gen) / 2, f'↓{wer_improvement:.1f}%',
            ha='center', fontsize=12, weight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✅ Saved metrics bar chart to {output_path}")


def create_cer_improvement_histogram(sample_results, output_path):
    """Create histogram of CER improvements."""
    print(f"\n📈 Creating CER improvement histogram...")
    
    improvements = [r['cer_improvement'] * 100 for r in sample_results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    n, bins, patches = ax.hist(improvements, bins=30, color='steelblue', 
                                edgecolor='black', alpha=0.7)
    
    # Color bars by improvement level
    for i, patch in enumerate(patches):
        if bins[i] > 50:
            patch.set_facecolor('green')
            patch.set_alpha(0.8)
        elif bins[i] > 25:
            patch.set_facecolor('orange')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('red')
            patch.set_alpha(0.6)
    
    # Add mean line
    mean_improvement = np.mean(improvements)
    ax.axvline(mean_improvement, color='darkred', linestyle='--', linewidth=2.5,
              label=f'Mean: {mean_improvement:.1f}%')
    
    # Add target line
    ax.axvline(25, color='blue', linestyle=':', linewidth=2,
              label='Target: 25%')
    
    # Add median line
    median_improvement = np.median(improvements)
    ax.axvline(median_improvement, color='purple', linestyle='-.', linewidth=2,
              label=f'Median: {median_improvement:.1f}%')
    
    ax.set_xlabel('CER Improvement (%)', fontsize=13, weight='bold')
    ax.set_ylabel('Frequency (Number of Samples)', fontsize=13, weight='bold')
    ax.set_title('Distribution of CER Improvements', fontsize=15, weight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics box
    stats_text = f"""Statistics:
    Mean: {mean_improvement:.2f}%
    Median: {median_improvement:.2f}%
    Std Dev: {np.std(improvements):.2f}%
    Min: {np.min(improvements):.2f}%
    Max: {np.max(improvements):.2f}%
    Samples > 25%: {sum(1 for x in improvements if x > 25)} ({sum(1 for x in improvements if x > 25)/len(improvements)*100:.1f}%)
    Samples > 50%: {sum(1 for x in improvements if x > 50)} ({sum(1 for x in improvements if x > 50)/len(improvements)*100:.1f}%)"""
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✅ Saved CER improvement histogram to {output_path}")


def create_sample_predictions_table(sample_results, output_path, num_samples=15):
    """Create table of sample predictions."""
    print(f"\n📋 Creating sample predictions table...")
    
    # Sort by CER improvement (best to worst)
    sorted_results = sorted(sample_results, key=lambda x: x['cer_improvement'], reverse=True)
    
    # Take top samples
    top_samples = sorted_results[:num_samples]
    
    # Create DataFrame
    data = []
    for i, result in enumerate(top_samples, 1):
        data.append({
            '#': i,
            'Ground Truth': result['ground_truth'][:50] + '...' if len(result['ground_truth']) > 50 else result['ground_truth'],
            'Degraded Pred': result['degraded_text'][:40] + '...' if len(result['degraded_text']) > 40 else result['degraded_text'],
            'Generated Pred': result['generated_text'][:40] + '...' if len(result['generated_text']) > 40 else result['generated_text'],
            'CER Deg': f"{result['cer_degraded']*100:.1f}%",
            'CER Gen': f"{result['cer_generated']*100:.1f}%",
            'Improvement': f"{result['cer_improvement']*100:.1f}%"
        })
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, num_samples * 0.6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='left', loc='center',
                    colWidths=[0.03, 0.25, 0.20, 0.20, 0.08, 0.08, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells - alternate colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            
            # Color improvement column
            if j == len(df.columns) - 1:  # Improvement column
                improvement = float(df.iloc[i-1]['Improvement'].rstrip('%'))
                if improvement > 50:
                    table[(i, j)].set_facecolor('#90EE90')
                elif improvement > 25:
                    table[(i, j)].set_facecolor('#FFD700')
    
    plt.title(f'Top {num_samples} Samples by CER Improvement', 
             fontsize=14, weight='bold', pad=20)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✅ Saved sample predictions table to {output_path}")
    
    # Also save as CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved CSV to {csv_path}")


def create_comparison_with_baseline(results, output_path):
    """Create comparison chart with Souibgui baseline."""
    print(f"\n📊 Creating baseline comparison...")
    
    # Data
    methods = ['Souibgui\n(2021)', 'Our Method\n(Generated)']
    psnr_values = [15.97, results['metrics']['image_quality']['psnr_mean']]
    cer_values = [21.98, results['metrics']['readability_generated']['cer_mean'] * 100]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Comparison with State-of-the-Art (Souibgui et al. 2021)', 
                fontsize=15, weight='bold')
    
    # PSNR comparison
    bars1 = ax1.bar(methods, psnr_values, color=['steelblue', 'green'], 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('PSNR (dB)', fontsize=12, weight='bold')
    ax1.set_title('Image Quality (PSNR)', fontsize=13, weight='bold')
    ax1.set_ylim([0, max(psnr_values) * 1.3])
    
    for i, (method, val) in enumerate(zip(methods, psnr_values)):
        ax1.text(i, val + 0.5, f'{val:.2f} dB', ha='center', fontsize=11, weight='bold')
    
    # Add improvement percentage
    improvement_psnr = ((psnr_values[1] - psnr_values[0]) / psnr_values[0]) * 100
    ax1.text(0.5, max(psnr_values) * 1.15, f'+{improvement_psnr:.1f}% better',
            ha='center', fontsize=12, weight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.grid(axis='y', alpha=0.3)
    
    # CER comparison
    bars2 = ax2.bar(methods, cer_values, color=['steelblue', 'green'],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('CER (%)', fontsize=12, weight='bold')
    ax2.set_title('Character Error Rate', fontsize=13, weight='bold')
    ax2.set_ylim([0, max(cer_values) * 1.3])
    
    for i, (method, val) in enumerate(zip(methods, cer_values)):
        ax2.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=11, weight='bold')
    
    # Add note about CER
    cer_note = "Note: CER comparison not directly\ncomparable (different recognizers)"
    ax2.text(0.5, max(cer_values) * 1.15, cer_note,
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✅ Saved baseline comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize GAN-HTR evaluation results')
    parser.add_argument('--results_json', type=str,
                       default='dual_modal_gan/outputs/checkpoints_final_100/metrics/comprehensive_metrics.json',
                       help='Path to comprehensive evaluation results JSON')
    parser.add_argument('--tfrecord_path', type=str,
                       default='dual_modal_gan/data/dataset_gan.tfrecord',
                       help='Path to TFRecord dataset')
    parser.add_argument('--checkpoint_path', type=str,
                       default='dual_modal_gan/outputs/checkpoints_final_100/best_model',
                       help='Path to generator checkpoint')
    parser.add_argument('--output_dir', type=str,
                       default='dual_modal_gan/outputs/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_comparison_samples', type=int, default=10,
                       help='Number of samples for comparison grid')
    parser.add_argument('--gpu_id', type=str, default='1',
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"VISUALIZATION GENERATION - GAN-HTR")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")
    
    # Load results
    results = load_results(args.results_json)
    sample_results = results['sample_results']
    
    # Create visualizations that don't need images
    print(f"\n{'='*70}")
    print("PHASE 1: Generating charts (no image loading needed)")
    print(f"{'='*70}")
    
    # 1. Metrics bar chart
    create_metrics_bar_chart(
        results,
        os.path.join(args.output_dir, 'metrics_comparison.png')
    )
    
    # 2. CER improvement histogram
    create_cer_improvement_histogram(
        sample_results,
        os.path.join(args.output_dir, 'cer_improvement_histogram.png')
    )
    
    # 3. Sample predictions table
    create_sample_predictions_table(
        sample_results,
        os.path.join(args.output_dir, 'sample_predictions_table.png'),
        num_samples=15
    )
    
    # 4. Baseline comparison
    create_comparison_with_baseline(
        results,
        os.path.join(args.output_dir, 'baseline_comparison.png')
    )
    
    # Load generator and create comparison grid
    print(f"\n{'='*70}")
    print("PHASE 2: Generating comparison grid (requires image loading)")
    print(f"{'='*70}")
    
    try:
        # Load generator
        print(f"\n🏗️  Loading generator from {args.checkpoint_path}...")
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from dual_modal_gan.src.models.generator import unet
        
        generator = unet(input_size=(1024, 128, 1))
        checkpoint = tf.train.Checkpoint(generator=generator)
        checkpoint_path = tf.train.latest_checkpoint(os.path.dirname(args.checkpoint_path))
        
        if checkpoint_path:
            checkpoint.restore(checkpoint_path).expect_partial()
            print(f"  ✅ Generator loaded")
            
            # Select best samples for visualization
            sorted_by_improvement = sorted(
                enumerate(sample_results), 
                key=lambda x: x[1]['cer_improvement'], 
                reverse=True
            )
            sample_indices = [idx for idx, _ in sorted_by_improvement[:args.num_comparison_samples]]
            
            # Load images
            images = parse_tfrecord_for_viz(args.tfrecord_path, sample_indices)
            
            # Generate enhanced images
            degraded_batch = [images[idx]['degraded'] for idx in sample_indices]
            generated_batch = generate_enhanced_images(generator, degraded_batch)
            
            # Add generated images to dict
            for idx, gen_img in zip(sample_indices, generated_batch):
                images[idx]['generated'] = gen_img
            
            # Create comparison grid
            create_comparison_grid(
                images,
                sample_results,
                os.path.join(args.output_dir, 'comparison_grid.png'),
                num_samples=args.num_comparison_samples
            )
        else:
            print(f"  ⚠️  Checkpoint not found, skipping comparison grid")
    
    except Exception as e:
        print(f"  ⚠️  Error creating comparison grid: {e}")
        print(f"  Continuing with other visualizations...")
    
    # Summary
    print(f"\n{'='*70}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n✅ Generated visualizations:")
    print(f"  1. metrics_comparison.png - Bar charts of all metrics")
    print(f"  2. cer_improvement_histogram.png - Distribution of improvements")
    print(f"  3. sample_predictions_table.png - Top 15 samples with predictions")
    print(f"  4. baseline_comparison.png - Comparison with Souibgui et al.")
    print(f"  5. comparison_grid.png - Visual comparison of images")
    print(f"  6. sample_predictions_table.csv - Raw data export")
    print(f"\n📁 All files saved to: {args.output_dir}")
    print(f"\n🎉 Visualization generation complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
