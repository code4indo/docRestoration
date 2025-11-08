#!/usr/bin/env python3
"""
Visualize DIBCO TFRecord Samples
=================================
Show resized DIBCO images from TFRecord to verify preprocessing quality.

Usage:
    poetry run python scripts/visualize_dibco_tfrecord.py \
        --tfrecord_path dual_modal_gan/data/dibco_finetuning.tfrecord \
        --num_samples 10 \
        --output_dir outputs/dibco_visualization
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def parse_tfrecord(example_proto):
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
    # Transpose from (H, W, C) to (W, H, C) for display
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
    
    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    # Transpose from (H, W, C) to (W, H, C) for display
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    
    return degraded_image, clean_image


def visualize_samples(tfrecord_path, num_samples=10, output_dir='outputs/dibco_visualization'):
    """Visualize DIBCO samples from TFRecord"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Visualizing DIBCO TFRecord samples...")
    print(f"   TFRecord: {tfrecord_path}")
    print(f"   Output: {output_dir}")
    print(f"   Samples: {num_samples}\n")
    
    # Load dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    
    # Visualize samples
    for idx, (degraded, clean) in enumerate(dataset.take(num_samples)):
        degraded_np = degraded.numpy().squeeze()  # (W, H, 1) -> (W, H)
        clean_np = clean.numpy().squeeze()
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        
        # Degraded image
        axes[0].imshow(degraded_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Sample {idx+1} - Degraded (Input)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Clean image
        axes[1].imshow(clean_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Sample {idx+1} - Clean (Ground Truth)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add info
        info_text = (
            f"Shape: {degraded_np.shape}\n"
            f"Degraded range: [{degraded_np.min():.3f}, {degraded_np.max():.3f}]\n"
            f"Clean range: [{clean_np.min():.3f}, {clean_np.max():.3f}]"
        )
        fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f'sample_{idx+1:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved: {output_path.name}")
        print(f"      Degraded: [{degraded_np.min():.3f}, {degraded_np.max():.3f}], "
              f"shape={degraded_np.shape}")
        print(f"      Clean:    [{clean_np.min():.3f}, {clean_np.max():.3f}], "
              f"shape={clean_np.shape}\n")
    
    print(f"\n✅ Visualization complete!")
    print(f"   Saved {num_samples} images to: {output_dir}")
    print(f"\n💡 Tip: Open images to verify:")
    print(f"   - Aspect ratio preservation")
    print(f"   - Padding quality (white borders)")
    print(f"   - Text readability after resize")
    print(f"   - No distortion or artifacts")


def create_grid_visualization(tfrecord_path, num_samples=9, output_path='outputs/dibco_grid.png'):
    """Create grid visualization of multiple samples"""
    
    print(f"\n📊 Creating grid visualization...")
    
    # Load dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    
    # Calculate grid size
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # Plot samples
    for idx, (degraded, clean) in enumerate(dataset.take(num_samples)):
        if idx >= len(axes):
            break
            
        degraded_np = degraded.numpy().squeeze()
        
        axes[idx].imshow(degraded_np, cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'Sample {idx+1}', fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('DIBCO Dataset Samples (Degraded Images After Resize)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Grid saved: {output_path}")
    print(f"   ✅ Quick preview available!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DIBCO TFRecord samples',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--tfrecord_path', type=str,
                        default='dual_modal_gan/data/dibco_finetuning.tfrecord',
                        help='Path to DIBCO TFRecord file')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize (default: 10)')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/dibco_visualization',
                        help='Output directory for individual images')
    parser.add_argument('--create_grid', action='store_true',
                        help='Also create grid visualization')
    parser.add_argument('--grid_samples', type=int, default=9,
                        help='Number of samples in grid (default: 9)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 DIBCO TFRecord Visualization")
    print("="*80)
    
    # Check TFRecord exists
    if not Path(args.tfrecord_path).exists():
        print(f"\n❌ Error: TFRecord not found: {args.tfrecord_path}")
        return 1
    
    # Visualize individual samples
    visualize_samples(args.tfrecord_path, args.num_samples, args.output_dir)
    
    # Create grid if requested
    if args.create_grid:
        grid_path = Path(args.output_dir) / 'grid_preview.png'
        create_grid_visualization(args.tfrecord_path, args.grid_samples, grid_path)
    
    print(f"\n{'='*80}")
    print(f"✅ Done! Check the images in: {args.output_dir}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
