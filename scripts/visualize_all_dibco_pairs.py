"""
Visualize ALL degraded-clean pairs from DIBCO TFRecord for manual verification.
Generates grid visualization of all samples with degraded on top, clean on bottom.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

def parse_tfrecord_fn(serialized_example):
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
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    # Decode images from raw bytes
    degraded_shape = tf.cast(example['degraded_image_shape'], tf.int32)
    clean_shape = tf.cast(example['clean_image_shape'], tf.int32)
    
    degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
    degraded_image = tf.reshape(degraded_image, degraded_shape)
    
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_shape)
    
    return degraded_image, clean_image

def load_all_samples(tfrecord_path):
    """Load all samples from TFRecord"""
    print(f"📂 Loading samples from: {tfrecord_path}")
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn)
    
    degraded_images = []
    clean_images = []
    
    for degraded, clean in tqdm(dataset, desc="Loading samples"):
        degraded_images.append(degraded.numpy())
        clean_images.append(clean.numpy())
    
    print(f"✅ Loaded {len(degraded_images)} pairs")
    return degraded_images, clean_images

def create_comparison_grid(degraded_images, clean_images, output_dir, samples_per_page=16):
    """
    Create comparison grid with degraded (top) and clean (bottom) for each sample.
    
    Args:
        degraded_images: List of degraded images
        clean_images: List of clean images
        output_dir: Directory to save visualizations
        samples_per_page: Number of samples per grid page (default: 16 = 4x4 grid)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = len(degraded_images)
    num_pages = (total_samples + samples_per_page - 1) // samples_per_page
    
    print(f"\n📊 Creating {num_pages} visualization pages ({samples_per_page} samples/page)")
    
    for page_idx in range(num_pages):
        start_idx = page_idx * samples_per_page
        end_idx = min(start_idx + samples_per_page, total_samples)
        page_samples = end_idx - start_idx
        
        # Calculate grid dimensions (2 rows per sample: degraded + clean)
        cols = int(np.sqrt(samples_per_page))
        rows = (samples_per_page + cols - 1) // cols
        total_rows = rows * 2  # 2 rows per sample (degraded + clean)
        
        # Create figure
        fig = plt.figure(figsize=(cols * 4, total_rows * 1.5))
        
        for i in range(page_samples):
            sample_idx = start_idx + i
            degraded = degraded_images[sample_idx]
            clean = clean_images[sample_idx]
            
            # Calculate position in grid
            col = i % cols
            row = (i // cols) * 2  # Each sample takes 2 rows
            
            # Degraded image (top)
            ax1 = plt.subplot(total_rows, cols, row * cols + col + 1)
            ax1.imshow(degraded.squeeze(), cmap='gray', vmin=0, vmax=1)
            ax1.set_title(f'#{sample_idx:03d} Degraded', fontsize=8)
            ax1.axis('off')
            
            # Clean image (bottom)
            ax2 = plt.subplot(total_rows, cols, (row + 1) * cols + col + 1)
            ax2.imshow(clean.squeeze(), cmap='gray', vmin=0, vmax=1)
            ax2.set_title(f'#{sample_idx:03d} Clean', fontsize=8)
            ax2.axis('off')
        
        plt.tight_layout()
        
        # Save page
        output_path = output_dir / f'dibco_pairs_page_{page_idx+1:02d}_of_{num_pages:02d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Page {page_idx+1}/{num_pages}: Samples {start_idx}-{end_idx-1} → {output_path.name}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    return num_pages

def create_single_column_view(degraded_images, clean_images, output_dir, samples_per_page=8):
    """
    Alternative view: Single column with degraded-clean pairs stacked vertically.
    Better for detailed inspection of individual pairs.
    """
    output_dir = Path(output_dir) / "single_column"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = len(degraded_images)
    num_pages = (total_samples + samples_per_page - 1) // samples_per_page
    
    print(f"\n📊 Creating single-column view: {num_pages} pages ({samples_per_page} pairs/page)")
    
    for page_idx in range(num_pages):
        start_idx = page_idx * samples_per_page
        end_idx = min(start_idx + samples_per_page, total_samples)
        page_samples = end_idx - start_idx
        
        # Create figure (1 column, 2 rows per sample)
        fig, axes = plt.subplots(page_samples * 2, 1, figsize=(12, page_samples * 3))
        if page_samples == 1:
            axes = [axes]
        
        for i in range(page_samples):
            sample_idx = start_idx + i
            degraded = degraded_images[sample_idx]
            clean = clean_images[sample_idx]
            
            # Degraded
            ax_deg = axes[i * 2]
            ax_deg.imshow(degraded.squeeze(), cmap='gray', vmin=0, vmax=1)
            ax_deg.set_title(f'Sample #{sample_idx:03d} - Degraded (mean: {degraded.mean():.3f})', fontsize=10)
            ax_deg.axis('off')
            
            # Clean
            ax_clean = axes[i * 2 + 1]
            ax_clean.imshow(clean.squeeze(), cmap='gray', vmin=0, vmax=1)
            ax_clean.set_title(f'Sample #{sample_idx:03d} - Clean (mean: {clean.mean():.3f})', fontsize=10)
            ax_clean.axis('off')
        
        plt.tight_layout()
        
        # Save page
        output_path = output_dir / f'dibco_pairs_detailed_{page_idx+1:02d}_of_{num_pages:02d}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Detailed page {page_idx+1}/{num_pages}: Samples {start_idx}-{end_idx-1}")
    
    return num_pages

def main():
    parser = argparse.ArgumentParser(description='Visualize all DIBCO degraded-clean pairs')
    parser.add_argument('--tfrecord_path', type=str, 
                       default='dual_modal_gan/data/dibco_tiled_no_palm.tfrecord',
                       help='Path to TFRecord file')
    parser.add_argument('--output_dir', type=str,
                       default='dual_modal_gan/outputs/dibco_pairs_verification',
                       help='Output directory for visualizations')
    parser.add_argument('--samples_per_page', type=int, default=16,
                       help='Number of samples per grid page (default: 16 = 4x4)')
    parser.add_argument('--skip_detailed', action='store_true',
                       help='Skip single-column detailed view (faster)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 DIBCO Dataset Pair Verification")
    print("=" * 80)
    
    # Load all samples
    degraded_images, clean_images = load_all_samples(args.tfrecord_path)
    
    # Create grid view
    print("\n" + "=" * 80)
    print("📊 GRID VIEW (Overview)")
    print("=" * 80)
    num_grid_pages = create_comparison_grid(
        degraded_images, clean_images, 
        args.output_dir, 
        args.samples_per_page
    )
    
    # Create detailed view (unless skipped)
    if not args.skip_detailed:
        print("\n" + "=" * 80)
        print("📋 DETAILED VIEW (Single Column)")
        print("=" * 80)
        num_detail_pages = create_single_column_view(
            degraded_images, clean_images,
            args.output_dir,
            samples_per_page=8
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Grid view: {num_grid_pages} pages ({args.samples_per_page} samples/page)")
    if not args.skip_detailed:
        print(f"📋 Detailed view: {num_detail_pages} pages (8 pairs/page)")
    print(f"🔢 Total samples verified: {len(degraded_images)}")
    print("\n💡 Review the images to verify all degraded-clean pairs are correctly matched")
    print("   Look for:")
    print("   - Degraded images should have noise/degradation")
    print("   - Clean images should be binary (black text on white background)")
    print("   - Pairs should show same content (degraded version vs clean version)")
    print("=" * 80)

if __name__ == '__main__':
    main()
