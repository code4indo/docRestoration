#!/usr/bin/env python3
"""
Extract and Visualize ALL Samples from Fine-tuning TFRecord

This script:
1. Reads TFRecord file (train/val/test)
2. Extracts ALL samples (degraded + clean pairs)
3. Saves each sample as individual image
4. Creates grid visualization showing all samples
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _parse_tfrecord_fn(example_proto):
    """Parse TFRecord example - same as training script"""
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
    degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])  # (H, W, C) → (W, H, C)
    
    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])  # (H, W, C) → (W, H, C)
    
    # Deserialize label
    label_shape = tf.cast(example['label_shape'], tf.int32)
    label = tf.io.decode_raw(example['label_raw'], tf.int64)
    label = tf.reshape(label, label_shape)
    
    return degraded_image, clean_image, label

def extract_all_samples(tfrecord_path, output_dir, split_name='train'):
    """Extract all samples from TFRecord and save as individual images"""
    
    print(f"\n{'='*80}")
    print(f"EXTRACTING SAMPLES: {split_name.upper()}")
    print(f"{'='*80}")
    print(f"Source: {tfrecord_path}")
    
    # Check if TFRecord exists
    if not os.path.exists(tfrecord_path):
        print(f"❌ TFRecord not found: {tfrecord_path}")
        print(f"   Please create dataset first:")
        print(f"   poetry run python scripts/create_finetuning_strips.py")
        return None
    
    # Create output directories
    output_dir = Path(output_dir)
    degraded_dir = output_dir / split_name / 'degraded'
    clean_dir = output_dir / split_name / 'clean'
    pair_dir = output_dir / split_name / 'pairs'
    
    degraded_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Count total samples
    total_samples = sum(1 for _ in dataset)
    print(f"✅ Found {total_samples} samples in {split_name} set")
    
    # Reset dataset iterator
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Extract all samples
    samples_data = []
    
    print(f"\n📦 Extracting samples...")
    for idx, (degraded, clean, label) in enumerate(tqdm(dataset, total=total_samples, desc=f"Processing {split_name}")):
        # Convert to numpy
        deg_np = degraded.numpy().squeeze()  # (W, H, 1) → (W, H)
        clean_np = clean.numpy().squeeze()
        
        # Transpose to (H, W) for proper image display
        deg_np = deg_np.T  # (W, H) → (H, W)
        clean_np = clean_np.T
        
        # Convert to uint8 [0, 255]
        deg_uint8 = (deg_np * 255).astype(np.uint8)
        clean_uint8 = (clean_np * 255).astype(np.uint8)
        
        # Save individual images
        deg_path = degraded_dir / f"sample_{idx:04d}_deg.png"
        clean_path = clean_dir / f"sample_{idx:04d}_clean.png"
        
        cv2.imwrite(str(deg_path), deg_uint8)
        cv2.imwrite(str(clean_path), clean_uint8)
        
        # Create pair visualization (side-by-side)
        separator = np.ones((128, 10), dtype=np.uint8) * 255
        pair = np.hstack([clean_uint8, separator, deg_uint8])
        pair_path = pair_dir / f"sample_{idx:04d}_pair.png"
        cv2.imwrite(str(pair_path), pair)
        
        # Store for grid visualization
        samples_data.append({
            'idx': idx,
            'degraded': deg_uint8,
            'clean': clean_uint8,
            'pair': pair
        })
    
    print(f"\n✅ Extracted {len(samples_data)} samples to:")
    print(f"   Degraded:  {degraded_dir}")
    print(f"   Clean:     {clean_dir}")
    print(f"   Pairs:     {pair_dir}")
    
    return samples_data

def create_grid_visualization(samples_data, output_path, split_name='train', max_display=50):
    """Create grid visualization showing multiple samples"""
    
    total = len(samples_data)
    display_count = min(total, max_display)
    
    print(f"\n📊 Creating grid visualization...")
    print(f"   Displaying: {display_count}/{total} samples")
    
    # Calculate grid size (roughly square)
    cols = int(np.ceil(np.sqrt(display_count)))
    rows = int(np.ceil(display_count / cols))
    
    # Create figure
    fig = plt.figure(figsize=(cols * 3, rows * 2))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.1)
    
    # Sample evenly if we have too many
    if total > max_display:
        step = total / max_display
        indices = [int(i * step) for i in range(max_display)]
    else:
        indices = range(total)
    
    # Plot samples
    for plot_idx, sample_idx in enumerate(indices):
        sample = samples_data[sample_idx]
        
        row = plot_idx // cols
        col = plot_idx % cols
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(sample['pair'], cmap='gray')
        ax.set_title(f"#{sample['idx']}", fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'{split_name.upper()} Set: {total} samples (showing {display_count})\n'
                 f'Left: Clean (GT) | Right: Degraded (Input)', 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

def create_summary_html(output_dir, train_count, val_count, test_count):
    """Create HTML summary with all samples"""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fine-tuning Dataset - All Samples</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .sample-card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .sample-card:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .sample-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .sample-label {{
            padding: 10px;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }}
        .tab {{
            padding: 10px 20px;
            background: #ecf0f1;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }}
        .tab.active {{
            background: #3498db;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
    <script>
        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            document.querySelector('[onclick="showTab(\\''+tabName+'\\')"]').classList.add('active');
        }}
    </script>
</head>
<body>
    <h1>🎯 Fine-tuning Dataset - All {train_count + val_count + test_count} Samples</h1>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value">{train_count + val_count + test_count}</div>
            <div class="stat-label">Total Samples</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{train_count}</div>
            <div class="stat-label">Train (70%)</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{val_count}</div>
            <div class="stat-label">Validation (15%)</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{test_count}</div>
            <div class="stat-label">Test (15%)</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Browse by Split</h2>
        <div class="tabs">
            <button class="tab active" onclick="showTab('train')">Train ({train_count})</button>
            <button class="tab" onclick="showTab('val')">Validation ({val_count})</button>
            <button class="tab" onclick="showTab('test')">Test ({test_count})</button>
        </div>
"""
    
    # Add content for each split
    for split_name, count in [('train', train_count), ('val', val_count), ('test', test_count)]:
        active = 'active' if split_name == 'train' else ''
        html_content += f"""
        <div id="{split_name}" class="tab-content {active}">
            <h3>{split_name.capitalize()} Set - {count} Samples</h3>
            <div class="grid-container">
"""
        # Add all samples
        pair_dir = Path(output_dir) / split_name / 'pairs'
        if pair_dir.exists():
            for img_path in sorted(pair_dir.glob("*.png")):
                sample_num = img_path.stem.split('_')[1]
                rel_path = f"{split_name}/pairs/{img_path.name}"
                html_content += f"""
                <div class="sample-card">
                    <img src="{rel_path}" alt="Sample {sample_num}">
                    <div class="sample-label">Sample #{sample_num}</div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>📥 Download Individual Folders</h2>
        <p>Semua samples juga tersimpan dalam folder terpisah:</p>
        <ul>
            <li><code>train/degraded/</code> - {train_count} degraded images</li>
            <li><code>train/clean/</code> - {train_count} clean (GT) images</li>
            <li><code>train/pairs/</code> - {train_count} side-by-side comparisons</li>
            <li>... (same for val/ and test/)</li>
        </ul>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; padding: 20px; color: #7f8c8d; border-top: 1px solid #ddd;">
        <p>Generated: October 30, 2025</p>
        <p>Total: {train_count + val_count + test_count} samples from 31 strips × 5 augmentations</p>
    </footer>
</body>
</html>
""".replace('{train_count}', str(train_count)).replace('{val_count}', str(val_count)).replace('{test_count}', str(test_count))
    
    html_path = Path(output_dir) / 'all_samples.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML summary created: {html_path}")
    return html_path

def main(args):
    """Main extraction function"""
    
    print("="*80)
    print("EXTRACT ALL FINE-TUNING SAMPLES FROM TFRECORD")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    all_samples = {}
    counts = {}
    
    for split in ['train', 'val', 'test']:
        tfrecord_path = Path(args.tfrecord_dir) / f"finetuning_{split}.tfrecord"
        
        if tfrecord_path.exists():
            samples = extract_all_samples(str(tfrecord_path), output_dir, split)
            if samples:
                all_samples[split] = samples
                counts[split] = len(samples)
                
                # Create grid visualization
                grid_path = output_dir / f"{split}_grid.png"
                create_grid_visualization(samples, grid_path, split, max_display=args.max_grid)
        else:
            print(f"\n⚠️  Skipping {split}: TFRecord not found")
            counts[split] = 0
    
    # Create summary HTML
    if all_samples:
        html_path = create_summary_html(
            output_dir,
            counts.get('train', 0),
            counts.get('val', 0),
            counts.get('test', 0)
        )
        
        print(f"\n{'='*80}")
        print("✅ EXTRACTION COMPLETE!")
        print(f"{'='*80}")
        print(f"\n📁 All samples saved to: {output_dir}")
        print(f"\n🌐 View in browser:")
        print(f"   file://{html_path.absolute()}")
        print(f"\n📊 Or open with:")
        print(f"   firefox {html_path}")
        print(f"   google-chrome {html_path}")
        
    else:
        print(f"\n❌ No TFRecord files found in: {args.tfrecord_dir}")
        print(f"   Please create dataset first:")
        print(f"   ./scripts/launch_finetuning.sh")
        print(f"   OR")
        print(f"   poetry run python scripts/create_finetuning_strips.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract all samples from fine-tuning TFRecord')
    parser.add_argument('--tfrecord_dir', type=str, default='dual_modal_gan/data/finetuning',
                        help='Directory containing TFRecord files')
    parser.add_argument('--output_dir', type=str, default='visualization/finetuning/all_samples',
                        help='Output directory for extracted samples')
    parser.add_argument('--max_grid', type=int, default=50,
                        help='Maximum samples to show in grid visualization (default: 50)')
    
    args = parser.parse_args()
    main(args)
