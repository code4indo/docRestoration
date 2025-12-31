#!/usr/bin/env python3
"""
Visualize HTR Weaknesses on Restored Images

Generate visual examples showing:
1. Degraded → Restored images
2. HTR predictions on both
3. Specific error patterns (n/m confusion, punctuation, space detection)
4. Highlighted problem areas
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from pathlib import Path
import editdistance
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data():
    """Load predictions and weakness analysis"""
    # Load predictions
    with open('dual_modal_gan/analysis/test_predictions_restored.json', 'r') as f:
        pred_data = json.load(f)
    
    # Load weaknesses
    with open('dual_modal_gan/analysis/htr_weaknesses_restored.json', 'r') as f:
        weaknesses = json.load(f)
    
    # Load character errors
    with open('dual_modal_gan/analysis/character_errors_restored.json', 'r') as f:
        errors_data = json.load(f)
    
    return pred_data['predictions'], weaknesses, errors_data['errors']

def load_images_from_tfrecord(sample_idx, tfrecord_path='dual_modal_gan/data/dataset_gan.tfrecord'):
    """Load degraded and clean images for a sample"""
    def _parse_tfrecord_fn(example_proto):
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
        degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
        degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
        degraded_image = tf.reshape(degraded_image, degraded_image_shape)
        degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
        degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])
        
        # Clean image (for reference)
        clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
        clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
        clean_image = tf.reshape(clean_image, clean_image_shape)
        clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
        clean_image = tf.ensure_shape(clean_image, [1024, 128, 1])
        
        return degraded_image, clean_image
    
    # Load dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    # Skip to test set
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_start = train_size + val_size
    
    # Get specific sample
    dataset = dataset.skip(test_start + sample_idx).take(1)
    dataset = dataset.map(_parse_tfrecord_fn)
    
    for degraded, clean in dataset:
        return degraded.numpy(), clean.numpy()
    
    return None, None

def restore_image_with_generator(degraded_img):
    """Restore image using generator"""
    from dual_modal_gan.src.models.generator_enhanced import unet_enhanced
    
    # Load generator
    generator = unet_enhanced(input_size=(1024, 128, 1))
    checkpoint = tf.train.Checkpoint(generator=generator)
    status = checkpoint.restore('dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/ckpt-96')
    status.expect_partial()
    
    # Restore
    degraded_tanh = degraded_img * 2.0 - 1.0
    degraded_batch = tf.expand_dims(degraded_tanh, 0)
    restored_batch = generator(degraded_batch, training=False)
    restored = (restored_batch[0] + 1.0) / 2.0
    
    return restored.numpy()

def align_and_annotate(gt_text, pred_text):
    """Align texts and identify specific error types"""
    # Simple character-by-character alignment
    errors = []
    
    # Use edit distance for alignment
    n, m = len(gt_text), len(pred_text)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_text[i-1] == pred_text[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrack
    alignment = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt_text[i-1] == pred_text[j-1]:
            alignment.append((gt_text[i-1], pred_text[j-1], 'correct'))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution - check for specific patterns
            gt_char = gt_text[i-1]
            pred_char = pred_text[j-1]
            
            error_type = 'substitution'
            if (gt_char, pred_char) in [('n', 'm'), ('m', 'n')]:
                error_type = 'n_m_confusion'
            elif gt_char in 'r' and pred_char in 'ean':
                error_type = 'r_confusion'
            elif gt_char in '.,;:' or pred_char in '.,;:':
                error_type = 'punctuation_error'
            
            alignment.append((gt_char, pred_char, error_type))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            alignment.append(('', pred_text[j-1], 'insertion'))
            j -= 1
        else:
            gt_char = gt_text[i-1]
            error_type = 'deletion'
            if gt_char == ' ':
                error_type = 'space_deletion'
            elif gt_char in '.,;:':
                error_type = 'punctuation_deletion'
            alignment.append((gt_char, '', error_type))
            i -= 1
    
    alignment.reverse()
    return alignment

def create_weakness_visualization(sample_idx, predictions, output_path):
    """Create visualization showing degraded → restored with error highlighting"""
    
    sample = predictions[sample_idx]
    gt_text = sample['gt_text']
    pred_degraded = sample['pred_text_degraded']
    pred_restored = sample['pred_text_restored']
    
    # Load images
    degraded_img, clean_img = load_images_from_tfrecord(sample_idx)
    if degraded_img is None:
        print(f"⚠️  Could not load images for sample {sample_idx}")
        return False
    
    # Restore image
    print(f"  Restoring image for sample {sample_idx}...")
    restored_img = restore_image_with_generator(degraded_img)
    
    # Transpose for display
    degraded_display = degraded_img.squeeze().T
    restored_display = restored_img.squeeze().T
    clean_display = clean_img.squeeze().T
    
    # Align texts
    alignment_degraded = align_and_annotate(gt_text, pred_degraded)
    alignment_restored = align_and_annotate(gt_text, pred_restored)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 2, height_ratios=[2, 2, 1, 1], hspace=0.3, wspace=0.2)
    
    # 1. Degraded Image
    ax_deg = fig.add_subplot(gs[0, 0])
    ax_deg.imshow(degraded_display, cmap='gray', aspect='auto')
    ax_deg.set_title('Degraded Image', fontsize=14, fontweight='bold')
    ax_deg.axis('off')
    
    # 2. Restored Image
    ax_res = fig.add_subplot(gs[0, 1])
    ax_res.imshow(restored_display, cmap='gray', aspect='auto')
    ax_res.set_title('Restored Image', fontsize=14, fontweight='bold')
    ax_res.axis('off')
    
    # 3. GT (with error highlighting from RESTORED)
    ax_gt = fig.add_subplot(gs[2, :])
    ax_gt.axis('off')
    ax_gt.set_xlim(0, 1)
    ax_gt.set_ylim(0, 1)
    ax_gt.text(0.01, 0.9, 'Ground Truth:', fontsize=11, fontweight='bold', va='top')
    
    x_pos = 0.02
    y_pos = 0.5
    char_width = 0.008
    
    # Highlight errors from RESTORED prediction
    for gt_char, pred_char, err_type in alignment_restored:
        if gt_char:
            # Color based on error type
            color = 'white'
            if err_type == 'n_m_confusion':
                color = '#FFB6B6'  # Light red
            elif err_type == 'r_confusion':
                color = '#FFD6A5'  # Light orange
            elif err_type in ['punctuation_error', 'punctuation_deletion']:
                color = '#A5D6FF'  # Light blue
            elif err_type == 'space_deletion':
                color = '#FFFA65'  # Yellow
            elif err_type in ['deletion', 'substitution']:
                color = '#FFE5E5'
            
            if color != 'white':
                rect = mpatches.Rectangle((x_pos-0.001, y_pos-0.15), char_width+0.001, 0.3,
                                         facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                ax_gt.add_patch(rect)
            
            ax_gt.text(x_pos, y_pos, gt_char, fontsize=9, ha='left', va='center',
                      family='monospace', fontweight='bold' if color != 'white' else 'normal')
            x_pos += char_width
    
    # 4. Prediction on RESTORED (with same highlighting)
    ax_pred = fig.add_subplot(gs[3, :])
    ax_pred.axis('off')
    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(0, 1)
    ax_pred.text(0.01, 0.9, 'HTR Prediction (Restored):', fontsize=11, fontweight='bold', va='top')
    
    x_pos = 0.02
    
    for gt_char, pred_char, err_type in alignment_restored:
        if pred_char:
            color = 'white'
            if err_type == 'n_m_confusion':
                color = '#FFB6B6'
            elif err_type == 'r_confusion':
                color = '#FFD6A5'
            elif err_type in ['punctuation_error', 'punctuation_deletion']:
                color = '#A5D6FF'
            elif err_type == 'space_deletion':
                color = '#FFFA65'
            elif err_type in ['deletion', 'substitution', 'insertion']:
                color = '#FFE5E5'
            
            if color != 'white':
                rect = mpatches.Rectangle((x_pos-0.001, y_pos-0.15), char_width+0.001, 0.3,
                                         facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                ax_pred.add_patch(rect)
            
            ax_pred.text(x_pos, y_pos, pred_char, fontsize=9, ha='left', va='center',
                        family='monospace', fontweight='bold' if color != 'white' else 'normal')
            x_pos += char_width
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#FFB6B6', alpha=0.6, label='n/m Confusion'),
        mpatches.Patch(facecolor='#FFD6A5', alpha=0.6, label='r Confusion'),
        mpatches.Patch(facecolor='#A5D6FF', alpha=0.6, label='Punctuation Error'),
        mpatches.Patch(facecolor='#FFFA65', alpha=0.6, label='Space Deletion'),
        mpatches.Patch(facecolor='#FFE5E5', alpha=0.6, label='Other Error')
    ]
    ax_pred.legend(handles=legend_elements, loc='lower right', ncol=3, fontsize=8)
    
    # Error counts
    error_counts = {}
    for _, _, err_type in alignment_restored:
        if err_type != 'correct':
            error_counts[err_type] = error_counts.get(err_type, 0) + 1
    
    stats_text = f"Sample {sample_idx} | GT len: {len(gt_text)} | Pred len: {len(pred_restored)}\n"
    stats_text += "Errors: " + ", ".join(f"{k}: {v}" for k, v in sorted(error_counts.items(), key=lambda x: -x[1])[:5])
    
    fig.text(0.5, 0.95, stats_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'HTR Weakness Visualization: Degraded → Restored (Sample {sample_idx})',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_path}")
    return True

def find_samples_with_specific_weaknesses(predictions, errors, weakness_type='n_m_confusion', n=3):
    """Find samples demonstrating specific weaknesses"""
    
    # Map errors to samples
    sample_weaknesses = {}
    for error in errors:
        sample_id = error['sample_id']
        if sample_id not in sample_weaknesses:
            sample_weaknesses[sample_id] = []
        sample_weaknesses[sample_id].append(error)
    
    # Find samples with target weakness
    candidates = []
    
    for sample_id, sample_errors in sample_weaknesses.items():
        count = 0
        for err in sample_errors:
            if weakness_type == 'n_m_confusion':
                if (err['gt_char'] in 'nm' and err['pred_char'] in 'nm' and 
                    err['gt_char'] != err['pred_char'] and err['error_type'] == 'substitution'):
                    count += 1
            elif weakness_type == 'punctuation':
                if err.get('is_punctuation', False):
                    count += 1
            elif weakness_type == 'space_deletion':
                if err['error_type'] == 'deletion' and err['gt_char'] == ' ':
                    count += 1
        
        if count > 0:
            # Also check prediction quality
            pred = predictions[sample_id]
            if len(pred['pred_text_restored']) > 5:  # Has some prediction
                candidates.append((sample_id, count))
    
    # Sort by count and return top n
    candidates.sort(key=lambda x: -x[1])
    return [sample_id for sample_id, _ in candidates[:n]]

def main():
    """Generate weakness visualizations"""
    print("="*80)
    print("GENERATING HTR WEAKNESS VISUALIZATIONS (RESTORED IMAGES)")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    predictions, weaknesses, errors = load_data()
    print(f"Loaded {len(predictions)} predictions, {len(errors)} errors")
    
    # Create output directory
    output_dir = Path('dual_modal_gan/analysis/weakness_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find diverse examples
    print("\nFinding samples with specific weaknesses...")
    
    # 1. n/m confusion examples
    nm_samples = find_samples_with_specific_weaknesses(predictions, errors, 'n_m_confusion', n=2)
    print(f"  Found {len(nm_samples)} samples with n/m confusion")
    
    # 2. Punctuation error examples
    punct_samples = find_samples_with_specific_weaknesses(predictions, errors, 'punctuation', n=2)
    print(f"  Found {len(punct_samples)} samples with punctuation errors")
    
    # 3. Space deletion examples
    space_samples = find_samples_with_specific_weaknesses(predictions, errors, 'space_deletion', n=2)
    print(f"  Found {len(space_samples)} samples with space deletion")
    
    # Combine (avoid duplicates)
    selected_samples = list(set(nm_samples + punct_samples + space_samples))[:5]
    
    print(f"\nGenerating visualizations for {len(selected_samples)} samples...")
    
    for i, sample_idx in enumerate(selected_samples, 1):
        print(f"\n[{i}/{len(selected_samples)}] Processing sample {sample_idx}...")
        output_path = output_dir / f'weakness_example_{i:02d}_sample_{sample_idx:03d}.png'
        create_weakness_visualization(sample_idx, predictions, str(output_path))
    
    print("\n" + "="*80)
    print("✅ ALL WEAKNESS VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nGenerated {len(selected_samples)} visualizations in:")
    print(f"  {output_dir}/")
    print("\nShowing:")
    print("  - Degraded → Restored images")
    print("  - HTR predictions on restored")
    print("  - Color-coded error highlighting:")
    print("    * Light Red: n/m confusion")
    print("    * Light Orange: r confusion")
    print("    * Light Blue: Punctuation errors")
    print("    * Yellow: Space deletion")

if __name__ == '__main__':
    main()
