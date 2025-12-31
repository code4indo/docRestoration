#!/usr/bin/env python3
"""
Generate Visual Examples of Character-Level Errors

Create publication-quality visualizations showing:
1. Degraded images with poor HTR predictions
2. GT vs Prediction alignment with error highlighting
3. Character-level error annotation
4. Multiple examples showing different patterns

Output: High-quality figures for thesis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from pathlib import Path
import sys

# Add project root
sys.path.append('/home/lambda_one/tesis/GAN-HTR-ORI/docRestoration')

def load_predictions_and_errors():
    """Load predictions and character errors"""
    # Load predictions
    with open('dual_modal_gan/analysis/test_predictions.json', 'r') as f:
        pred_data = json.load(f)
    
    # Load character errors
    with open('dual_modal_gan/analysis/character_errors_full.json', 'r') as f:
        error_data = json.load(f)
    
    return pred_data['predictions'], error_data['errors']

def calculate_sample_cer(gt_text, pred_text):
    """Calculate CER for a sample"""
    import editdistance
    if len(gt_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0
    return editdistance.eval(gt_text, pred_text) / len(gt_text)

def align_and_highlight(gt_text, pred_text):
    """
    Align GT and pred texts, return colored representation
    
    Returns:
        List of (char, color, label) tuples
        - color: 'green' (correct), 'red' (substitution/deletion), 'yellow' (insertion)
    """
    import editdistance
    
    # Simple alignment using edit distance
    n, m = len(gt_text), len(pred_text)
    
    # DP matrix
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
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    # Backtrack
    alignment = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt_text[i-1] == pred_text[j-1]:
            alignment.append((gt_text[i-1], 'green', 'correct', pred_text[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((gt_text[i-1], 'red', 'substitution', pred_text[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append(('', 'yellow', 'insertion', pred_text[j-1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append((gt_text[i-1], 'orange', 'deletion', ''))
            i -= 1
    
    alignment.reverse()
    return alignment

def load_image_from_tfrecord(sample_idx, tfrecord_path='dual_modal_gan/data/dataset_gan.tfrecord'):
    """
    Load degraded image for a specific sample index from TFRecord
    """
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
        
        # Deserialize degraded image
        degraded_image_shape = tf.cast(example['degraded_image_shape'], tf.int32)
        degraded_image = tf.io.decode_raw(example['degraded_image_raw'], tf.float32)
        degraded_image = tf.reshape(degraded_image, degraded_image_shape)
        degraded_image = tf.transpose(degraded_image, perm=[1, 0, 2])
        degraded_image = tf.ensure_shape(degraded_image, [1024, 128, 1])
        
        return degraded_image
    
    # Load dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    total_size = sum(1 for _ in dataset)
    
    # Skip to test set (70% train, 15% val)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_start = train_size + val_size
    
    # Get specific sample
    dataset = dataset.skip(test_start + sample_idx).take(1)
    dataset = dataset.map(_parse_tfrecord_fn)
    
    for img in dataset:
        return img.numpy()
    
    return None

def create_visual_example(sample_idx, predictions, output_path):
    """
    Create visual example for one sample
    
    Shows:
    - Degraded image
    - GT vs Prediction with color coding
    - Error statistics
    """
    sample = predictions[sample_idx]
    gt_text = sample['gt_text']
    pred_text = sample['pred_text_degraded']
    
    # Calculate CER
    cer = calculate_sample_cer(gt_text, pred_text)
    
    # Load image
    img = load_image_from_tfrecord(sample_idx)
    if img is None:
        print(f"⚠️  Could not load image for sample {sample_idx}")
        return False
    
    # Transpose for display (1024, 128, 1) → (128, 1024)
    img_display = img.squeeze().T
    
    # Align texts
    alignment = align_and_highlight(gt_text, pred_text)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # 1. Degraded Image
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_display, cmap='gray', aspect='auto')
    ax_img.set_title(f'Sample {sample_idx}: Degraded Image (CER = {cer:.1%})', 
                     fontsize=14, fontweight='bold')
    ax_img.axis('off')
    
    # 2. Ground Truth (with color coding)
    ax_gt = fig.add_subplot(gs[1])
    ax_gt.axis('off')
    ax_gt.set_xlim(0, 1)
    ax_gt.set_ylim(0, 1)
    
    # Draw GT text with color coding
    x_pos = 0.02
    y_pos = 0.5
    char_width = 0.012
    
    ax_gt.text(0.01, 0.9, 'Ground Truth:', fontsize=12, fontweight='bold', va='top')
    
    for gt_char, color, error_type, pred_char in alignment:
        if gt_char:  # Not insertion
            # Background box
            if color != 'green':
                rect = mpatches.Rectangle((x_pos-0.002, y_pos-0.15), char_width+0.002, 0.3,
                                         facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5)
                ax_gt.add_patch(rect)
            
            # Character
            ax_gt.text(x_pos, y_pos, gt_char, fontsize=10, ha='left', va='center',
                      family='monospace', fontweight='bold' if color != 'green' else 'normal')
            x_pos += char_width
    
    # 3. Prediction (with color coding)
    ax_pred = fig.add_subplot(gs[2])
    ax_pred.axis('off')
    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(0, 1)
    
    x_pos = 0.02
    
    ax_pred.text(0.01, 0.9, 'HTR Prediction:', fontsize=12, fontweight='bold', va='top')
    
    for gt_char, color, error_type, pred_char in alignment:
        if pred_char:  # Not deletion
            # Background box
            if color != 'green':
                rect = mpatches.Rectangle((x_pos-0.002, y_pos-0.15), char_width+0.002, 0.3,
                                         facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5)
                ax_pred.add_patch(rect)
            
            # Character
            ax_pred.text(x_pos, y_pos, pred_char, fontsize=10, ha='left', va='center',
                        family='monospace', fontweight='bold' if color != 'green' else 'normal')
            x_pos += char_width
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.3, label='Correct'),
        mpatches.Patch(facecolor='red', alpha=0.3, label='Substitution'),
        mpatches.Patch(facecolor='orange', alpha=0.3, label='Deletion (missing in pred)'),
        mpatches.Patch(facecolor='yellow', alpha=0.3, label='Insertion (extra in pred)')
    ]
    ax_pred.legend(handles=legend_elements, loc='lower right', ncol=2, fontsize=9)
    
    # Statistics box
    error_counts = {
        'correct': sum(1 for _, c, _, _ in alignment if c == 'green'),
        'substitution': sum(1 for _, c, _, _ in alignment if c == 'red'),
        'deletion': sum(1 for _, c, _, _ in alignment if c == 'orange'),
        'insertion': sum(1 for _, c, _, _ in alignment if c == 'yellow')
    }
    
    stats_text = f"Errors: {len(alignment) - error_counts['correct']}/{len(alignment)} chars\n"
    stats_text += f"Deletions: {error_counts['deletion']}, "
    stats_text += f"Substitutions: {error_counts['substitution']}, "
    stats_text += f"Insertions: {error_counts['insertion']}"
    
    ax_img.text(0.02, 0.98, stats_text, transform=ax_img.transAxes,
               fontsize=10, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created visual example: {output_path}")
    return True

def select_diverse_examples(predictions, n=6):
    """
    Select diverse examples with different error patterns
    
    Criteria:
    - Range of CER values (low, medium, high)
    - Different text lengths
    - Mix of error types
    """
    # Calculate CER for all samples
    samples_with_cer = []
    for i, sample in enumerate(predictions):
        cer = calculate_sample_cer(sample['gt_text'], sample['pred_text_degraded'])
        # Only include samples with some predictions (not completely empty)
        if len(sample['pred_text_degraded']) > 5:
            samples_with_cer.append((i, cer, len(sample['gt_text'])))
    
    # Sort by CER
    samples_with_cer.sort(key=lambda x: x[1])
    
    # Select diverse samples
    selected = []
    
    # Different CER ranges
    total = len(samples_with_cer)
    if total >= n:
        # Low CER (best predictions)
        selected.append(samples_with_cer[min(5, total//10)])
        # Medium-low CER
        selected.append(samples_with_cer[total//4])
        # Medium CER
        selected.append(samples_with_cer[total//2])
        # Medium-high CER
        selected.append(samples_with_cer[3*total//4])
        # High CER
        selected.append(samples_with_cer[min(total-5, 9*total//10)])
        # Very high CER (worst)
        selected.append(samples_with_cer[-1])
    else:
        # Just take what we have
        step = max(1, total // n)
        selected = samples_with_cer[::step][:n]
    
    return [s[0] for s in selected[:n]]

def main():
    """Generate visual examples"""
    print("="*80)
    print("GENERATING VISUAL EXAMPLES OF CHARACTER-LEVEL ERRORS")
    print("="*80)
    
    # Create output directory
    output_dir = Path('dual_modal_gan/analysis/visual_examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading predictions...")
    predictions, errors = load_predictions_and_errors()
    print(f"Loaded {len(predictions)} samples")
    
    # Select diverse examples
    print("\nSelecting diverse examples...")
    selected_indices = select_diverse_examples(predictions, n=6)
    
    print(f"\nSelected {len(selected_indices)} samples:")
    for i, idx in enumerate(selected_indices, 1):
        sample = predictions[idx]
        cer = calculate_sample_cer(sample['gt_text'], sample['pred_text_degraded'])
        print(f"  {i}. Sample {idx}: CER={cer:.1%}, GT len={len(sample['gt_text'])}, Pred len={len(sample['pred_text_degraded'])}")
    
    # Generate visual examples
    print("\nGenerating visual examples...")
    for i, idx in enumerate(selected_indices, 1):
        output_path = output_dir / f'example_{i:02d}_sample_{idx:03d}.png'
        success = create_visual_example(idx, predictions, str(output_path))
        if not success:
            print(f"  ⚠️  Skipped sample {idx}")
    
    print("\n" + "="*80)
    print("✅ VISUAL EXAMPLES COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print(f"Generated {len(selected_indices)} examples (PNG + PDF)")
    print("\nFiles:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
