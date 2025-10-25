#!/usr/bin/env python3
"""
Analyze stroke width distribution in training dataset.
Determines if dataset has sufficient thin stroke representation for fine-tuning.

Usage:
    poetry run python scripts/check_dataset_stroke_distribution.py

Output:
    - JSON summary with stroke width statistics
    - Histogram visualization
    - Decision recommendation (fine-tune vs augment)
"""

import tensorflow as tf
import numpy as np
import cv2
import json
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
TFRECORD_PATH = "dual_modal_gan/data/dataset_gan.tfrecord"
NUM_SAMPLES = 500  # Sample 500 images to analyze
OUTPUT_DIR = Path("metrics/dataset_analysis")
OUTPUT_JSON = OUTPUT_DIR / "stroke_distribution.json"
OUTPUT_HIST = OUTPUT_DIR / "stroke_width_histogram.png"

# Stroke width thresholds (pixels)
THIN_THRESHOLD = 3
MEDIUM_THRESHOLD = 6


def parse_tfrecord(example_proto):
    """Parse TFRecord example using the actual schema."""
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
    
    # Deserialize clean image
    clean_image_shape = tf.cast(example['clean_image_shape'], tf.int32)
    clean_image = tf.io.decode_raw(example['clean_image_raw'], tf.float32)
    clean_image = tf.reshape(clean_image, clean_image_shape)
    # Transpose from (H, W, C) to (W, H, C)
    clean_image = tf.transpose(clean_image, perm=[1, 0, 2])
    
    return clean_image


def decode_image(image_tensor):
    """Convert image tensor to numpy array."""
    # Already in range [0, 1], convert to [0, 255] for processing
    image = image_tensor.numpy()
    image = (image * 255.0).astype(np.uint8)
    return image


def measure_stroke_widths(binary_image):
    """
    Measure stroke widths using distance transform.
    Returns array of stroke widths for all text pixels.
    """
    # Ensure binary (0 or 255)
    binary = (binary_image > 127).astype(np.uint8) * 255
    
    # Distance transform on text pixels
    # For white text on black background, invert first
    if np.mean(binary) > 127:
        binary = 255 - binary
    
    # Distance transform gives distance to nearest background pixel
    dist_transform = distance_transform_edt(binary > 0)
    
    # Stroke width = 2 * distance (radius to width)
    stroke_widths = dist_transform[binary > 0] * 2
    
    return stroke_widths


def analyze_dataset(tfrecord_path, num_samples=500):
    """
    Analyze stroke width distribution in dataset.
    
    Returns:
        dict with statistics and per-image data
    """
    print(f"Analyzing stroke distribution in {tfrecord_path}")
    print(f"Sampling {num_samples} images...")
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.take(num_samples)
    
    all_stroke_widths = []
    per_image_stats = []
    
    for i, record in enumerate(tqdm(dataset, total=num_samples, desc="Analyzing images")):
        # Get clean image
        clean_img = decode_image(record)
        
        # Measure stroke widths
        stroke_widths = measure_stroke_widths(clean_img)
        
        if len(stroke_widths) == 0:
            continue
        
        all_stroke_widths.extend(stroke_widths)
        
        # Per-image statistics
        thin_pixels = np.sum(stroke_widths < THIN_THRESHOLD)
        medium_pixels = np.sum((stroke_widths >= THIN_THRESHOLD) & (stroke_widths < MEDIUM_THRESHOLD))
        thick_pixels = np.sum(stroke_widths >= MEDIUM_THRESHOLD)
        total_pixels = len(stroke_widths)
        
        per_image_stats.append({
            'image_idx': i,
            'mean_width': float(np.mean(stroke_widths)),
            'median_width': float(np.median(stroke_widths)),
            'min_width': float(np.min(stroke_widths)),
            'max_width': float(np.max(stroke_widths)),
            'thin_ratio': thin_pixels / total_pixels,
            'medium_ratio': medium_pixels / total_pixels,
            'thick_ratio': thick_pixels / total_pixels,
            'total_pixels': int(total_pixels)
        })
    
    # Aggregate statistics
    all_stroke_widths = np.array(all_stroke_widths)
    thin_pixels = np.sum(all_stroke_widths < THIN_THRESHOLD)
    medium_pixels = np.sum((all_stroke_widths >= THIN_THRESHOLD) & (all_stroke_widths < MEDIUM_THRESHOLD))
    thick_pixels = np.sum(all_stroke_widths >= MEDIUM_THRESHOLD)
    total_pixels = len(all_stroke_widths)
    
    summary = {
        'dataset_path': str(tfrecord_path),
        'num_samples_analyzed': num_samples,
        'total_text_pixels': int(total_pixels),
        'stroke_width_statistics': {
            'mean': float(np.mean(all_stroke_widths)),
            'median': float(np.median(all_stroke_widths)),
            'std': float(np.std(all_stroke_widths)),
            'min': float(np.min(all_stroke_widths)),
            'max': float(np.max(all_stroke_widths)),
            'percentiles': {
                'p10': float(np.percentile(all_stroke_widths, 10)),
                'p25': float(np.percentile(all_stroke_widths, 25)),
                'p50': float(np.percentile(all_stroke_widths, 50)),
                'p75': float(np.percentile(all_stroke_widths, 75)),
                'p90': float(np.percentile(all_stroke_widths, 90)),
            }
        },
        'stroke_distribution': {
            'thin_strokes': {
                'threshold': f'< {THIN_THRESHOLD}px',
                'pixel_count': int(thin_pixels),
                'percentage': float(thin_pixels / total_pixels * 100)
            },
            'medium_strokes': {
                'threshold': f'{THIN_THRESHOLD}-{MEDIUM_THRESHOLD}px',
                'pixel_count': int(medium_pixels),
                'percentage': float(medium_pixels / total_pixels * 100)
            },
            'thick_strokes': {
                'threshold': f'> {MEDIUM_THRESHOLD}px',
                'pixel_count': int(thick_pixels),
                'percentage': float(thick_pixels / total_pixels * 100)
            }
        },
        'per_image_stats': per_image_stats
    }
    
    # Decision recommendation
    thin_percentage = summary['stroke_distribution']['thin_strokes']['percentage']
    if thin_percentage >= 20:
        recommendation = "PHASE 1: Fine-tune with loss rebalancing (dataset has sufficient thin strokes)"
    elif thin_percentage >= 10:
        recommendation = "PHASE 2: Augment dataset with erosion/fading (moderate thin stroke representation)"
    else:
        recommendation = "PHASE 2-3: Strong augmentation or real data collection needed (low thin stroke representation)"
    
    summary['recommendation'] = recommendation
    summary['decision_criteria'] = {
        'thin_stroke_percentage': float(thin_percentage),
        'threshold_phase1': 20.0,
        'threshold_phase2': 10.0,
        'action': 'FINE_TUNE' if thin_percentage >= 20 else 'AUGMENT'
    }
    
    return summary, all_stroke_widths


def plot_histogram(stroke_widths, output_path):
    """Plot stroke width histogram."""
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.hist(stroke_widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # Add threshold lines
    plt.axvline(THIN_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Thin threshold ({THIN_THRESHOLD}px)')
    plt.axvline(MEDIUM_THRESHOLD, color='orange', linestyle='--', linewidth=2, label=f'Medium threshold ({MEDIUM_THRESHOLD}px)')
    
    plt.xlabel('Stroke Width (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Stroke Width Distribution in Training Dataset', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Mean: {np.mean(stroke_widths):.2f}px\n"
    stats_text += f"Median: {np.median(stroke_widths):.2f}px\n"
    stats_text += f"Std: {np.std(stroke_widths):.2f}px"
    plt.text(0.98, 0.97, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {output_path}")


def main():
    """Main execution."""
    # Check if TFRecord exists
    if not Path(TFRECORD_PATH).exists():
        print(f"ERROR: TFRecord not found at {TFRECORD_PATH}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Analyze dataset
    print("\n" + "="*80)
    print("DATASET STROKE DISTRIBUTION ANALYSIS")
    print("="*80 + "\n")
    
    summary, stroke_widths = analyze_dataset(TFRECORD_PATH, NUM_SAMPLES)
    
    # Save JSON summary
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {OUTPUT_JSON}")
    
    # Plot histogram
    plot_histogram(stroke_widths, OUTPUT_HIST)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print(f"\nDataset: {TFRECORD_PATH}")
    print(f"Samples analyzed: {summary['num_samples_analyzed']}")
    print(f"Total text pixels: {summary['total_text_pixels']:,}")
    
    print("\n--- Stroke Width Statistics ---")
    stats = summary['stroke_width_statistics']
    print(f"Mean:   {stats['mean']:.2f}px")
    print(f"Median: {stats['median']:.2f}px")
    print(f"Std:    {stats['std']:.2f}px")
    print(f"Range:  {stats['min']:.2f}px - {stats['max']:.2f}px")
    
    print("\n--- Stroke Distribution ---")
    dist = summary['stroke_distribution']
    print(f"Thin strokes   ({dist['thin_strokes']['threshold']}):   {dist['thin_strokes']['percentage']:5.2f}%  ({dist['thin_strokes']['pixel_count']:,} pixels)")
    print(f"Medium strokes ({dist['medium_strokes']['threshold']}): {dist['medium_strokes']['percentage']:5.2f}%  ({dist['medium_strokes']['pixel_count']:,} pixels)")
    print(f"Thick strokes  ({dist['thick_strokes']['threshold']}):  {dist['thick_strokes']['percentage']:5.2f}%  ({dist['thick_strokes']['pixel_count']:,} pixels)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"\n{summary['recommendation']}")
    print(f"\nAction: {summary['decision_criteria']['action']}")
    print(f"Thin stroke percentage: {summary['decision_criteria']['thin_stroke_percentage']:.2f}%")
    print(f"Phase 1 threshold: {summary['decision_criteria']['threshold_phase1']:.2f}%")
    print(f"Phase 2 threshold: {summary['decision_criteria']['threshold_phase2']:.2f}%")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
