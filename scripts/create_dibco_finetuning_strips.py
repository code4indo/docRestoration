#!/usr/bin/env python3
"""
Create DIBCO Fine-tuning Dataset with ANRI-style Strip Method

Strategy (same as ANRI fine-tuning):
1. Slice full-page images into horizontal strips (1024x128)
2. Resize dengan ASPECT RATIO PRESERVATION + center padding
3. Use SEQUENTIAL split to avoid data leakage (70/15/15)
4. Apply augmentation (5x) untuk increase diversity
5. Generate TFRecord untuk training

Differences from ANRI:
- NO transcription labels (DIBCO is visual-only)
- Dummy labels untuk compatibility dengan training script
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def preprocess_image_strip(image, target_height=128, target_width=1024):
    """
    Preprocess image strip dengan aspect ratio preservation (ANRI method).
    
    Args:
        image: numpy array (H, W) or (H, W, C)
        target_height: 128 (fixed)
        target_width: 1024 (fixed)
    
    Returns:
        Preprocessed image (target_height, target_width, 1) in [0, 1] range
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = image.squeeze()
    
    h, w = image.shape
    
    # Calculate scaling factor to fit height to target_height
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height
    
    # Resize maintaining aspect ratio
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad or crop to target_width
    if new_w < target_width:
        # Center padding dengan white (255)
        pad_total = target_width - new_w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        image_final = cv2.copyMakeBorder(
            image_resized,
            top=0, bottom=0,
            left=pad_left, right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=255  # White padding
        )
    elif new_w > target_width:
        # Crop center region
        crop_start = (new_w - target_width) // 2
        image_final = image_resized[:, crop_start:crop_start+target_width]
    else:
        image_final = image_resized
    
    # Normalize to [0, 1]
    if image_final.max() > 1.0:
        image_final = image_final.astype(np.float32) / 255.0
    else:
        image_final = image_final.astype(np.float32)
    
    # Expand dims to (H, W, 1)
    image_final = np.expand_dims(image_final, axis=-1)
    
    return image_final

def slice_image_into_strips(image, strip_height=128, overlap=0):
    """
    Slice image into horizontal strips.
    
    Args:
        image: numpy array (H, W) or (H, W, C)
        strip_height: Height of each strip (before resize)
        overlap: Overlap between strips (0 = no overlap)
    
    Returns:
        List of strips (numpy arrays)
    """
    h, w = image.shape[:2]
    strips = []
    
    if h <= strip_height:
        # Image too small, return as-is
        return [image]
    
    # Calculate step
    step = int(strip_height * (1 - overlap))
    
    # Slice from top to bottom
    for y in range(0, h - strip_height + 1, step):
        strip = image[y:y+strip_height, :]
        strips.append(strip)
    
    # Handle last strip if not covered
    if y + strip_height < h:
        last_strip = image[-strip_height:, :]
        strips.append(last_strip)
    
    return strips

def augment_strip(strip, augmentation_factor=5):
    """
    Apply augmentation to single strip (same as ANRI method).
    
    Args:
        strip: numpy array (H, W, 1) in [0, 1] range
        augmentation_factor: Number of augmented versions
    
    Returns:
        List of augmented strips (including original)
    """
    strips = [strip]  # Include original
    
    h, w, c = strip.shape
    
    # Denormalize for augmentation
    strip_uint8 = (strip * 255).astype(np.uint8).squeeze()
    
    for i in range(augmentation_factor - 1):
        aug = strip_uint8.copy()
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.9, 1.1)  # Contrast
        beta = np.random.randint(-10, 10)  # Brightness
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
        
        # Random noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 3, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Normalize back
        aug_normalized = aug.astype(np.float32) / 255.0
        aug_normalized = np.expand_dims(aug_normalized, axis=-1)
        
        strips.append(aug_normalized)
    
    return strips

def create_tf_example(degraded_strip, clean_strip, label_ids=None):
    """
    Create TensorFlow Example (format SAMA dengan ANRI finetuning).
    
    Args:
        degraded_strip: numpy array (H, W, 1) in [0, 1] range
        clean_strip: numpy array (H, W, 1) in [0, 1] range
        label_ids: numpy array of label IDs (dummy untuk DIBCO)
    """
    # Dummy label jika tidak ada
    if label_ids is None:
        label_ids = np.array([0], dtype=np.int64)  # Blank token
    
    feature = {
        'degraded_image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[degraded_strip.tobytes()])),
        'degraded_image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=degraded_strip.shape)),
        'degraded_image_dtype': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'float32'])),
        'clean_image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[clean_strip.tobytes()])),
        'clean_image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=clean_strip.shape)),
        'clean_image_dtype': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'float32'])),
        'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_ids.tobytes()])),
        'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_ids.shape[0]])),
        'label_dtype': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'int64'])),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def process_dibco_dataset(dibco_root, output_dir, strip_height=128, augmentation_factor=5, 
                          train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Process DIBCO dataset with ANRI-style strip method.
    """
    dibco_root = Path(dibco_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📄 CREATING DIBCO FINE-TUNING DATASET (ANRI METHOD)")
    print("=" * 80)
    print()
    print(f"Source: {dibco_root}")
    print(f"Output: {output_dir}")
    print(f"Strip height: {strip_height}px (before resize to 128)")
    print(f"Target size: 1024x128")
    print(f"Augmentation: {augmentation_factor}x")
    print(f"Split ratio: Train {train_ratio:.0%}, Val {val_ratio:.0%}, Test {test_ratio:.0%}")
    print()
    
    # Collect all image pairs
    all_strips = []
    stats = {}
    
    print("📊 Processing DIBCO years...")
    for year_dir in sorted(dibco_root.iterdir()):
        if not year_dir.is_dir():
            continue
        
        year_name = year_dir.name
        imgs_dir = year_dir / "imgs"
        gt_dir = year_dir / "gt_imgs"
        
        if not (imgs_dir.exists() and gt_dir.exists()):
            print(f"⚠️  Skipping {year_name}: missing imgs or gt_imgs")
            continue
        
        print(f"\n{year_name}:")
        
        # Get all degraded images
        degraded_files = sorted(list(imgs_dir.glob("*.png")) + list(imgs_dir.glob("*.jpg")))
        year_strips = []
        
        for deg_file in tqdm(degraded_files, desc=f"  Processing {year_name}"):
            # Find corresponding GT
            gt_file = gt_dir / deg_file.name
            if not gt_file.exists():
                # Try different extension
                gt_file = gt_dir / deg_file.with_suffix('.png').name
                if not gt_file.exists():
                    print(f"    ⚠️  No GT for {deg_file.name}, skipping")
                    continue
            
            # Read images
            degraded = cv2.imread(str(deg_file), cv2.IMREAD_GRAYSCALE)
            clean = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            
            if degraded is None or clean is None:
                print(f"    ⚠️  Failed to read {deg_file.name}, skipping")
                continue
            
            # Slice into strips
            degraded_strips = slice_image_into_strips(degraded, strip_height=strip_height)
            clean_strips = slice_image_into_strips(clean, strip_height=strip_height)
            
            if len(degraded_strips) != len(clean_strips):
                print(f"    ⚠️  Strip count mismatch for {deg_file.name}, skipping")
                continue
            
            # Process each strip pair
            for deg_strip, clean_strip in zip(degraded_strips, clean_strips):
                # Preprocess (resize + aspect ratio preservation)
                deg_processed = preprocess_image_strip(deg_strip)
                clean_processed = preprocess_image_strip(clean_strip)
                
                year_strips.append((deg_processed, clean_processed, year_name))
        
        print(f"  Total strips: {len(year_strips)}")
        all_strips.extend(year_strips)
        stats[year_name] = len(year_strips)
    
    print()
    print("=" * 80)
    print(f"📊 TOTAL STRIPS BEFORE AUGMENTATION: {len(all_strips)}")
    print("=" * 80)
    print()
    
    # Sequential split (ANRI method - no randomization)
    total = len(all_strips)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_strips = all_strips[:train_end]
    val_strips = all_strips[train_end:val_end]
    test_strips = all_strips[val_end:]
    
    print(f"Sequential Split:")
    print(f"  Train: {len(train_strips)} strips (0 to {train_end})")
    print(f"  Val:   {len(val_strips)} strips ({train_end} to {val_end})")
    print(f"  Test:  {len(test_strips)} strips ({val_end} to {total})")
    print()
    
    # Apply augmentation and create TFRecords
    splits = {
        'train': (train_strips, True),  # Apply augmentation
        'val': (val_strips, False),      # No augmentation
        'test': (test_strips, False)     # No augmentation
    }
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'method': 'ANRI-style strips (horizontal slicing + aspect ratio preservation)',
        'strip_height_before_resize': strip_height,
        'target_dimensions': {'height': 128, 'width': 1024, 'channels': 1},
        'augmentation_factor': augmentation_factor,
        'split_method': 'sequential (70/15/15)',
        'stats_per_year': stats,
        'splits': {}
    }
    
    for split_name, (strips, apply_aug) in splits.items():
        print(f"\n{'='*80}")
        print(f"Creating {split_name.upper()} TFRecord...")
        print(f"{'='*80}")
        
        tfrecord_path = output_dir / f"finetuning_{split_name}.tfrecord"
        
        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            total_samples = 0
            
            for deg, clean, year in tqdm(strips, desc=f"  Writing {split_name}"):
                if apply_aug:
                    # Apply augmentation
                    deg_augs = augment_strip(deg, augmentation_factor)
                    clean_augs = augment_strip(clean, augmentation_factor)
                    
                    for deg_aug, clean_aug in zip(deg_augs, clean_augs):
                        example = create_tf_example(deg_aug, clean_aug)
                        writer.write(example.SerializeToString())
                        total_samples += 1
                else:
                    # No augmentation
                    example = create_tf_example(deg, clean)
                    writer.write(example.SerializeToString())
                    total_samples += 1
        
        metadata['splits'][split_name] = {
            'original_strips': len(strips),
            'total_samples': total_samples,
            'augmentation': 'yes' if apply_aug else 'no',
            'tfrecord_path': str(tfrecord_path.name)
        }
        
        print(f"  ✅ Saved: {tfrecord_path}")
        print(f"  📊 Samples: {total_samples}")
    
    # Save metadata
    metadata_path = output_dir / "finetuning_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 80)
    print("✅ DIBCO FINE-TUNING DATASET CREATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print(f"Metadata: {metadata_path}")
    print()
    print("Summary:")
    for split_name, info in metadata['splits'].items():
        print(f"  {split_name.upper()}: {info['total_samples']} samples " +
              f"({info['original_strips']} strips, aug={info['augmentation']})")
    print()
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Create DIBCO fine-tuning dataset (ANRI method)')
    parser.add_argument('--dibco_root', type=str, 
                       default='real_data_preparation/dataset_dibco',
                       help='Root directory of DIBCO dataset')
    parser.add_argument('--output_dir', type=str,
                       default='dual_modal_gan/data/dibco_finetuning',
                       help='Output directory for TFRecords')
    parser.add_argument('--strip_height', type=int, default=128,
                       help='Strip height before resize (default: 128)')
    parser.add_argument('--augmentation_factor', type=int, default=5,
                       help='Augmentation factor for training set (default: 5)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Train split ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test split ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    metadata = process_dibco_dataset(
        dibco_root=args.dibco_root,
        output_dir=args.output_dir,
        strip_height=args.strip_height,
        augmentation_factor=args.augmentation_factor,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

if __name__ == '__main__':
    main()
