#!/usr/bin/env python3
"""
Create Fine-tuning Dataset from Full-Page Documents

Strategy:
1. Slice full-page images into 1024x128 strips (horizontal slices)
2. Use SEQUENTIAL split to avoid data leakage (not random)
3. Apply augmentation to increase diversity
4. Generate TFRecord for training

Academic Protocol:
- Sequential split: top 70% → train, middle 15% → val, bottom 15% → test
- This ensures no correlation between train/val/test sets
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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def read_charlist(path):
    """Read character list from file"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def preprocess_image_strip(image, target_height=128, target_width=1024, maintain_aspect_ratio=True):
    """
    Preprocess image strip to match training format with PROPER aspect ratio handling.
    
    ✅ FIX (2025-10-30): Maintain aspect ratio dengan center padding untuk prevent text distortion
    
    Args:
        image: numpy array (H, W) or (H, W, C)
        target_height: 128 (fixed)
        target_width: 1024 (fixed)
        maintain_aspect_ratio: If True, resize maintaining aspect ratio and pad with white
    
    Returns:
        Preprocessed image (target_width, target_height, 1) in [0, 1] range
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = image.squeeze()
    
    # Get current dimensions
    h, w = image.shape
    
    if maintain_aspect_ratio:
        # Calculate scaling factor to fit height to target_height
        scale = target_height / h
        new_w = int(w * scale)
        new_h = target_height
        
        # Resize maintaining aspect ratio
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad or crop to target_width
        if new_w < target_width:
            # Pad with white (255) to reach target_width (center padding)
            pad_total = target_width - new_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
            image_padded = cv2.copyMakeBorder(
                image_resized,
                top=0, bottom=0,
                left=pad_left, right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=255  # White padding (document background)
            )
            image_final_hw = image_padded
        elif new_w > target_width:
            # Crop center region to target_width
            crop_start = (new_w - target_width) // 2
            image_final_hw = image_resized[:, crop_start:crop_start+target_width]
        else:
            image_final_hw = image_resized
    else:
        # OLD BEHAVIOR: Force resize (causes aspect ratio distortion)
        # ⚠️ This will compress/stretch text!
        image_final_hw = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    if image_final_hw.max() > 1.0:
        image_final_hw = image_final_hw.astype(np.float32) / 255.0
    else:
        image_final_hw = image_final_hw.astype(np.float32)
    
    # Add channel dimension and transpose to (W, H, C) format (as expected by model)
    image_final = image_final_hw[np.newaxis, :, :].transpose(2, 1, 0)  # (1, H, W) → (W, H, 1)
    
    return image_final

def extract_strips_from_image(image_path, strip_height=128, overlap=0):
    """
    Extract horizontal strips from full-page image.
    
    Args:
        image_path: Path to full-page image
        strip_height: Height of each strip (default: 128)
        overlap: Overlap between strips in pixels (default: 0)
    
    Returns:
        List of image strips (numpy arrays)
    """
    # Read image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    h, w = image.shape
    strips = []
    
    # Calculate stride
    stride = strip_height - overlap
    
    # Extract strips from top to bottom
    y = 0
    while y + strip_height <= h:
        strip = image[y:y+strip_height, :]
        strips.append(strip)
        y += stride
    
    # Handle last strip if remaining height > 50% of strip_height
    if y < h and (h - y) > strip_height // 2:
        # Take last strip_height pixels
        strip = image[h-strip_height:h, :]
        strips.append(strip)
    
    print(f"   Extracted {len(strips)} strips from {image_path.name} (size: {h}x{w})")
    
    return strips

def augment_strip(image):
    """
    Apply augmentation to image strip.
    
    Args:
        image: numpy array (H, W) grayscale
    
    Returns:
        List of augmented images
    """
    augmented = [image]  # Original
    
    # Horizontal flip
    augmented.append(cv2.flip(image, 1))
    
    # Brightness variations (±10%)
    for factor in [0.9, 1.1]:
        aug = np.clip(image * factor, 0, 255).astype(np.uint8)
        augmented.append(aug)
    
    # Small contrast adjustment
    alpha = 1.1  # Contrast
    beta = 0     # Brightness
    aug = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented.append(aug)
    
    return augmented

def create_tfrecord_example(degraded_image, clean_image, label_ids):
    """
    Create TFRecord example.
    
    Args:
        degraded_image: numpy array (W, H, C) in [0, 1] range
        clean_image: numpy array (W, H, C) in [0, 1] range
        label_ids: numpy array of label IDs
    
    Returns:
        tf.train.Example
    """
    # Transpose from (W, H, C) to (H, W, C) for storage
    degraded_image_hwc = degraded_image.transpose(1, 0, 2)
    clean_image_hwc = clean_image.transpose(1, 0, 2)
    
    feature = {
        'degraded_image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[degraded_image_hwc.tobytes()])),
        'degraded_image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=degraded_image_hwc.shape)),
        'degraded_image_dtype': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'float32'])),
        'clean_image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[clean_image_hwc.tobytes()])),
        'clean_image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=clean_image_hwc.shape)),
        'clean_image_dtype': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'float32'])),
        'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_ids.tobytes()])),
        'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_ids.shape[0]])),
        'label_dtype': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'int64'])),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def main(args):
    """
    Main function to create fine-tuning dataset.
    """
    print("="*80)
    print("FINE-TUNING DATASET CREATION")
    print("="*80)
    
    # Setup paths
    gt_dir = Path(args.gt_dir)
    deg_dir = Path(args.deg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read charset
    charset = read_charlist(args.charset_path)
    vocab_size = len(charset) + 1
    print(f"\n✅ Charset loaded: {vocab_size} characters")
    
    # Find image pairs
    gt_images = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))
    deg_images = sorted(list(deg_dir.glob("*.png")) + list(deg_dir.glob("*.jpg")))
    
    print(f"\n📁 Found {len(gt_images)} GT images and {len(deg_images)} degraded images")
    
    # Match pairs
    pairs = []
    for gt_path in gt_images:
        # Try to find matching degraded image
        stem = gt_path.stem
        for deg_path in deg_images:
            if deg_path.stem == stem:
                pairs.append((gt_path, deg_path))
                break
    
    print(f"✅ Matched {len(pairs)} image pairs")
    
    if len(pairs) == 0:
        print("❌ No image pairs found! Check your GT and DEG directories.")
        return
    
    # Process each pair
    all_strips = []
    
    for gt_path, deg_path in pairs:
        print(f"\n📄 Processing: {gt_path.name}")
        
        # Extract strips
        gt_strips = extract_strips_from_image(gt_path, strip_height=128, overlap=args.overlap)
        deg_strips = extract_strips_from_image(deg_path, strip_height=128, overlap=args.overlap)
        
        if len(gt_strips) != len(deg_strips):
            print(f"   ⚠️  Mismatch: GT has {len(gt_strips)} strips, DEG has {len(deg_strips)} strips")
            min_strips = min(len(gt_strips), len(deg_strips))
            gt_strips = gt_strips[:min_strips]
            deg_strips = deg_strips[:min_strips]
            print(f"   Using first {min_strips} strips")
        
        # Apply augmentation if enabled
        for i, (gt_strip, deg_strip) in enumerate(zip(gt_strips, deg_strips)):
            if args.augment:
                gt_augmented = augment_strip(gt_strip)
                deg_augmented = augment_strip(deg_strip)
                
                for gt_aug, deg_aug in zip(gt_augmented, deg_augmented):
                    all_strips.append((gt_aug, deg_aug, i))
            else:
                all_strips.append((gt_strip, deg_strip, i))
    
    print(f"\n✅ Total strips (with augmentation): {len(all_strips)}")
    
    # Sequential split (ACADEMIC PROTOCOL)
    total_strips = len(all_strips)
    train_idx = int(total_strips * args.train_split)
    val_idx = int(total_strips * (args.train_split + args.val_split))
    
    train_strips = all_strips[:train_idx]
    val_strips = all_strips[train_idx:val_idx]
    test_strips = all_strips[val_idx:]
    
    print(f"\n📊 Sequential Split (NO data leakage):")
    print(f"   Train: {len(train_strips)} strips ({len(train_strips)/total_strips*100:.1f}%)")
    print(f"   Val:   {len(val_strips)} strips ({len(val_strips)/total_strips*100:.1f}%)")
    print(f"   Test:  {len(test_strips)} strips ({len(test_strips)/total_strips*100:.1f}%)")
    
    # Create TFRecord files
    splits = {
        'train': train_strips,
        'val': val_strips,
        'test': test_strips
    }
    
    # Dummy labels (empty labels for now - fine-tuning is for visual quality, not HTR)
    dummy_label = np.array([0], dtype=np.int64)  # Blank token
    
    for split_name, strips in splits.items():
        if len(strips) == 0:
            print(f"⚠️  Skipping {split_name} (no strips)")
            continue
        
        tfrecord_path = output_dir / f"finetuning_{split_name}.tfrecord"
        print(f"\n📝 Creating {split_name} TFRecord: {tfrecord_path}")
        
        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for gt_strip, deg_strip, strip_idx in tqdm(strips, desc=f"Writing {split_name}"):
                # Preprocess to match training format with aspect ratio preservation
                gt_processed = preprocess_image_strip(
                    gt_strip, 
                    target_height=128, 
                    target_width=1024,
                    maintain_aspect_ratio=args.maintain_aspect_ratio
                )
                deg_processed = preprocess_image_strip(
                    deg_strip, 
                    target_height=128, 
                    target_width=1024,
                    maintain_aspect_ratio=args.maintain_aspect_ratio
                )
                
                # Create TFRecord example
                example = create_tfrecord_example(deg_processed, gt_processed, dummy_label)
                writer.write(example.SerializeToString())
        
        print(f"   ✅ Saved {len(strips)} examples to {tfrecord_path}")
    
    # Save metadata
    metadata = {
        'creation_date': str(Path(__file__).stat().st_mtime),
        'source_images': [str(p) for p, _ in pairs],
        'total_strips': total_strips,
        'splits': {
            'train': len(train_strips),
            'val': len(val_strips),
            'test': len(test_strips)
        },
        'augmentation': args.augment,
        'overlap': args.overlap,
        'strip_dimensions': '1024x128',
        'aspect_ratio_preserved': args.maintain_aspect_ratio,
        'resize_method': 'maintain_aspect_ratio_with_padding' if args.maintain_aspect_ratio else 'force_resize',
        'charset_size': vocab_size,
        'note': 'Sequential split used to prevent data leakage (top→train, middle→val, bottom→test)'
    }
    
    metadata_path = output_dir / 'finetuning_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved to: {metadata_path}")
    print(f"\n🎉 Fine-tuning dataset creation COMPLETE!")
    print(f"   Output directory: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create fine-tuning dataset from full-page documents')
    parser.add_argument('--gt_dir', type=str, default='DokumenRusak/manual_restoration/gt',
                        help='Directory with ground truth images')
    parser.add_argument('--deg_dir', type=str, default='DokumenRusak/manual_restoration/deg',
                        help='Directory with degraded images')
    parser.add_argument('--output_dir', type=str, default='dual_modal_gan/data/finetuning',
                        help='Output directory for TFRecord files')
    parser.add_argument('--charset_path', type=str, default='real_data_preparation/real_data_charlist.txt',
                        help='Path to charset file')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Fraction for training (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction for validation (default: 0.15)')
    parser.add_argument('--overlap', type=int, default=0,
                        help='Overlap between strips in pixels (default: 0)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Apply augmentation (flip, brightness, contrast)')
    parser.add_argument('--no_augment', dest='augment', action='store_false',
                        help='Disable augmentation')
    parser.add_argument('--maintain_aspect_ratio', action='store_true', default=True,
                        help='Maintain aspect ratio when resizing (with padding) - RECOMMENDED to prevent text distortion')
    parser.add_argument('--force_resize', dest='maintain_aspect_ratio', action='store_false',
                        help='Force resize without maintaining aspect ratio (may cause text distortion - NOT recommended)')
    
    args = parser.parse_args()
    main(args)
