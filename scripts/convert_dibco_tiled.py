#!/usr/bin/env python3
"""
DIBCO to TFRecord Converter with Tiling Augmentation
Converts DIBCO binarization datasets to TFRecord format with intelligent tiling
for large images to maximize dataset size and preserve details.

Strategy:
- Large images (>2048px width): Extract overlapping rectangular tiles
- Small images: Use padding strategy (preserve aspect ratio)
- Target: 152 source images → ~600 training samples

Author: ML Pipeline
Date: 2025-10-26
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def extract_tiles(
    image: np.ndarray,
    tile_height: int = 320,
    tile_width: int = 1280,
    overlap: float = 0.25,
    min_text_ratio: float = 0.01
) -> List[np.ndarray]:
    """
    Extract overlapping rectangular tiles from large image.
    
    Args:
        image: Input image (H x W)
        tile_height: Target tile height before resize
        tile_width: Target tile width before resize
        overlap: Overlap ratio between adjacent tiles (0.0 - 0.5)
        min_text_ratio: Minimum ratio of dark pixels to keep tile
        
    Returns:
        List of tile images
    """
    h, w = image.shape
    tiles = []
    
    # Calculate stride
    stride_h = int(tile_height * (1 - overlap))
    stride_w = int(tile_width * (1 - overlap))
    
    # Ensure minimum stride
    stride_h = max(stride_h, tile_height // 2)
    stride_w = max(stride_w, tile_width // 2)
    
    # Extract tiles with sliding window
    for y in range(0, h - tile_height + 1, stride_h):
        for x in range(0, w - tile_width + 1, stride_w):
            tile = image[y:y+tile_height, x:x+tile_width]
            
            # Quality check: skip tiles that are mostly blank
            # (for clean/GT images, check if has enough text content)
            dark_pixel_ratio = np.sum(tile < 128) / tile.size
            
            if dark_pixel_ratio >= min_text_ratio or dark_pixel_ratio > 0.5:
                # Either has text content OR is degraded image (mostly dark)
                tiles.append(tile)
    
    # If no tiles extracted (image too small), return original padded
    if len(tiles) == 0:
        tiles.append(image)
    
    return tiles


def load_and_preprocess_image(
    image_path: str,
    target_height: int = 128,
    target_width: int = 1024,
    use_tiling: bool = True,
    tile_height: int = 256,
    tile_width: int = 2048,
    overlap: float = 0.25
) -> List[np.ndarray]:
    """
    Load image and apply tiling or padding strategy.
    
    Returns:
        List of preprocessed images (can be multiple tiles or single padded image)
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    h, w = img.shape
    
    # Decide: Tiling vs Padding
    # Use tiling if image meets or exceeds tile size
    # This ensures we extract maximum tiles from available data
    needs_tiling = (w >= tile_width) or (h >= tile_height)
    
    if use_tiling and needs_tiling:
        # TILING STRATEGY for large images
        tiles = extract_tiles(img, tile_height, tile_width, overlap)
        
        # Resize each tile to target size
        processed_tiles = []
        for tile in tiles:
            resized = cv2.resize(tile, (target_width, target_height), 
                                interpolation=cv2.INTER_AREA)
            processed_tiles.append(resized)
        
        return processed_tiles
    
    else:
        # PADDING STRATEGY for small images
        # Calculate scaling to fit within target while preserving aspect ratio
        scale_h = target_height / h
        scale_w = target_width / w
        scale = min(scale_h, scale_w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create white canvas and paste resized image
        canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
        
        # Center the image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return [canvas]


def create_dummy_label(text_length: int = 10) -> np.ndarray:
    """
    Create dummy text label for visual-only training.
    DIBCO has no text annotations, so we use placeholder.
    """
    # Return array of zeros (will be ignored during training with CTC weight = 0)
    return np.zeros(text_length, dtype=np.int32)


def _bytes_feature(value):
    """Helper to create bytes feature."""
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Helper to create int64 feature."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(
    tfrecord_path: str,
    degraded_clean_pairs: List[Tuple[np.ndarray, np.ndarray]],
    verify: bool = True
) -> Dict:
    """
    Write image pairs to TFRecord file.
    Format MUST match train_enhanced.py schema EXACTLY (raw bytes + metadata).
    
    Args:
        tfrecord_path: Output TFRecord file path
        degraded_clean_pairs: List of (degraded, clean) image pairs (uint8, 0-255)
        verify: Whether to verify written records
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'total_samples': len(degraded_clean_pairs),
        'successful': 0,
        'failed': 0
    }
    
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for degraded_img, clean_img in tqdm(degraded_clean_pairs, desc="Writing TFRecord"):
            try:
                # Normalize to [0, 1] float32 AND add channel dimension
                degraded_normalized = (degraded_img.astype(np.float32) / 255.0).reshape(
                    degraded_img.shape[0], degraded_img.shape[1], 1
                )  # (H, W) → (H, W, 1)
                clean_normalized = (clean_img.astype(np.float32) / 255.0).reshape(
                    clean_img.shape[0], clean_img.shape[1], 1
                )  # (H, W) → (H, W, 1)
                
                # Create dummy label (will be ignored with CTC weight = 0)
                label = create_dummy_label()
                
                # Create TF Example with EXACT schema from convert_dibco_to_tfrecord.py
                feature = {
                    'degraded_image_raw': _bytes_feature(degraded_normalized.astype(np.float32).tobytes()),
                    'degraded_image_shape': _int64_feature(list(degraded_normalized.shape)),
                    'degraded_image_dtype': _bytes_feature(b'float32'),
                    'clean_image_raw': _bytes_feature(clean_normalized.astype(np.float32).tobytes()),
                    'clean_image_shape': _int64_feature(list(clean_normalized.shape)),
                    'clean_image_dtype': _bytes_feature(b'float32'),
                    'label_raw': _bytes_feature(label.astype(np.int64).tobytes()),
                    'label_shape': _int64_feature([len(label)]),
                    'label_dtype': _bytes_feature(b'int64'),
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                stats['successful'] += 1
                
            except Exception as e:
                print(f"Error writing sample: {e}")
                stats['failed'] += 1
    
    # Verify if requested
    if verify and stats['successful'] > 0:
        print(f"\nVerifying TFRecord file...")
        
        # Use matching schema from train_enhanced.py
        feature_desc = {
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
        
        count = 0
        for raw_record in tf.data.TFRecordDataset(tfrecord_path):
            try:
                parsed = tf.io.parse_single_example(raw_record, feature_desc)
                # Try to decode to verify integrity
                deg_shape = parsed['degraded_image_shape'].numpy()
                cln_shape = parsed['clean_image_shape'].numpy()
                count += 1
            except Exception as e:
                print(f"⚠️  Verification error on record {count}: {e}")
                break
        
        print(f"✅ Verified {count} records in TFRecord")
        
        if count != stats['successful']:
            print(f"⚠️  Warning: Write count ({stats['successful']}) != Verify count ({count})")
    
    return stats


def convert_dibco_to_tfrecord(
    dibco_root: str,
    output_tfrecord: str,
    exclude_years: List[str] = None,
    max_samples: int = None,
    verify: bool = True,
    use_tiling: bool = True,
    tile_height: int = 256,
    tile_width: int = 2048,
    overlap: float = 0.25
) -> Dict:
    """
    Convert DIBCO dataset to TFRecord with tiling augmentation.
    
    Args:
        dibco_root: Root directory of DIBCO datasets
        output_tfrecord: Output TFRecord file path
        exclude_years: List of years to exclude (e.g., ['2012'] for test set)
        max_samples: Maximum number of source images to process (for testing)
        verify: Whether to verify the TFRecord after creation
        use_tiling: Whether to use tiling for large images
        tile_height: Tile height before resize
        tile_width: Tile width before resize
        overlap: Overlap ratio between tiles
        
    Returns:
        Statistics dictionary
    """
    dibco_root = Path(dibco_root)
    exclude_years = exclude_years or []
    
    # Available DIBCO years
    available_years = ['2009', '2010', '2011', '2013', '2014', '2016', '2017', '2018', 'PALM']
    years_to_process = [y for y in available_years if y not in exclude_years]
    
    print("=" * 80)
    print("DIBCO to TFRecord Converter with Tiling Augmentation")
    print("=" * 80)
    print(f"Source: {dibco_root}")
    print(f"Output: {output_tfrecord}")
    print(f"Years: {', '.join(years_to_process)}")
    print(f"Excluded: {', '.join(exclude_years) if exclude_years else 'None'}")
    print(f"Tiling: {'ENABLED' if use_tiling else 'DISABLED'}")
    if use_tiling:
        print(f"  Tile size: {tile_height} x {tile_width}")
        print(f"  Overlap: {overlap*100:.0f}%")
    print("=" * 80)
    print()
    
    # Collect image pairs
    degraded_clean_pairs = []
    source_image_count = 0
    tile_count = 0
    
    year_stats = {}
    
    for year in years_to_process:
        degraded_dir = dibco_root / year / 'imgs'
        clean_dir = dibco_root / year / 'gt_imgs'
        
        if not degraded_dir.exists() or not clean_dir.exists():
            print(f"⚠️  Skipping {year}: Missing imgs or gt_imgs directory")
            continue
        
        # Get matching pairs
        degraded_files = sorted(degraded_dir.glob('*.png')) + sorted(degraded_dir.glob('*.bmp'))
        clean_files = sorted(clean_dir.glob('*.png')) + sorted(clean_dir.glob('*.bmp'))
        
        # Match by filename
        degraded_dict = {f.stem: f for f in degraded_files}
        clean_dict = {f.stem: f for f in clean_files}
        
        common_stems = set(degraded_dict.keys()) & set(clean_dict.keys())
        
        year_tiles = 0
        year_sources = 0
        
        for stem in sorted(common_stems):
            if max_samples and source_image_count >= max_samples:
                break
            
            degraded_path = str(degraded_dict[stem])
            clean_path = str(clean_dict[stem])
            
            try:
                # Load and process (potentially multiple tiles)
                degraded_tiles = load_and_preprocess_image(
                    degraded_path, use_tiling=use_tiling,
                    tile_height=tile_height, tile_width=tile_width, overlap=overlap
                )
                clean_tiles = load_and_preprocess_image(
                    clean_path, use_tiling=use_tiling,
                    tile_height=tile_height, tile_width=tile_width, overlap=overlap
                )
                
                # Ensure same number of tiles
                if len(degraded_tiles) != len(clean_tiles):
                    print(f"⚠️  Tile count mismatch for {stem}: "
                          f"degraded={len(degraded_tiles)}, clean={len(clean_tiles)}")
                    # Use minimum count
                    min_tiles = min(len(degraded_tiles), len(clean_tiles))
                    degraded_tiles = degraded_tiles[:min_tiles]
                    clean_tiles = clean_tiles[:min_tiles]
                
                # Add all tile pairs
                for deg_tile, cln_tile in zip(degraded_tiles, clean_tiles):
                    degraded_clean_pairs.append((deg_tile, cln_tile))
                    tile_count += 1
                
                year_tiles += len(degraded_tiles)
                year_sources += 1
                source_image_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {stem}: {e}")
                continue
        
        year_stats[year] = {
            'source_images': year_sources,
            'tiles': year_tiles
        }
        
        print(f"✅ {year}: {year_sources} images → {year_tiles} tiles "
              f"(avg {year_tiles/year_sources:.1f} tiles/image)" if year_sources > 0 else "")
    
    print()
    print(f"📊 Total collected: {source_image_count} source images → {tile_count} tiles")
    print()
    
    # Write TFRecord
    if tile_count == 0:
        print("❌ No samples to write!")
        return {'error': 'No samples collected'}
    
    stats = write_tfrecord(output_tfrecord, degraded_clean_pairs, verify)
    
    # Save metadata
    metadata = {
        'created_at': str(pd.Timestamp.now()) if 'pd' in dir() else str(np.datetime64('now')),
        'source': 'DIBCO binarization datasets',
        'purpose': 'Fine-tuning GAN with tiling augmentation (CTC-free)',
        'target_dimensions': {
            'height': 128,
            'width': 1024,
            'channels': 1
        },
        'tiling_strategy': {
            'enabled': use_tiling,
            'tile_height': tile_height,
            'tile_width': tile_width,
            'overlap': overlap
        },
        'preprocessing': 'Tiling for large images, padding for small images',
        'label_type': 'dummy (visual-only training)',
        'statistics': {
            'source_images': source_image_count,
            'total_samples': stats['successful'],
            'successful': stats['successful'],
            'failed': stats['failed'],
            'augmentation_ratio': stats['successful'] / source_image_count if source_image_count > 0 else 0,
            'years': year_stats
        }
    }
    
    metadata_path = output_tfrecord.replace('.tfrecord', '.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved: {metadata_path}")
    print(f"✅ TFRecord created: {output_tfrecord}")
    print(f"   Source images: {source_image_count}")
    print(f"   Output samples: {stats['successful']}")
    print(f"   Augmentation: {stats['successful']/source_image_count:.1f}x")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert DIBCO datasets to TFRecord with tiling augmentation'
    )
    parser.add_argument('--dibco_root', type=str, required=True,
                       help='Root directory of DIBCO datasets')
    parser.add_argument('--output_tfrecord', type=str, required=True,
                       help='Output TFRecord file path')
    parser.add_argument('--exclude_years', type=str, nargs='+', default=[],
                       help='Years to exclude (e.g., 2012 for test set)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of source images to process (for testing)')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify TFRecord after creation')
    parser.add_argument('--no_tiling', action='store_true',
                       help='Disable tiling (use padding only)')
    parser.add_argument('--tile_height', type=int, default=320,
                       help='Tile height before resize (default: 320)')
    parser.add_argument('--tile_width', type=int, default=1280,
                       help='Tile width before resize (default: 1280)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap ratio between tiles (default: 0.25)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_tfrecord), exist_ok=True)
    
    # Convert
    stats = convert_dibco_to_tfrecord(
        dibco_root=args.dibco_root,
        output_tfrecord=args.output_tfrecord,
        exclude_years=args.exclude_years,
        max_samples=args.max_samples,
        verify=args.verify,
        use_tiling=not args.no_tiling,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        overlap=args.overlap
    )
    
    if stats.get('successful', 0) > 0:
        print("\n🎉 Conversion completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
