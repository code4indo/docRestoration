#!/usr/bin/env python3
"""
Extract patches from full-page ANRI documents for pseudo-labeling fine-tuning

Strategy:
- Grid-based tiling with overlap
- Split into train/validation sets
- Preserve patch metadata for reconstruction

Author: Pseudo-Labeling Pipeline
Date: 2025-10-28
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_patches_from_image(
    image_path: Path,
    patch_size: int = 256,
    overlap: int = 64,
    min_text_ratio: float = 0.01,
) -> List[Tuple[np.ndarray, dict]]:
    """
    Extract patches from a single full-page image
    
    Args:
        image_path: Path to input image
        patch_size: Size of square patches
        overlap: Overlap between adjacent patches
        min_text_ratio: Minimum ratio of dark pixels to keep patch
    
    Returns:
        List of (patch_array, metadata) tuples
    """
    with Image.open(image_path) as img:
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img)
        height, width = img_array.shape
    
    stride = patch_size - overlap
    patches = []
    
    # Grid-based extraction
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = img_array[y:y+patch_size, x:x+patch_size]
            
            # Filter patches with too little content (mostly background)
            text_ratio = (patch < 128).sum() / patch.size
            
            if text_ratio >= min_text_ratio:
                metadata = {
                    'source_image': image_path.name,
                    'x': int(x),
                    'y': int(y),
                    'width': patch_size,
                    'height': patch_size,
                    'text_ratio': float(text_ratio),
                }
                patches.append((patch, metadata))
    
    return patches


def save_patch(patch_array: np.ndarray, output_path: Path):
    """Save patch as image"""
    img = Image.fromarray(patch_array)
    img.save(output_path, 'JPEG', quality=95)


def main():
    parser = argparse.ArgumentParser(
        description='Extract patches from full-page ANRI documents'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='DokumenRusak/full_pages_ANRI',
        help='Directory containing full-page images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='DokumenRusak/anri_patches',
        help='Output directory for patches'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=256,
        help='Size of square patches (default: 256)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=64,
        help='Overlap between patches (default: 64)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.85,
        help='Ratio of documents for training (default: 0.85 → 28/33)'
    )
    parser.add_argument(
        '--min_text_ratio',
        type=float,
        default=0.01,
        help='Minimum text ratio to keep patch (default: 0.01)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    train_degraded_dir = output_dir / 'train' / 'degraded'
    val_degraded_dir = output_dir / 'val' / 'degraded'
    
    train_degraded_dir.mkdir(parents=True, exist_ok=True)
    val_degraded_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PATCH EXTRACTION FROM FULL-PAGE ANRI DOCUMENTS")
    print("="*80)
    print()
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Overlap: {args.overlap} px")
    print(f"Stride: {args.patch_size - args.overlap} px")
    print(f"Min text ratio: {args.min_text_ratio}")
    print()
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f'*{ext}')))
        image_files.extend(list(input_dir.glob(f'*{ext.upper()}')))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} full-page documents")
    print()
    
    # Split into train/val
    random.seed(args.seed)
    random.shuffle(image_files)
    
    n_train = int(len(image_files) * args.train_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    
    print(f"Split: {len(train_files)} train / {len(val_files)} validation")
    print()
    
    # Extract patches
    train_patches = []
    val_patches = []
    train_metadata = []
    val_metadata = []
    
    print("Extracting training patches...")
    for img_path in tqdm(train_files, desc="Train"):
        patches = extract_patches_from_image(
            img_path,
            patch_size=args.patch_size,
            overlap=args.overlap,
            min_text_ratio=args.min_text_ratio,
        )
        
        for i, (patch, meta) in enumerate(patches):
            patch_filename = f"{img_path.stem}_patch_{i:04d}.jpg"
            patch_path = train_degraded_dir / patch_filename
            save_patch(patch, patch_path)
            
            meta['patch_filename'] = patch_filename
            train_metadata.append(meta)
            train_patches.append(patch_path)
    
    print()
    print("Extracting validation patches...")
    for img_path in tqdm(val_files, desc="Val"):
        patches = extract_patches_from_image(
            img_path,
            patch_size=args.patch_size,
            overlap=args.overlap,
            min_text_ratio=args.min_text_ratio,
        )
        
        for i, (patch, meta) in enumerate(patches):
            patch_filename = f"{img_path.stem}_patch_{i:04d}.jpg"
            patch_path = val_degraded_dir / patch_filename
            save_patch(patch, patch_path)
            
            meta['patch_filename'] = patch_filename
            val_metadata.append(meta)
            val_patches.append(patch_path)
    
    print()
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print()
    print(f"Training set:")
    print(f"  Documents: {len(train_files)}")
    print(f"  Patches: {len(train_patches)}")
    print(f"  Avg patches/doc: {len(train_patches)/len(train_files):.1f}")
    print()
    print(f"Validation set:")
    print(f"  Documents: {len(val_files)}")
    print(f"  Patches: {len(val_patches)}")
    print(f"  Avg patches/doc: {len(val_patches)/len(val_files):.1f}")
    print()
    print(f"Total patches: {len(train_patches) + len(val_patches)}")
    print()
    
    # Save metadata
    metadata = {
        'config': {
            'patch_size': args.patch_size,
            'overlap': args.overlap,
            'stride': args.patch_size - args.overlap,
            'min_text_ratio': args.min_text_ratio,
            'train_ratio': args.train_ratio,
            'seed': args.seed,
        },
        'train_files': [str(f.name) for f in train_files],
        'val_files': [str(f.name) for f in val_files],
        'train_patches': train_metadata,
        'val_patches': val_metadata,
        'statistics': {
            'n_train_docs': len(train_files),
            'n_val_docs': len(val_files),
            'n_train_patches': len(train_patches),
            'n_val_patches': len(val_patches),
            'avg_patches_per_train_doc': len(train_patches) / len(train_files) if train_files else 0,
            'avg_patches_per_val_doc': len(val_patches) / len(val_files) if val_files else 0,
        }
    }
    
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    print()
    
    # Calculate text ratio statistics
    train_text_ratios = [m['text_ratio'] for m in train_metadata]
    val_text_ratios = [m['text_ratio'] for m in val_metadata]
    
    if train_text_ratios:
        print(f"Training patches text ratio:")
        print(f"  Mean: {np.mean(train_text_ratios):.3f}")
        print(f"  Std: {np.std(train_text_ratios):.3f}")
        print(f"  Min: {np.min(train_text_ratios):.3f}")
        print(f"  Max: {np.max(train_text_ratios):.3f}")
        print()
    
    if val_text_ratios:
        print(f"Validation patches text ratio:")
        print(f"  Mean: {np.mean(val_text_ratios):.3f}")
        print(f"  Std: {np.std(val_text_ratios):.3f}")
        print(f"  Min: {np.min(val_text_ratios):.3f}")
        print(f"  Max: {np.max(val_text_ratios):.3f}")
        print()
    
    print("="*80)
    print("✅ PATCH EXTRACTION COMPLETE")
    print("="*80)
    print()
    print("Next step:")
    print("  python scripts/generate_pseudo_labels_patches.py \\")
    print(f"    --input_dir {output_dir} \\")
    print("    --checkpoint_dir dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model \\")
    print("    --checkpoint_name ckpt-99")
    print()


if __name__ == '__main__':
    main()
