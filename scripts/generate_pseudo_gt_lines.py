#!/usr/bin/env python3
"""
Generate pseudo-GT for ANRI lines using ckpt-99

Strategy:
- Load model ckpt-99 (trained on 1024×128 lines)
- Run inference on extracted degraded lines
- Save pseudo-GT for fine-tuning

Author: Line-Level Pseudo-Labeling Pipeline
Date: 2025-10-28
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add dual_modal_gan to path
project_root = Path(__file__).parent.parent
dual_modal_dir = project_root / "dual_modal_gan"
sys.path.insert(0, str(dual_modal_dir))

# Import model architecture
from src.models.generator import unet as unet_enhanced


def load_generator(checkpoint_path: Path, input_size: Tuple[int, int] = (1024, 128)) -> tf.keras.Model:
    """Load generator from checkpoint"""
    print("Loading generator...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Input size: {input_size}")
    
    # Create model
    generator = unet_enhanced(input_size=(*input_size, 1))
    
    # Create checkpoint
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Restore weights
    status = checkpoint.restore(str(checkpoint_path)).expect_partial()
    
    print("✓ Generator loaded")
    print()
    
    return generator


def generate_pseudo_gt_batch(
    degraded_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 8,
    input_size: Tuple[int, int] = (1024, 128)
) -> List[str]:
    """
    Generate pseudo-GT for all lines in degraded_dir
    
    Returns:
        List of generated pseudo-GT filenames
    """
    # Load model
    generator = load_generator(checkpoint_path, input_size)
    
    # Find all degraded lines
    degraded_files = sorted(list(degraded_dir.glob('*.jpg')))
    
    if not degraded_files:
        raise ValueError(f"No images found in {degraded_dir}")
    
    print(f"Found {len(degraded_files)} degraded lines")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    generated_files = []
    
    for i in tqdm(range(0, len(degraded_files), batch_size), desc="Generating pseudo-GT"):
        batch_files = degraded_files[i:i+batch_size]
        
        # Load batch
        batch_images = []
        for file_path in batch_files:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            
            # Resize if needed
            if img.shape != (input_size[1], input_size[0]):  # (H, W)
                img = cv2.resize(img, input_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to [-1, 1]
            img_norm = (img.astype(np.float32) - 127.5) / 127.5
            batch_images.append(img_norm)
        
        # Stack to batch
        batch_array = np.array(batch_images)[..., np.newaxis]  # (B, H, W, 1)
        
        # Run inference
        pseudo_gt_batch = generator(batch_array, training=False).numpy()
        
        # Save each result
        for j, (file_path, pseudo_gt) in enumerate(zip(batch_files, pseudo_gt_batch)):
            # Denormalize from [-1, 1] to [0, 255]
            pseudo_gt_uint8 = ((pseudo_gt + 1.0) * 127.5).astype(np.uint8)
            
            # Remove channel dimension if present
            if pseudo_gt_uint8.ndim == 3 and pseudo_gt_uint8.shape[-1] == 1:
                pseudo_gt_uint8 = pseudo_gt_uint8[..., 0]
            
            # Save with same filename
            output_path = output_dir / file_path.name
            cv2.imwrite(str(output_path), pseudo_gt_uint8, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            generated_files.append(file_path.name)
    
    return generated_files


def compute_quality_metrics(degraded_path: Path, pseudo_gt_path: Path) -> dict:
    """Compute quality metrics for pseudo-GT"""
    degraded = cv2.imread(str(degraded_path), cv2.IMREAD_GRAYSCALE)
    pseudo_gt = cv2.imread(str(pseudo_gt_path), cv2.IMREAD_GRAYSCALE)
    
    if degraded is None or pseudo_gt is None:
        return {}
    
    # Basic statistics
    metrics = {
        'degraded_mean': float(degraded.mean()),
        'pseudo_gt_mean': float(pseudo_gt.mean()),
        'degraded_std': float(degraded.std()),
        'pseudo_gt_std': float(pseudo_gt.std()),
        'degraded_text_ratio': float((degraded < 200).sum() / degraded.size),
        'pseudo_gt_text_ratio': float((pseudo_gt < 200).sum() / pseudo_gt.size),
        'contrast_improvement': float(pseudo_gt.std() - degraded.std()),
    }
    
    # SSIM (TensorFlow implementation)
    degraded_tf = tf.convert_to_tensor(degraded[None, ..., None], dtype=tf.float32)
    pseudo_gt_tf = tf.convert_to_tensor(pseudo_gt[None, ..., None], dtype=tf.float32)
    
    ssim_val = tf.image.ssim(degraded_tf, pseudo_gt_tf, max_val=255.0)
    metrics['ssim'] = float(ssim_val.numpy()[0])
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-GT for ANRI lines')
    parser.add_argument('--lines_dir', type=str, default='outputs/anri_lines_laypa',
                        help='Directory with extracted lines (train/degraded, val/degraded)')
    parser.add_argument('--output_dir', type=str, default='outputs/anri_lines_laypa',
                        help='Output directory (will create train/pseudo_gt, val/pseudo_gt)')
    parser.add_argument('--checkpoint', type=str, 
                        default='models/thin_stroke_preservation_v1_academic/ckpt-99',
                        help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_width', type=int, default=1024)
    parser.add_argument('--input_height', type=int, default=128)
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute quality metrics for sample')
    
    args = parser.parse_args()
    
    # Setup paths
    lines_dir = Path(args.lines_dir)
    output_dir = Path(args.output_dir)
    
    train_degraded_dir = lines_dir / 'train' / 'degraded'
    val_degraded_dir = lines_dir / 'val' / 'degraded'
    
    train_pseudo_gt_dir = output_dir / 'train' / 'pseudo_gt'
    val_pseudo_gt_dir = output_dir / 'val' / 'pseudo_gt'
    
    input_size = (args.input_width, args.input_height)
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        # Try with .index suffix
        if not Path(str(checkpoint_path) + '.index').exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("="*80)
    print("PSEUDO-GT GENERATION FOR ANRI LINES")
    print("="*80)
    print()
    print(f"Lines directory: {lines_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input size: {input_size}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Generate training pseudo-GT
    if train_degraded_dir.exists():
        print("="*80)
        print("TRAINING SET")
        print("="*80)
        print()
        
        train_files = generate_pseudo_gt_batch(
            train_degraded_dir,
            train_pseudo_gt_dir,
            checkpoint_path,
            args.batch_size,
            input_size
        )
        
        print()
        print(f"✅ Generated {len(train_files)} training pseudo-GT images")
        print()
    else:
        print(f"⚠️  Training degraded directory not found: {train_degraded_dir}")
        print()
    
    # Generate validation pseudo-GT
    if val_degraded_dir.exists():
        print("="*80)
        print("VALIDATION SET")
        print("="*80)
        print()
        
        val_files = generate_pseudo_gt_batch(
            val_degraded_dir,
            val_pseudo_gt_dir,
            checkpoint_path,
            args.batch_size,
            input_size
        )
        
        print()
        print(f"✅ Generated {len(val_files)} validation pseudo-GT images")
        print()
    else:
        print(f"⚠️  Validation degraded directory not found: {val_degraded_dir}")
        print()
    
    # Compute metrics on sample
    if args.compute_metrics and train_pseudo_gt_dir.exists():
        print("="*80)
        print("QUALITY METRICS (SAMPLE)")
        print("="*80)
        print()
        
        # Sample 10 random pairs
        train_degraded_files = sorted(list(train_degraded_dir.glob('*.jpg')))
        sample_indices = np.random.choice(len(train_degraded_files), min(10, len(train_degraded_files)), replace=False)
        
        all_metrics = []
        for idx in sample_indices:
            degraded_path = train_degraded_files[idx]
            pseudo_gt_path = train_pseudo_gt_dir / degraded_path.name
            
            if pseudo_gt_path.exists():
                metrics = compute_quality_metrics(degraded_path, pseudo_gt_path)
                all_metrics.append(metrics)
                
                print(f"{degraded_path.name}:")
                print(f"  Degraded: mean={metrics['degraded_mean']:.1f}, std={metrics['degraded_std']:.1f}, text={metrics['degraded_text_ratio']*100:.1f}%")
                print(f"  Pseudo-GT: mean={metrics['pseudo_gt_mean']:.1f}, std={metrics['pseudo_gt_std']:.1f}, text={metrics['pseudo_gt_text_ratio']*100:.1f}%")
                print(f"  SSIM: {metrics['ssim']:.3f}")
                print(f"  Contrast improvement: {metrics['contrast_improvement']:.1f}")
                print()
        
        # Aggregate statistics
        if all_metrics:
            avg_ssim = np.mean([m['ssim'] for m in all_metrics])
            avg_text_preserved = np.mean([m['pseudo_gt_text_ratio'] / max(m['degraded_text_ratio'], 0.001) for m in all_metrics])
            avg_contrast_improvement = np.mean([m['contrast_improvement'] for m in all_metrics])
            
            print("="*80)
            print("AGGREGATE METRICS")
            print("="*80)
            print()
            print(f"Average SSIM: {avg_ssim:.3f}")
            print(f"Average text preservation ratio: {avg_text_preserved:.2f}x")
            print(f"Average contrast improvement: {avg_contrast_improvement:.1f}")
            print()
            
            # Quality assessment
            if avg_ssim > 0.7 and avg_text_preserved > 0.5:
                print("✅ QUALITY: GOOD - Suitable for fine-tuning")
            elif avg_ssim > 0.6:
                print("⚠️  QUALITY: ACCEPTABLE - May need filtering")
            else:
                print("❌ QUALITY: POOR - Review model/approach")
            print()
    
    # Save summary
    summary = {
        'checkpoint': str(checkpoint_path),
        'input_size': input_size,
        'batch_size': args.batch_size,
        'train_pseudo_gt_count': len(train_files) if train_degraded_dir.exists() else 0,
        'val_pseudo_gt_count': len(val_files) if val_degraded_dir.exists() else 0,
    }
    
    summary_path = output_dir / 'pseudo_gt_generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("="*80)
    print("✅ PSEUDO-GT GENERATION COMPLETE")
    print("="*80)
    print()
    print(f"Summary: {summary_path}")
    print()
    print("Next steps:")
    print("  1. Inspect sample pseudo-GT images")
    print("  2. Filter by quality metrics (SSIM, text preservation)")
    print("  3. Create TFRecord for fine-tuning")
    print()


if __name__ == '__main__':
    main()
