#!/usr/bin/env python3
"""
Generate pseudo ground-truth labels for extracted patches using trained model

Strategy:
- Run inference on degraded patches with ckpt-99
- Save restored patches as pseudo-GT
- Calculate confidence metrics for quality filtering

Author: Pseudo-Labeling Pipeline
Date: 2025-10-28
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_model(checkpoint_dir: Path, checkpoint_name: str):
    """Load the trained restoration model"""
    # Add dual_modal_gan to path for imports (same pattern as working inference scripts)
    dual_modal_dir = project_root / "dual_modal_gan"
    sys.path.insert(0, str(dual_modal_dir))
    
    from src.models.generator_enhanced import unet_enhanced
    
    print(f"Loading model from: {checkpoint_dir / checkpoint_name}")
    
    # Build generator (None, None for variable size patches)
    generator = unet_enhanced(input_size=(None, None, 1))
    
    # Load weights
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint = tf.train.Checkpoint(generator=generator)
    status = checkpoint.restore(str(checkpoint_path))
    
    print("✅ Model loaded successfully")
    return generator


def process_patch(
    patch_path: Path,
    generator: tf.keras.Model,
    output_size: int = 256
) -> tuple:
    """
    Process a single patch through the model
    
    Returns:
        (restored_patch, confidence_metrics)
    """
    # Load and preprocess
    with Image.open(patch_path) as img:
        if img.mode != 'L':
            img = img.convert('L')
        
        degraded = np.array(img, dtype=np.float32)
    
    # Normalize to [-1, 1]
    degraded_normalized = (degraded / 127.5) - 1.0
    
    # Add batch and channel dimensions
    input_tensor = tf.convert_to_tensor(
        degraded_normalized[np.newaxis, :, :, np.newaxis],
        dtype=tf.float32
    )
    
    # Run inference
    restored_tensor = generator(input_tensor, training=False)
    
    # Denormalize to [0, 255]
    restored = ((restored_tensor[0, :, :, 0].numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Calculate confidence metrics
    confidence_metrics = calculate_confidence_metrics(degraded, restored)
    
    return restored, confidence_metrics


def calculate_confidence_metrics(degraded: np.ndarray, restored: np.ndarray) -> Dict:
    """
    Calculate confidence metrics for quality filtering
    
    Metrics:
    - Contrast improvement
    - Text preservation ratio
    - SSIM (similarity)
    - Standard deviation (not over-cleaned)
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Normalize to [0, 1] for SSIM
    deg_norm = degraded / 255.0
    res_norm = restored / 255.0
    
    ssim_score = ssim(deg_norm, res_norm, data_range=1.0)
    
    # Text ratio (dark pixels)
    degraded_text_ratio = (degraded < 128).sum() / degraded.size
    restored_text_ratio = (restored < 128).sum() / restored.size
    text_preservation = restored_text_ratio / (degraded_text_ratio + 1e-8)
    
    # Contrast metrics
    degraded_std = degraded.std()
    restored_std = restored.std()
    contrast_improvement = restored_std / (degraded_std + 1e-8)
    
    # Intensity metrics
    degraded_mean = degraded.mean()
    restored_mean = restored.mean()
    
    return {
        'ssim': float(ssim_score),
        'text_preservation_ratio': float(text_preservation),
        'contrast_improvement': float(contrast_improvement),
        'degraded_std': float(degraded_std),
        'restored_std': float(restored_std),
        'degraded_mean': float(degraded_mean),
        'restored_mean': float(restored_mean),
        'degraded_text_ratio': float(degraded_text_ratio),
        'restored_text_ratio': float(restored_text_ratio),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate pseudo-labels for extracted patches'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='DokumenRusak/anri_patches',
        help='Directory containing extracted patches'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Directory containing model checkpoint'
    )
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        default='ckpt-99',
        help='Checkpoint name (default: ckpt-99)'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU ID to use (default: 0)'
    )
    parser.add_argument(
        '--batch_process',
        action='store_true',
        help='Process in batches (faster but more memory)'
    )
    
    args = parser.parse_args()
    
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Setup paths
    input_dir = Path(args.input_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    train_degraded_dir = input_dir / 'train' / 'degraded'
    train_pseudo_gt_dir = input_dir / 'train' / 'pseudo_gt'
    val_degraded_dir = input_dir / 'val' / 'degraded'
    val_pseudo_gt_dir = input_dir / 'val' / 'pseudo_gt'
    
    train_pseudo_gt_dir.mkdir(parents=True, exist_ok=True)
    val_pseudo_gt_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PSEUDO-LABEL GENERATION FOR PATCHES")
    print("="*80)
    print()
    print(f"Input directory: {input_dir}")
    print(f"Checkpoint: {checkpoint_dir / args.checkpoint_name}")
    print(f"GPU: {args.gpu_id}")
    print()
    
    # Load model
    generator = load_model(checkpoint_dir, args.checkpoint_name)
    print()
    
    # Load extraction metadata
    metadata_path = input_dir / 'extraction_metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            extraction_meta = json.load(f)
        print(f"Loaded extraction metadata:")
        stats = extraction_meta['statistics']
        print(f"  Train patches: {stats['n_train_patches']}")
        print(f"  Val patches: {stats['n_val_patches']}")
        print()
    
    # Process training patches
    print("Processing training patches...")
    train_patches = sorted(list(train_degraded_dir.glob('*.jpg')))
    train_metrics = []
    
    for patch_path in tqdm(train_patches, desc="Train"):
        # Generate pseudo-GT
        restored, metrics = process_patch(patch_path, generator)
        
        # Save pseudo-GT
        output_path = train_pseudo_gt_dir / patch_path.name
        Image.fromarray(restored).save(output_path, 'JPEG', quality=95)
        
        # Store metrics
        metrics['patch_filename'] = patch_path.name
        train_metrics.append(metrics)
    
    print()
    
    # Process validation patches
    print("Processing validation patches...")
    val_patches = sorted(list(val_degraded_dir.glob('*.jpg')))
    val_metrics = []
    
    for patch_path in tqdm(val_patches, desc="Val"):
        # Generate pseudo-GT
        restored, metrics = process_patch(patch_path, generator)
        
        # Save pseudo-GT
        output_path = val_pseudo_gt_dir / patch_path.name
        Image.fromarray(restored).save(output_path, 'JPEG', quality=95)
        
        # Store metrics
        metrics['patch_filename'] = patch_path.name
        val_metrics.append(metrics)
    
    print()
    print("="*80)
    print("PSEUDO-LABEL GENERATION SUMMARY")
    print("="*80)
    print()
    print(f"Training patches processed: {len(train_metrics)}")
    print(f"Validation patches processed: {len(val_metrics)}")
    print()
    
    # Calculate aggregate statistics
    if train_metrics:
        train_ssim = [m['ssim'] for m in train_metrics]
        train_text_pres = [m['text_preservation_ratio'] for m in train_metrics]
        train_contrast = [m['contrast_improvement'] for m in train_metrics]
        
        print("Training pseudo-labels quality:")
        print(f"  SSIM: {np.mean(train_ssim):.3f} ± {np.std(train_ssim):.3f}")
        print(f"  Text preservation: {np.mean(train_text_pres):.3f} ± {np.std(train_text_pres):.3f}")
        print(f"  Contrast improvement: {np.mean(train_contrast):.3f} ± {np.std(train_contrast):.3f}")
        print()
    
    if val_metrics:
        val_ssim = [m['ssim'] for m in val_metrics]
        val_text_pres = [m['text_preservation_ratio'] for m in val_metrics]
        val_contrast = [m['contrast_improvement'] for m in val_metrics]
        
        print("Validation pseudo-labels quality:")
        print(f"  SSIM: {np.mean(val_ssim):.3f} ± {np.std(val_ssim):.3f}")
        print(f"  Text preservation: {np.mean(val_text_pres):.3f} ± {np.std(val_text_pres):.3f}")
        print(f"  Contrast improvement: {np.mean(val_contrast):.3f} ± {np.std(val_contrast):.3f}")
        print()
    
    # Save metrics
    metrics_output = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_statistics': {
            'ssim_mean': float(np.mean(train_ssim)) if train_metrics else 0,
            'ssim_std': float(np.std(train_ssim)) if train_metrics else 0,
            'text_preservation_mean': float(np.mean(train_text_pres)) if train_metrics else 0,
            'text_preservation_std': float(np.std(train_text_pres)) if train_metrics else 0,
            'contrast_improvement_mean': float(np.mean(train_contrast)) if train_metrics else 0,
            'contrast_improvement_std': float(np.std(train_contrast)) if train_metrics else 0,
        },
        'val_statistics': {
            'ssim_mean': float(np.mean(val_ssim)) if val_metrics else 0,
            'ssim_std': float(np.std(val_ssim)) if val_metrics else 0,
            'text_preservation_mean': float(np.mean(val_text_pres)) if val_metrics else 0,
            'text_preservation_std': float(np.std(val_text_pres)) if val_metrics else 0,
            'contrast_improvement_mean': float(np.mean(val_contrast)) if val_metrics else 0,
            'contrast_improvement_std': float(np.std(val_contrast)) if val_metrics else 0,
        }
    }
    
    metrics_path = input_dir / 'pseudo_label_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")
    print()
    
    print("="*80)
    print("✅ PSEUDO-LABEL GENERATION COMPLETE")
    print("="*80)
    print()
    print("Next step:")
    print("  python scripts/filter_pseudo_patches.py \\")
    print(f"    --input_dir {input_dir}")
    print()


if __name__ == '__main__':
    main()
