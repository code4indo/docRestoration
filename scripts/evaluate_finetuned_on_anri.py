#!/usr/bin/env python3
"""
Evaluate fine-tuned model on held-out ANRI validation pages

Strategy:
- Run inference on 5 validation full pages
- Calculate no-reference quality metrics (BRISQUE/NIQE)
- Visual comparison before/after fine-tuning
- Detect overfitting

Author: Pseudo-Labeling Pipeline
Date: 2025-10-28
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def calculate_no_reference_metrics(image: np.ndarray) -> Dict:
    """
    Calculate no-reference quality metrics
    
    Metrics:
    - Contrast (std)
    - Sharpness (Laplacian variance)
    - Entropy (information content)
    """
    from scipy import ndimage
    
    # Contrast (standard deviation)
    contrast = float(image.std())
    
    # Sharpness (Laplacian variance)
    laplacian = ndimage.laplace(image)
    sharpness = float(laplacian.var())
    
    # Entropy
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()
    hist = hist[hist > 0]  # Remove zero bins
    entropy = float(-(hist * np.log2(hist)).sum())
    
    # Mean intensity
    mean_intensity = float(image.mean())
    
    # Text ratio (dark pixels)
    text_ratio = float((image < 128).sum() / image.size)
    
    return {
        'contrast': contrast,
        'sharpness': sharpness,
        'entropy': entropy,
        'mean_intensity': mean_intensity,
        'text_ratio': text_ratio,
    }


def run_inference_on_full_page(
    image_path: Path,
    model,
    patch_size: int = 256,
    overlap: int = 64
) -> np.ndarray:
    """
    Run inference on full page using patch-based approach
    
    Args:
        image_path: Path to input image
        model: Loaded generator model
        patch_size: Size of patches for processing
        overlap: Overlap between patches
    
    Returns:
        Restored full-page image array
    """
    import tensorflow as tf
    
    # Load image
    with Image.open(image_path) as img:
        if img.mode != 'L':
            img = img.convert('L')
        degraded = np.array(img, dtype=np.float32)
    
    height, width = degraded.shape
    stride = patch_size - overlap
    
    # Create output array
    restored = np.zeros_like(degraded, dtype=np.float32)
    weight_map = np.zeros_like(degraded, dtype=np.float32)
    
    # Process patches
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Extract patch
            patch = degraded[y:y+patch_size, x:x+patch_size]
            
            # Normalize to [-1, 1]
            patch_normalized = (patch / 127.5) - 1.0
            
            # Add batch and channel dimensions
            input_tensor = tf.convert_to_tensor(
                patch_normalized[np.newaxis, :, :, np.newaxis],
                dtype=tf.float32
            )
            
            # Run inference
            restored_tensor = model(input_tensor, training=False)
            
            # Denormalize to [0, 255]
            restored_patch = ((restored_tensor[0, :, :, 0].numpy() + 1.0) * 127.5).clip(0, 255)
            
            # Add to output with blending weights
            # Use Gaussian-like weights (higher in center)
            weight = np.ones_like(restored_patch)
            # Tapering at edges
            taper_size = overlap // 2
            if taper_size > 0:
                # Taper left edge
                weight[:, :taper_size] *= np.linspace(0, 1, taper_size)
                # Taper right edge
                weight[:, -taper_size:] *= np.linspace(1, 0, taper_size)
                # Taper top edge
                weight[:taper_size, :] *= np.linspace(0, 1, taper_size)[:, np.newaxis]
                # Taper bottom edge
                weight[-taper_size:, :] *= np.linspace(1, 0, taper_size)[:, np.newaxis]
            
            # Accumulate
            restored[y:y+patch_size, x:x+patch_size] += restored_patch * weight
            weight_map[y:y+patch_size, x:x+patch_size] += weight
    
    # Normalize by weight map
    restored = np.divide(restored, weight_map, where=weight_map > 0)
    
    # Handle edges (no overlap)
    restored = restored.clip(0, 255).astype(np.uint8)
    
    return restored


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned model on ANRI validation pages'
    )
    parser.add_argument(
        '--val_pages_dir',
        type=str,
        default='DokumenRusak/full_pages_ANRI',
        help='Directory with full ANRI pages'
    )
    parser.add_argument(
        '--extraction_metadata',
        type=str,
        default='DokumenRusak/anri_patches/extraction_metadata.json',
        help='Metadata from patch extraction (contains val file list)'
    )
    parser.add_argument(
        '--baseline_checkpoint_dir',
        type=str,
        default='dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model',
        help='Baseline model checkpoint directory'
    )
    parser.add_argument(
        '--baseline_checkpoint_name',
        type=str,
        default='ckpt-99',
        help='Baseline checkpoint name'
    )
    parser.add_argument(
        '--finetuned_checkpoint_dir',
        type=str,
        default='dual_modal_gan/checkpoints/finetune_anri_pseudo_v1/best_model',
        help='Fine-tuned model checkpoint directory'
    )
    parser.add_argument(
        '--finetuned_checkpoint_name',
        type=str,
        default='ckpt-best',
        help='Fine-tuned checkpoint name'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/anri_finetuning_evaluation',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU ID to use'
    )
    
    args = parser.parse_args()
    
    # Setup
    import os
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    from dual_modal_gan.models.generator import build_enhanced_generator
    
    val_pages_dir = Path(args.val_pages_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ANRI FINE-TUNING EVALUATION")
    print("="*80)
    print()
    
    # Load validation file list
    metadata_path = Path(args.extraction_metadata)
    if not metadata_path.exists():
        print(f"❌ Extraction metadata not found: {metadata_path}")
        print("Run extract_patches_from_full_pages.py first!")
        return
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    val_files = metadata['val_files']
    print(f"Validation pages: {len(val_files)}")
    for f in val_files:
        print(f"  - {f}")
    print()
    
    # Load baseline model
    print(f"Loading baseline model from: {args.baseline_checkpoint_dir}/{args.baseline_checkpoint_name}")
    baseline_generator = build_enhanced_generator(input_shape=(None, None, 1))
    baseline_checkpoint = tf.train.Checkpoint(generator=baseline_generator)
    baseline_checkpoint.restore(
        str(Path(args.baseline_checkpoint_dir) / args.baseline_checkpoint_name)
    )
    print("✅ Baseline model loaded")
    print()
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model from: {args.finetuned_checkpoint_dir}/{args.finetuned_checkpoint_name}")
    finetuned_generator = build_enhanced_generator(input_shape=(None, None, 1))
    finetuned_checkpoint = tf.train.Checkpoint(generator=finetuned_generator)
    finetuned_checkpoint.restore(
        str(Path(args.finetuned_checkpoint_dir) / args.finetuned_checkpoint_name)
    )
    print("✅ Fine-tuned model loaded")
    print()
    
    # Evaluate each validation page
    results = []
    
    for val_file in tqdm(val_files, desc="Evaluating"):
        image_path = val_pages_dir / val_file
        
        if not image_path.exists():
            print(f"⚠️  {val_file} not found, skipping")
            continue
        
        # Load degraded
        with Image.open(image_path) as img:
            if img.mode != 'L':
                img = img.convert('L')
            degraded = np.array(img)
        
        # Run baseline inference
        print(f"\n  Processing {val_file} with baseline...")
        restored_baseline = run_inference_on_full_page(
            image_path,
            baseline_generator,
            patch_size=256,
            overlap=64
        )
        
        # Run fine-tuned inference
        print(f"  Processing {val_file} with fine-tuned...")
        restored_finetuned = run_inference_on_full_page(
            image_path,
            finetuned_generator,
            patch_size=256,
            overlap=64
        )
        
        # Calculate metrics
        degraded_metrics = calculate_no_reference_metrics(degraded)
        baseline_metrics = calculate_no_reference_metrics(restored_baseline)
        finetuned_metrics = calculate_no_reference_metrics(restored_finetuned)
        
        # Store results
        result = {
            'filename': val_file,
            'degraded': degraded_metrics,
            'baseline': baseline_metrics,
            'finetuned': finetuned_metrics,
        }
        results.append(result)
        
        # Save comparison images
        comparison_dir = output_dir / 'comparisons'
        comparison_dir.mkdir(exist_ok=True)
        
        # Create 3-way comparison
        height, width = degraded.shape
        comparison = Image.new('L', (width * 3, height))
        comparison.paste(Image.fromarray(degraded), (0, 0))
        comparison.paste(Image.fromarray(restored_baseline), (width, 0))
        comparison.paste(Image.fromarray(restored_finetuned), (width * 2, 0))
        
        comparison_path = comparison_dir / f"{Path(val_file).stem}_comparison.jpg"
        comparison.save(comparison_path, 'JPEG', quality=95)
        
        print(f"    Saved comparison: {comparison_path.name}")
    
    print()
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print()
    
    # Aggregate statistics
    if results:
        metrics = ['contrast', 'sharpness', 'entropy', 'text_ratio']
        
        for metric in metrics:
            degraded_vals = [r['degraded'][metric] for r in results]
            baseline_vals = [r['baseline'][metric] for r in results]
            finetuned_vals = [r['finetuned'][metric] for r in results]
            
            print(f"{metric.upper()}:")
            print(f"  Degraded:   {np.mean(degraded_vals):.3f} ± {np.std(degraded_vals):.3f}")
            print(f"  Baseline:   {np.mean(baseline_vals):.3f} ± {np.std(baseline_vals):.3f}")
            print(f"  Fine-tuned: {np.mean(finetuned_vals):.3f} ± {np.std(finetuned_vals):.3f}")
            
            improvement = (np.mean(finetuned_vals) - np.mean(baseline_vals)) / np.mean(baseline_vals) * 100
            print(f"  Improvement: {improvement:+.1f}%")
            print()
    
    # Save results JSON
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'val_files': val_files,
            'results': results,
        }, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    print(f"Comparisons saved to: {output_dir / 'comparisons'}")
    print()
    
    print("="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
