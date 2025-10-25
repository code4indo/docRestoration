#!/usr/bin/env python3
"""
Evaluate Grid-Based Restoration on DIBCO Dataset
=================================================
Objective evaluation with ground truth comparison.

Metrics:
- Stroke width preservation
- Fragmentation reduction
- Contrast recovery
- PSNR, SSIM (if applicable)
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from inference pipeline
import tensorflow as tf
from network.unet import unet_enhanced


def analyze_stroke_metrics(binary_image: np.ndarray) -> dict:
    """Calculate stroke quality metrics from binary image"""
    # Invert for text analysis (text = white on black)
    binary_inv = 255 - binary_image
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_inv, connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
    
    # Stroke width using distance transform
    dist = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 3)
    stroke_widths = dist[dist > 0]
    
    # Edge detection for sharpness
    edges = cv2.Canny(binary_image, 50, 150)
    edge_count = np.sum(edges > 0)
    
    return {
        'num_components': int(num_labels - 1),
        'small_fragments': int(np.sum(areas < 50)),
        'tiny_fragments': int(np.sum(areas < 10)),
        'avg_component_size': float(areas.mean()) if len(areas) > 0 else 0.0,
        'median_component_size': float(np.median(areas)) if len(areas) > 0 else 0.0,
        'mean_stroke_width': float(stroke_widths.mean()) if len(stroke_widths) > 0 else 0.0,
        'median_stroke_width': float(np.median(stroke_widths)) if len(stroke_widths) > 0 else 0.0,
        'edge_pixels': int(edge_count)
    }


def calculate_image_quality(image: np.ndarray) -> dict:
    """Calculate basic image quality metrics"""
    return {
        'mean': float(image.mean()),
        'std': float(image.std()),
        'contrast': float(image.std() / image.mean()) if image.mean() > 0 else 0.0,
        'min': int(image.min()),
        'max': int(image.max())
    }


def evaluate_document(degraded_path: str, 
                     restored_path: str,
                     gt_path: str = None) -> dict:
    """
    Evaluate restoration quality.
    
    Args:
        degraded_path: Path to degraded input
        restored_path: Path to restored output
        gt_path: Path to ground truth (optional)
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Load images
    degraded = cv2.imread(degraded_path, cv2.IMREAD_GRAYSCALE)
    restored = cv2.imread(restored_path, cv2.IMREAD_GRAYSCALE)
    
    # Binarize for stroke analysis
    _, deg_bin = cv2.threshold(degraded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, res_bin = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate metrics
    metrics = {
        'document': os.path.basename(degraded_path),
        'degraded_quality': calculate_image_quality(degraded),
        'restored_quality': calculate_image_quality(restored),
        'degraded_strokes': analyze_stroke_metrics(deg_bin),
        'restored_strokes': analyze_stroke_metrics(res_bin)
    }
    
    # Calculate improvements
    deg_stroke = metrics['degraded_strokes']
    res_stroke = metrics['restored_strokes']
    
    metrics['improvements'] = {
        'fragmentation_reduction': float(
            (deg_stroke['num_components'] - res_stroke['num_components']) / 
            deg_stroke['num_components'] * 100
        ) if deg_stroke['num_components'] > 0 else 0.0,
        'small_fragments_reduction': float(
            (deg_stroke['small_fragments'] - res_stroke['small_fragments']) / 
            deg_stroke['small_fragments'] * 100
        ) if deg_stroke['small_fragments'] > 0 else 0.0,
        'stroke_width_change': float(
            (res_stroke['mean_stroke_width'] - deg_stroke['mean_stroke_width']) /
            deg_stroke['mean_stroke_width'] * 100
        ) if deg_stroke['mean_stroke_width'] > 0 else 0.0,
        'contrast_change': float(
            (metrics['restored_quality']['contrast'] - metrics['degraded_quality']['contrast']) /
            metrics['degraded_quality']['contrast'] * 100
        ) if metrics['degraded_quality']['contrast'] > 0 else 0.0
    }
    
    # Ground truth comparison (if available)
    if gt_path and os.path.exists(gt_path):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure same size
        if gt.shape != restored.shape:
            gt = cv2.resize(gt, (restored.shape[1], restored.shape[0]))
        
        # Binarize ground truth
        _, gt_bin = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate similarity
        mse = np.mean((res_bin.astype(float) - gt_bin.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        # F-measure for binarization (precision, recall)
        tp = np.sum((res_bin == 0) & (gt_bin == 0))  # True positives (text pixels)
        fp = np.sum((res_bin == 0) & (gt_bin == 255))  # False positives
        fn = np.sum((res_bin == 255) & (gt_bin == 0))  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['ground_truth_comparison'] = {
            'psnr': float(psnr),
            'precision': float(precision),
            'recall': float(recall),
            'f_measure': float(f_measure)
        }
    
    return metrics


def load_generator(checkpoint_path: str) -> tf.keras.Model:
    """Load GAN generator model"""
    print(f"📦 Loading generator from: {checkpoint_path}")
    
    # Build model
    generator = unet_enhanced(input_size=(1024, 128, 1))
    
    # Handle checkpoint path
    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path.replace('.index', '').replace('.data-00000-of-00001', '')
    
    if not checkpoint_path:
        raise ValueError(f"No checkpoint found")
    
    # Load weights
    checkpoint = tf.train.Checkpoint(generator=generator)
    status = checkpoint.restore(checkpoint_path)
    status.expect_partial()
    
    print(f"✅ Generator loaded: {generator.count_params():,} parameters")
    return generator


def process_single_document(input_path: str,
                            output_dir: str,
                            generator: tf.keras.Model,
                            batch_size: int) -> dict:
    """Process single document with grid-based approach"""
    from dual_modal_gan.scripts.inference_pipeline_grid_based import (
        split_document_to_tiles, reconstruct_from_tiles, 
        preprocess_tile_for_gan, postprocess_gan_output,
        create_comparison_image, TILE_WIDTH, TILE_HEIGHT, OVERLAP
    )
    
    doc_name = Path(input_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Load and split
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    orig_h, orig_w = gray.shape
    
    tiles_info = split_document_to_tiles(gray, TILE_WIDTH, TILE_HEIGHT, OVERLAP)
    num_tiles = len(tiles_info)
    
    # Process tiles in batches
        try:
            # Run grid-based restoration
            result = process_single_document(
                input_path=str(img_path),
                output_dir=str(doc_output),
                generator=generator,
                batch_size=batch_size
            )info = [(restored, x, y, w, h) for restored, (_, x, y, w, h) in zip(restored_tiles, tiles_info)]
    reconstructed = reconstruct_from_tiles(restored_info, (orig_h, orig_w), OVERLAP)
    
    # Post-processing (V5: CLAHE + Closing + Unsharp)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(reconstructed)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    gaussian = cv2.GaussianBlur(closed, (3, 3), 1.0)
    unsharp_mask = cv2.addWeighted(closed, 1.5, gaussian, -0.5, 0)
    processed = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
    
    # Save
    output_image_path = output_path / f"{doc_name}_restored.png"
    cv2.imwrite(str(output_image_path), processed)
    
    elapsed = time.time() - start_time
    
    return {
        'document': doc_name,
        'num_tiles': num_tiles,
        'processing_time': elapsed,
        'output_path': str(output_image_path)
    }


def batch_evaluate_dibco(dibco_input_dir: str,
                         dibco_gt_dir: str,
                         output_base_dir: str,
                         checkpoint_path: str,
                         batch_size: int = 2) -> dict:
    """
    Batch process and evaluate DIBCO dataset.
    
    Args:
        dibco_input_dir: Directory with degraded images
        dibco_gt_dir: Directory with ground truth
        output_base_dir: Base output directory
        checkpoint_path: GAN model checkpoint
        batch_size: Batch size for processing
    
    Returns:
        summary: Summary statistics
    """
    input_path = Path(dibco_input_dir)
    gt_path = Path(dibco_gt_dir)
    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load generator once
    generator = load_generator(checkpoint_path)
    
    # Find all input images
    input_images = sorted(input_path.glob("*.bmp"))
    
    print("=" * 80)
    print("DIBCO DATASET EVALUATION - GRID-BASED RESTORATION (V5 Closing)")
    print("=" * 80)
    print(f"Input directory: {dibco_input_dir}")
    print(f"Ground truth: {dibco_gt_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Documents to process: {len(input_images)}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)
    
    all_metrics = []
    
    # Process each document
    for img_path in tqdm(input_images, desc="Processing DIBCO"):
        doc_name = img_path.stem
        
        # Find ground truth
        gt_file = gt_path / f"{doc_name}_gt.bmp"
        
        # Output directory for this document
        doc_output = output_path / doc_name
        doc_output.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📄 Processing: {doc_name}")
        
        try:
            # Run grid-based restoration
            result = process_document_grid_based(
                document_path=str(img_path),
                output_dir=str(doc_output),
                checkpoint_path=checkpoint_path,
                batch_size=batch_size,
                save_tiles=False  # Save space
            )
            
            # Evaluate results
            restored_path = doc_output / f"{doc_name}_restored.png"
            
            metrics = evaluate_document(
                degraded_path=str(img_path),
                restored_path=str(restored_path),
                gt_path=str(gt_file) if gt_file.exists() else None
            )
            
            metrics['processing_time'] = result['processing_time']
            all_metrics.append(metrics)
            
            # Print quick summary
            imp = metrics['improvements']
            print(f"   ✅ Fragmentation: {imp['fragmentation_reduction']:.1f}%")
            print(f"   ✅ Small fragments: {imp['small_fragments_reduction']:.1f}%")
            print(f"   ✅ Stroke width: {imp['stroke_width_change']:+.1f}%")
            
            if 'ground_truth_comparison' in metrics:
                gt_comp = metrics['ground_truth_comparison']
                print(f"   📊 F-measure: {gt_comp['f_measure']:.4f}")
                print(f"   📊 PSNR: {gt_comp['psnr']:.2f} dB")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            continue
    
    # Calculate summary statistics
    if all_metrics:
        summary = calculate_summary_statistics(all_metrics)
        
        # Save detailed results
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'per_document': all_metrics
            }, f, indent=2)
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print_summary(summary)
        print(f"\n💾 Detailed results saved: {results_file}")
        print("=" * 80)
        
        return summary
    
    return {}


def calculate_summary_statistics(all_metrics: list) -> dict:
    """Calculate aggregate statistics across all documents"""
    
    # Extract improvements
    frag_reductions = [m['improvements']['fragmentation_reduction'] for m in all_metrics]
    small_frag_reductions = [m['improvements']['small_fragments_reduction'] for m in all_metrics]
    stroke_changes = [m['improvements']['stroke_width_change'] for m in all_metrics]
    contrast_changes = [m['improvements']['contrast_change'] for m in all_metrics]
    
    summary = {
        'num_documents': len(all_metrics),
        'fragmentation_reduction': {
            'mean': float(np.mean(frag_reductions)),
            'median': float(np.median(frag_reductions)),
            'std': float(np.std(frag_reductions)),
            'min': float(np.min(frag_reductions)),
            'max': float(np.max(frag_reductions))
        },
        'small_fragments_reduction': {
            'mean': float(np.mean(small_frag_reductions)),
            'median': float(np.median(small_frag_reductions)),
            'std': float(np.std(small_frag_reductions))
        },
        'stroke_width_change': {
            'mean': float(np.mean(stroke_changes)),
            'median': float(np.median(stroke_changes)),
            'std': float(np.std(stroke_changes))
        },
        'contrast_change': {
            'mean': float(np.mean(contrast_changes)),
            'median': float(np.median(contrast_changes)),
            'std': float(np.std(contrast_changes))
        }
    }
    
    # Ground truth metrics (if available)
    gt_metrics = [m for m in all_metrics if 'ground_truth_comparison' in m]
    if gt_metrics:
        f_measures = [m['ground_truth_comparison']['f_measure'] for m in gt_metrics]
        psnrs = [m['ground_truth_comparison']['psnr'] for m in gt_metrics if m['ground_truth_comparison']['psnr'] != float('inf')]
        
        summary['ground_truth'] = {
            'f_measure': {
                'mean': float(np.mean(f_measures)),
                'median': float(np.median(f_measures)),
                'std': float(np.std(f_measures))
            }
        }
        
        if psnrs:
            summary['ground_truth']['psnr'] = {
                'mean': float(np.mean(psnrs)),
                'median': float(np.median(psnrs)),
                'std': float(np.std(psnrs))
            }
    
    return summary


def print_summary(summary: dict):
    """Pretty print summary statistics"""
    print(f"Documents processed: {summary['num_documents']}")
    print(f"\n📊 FRAGMENTATION REDUCTION:")
    print(f"   Mean: {summary['fragmentation_reduction']['mean']:.1f}%")
    print(f"   Median: {summary['fragmentation_reduction']['median']:.1f}%")
    print(f"   Range: [{summary['fragmentation_reduction']['min']:.1f}%, {summary['fragmentation_reduction']['max']:.1f}%]")
    
    print(f"\n📊 SMALL FRAGMENTS REDUCTION:")
    print(f"   Mean: {summary['small_fragments_reduction']['mean']:.1f}%")
    print(f"   Median: {summary['small_fragments_reduction']['median']:.1f}%")
    
    print(f"\n📏 STROKE WIDTH CHANGE:")
    print(f"   Mean: {summary['stroke_width_change']['mean']:+.1f}%")
    print(f"   Median: {summary['stroke_width_change']['median']:+.1f}%")
    
    print(f"\n🎨 CONTRAST CHANGE:")
    print(f"   Mean: {summary['contrast_change']['mean']:+.1f}%")
    print(f"   Median: {summary['contrast_change']['median']:+.1f}%")
    
    if 'ground_truth' in summary:
        print(f"\n✅ GROUND TRUTH COMPARISON:")
        print(f"   F-measure: {summary['ground_truth']['f_measure']['mean']:.4f} ± {summary['ground_truth']['f_measure']['std']:.4f}")
        if 'psnr' in summary['ground_truth']:
            print(f"   PSNR: {summary['ground_truth']['psnr']['mean']:.2f} ± {summary['ground_truth']['psnr']['std']:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Grid-Based Restoration on DIBCO")
    parser.add_argument("--input_dir", type=str, 
                       default="dibco_datasets/DIPCO2016_dataset",
                       help="DIBCO input directory")
    parser.add_argument("--gt_dir", type=str,
                       default="dibco_datasets/DIPCO2016_Dataset_GT",
                       help="DIBCO ground truth directory")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/dibco_evaluation_grid_v5",
                       help="Output directory")
    parser.add_argument("--checkpoint", type=str,
                       default="dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model/ckpt-88",
                       help="GAN checkpoint path")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    batch_evaluate_dibco(
        dibco_input_dir=args.input_dir,
        dibco_gt_dir=args.gt_dir,
        output_base_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
