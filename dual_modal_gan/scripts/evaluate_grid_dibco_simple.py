#!/usr/bin/env python3
"""
Simple DIBCO Evaluation for Grid-Based Restoration
===================================================
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Import GAN dependencies
sys.path.append(str(Path(__file__).parent.parent.parent))
import tensorflow as tf
from dual_modal_gan.src.models.generator_enhanced import unet_enhanced


def load_generator(checkpoint_path):
    """Load GAN generator"""
    generator = unet_enhanced(input_size=(1024, 128, 1))
    checkpoint_path = checkpoint_path.replace('.index', '').replace('.data-00000-of-00001', '')
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_path).expect_partial()
    return generator


def analyze_strokes(binary_img):
    """Analyze stroke metrics"""
    binary_inv = 255 - binary_img
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    dist = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 3)
    widths = dist[dist > 0]
    
    return {
        'components': int(num_labels - 1),
        'small_frags': int(np.sum(areas < 50)),
        'mean_width': float(widths.mean()) if len(widths) > 0 else 0.0
    }


def process_document(img_path, generator, output_dir):
    """Process one document"""
    # Import processing functions
    from inference_pipeline_grid_based import (
        split_document_to_tiles, reconstruct_from_tiles,
        preprocess_tile_for_gan, postprocess_gan_output,
        TILE_WIDTH, TILE_HEIGHT, OVERLAP
    )
    
    # Load
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    
    # Split
    tiles_info = split_document_to_tiles(gray, TILE_WIDTH, TILE_HEIGHT, OVERLAP)
    
    # Process tiles
    restored_tiles = []
    for tile, x, y, tw, th in tqdm(tiles_info, desc="Tiles", leave=False):
        preprocessed = preprocess_tile_for_gan(tile)
        output = generator(np.expand_dims(preprocessed, 0), training=False)
        restored = postprocess_gan_output(output[0].numpy())
        restored_tiles.append((restored, x, y, tw, th))
    
    # Reconstruct
    reconstructed = reconstruct_from_tiles(restored_tiles, (h, w), OVERLAP)
    
    # Post-processing V5
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(reconstructed)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    gaussian = cv2.GaussianBlur(closed, (3, 3), 1.0)
    unsharp = cv2.addWeighted(closed, 1.5, gaussian, -0.5, 0)
    processed = np.clip(unsharp, 0, 255).astype(np.uint8)
    
    # Save
    doc_name = Path(img_path).stem
    output_path = Path(output_dir) / f"{doc_name}_restored.png"
    cv2.imwrite(str(output_path), processed)
    
    return str(output_path)


def main():
    checkpoint = "dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model/ckpt-88"
    input_dir = Path("dibco_datasets/DIPCO2016_dataset")
    gt_dir = Path("dibco_datasets/DIPCO2016_Dataset_GT")
    output_dir = Path("outputs/dibco_eval_grid_v5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading generator...")
    generator = load_generator(checkpoint)
    
    results = []
    
    for img_path in tqdm(sorted(input_dir.glob("*.bmp")), desc="DIBCO"):
        doc_name = img_path.stem
        gt_path = gt_dir / f"{doc_name}_gt.bmp"
        
        print(f"\nProcessing: {doc_name}")
        
        # Process
        restored_path = process_document(img_path, generator, output_dir)
        
        # Evaluate
        degraded = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        restored = cv2.imread(restored_path, cv2.IMREAD_GRAYSCALE)
        
        _, deg_bin = cv2.threshold(degraded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, res_bin = cv2.threshold(restored, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        deg_metrics = analyze_strokes(deg_bin)
        res_metrics = analyze_strokes(res_bin)
        
        frag_reduction = (deg_metrics['components'] - res_metrics['components']) / deg_metrics['components'] * 100
        small_reduction = (deg_metrics['small_frags'] - res_metrics['small_frags']) / max(deg_metrics['small_frags'], 1) * 100
        width_change = (res_metrics['mean_width'] - deg_metrics['mean_width']) / deg_metrics['mean_width'] * 100
        
        result = {
            'document': doc_name,
            'degraded': deg_metrics,
            'restored': res_metrics,
            'frag_reduction_%': frag_reduction,
            'small_frag_reduction_%': small_reduction,
            'width_change_%': width_change
        }
        
        results.append(result)
        
        print(f"  Fragmentation: {frag_reduction:.1f}%")
        print(f"  Small frags: {small_reduction:.1f}%")
        print(f"  Width: {width_change:+.1f}%")
    
    # Summary
    frag_reductions = [r['frag_reduction_%'] for r in results]
    small_reductions = [r['small_frag_reduction_%'] for r in results]
    width_changes = [r['width_change_%'] for r in results]
    
    summary = {
        'num_documents': len(results),
        'frag_reduction': {'mean': np.mean(frag_reductions), 'std': np.std(frag_reductions)},
        'small_frag_reduction': {'mean': np.mean(small_reductions), 'std': np.std(small_reductions)},
        'width_change': {'mean': np.mean(width_changes), 'std': np.std(width_changes)}
    }
    
    # Save
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({'summary': summary, 'per_document': results}, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Documents: {summary['num_documents']}")
    print(f"Fragmentation reduction: {summary['frag_reduction']['mean']:.1f}% ± {summary['frag_reduction']['std']:.1f}%")
    print(f"Small fragments reduction: {summary['small_frag_reduction']['mean']:.1f}% ± {summary['small_frag_reduction']['std']:.1f}%")
    print(f"Stroke width change: {summary['width_change']['mean']:+.1f}% ± {summary['width_change']['std']:.1f}%")
    print(f"\nResults saved: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
