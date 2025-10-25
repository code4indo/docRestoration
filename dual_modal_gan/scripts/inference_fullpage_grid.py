#!/usr/bin/env python3
"""
FULL PAGE GRID-BASED DOCUMENT RESTORATION
=========================================

Processes ENTIRE image using overlapping grid tiles (NO line detection).
Ensures ALL parts of the page are restored, including top/bottom margins.

Author: Enhanced from production_v3
Date: 2024-10-24
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import json

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Constants
TARGET_TILE_SIZE = (1024, 128)  # W×H for model input
OVERLAP = 64  # Overlap between tiles for smooth blending


# ============================================================================
# GPU Configuration
# ============================================================================

def setup_gpu(gpu_id: int = 1):
    """Configure GPU for TensorFlow."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            logging.info(f"✓ Using GPU {gpu_id}: {gpus[gpu_id]}")
        except RuntimeError as e:
            logging.error(f"GPU setup failed: {e}")
            raise


# ============================================================================
# Model Loading
# ============================================================================

def load_generator(checkpoint_dir: str, checkpoint_name: str):
    """Load trained generator model."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.generator_enhanced import unet_enhanced
    
    logging.info(f"Loading generator from: {checkpoint_dir}/{checkpoint_name}")
    
    generator = unet_enhanced(input_size=(TARGET_TILE_SIZE[0], TARGET_TILE_SIZE[1], 1))
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(str(checkpoint_path)).expect_partial()
    
    logging.info(f"✓ Generator loaded successfully")
    return generator


# ============================================================================
# Image Preprocessing
# ============================================================================

def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """Preprocess tile for model input."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Transpose (H, W) -> (W, H)
    image = image.T
    
    # Add channel
    image = image[..., np.newaxis]
    
    return image


def postprocess_from_model(image: np.ndarray) -> np.ndarray:
    """Postprocess model output."""
    if len(image.shape) == 3:
        image = image[..., 0]
    
    # Transpose back (W, H) -> (H, W)
    image = image.T
    
    # Denormalize from [-1, 1] to [0, 255]
    image = ((image + 1.0) / 2.0 * 255.0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


# ============================================================================
# Full Page Grid Tiling
# ============================================================================

def extract_grid_tiles(image: np.ndarray, 
                       tile_size: Tuple[int, int] = TARGET_TILE_SIZE,
                       overlap: int = OVERLAP,
                       adaptive_width: bool = True) -> List[Dict]:
    """
    Extract overlapping grid tiles covering ENTIRE image.
    
    Args:
        adaptive_width: If True, adjust tile width to preserve aspect ratio
                       for narrow images (prevents extreme stretching)
    
    Returns tiles that cover every single pixel of the input image.
    """
    img_h, img_w = image.shape[:2]
    tile_w, tile_h = tile_size
    
    # ADAPTIVE: Adjust tile width for narrow images to preserve aspect ratio
    if adaptive_width:
        img_aspect = img_w / img_h
        tile_aspect = tile_w / tile_h  # 1024/128 = 8.0
        
        # If image is narrower than tile aspect ratio, reduce tile width
        if img_aspect < tile_aspect * 0.6:  # Image significantly narrower
            # Calculate better tile width to match image aspect
            adjusted_tile_w = min(tile_w, int(img_aspect * tile_h * 1.2))
            adjusted_tile_w = max(256, adjusted_tile_w)  # Minimum 256px width
            
            if adjusted_tile_w < tile_w:
                logging.info(f"  📐 ADAPTIVE: Reducing tile width {tile_w}→{adjusted_tile_w}px for aspect ratio {img_aspect:.2f}")
                tile_w = adjusted_tile_w
    
    tiles = []
    
    # Calculate stride (tile_size - overlap)
    stride_w = tile_w - overlap
    stride_h = tile_h - overlap
    
    # Calculate number of tiles needed
    n_tiles_w = max(1, int(np.ceil((img_w - overlap) / stride_w)))
    n_tiles_h = max(1, int(np.ceil((img_h - overlap) / stride_h)))
    
    logging.info(f"  Image size: {img_w}×{img_h} (aspect={img_w/img_h:.2f})")
    logging.info(f"  Grid: {n_tiles_w}×{n_tiles_h} tiles ({tile_w}×{tile_h}, stride={stride_w}×{stride_h}, overlap={overlap})")
    
    tile_id = 0
    for row in range(n_tiles_h):
        for col in range(n_tiles_w):
            # Calculate tile position
            y_start = row * stride_h
            x_start = col * stride_w
            
            # Ensure we don't go beyond image bounds
            y_end = min(y_start + tile_h, img_h)
            x_end = min(x_start + tile_w, img_w)
            
            # Adjust start if we're at the edge
            if y_end == img_h:
                y_start = max(0, img_h - tile_h)
            if x_end == img_w:
                x_start = max(0, img_w - tile_w)
            
            y_end = min(y_start + tile_h, img_h)
            x_end = min(x_start + tile_w, img_w)
            
            # Extract tile
            tile = image[y_start:y_end, x_start:x_end]
            
            # Pad if needed (edge tiles might be smaller)
            if tile.shape[0] < tile_h or tile.shape[1] < tile_w:
                padded = np.ones((tile_h, tile_w), dtype=np.uint8) * 255
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            tiles.append({
                'id': tile_id,
                'image': tile,
                'position': (x_start, y_start),
                'size': (x_end - x_start, y_end - y_start),
                'bbox': (x_start, y_start, x_end, y_end)
            })
            
            tile_id += 1
    
    logging.info(f"  ✓ Extracted {len(tiles)} tiles covering entire page")
    return tiles


def merge_tiles_weighted(original_shape: Tuple[int, int],
                        tiles: List[Dict],
                        overlap: int = OVERLAP) -> np.ndarray:
    """
    Merge restored tiles with weighted blending in overlap regions.
    """
    img_h, img_w = original_shape
    
    # Accumulator for weighted averaging
    reconstructed = np.zeros((img_h, img_w), dtype=np.float32)
    weight_map = np.zeros((img_h, img_w), dtype=np.float32)
    
    for tile_info in tiles:
        restored_tile = tile_info['restored']
        x_start, y_start = tile_info['position']
        actual_w, actual_h = tile_info['size']
        
        # Use only the actual content (not padding)
        tile_content = restored_tile[:actual_h, :actual_w]
        
        # Create weight matrix (higher in center, lower at edges)
        weight = np.ones((actual_h, actual_w), dtype=np.float32)
        
        # Apply distance-based weights for smooth blending
        if overlap > 0:
            # Horizontal weights
            for x in range(min(overlap, actual_w)):
                weight[:, x] *= (x + 1) / (overlap + 1)
                if actual_w - x - 1 < actual_w:
                    weight[:, actual_w - x - 1] *= (x + 1) / (overlap + 1)
            
            # Vertical weights  
            for y in range(min(overlap, actual_h)):
                weight[y, :] *= (y + 1) / (overlap + 1)
                if actual_h - y - 1 < actual_h:
                    weight[actual_h - y - 1, :] *= (y + 1) / (overlap + 1)
        
        # Add to accumulator
        y_end = min(y_start + actual_h, img_h)
        x_end = min(x_start + actual_w, img_w)
        
        reconstructed[y_start:y_end, x_start:x_end] += tile_content.astype(np.float32) * weight
        weight_map[y_start:y_end, x_start:x_end] += weight
    
    # Normalize by weights
    reconstructed = np.divide(reconstructed, weight_map, 
                             where=weight_map > 0,
                             out=np.ones_like(reconstructed) * 255)
    
    return reconstructed.astype(np.uint8)


# ============================================================================
# Document Processing
# ============================================================================

def process_document(image_path: Path,
                    output_dir: Path,
                    generator,
                    tile_size: Tuple[int, int] = TARGET_TILE_SIZE,
                    overlap: int = OVERLAP,
                    blend_alpha: float = 0.0,
                    batch_size: int = 4,
                    save_tiles: bool = False):
    """Process entire document using grid-based tiling."""
    image_name = image_path.stem
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Processing: {image_name}")
    logging.info(f"{'='*70}")
    
    # Load document
    document = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if document is None:
        raise ValueError(f"Failed to load: {image_path}")
    
    doc_h, doc_w = document.shape
    logging.info(f"  Document size: {doc_w}×{doc_h}")
    
    # Extract tiles
    tiles = extract_grid_tiles(document, tile_size, overlap, adaptive_width=True)
    
    # Process tiles in batches
    logging.info(f"  Processing {len(tiles)} tiles in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(tiles), batch_size), desc="  Batches"):
        batch_tiles = tiles[i:i+batch_size]
        
        # Prepare batch
        batch_input = []
        for tile_info in batch_tiles:
            # Resize to exact target size
            tile_img = tile_info['image']
            if tile_img.shape != (tile_size[1], tile_size[0]):
                tile_img = cv2.resize(tile_img, tile_size, interpolation=cv2.INTER_CUBIC)
            
            preprocessed = preprocess_for_model(tile_img)
            batch_input.append(preprocessed)
        
        batch_input = np.array(batch_input)
        
        # Run inference
        restored_batch = generator(batch_input, training=False)
        
        # Postprocess and store
        for j, tile_info in enumerate(batch_tiles):
            restored = postprocess_from_model(restored_batch.numpy()[j])
            
            # Resize back to original tile size
            actual_w, actual_h = tile_info['size']
            if restored.shape != (actual_h, actual_w):
                restored = cv2.resize(restored, (actual_w, actual_h), 
                                    interpolation=cv2.INTER_CUBIC)
            
            tile_info['restored'] = restored
            
            # Save debug tiles if requested
            if save_tiles:
                tiles_dir = output_dir / f"{image_name}_tiles"
                tiles_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(tiles_dir / f"tile_{tile_info['id']:03d}_input.png"), 
                           tile_info['image'])
                cv2.imwrite(str(tiles_dir / f"tile_{tile_info['id']:03d}_restored.png"), 
                           restored)
    
    # Merge tiles
    logging.info(f"  Merging tiles with weighted blending...")
    reconstructed = merge_tiles_weighted((doc_h, doc_w), tiles, overlap)
    
    # Optional blending with original
    if blend_alpha > 0:
        logging.info(f"  Applying alpha blending: {blend_alpha:.2f}")
        reconstructed = cv2.addWeighted(
            reconstructed.astype(np.float32), 1.0 - blend_alpha,
            document.astype(np.float32), blend_alpha,
            0
        ).astype(np.uint8)
    
    # Save result
    output_path = output_dir / f"{image_name}_restored.png"
    cv2.imwrite(str(output_path), reconstructed)
    logging.info(f"  ✓ Saved: {output_path.name}")
    
    return {
        'image_name': image_name,
        'document_size': (doc_w, doc_h),
        'num_tiles': len(tiles),
        'tile_size': tile_size,
        'overlap': overlap,
        'output_path': str(output_path)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full-page grid-based restoration')
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--checkpoint_name', default='ckpt-88')
    parser.add_argument('--input', type=str, help='Single image path')
    parser.add_argument('--input_dir', type=str, help='Directory of images')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--tile_width', type=int, default=1024)
    parser.add_argument('--tile_height', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--blend_alpha', type=float, default=0.0,
                       help='Blending with original (0.0-0.3)')
    parser.add_argument('--save_tiles', action='store_true')
    parser.add_argument('--image_ext', type=str, default='.jpg')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input_dir required")
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f'inference_fullpage_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("FULL PAGE GRID-BASED DOCUMENT RESTORATION")
    logging.info("="*70)
    logging.info(f"Checkpoint: {args.checkpoint_dir}/{args.checkpoint_name}")
    logging.info(f"Tile size: {args.tile_width}×{args.tile_height}")
    logging.info(f"Overlap: {args.overlap}px")
    logging.info(f"Batch size: {args.batch_size}")
    if args.blend_alpha > 0:
        logging.info(f"Alpha blending: {args.blend_alpha}")
    logging.info(f"Output: {args.output_dir}")
    
    # Setup GPU
    setup_gpu(args.gpu_id)
    
    # Load model
    generator = load_generator(args.checkpoint_dir, args.checkpoint_name)
    
    # Get images
    if args.input:
        images = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        images = sorted(input_dir.glob(f'*{args.image_ext}'))
    
    logging.info(f"Found {len(images)} images")
    
    # Process all images
    tile_size = (args.tile_width, args.tile_height)
    results = []
    
    for img_path in images:
        try:
            result = process_document(
                img_path, output_dir, generator,
                tile_size=tile_size,
                overlap=args.overlap,
                blend_alpha=args.blend_alpha,
                batch_size=args.batch_size,
                save_tiles=args.save_tiles
            )
            results.append(result)
        except Exception as e:
            logging.error(f"Failed to process {img_path.name}: {e}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': f"{args.checkpoint_dir}/{args.checkpoint_name}",
        'tile_size': tile_size,
        'overlap': args.overlap,
        'blend_alpha': args.blend_alpha,
        'num_documents': len(images),
        'processed': len(results),
        'results': results
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"✓ COMPLETED: {len(results)}/{len(images)} documents")
    logging.info(f"✓ Summary: {summary_path}")
    logging.info(f"{'='*70}")


if __name__ == '__main__':
    main()
