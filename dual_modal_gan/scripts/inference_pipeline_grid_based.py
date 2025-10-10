#!/usr/bin/env python3
"""
GRID-BASED DOCUMENT RESTORATION PIPELINE
========================================

Approach: Split document into uniform 1024×128 tiles → GAN restoration → Blend

Advantages:
- ✅ Native GAN resolution (no resize artifacts)
- ✅ Simple code (~300 lines vs 1098 lines)
- ✅ Uniform tile size (consistent quality)
- ✅ No detection dependency (works on any document)
- ✅ Reproducible (fixed grid coordinates)
- ✅ Smooth blending with overlap

Disadvantages:
- ❌ Slower (process entire document area)
- ❌ Process margins (wasted compute on empty areas)

Author: AI/ML Research Team
Date: October 7, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import json
from tqdm import tqdm
import time
from typing import List, Tuple, Dict
from PIL import Image

# Hyperparameters
TILE_WIDTH = 1024
TILE_HEIGHT = 128
OVERLAP = 32  # Overlap pixels untuk smooth blending
IMG_WIDTH = 1024
IMG_HEIGHT = 128

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")


def create_blend_mask(width: int, height: int, overlap: int) -> np.ndarray:
    """
    Create blend mask untuk smooth tile blending dengan overlap.
    
    IMPORTANT: For grid-based tiling, we use UNIFORM weight (1.0) in the center
    and only fade at overlap regions. This prevents weight accumulation issues.
    
    Args:
        width: Tile width
        height: Tile height
        overlap: Overlap pixels
    
    Returns:
        mask: (H, W) float array [0, 1]
    """
    # Start with uniform weights (no fade)
    mask = np.ones((height, width), dtype=np.float32)
    
    # For grid-based approach, we don't need edge fading
    # The weighted average in overlap regions will handle blending
    # This keeps weights close to 1.0, avoiding the division problem
    
    return mask


def split_document_to_tiles(image: np.ndarray, 
                            tile_w: int = TILE_WIDTH, 
                            tile_h: int = TILE_HEIGHT,
                            overlap: int = OVERLAP) -> List[Tuple[np.ndarray, int, int, int, int]]:
    """
    Split document ke uniform tiles dengan overlap.
    
    Args:
        image: Document image (H, W) or (H, W, C)
        tile_w: Tile width (default 1024)
        tile_h: Tile height (default 128)
        overlap: Overlap pixels (default 32)
    
    Returns:
        tiles: List of (tile_img, x, y, w, h)
               tile_img: (tile_h, tile_w) grayscale
               x, y: Top-left coordinate dalam dokumen asli
               w, h: Tile dimensions
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = image.shape
    tiles = []
    
    # Calculate step size (tile size - overlap)
    step_h = tile_h - overlap
    step_w = tile_w - overlap
    
    # Sliding window dengan overlap
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Calculate tile boundaries
            y_end = min(y + tile_h, h)
            x_end = min(x + tile_w, w)
            
            # Extract tile
            tile = image[y:y_end, x:x_end].copy()
            
            # Pad jika tile lebih kecil dari target (edges)
            actual_h, actual_w = tile.shape
            if actual_h < tile_h or actual_w < tile_w:
                # Pad dengan white (255) untuk match document background
                padded = np.ones((tile_h, tile_w), dtype=np.uint8) * 255
                padded[:actual_h, :actual_w] = tile
                tile = padded
            
            tiles.append((tile, x, y, actual_w, actual_h))
            
            # Move to next column
            x += step_w
            if x >= w:
                break
        
        # Move to next row
        y += step_h
        if y >= h:
            break
    
    return tiles


def preprocess_tile_for_gan(tile: np.ndarray) -> np.ndarray:
    """
    Preprocess tile untuk GAN input.
    Tile sudah dalam ukuran 1024×128 (native GAN size).
    
    Args:
        tile: (H, W) grayscale image
    
    Returns:
        tensor: (1, W, H, 1) float32 [0, 1]
    """
    # Normalize ke [0, 1]
    normalized = tile.astype(np.float32) / 255.0
    
    # Transpose ke (W, H) untuk model
    transposed = np.transpose(normalized)
    
    # Add batch and channel dimension: (1, W, H, 1)
    tensor = transposed[np.newaxis, :, :, np.newaxis]
    
    return tensor


def postprocess_gan_output(generated: np.ndarray, 
                           original_h: int, 
                           original_w: int) -> np.ndarray:
    """
    Postprocess GAN output kembali ke tile format.
    
    Args:
        generated: (1, W, H, 1) float32 [0, 1]
        original_h: Target height
        original_w: Target width
    
    Returns:
        tile: (H, W) uint8 [0, 255]
    """
    # Remove batch & channel: (1, W, H, 1) → (W, H)
    squeezed = np.squeeze(generated, axis=(0, 3))
    
    # Transpose back: (W, H) → (H, W)
    transposed = np.transpose(squeezed)
    
    # Denormalize to [0, 255]
    denormalized = np.clip(transposed * 255.0, 0, 255).astype(np.uint8)
    
    # Crop to original tile size (remove padding if any)
    result = denormalized[:original_h, :original_w]
    
    return result


def restore_tiles_batch(tiles: List[np.ndarray], 
                        generator, 
                        batch_size: int = 8) -> List[np.ndarray]:
    """
    Restore multiple tiles dengan batch processing.
    
    Args:
        tiles: List of tile images (H, W)
        generator: GAN generator model
        batch_size: Batch size for processing
    
    Returns:
        restored_tiles: List of restored tile images (H, W)
    """
    restored = []
    
    # Preprocess all tiles
    preprocessed = [preprocess_tile_for_gan(tile) for tile in tiles]
    
    # Process in batches
    for i in tqdm(range(0, len(preprocessed), batch_size), desc="   Restoring tiles"):
        batch = preprocessed[i:i+batch_size]
        
        # Stack to single tensor
        batch_tensor = np.concatenate(batch, axis=0)  # (B, W, H, 1)
        
        # Run GAN
        generated_batch = generator(batch_tensor, training=False)
        
        # Postprocess each item in batch
        for j, generated in enumerate(generated_batch.numpy()):
            # Get original dimensions
            orig_tile = tiles[i + j]
            orig_h, orig_w = orig_tile.shape
            
            # Postprocess
            restored_tile = postprocess_gan_output(
                generated[np.newaxis, :, :, np.newaxis],
                orig_h,
                orig_w
            )
            restored.append(restored_tile)
    
    return restored


def reconstruct_from_tiles(tiles_info: List[Tuple[np.ndarray, int, int, int, int]],
                           doc_shape: Tuple[int, int],
                           overlap: int = OVERLAP) -> np.ndarray:
    """
    Reconstruct full document dari restored tiles dengan smooth blending.
    
    Args:
        tiles_info: List of (restored_tile, x, y, w, h)
        doc_shape: Original document shape (H, W)
        overlap: Overlap pixels untuk blending
    
    Returns:
        reconstructed: (H, W) uint8 document
    """
    h, w = doc_shape
    canvas = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    
    # Create blend mask once (reused for all tiles)
    blend_mask = create_blend_mask(TILE_WIDTH, TILE_HEIGHT, overlap)
    
    for tile, x, y, orig_w, orig_h in tiles_info:
        # Ensure tile is 2D (H, W)
        # Tile might be (1, H, W) from batch processing or (H, W, C) from color
        while len(tile.shape) > 2:
            if tile.shape[0] == 1:
                tile = tile[0]  # Remove batch dimension
            elif tile.shape[-1] == 1:
                tile = tile[:, :, 0]  # Remove channel dimension
            elif len(tile.shape) == 3:
                tile = tile[:, :, 0]  # Take first channel
            else:
                break
        
        # Validate tile shape
        if tile.shape[0] == 0 or tile.shape[1] == 0:
            continue  # Skip empty tiles
        
        # Get actual boundaries (considering document edges)
        y_end = min(y + orig_h, h)
        x_end = min(x + orig_w, w)
        
        actual_h = y_end - y
        actual_w = x_end - x
        
        # Skip if no overlap
        if actual_h <= 0 or actual_w <= 0:
            continue
        
        # Get corresponding mask region
        mask = blend_mask[:actual_h, :actual_w]
        
        # Ensure tile matches expected dimensions (H, W)
        tile_h, tile_w = tile.shape[:2]
        crop_h = min(actual_h, tile_h)
        crop_w = min(actual_w, tile_w)
        
        tile_region = tile[:crop_h, :crop_w].astype(np.float32)
        mask_region = mask[:crop_h, :crop_w]
        
        # Accumulate weighted tile
        canvas[y:y+crop_h, x:x+crop_w] += tile_region * mask_region
        weights[y:y+crop_h, x:x+crop_w] += mask_region
    
    # Normalize by weights (avoid division by zero)
    reconstructed = canvas / (weights + 1e-8)
    
    return reconstructed.astype(np.uint8)


def load_gan_model(checkpoint_path: str):
    """Load GAN generator model."""
    import sys
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from models.generator import unet
    
    # Build U-Net generator (grayscale: 1024×128×1)
    generator = unet(input_size=(IMG_WIDTH, IMG_HEIGHT, 1))
    
    # Load weights menggunakan tf.train.Checkpoint (TensorFlow native format)
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    # Handle checkpoint path
    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path.replace('.index', '').replace('.data-00000-of-00001', '')
    
    if checkpoint_path and os.path.exists(checkpoint_path + '.index'):
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
        print(f"  ✅ Generator loaded: {generator.count_params():,} parameters")
    else:
        raise ValueError(f"No checkpoint found at {checkpoint_path}")
    
    return generator


def process_document_grid_based(document_path: str,
                                output_dir: str,
                                generator,
                                batch_size: int = 8,
                                save_tiles: bool = False) -> Dict:
    """
    Process dokumen dengan grid-based approach.
    
    Args:
        document_path: Path ke document image
        output_dir: Output directory
        generator: GAN generator model
        batch_size: Batch size for GAN processing
        save_tiles: Save individual tiles (debug)
    
    Returns:
        result: Dict dengan statistics dan paths
    """
    doc_name = Path(document_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Load document
    print(f"\n📄 Processing: {doc_name}")
    image = cv2.imread(document_path)
    if image is None:
        raise ValueError(f"Failed to load image: {document_path}")
    
    orig_h, orig_w = image.shape[:2]
    print(f"   Original size: {orig_w}×{orig_h}")
    
    # Step 1: Split ke tiles
    print(f"   📐 Splitting to {TILE_WIDTH}×{TILE_HEIGHT} tiles (overlap={OVERLAP})...")
    tiles_info = split_document_to_tiles(image, TILE_WIDTH, TILE_HEIGHT, OVERLAP)
    num_tiles = len(tiles_info)
    print(f"   ✅ Created {num_tiles} tiles")
    
    # Calculate theoretical tiles (for comparison)
    step_h = TILE_HEIGHT - OVERLAP
    step_w = TILE_WIDTH - OVERLAP
    theo_rows = int(np.ceil(orig_h / step_h))
    theo_cols = int(np.ceil(orig_w / step_w))
    print(f"   📊 Grid: {theo_cols}×{theo_rows} tiles")
    
    # Step 2: GAN restoration
    print(f"   🎨 Restoring tiles with GAN (batch_size={batch_size})...")
    tiles_only = [tile for tile, _, _, _, _ in tiles_info]
    restored_tiles = restore_tiles_batch(tiles_only, generator, batch_size)
    print(f"   ✅ Restored {len(restored_tiles)} tiles")
    
    # DEBUG: Check restored tile statistics
    if len(restored_tiles) > 0:
        sample_stats = [f"mean={t.mean():.1f}" for t in restored_tiles[:5]]
        print(f"   📊 Sample tile stats: {', '.join(sample_stats)}")
    
    # Save individual tiles if requested
    if save_tiles:
        tiles_dir = output_path / "tiles_grid"
        tiles_dir.mkdir(exist_ok=True)
        
        for idx, (orig_tile, restored_tile) in enumerate(zip(tiles_only, restored_tiles)):
            cv2.imwrite(str(tiles_dir / f"tile_{idx:04d}_original.png"), orig_tile)
            cv2.imwrite(str(tiles_dir / f"tile_{idx:04d}_restored.png"), restored_tile)
        
        print(f"   💾 Saved tiles to: {tiles_dir}")
    
    # Step 3: Reconstruct document dengan blending
    print(f"   🔨 Reconstructing document with smooth blending...")
    restored_info = [(restored, x, y, w, h) 
                     for restored, (_, x, y, w, h) in zip(restored_tiles, tiles_info)]
    
    reconstructed = reconstruct_from_tiles(restored_info, (orig_h, orig_w), OVERLAP)
    print(f"   ✅ Reconstructed: {reconstructed.shape[1]}×{reconstructed.shape[0]}")
    print(f"   📊 Stats: min={reconstructed.min()}, max={reconstructed.max()}, mean={reconstructed.mean():.2f}")
    
    # Apply post-processing to reduce perceived pixelation
    # Fix: bilateral filter + reduced sharpness + DPI metadata
    print(f"   🔧 Post-processing (bilateral + reduced sharpness)...")
    
    # Step 1: Bilateral filter (edge-preserving smoothing)
    processed = cv2.bilateralFilter(reconstructed, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Step 2: Slightly reduce sharpness (counter GAN over-sharpening)
    blurred = cv2.GaussianBlur(processed, (3, 3), 0.5)
    processed = cv2.addWeighted(processed, 0.85, blurred, 0.15, 0).astype(np.uint8)
    
    # Save results with DPI metadata
    output_image_path = output_path / f"{doc_name}_restored.png"
    pil_img = Image.fromarray(processed)
    pil_img.save(str(output_image_path), dpi=(300, 300))
    print(f"   💾 Saved (with DPI 300): {output_image_path}")
    
    # Save original for comparison (with DPI)
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_path = output_path / f"{doc_name}_original.png"
    pil_orig = Image.fromarray(original_gray)
    pil_orig.save(str(original_path), dpi=(300, 300))
    
    # Create side-by-side comparison (use processed version)
    comparison = create_comparison_image(original_gray, processed, doc_name)
    comparison_path = output_path / f"{doc_name}_comparison.png"
    pil_comp = Image.fromarray(comparison)
    pil_comp.save(str(comparison_path), dpi=(300, 300))
    print(f"   📊 Comparison (with DPI 300): {comparison_path}")
    
    elapsed = time.time() - start_time
    
    # Statistics
    result = {
        'document': doc_name,
        'original_size': (orig_w, orig_h),
        'tile_size': (TILE_WIDTH, TILE_HEIGHT),
        'overlap': OVERLAP,
        'num_tiles': num_tiles,
        'grid_layout': (theo_cols, theo_rows),
        'processing_time': elapsed,
        'time_per_tile': elapsed / num_tiles,
        'output_path': str(output_image_path)
    }
    
    # Save statistics
    stats_path = output_path / f"{doc_name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"   ⏱️  Processing time: {elapsed:.2f}s ({elapsed/num_tiles:.3f}s/tile)")
    
    return result


def create_comparison_image(original: np.ndarray, 
                           restored: np.ndarray,
                           title: str) -> np.ndarray:
    """
    Create side-by-side comparison image.
    
    Args:
        original: Original grayscale image
        restored: Restored grayscale image
        title: Title text
    
    Returns:
        comparison: Side-by-side comparison
    """
    h, w = original.shape
    
    # Convert to color untuk labels
    orig_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    rest_color = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
    
    # Gap separator
    gap = 30
    gap_img = np.ones((h, gap, 3), dtype=np.uint8) * 200
    
    # Stack horizontally
    comparison = np.hstack([orig_color, gap_img, rest_color])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    cv2.putText(comparison, "ORIGINAL (Degraded)", (20, 50),
               font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    cv2.putText(comparison, "RESTORED (Grid-Based GAN)", (w + gap + 20, 50),
               font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # Add title at bottom
    cv2.putText(comparison, f"Document: {title}", (20, h - 20),
               font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return comparison


def main():
    """Main function untuk testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grid-Based Document Restoration")
    parser.add_argument('--input', type=str, required=True, help='Input document path')
    parser.add_argument('--output_dir', type=str, default='outputs/grid_based',
                       help='Output directory')
    parser.add_argument('--generator', type=str,
                       default='dual_modal_gan/outputs/checkpoints_fp32_smoke_test/best_model-12',
                       help='Generator checkpoint path')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for GAN processing')
    parser.add_argument('--save_tiles', action='store_true',
                       help='Save individual tiles (debug mode)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GRID-BASED DOCUMENT RESTORATION PIPELINE")
    print("=" * 80)
    print(f"Tile size: {TILE_WIDTH}×{TILE_HEIGHT}")
    print(f"Overlap: {OVERLAP}px")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    
    # Load GAN model
    print("\n[1/2] Loading GAN generator...")
    generator = load_gan_model(args.generator)
    
    # Process document
    print("\n[2/2] Processing document...")
    result = process_document_grid_based(
        args.input,
        args.output_dir,
        generator,
        batch_size=args.batch_size,
        save_tiles=args.save_tiles
    )
    
    print("\n" + "=" * 80)
    print("✅ GRID-BASED RESTORATION COMPLETE")
    print("=" * 80)
    print(f"📄 Document: {result['document']}")
    print(f"📐 Original: {result['original_size'][0]}×{result['original_size'][1]}")
    print(f"🎯 Tiles: {result['num_tiles']} ({result['grid_layout'][0]}×{result['grid_layout'][1]})")
    print(f"⏱️  Time: {result['processing_time']:.2f}s ({result['time_per_tile']:.3f}s/tile)")
    print(f"💾 Output: {result['output_path']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
