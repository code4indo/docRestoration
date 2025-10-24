#!/usr/bin/env python3
"""
GRID-BASED DOCUMENT RESTORATION PIPELINE - PRODUCTION V3 COMPATIBLE
=====================================================================

Approach: Split document into uniform 1024×128 tiles → GAN restoration → Blend

Version: Updated for Production V3 Academic Split Model
Compatibility: Enhanced U-Net Generator with [-1, 1] normalization

Advantages:
- ✅ Native GAN resolution (no resize artifacts)
- ✅ Simple code (~350 lines)
- ✅ Uniform tile size (consistent quality)
- ✅ No detection dependency (works on any document)
- ✅ Reproducible (fixed grid coordinates)
- ✅ Smooth blending with gradient overlap

Updates from Original:
- ✅ Compatible dengan production_v3 model (unet_enhanced)
- ✅ Proper [-1, 1] normalization (match training)
- ✅ Gradient blend mask (smooth transitions)
- ✅ Correct transpose operations (H,W ↔ W,H)

Author: AI/ML Research Team
Date: October 23, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import time
from typing import List, Tuple, Dict
from PIL import Image
import json

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced

# Hyperparameters (MATCH TRAINING)
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


def create_blend_mask_gradient(width: int, height: int, overlap: int) -> np.ndarray:
    """
    Create gradient blend mask untuk smooth tile blending.
    
    IMPROVED: Uses gradient fade in overlap regions instead of uniform weights.
    This prevents visible seams while avoiding weight accumulation issues.
    
    Args:
        width: Tile width
        height: Tile height
        overlap: Overlap pixels
    
    Returns:
        mask: (H, W) float array [0, 1]
    """
    mask = np.ones((height, width), dtype=np.float32)
    
    if overlap <= 0:
        return mask
    
    # Create fade gradient [0 -> 1]
    fade = np.linspace(0, 1, overlap)
    
    # Horizontal fade (left edge)
    mask[:, :overlap] *= fade[np.newaxis, :]
    
    # Horizontal fade (right edge)
    mask[:, -overlap:] *= fade[::-1][np.newaxis, :]
    
    # Vertical fade (top edge)
    mask[:overlap, :] *= fade[:, np.newaxis]
    
    # Vertical fade (bottom edge)
    mask[-overlap:, :] *= fade[::-1][:, np.newaxis]
    
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
               w, h: Tile dimensions (actual size before padding)
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
            
            # Store actual dimensions before padding
            actual_h, actual_w = tile.shape
            
            # Pad jika tile lebih kecil dari target (edges)
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
    
    CRITICAL: Must match training preprocessing!
    
    Transformations:
    1. Ensure grayscale (already done)
    2. Normalize to [0, 1]
    3. Transpose (H, W) -> (W, H) for model
    4. Add channel dimension
    5. Normalize to [-1, 1] for tanh generator
    
    Args:
        tile: (H, W) grayscale image, range [0, 255]
    
    Returns:
        tensor: (1, W, H, 1) float32 [-1, 1]
    """
    # Normalize to [0, 1]
    normalized = tile.astype(np.float32) / 255.0
    
    # Transpose (H, W) -> (W, H)
    transposed = normalized.T  # (1024, 128)
    
    # Add channel dimension
    transposed = transposed[..., np.newaxis]  # (1024, 128, 1)
    
    # Normalize to [-1, 1] for tanh generator
    transposed = transposed * 2.0 - 1.0
    
    # Add batch dimension
    tensor = transposed[np.newaxis, ...]  # (1, 1024, 128, 1)
    
    return tensor


def postprocess_gan_output(generated: np.ndarray, 
                           original_h: int, 
                           original_w: int) -> np.ndarray:
    """
    Postprocess GAN output kembali ke tile format.
    
    CRITICAL: Inverse of preprocess_tile_for_gan!
    
    Transformations:
    1. Remove batch & channel dimensions
    2. Denormalize from [-1, 1] to [0, 255]
    3. Transpose (W, H) -> (H, W)
    4. Clip and convert to uint8
    5. Crop to original size
    
    Args:
        generated: (1, W, H, 1) float32 [-1, 1]
        original_h: Target height
        original_w: Target width
    
    Returns:
        tile: (H, W) uint8 [0, 255]
    """
    # Remove batch & channel: (1, W, H, 1) -> (W, H)
    squeezed = np.squeeze(generated, axis=(0, 3))
    
    # Denormalize from [-1, 1] to [0, 255]
    denormalized = (squeezed + 1.0) * 127.5
    
    # Transpose back: (W, H) -> (H, W)
    transposed = denormalized.T
    
    # Clip and convert
    result = np.clip(transposed, 0, 255).astype(np.uint8)
    
    # Crop to original tile size (remove padding if any)
    result = result[:original_h, :original_w]
    
    return result


def restore_tiles_batch(tiles: List[np.ndarray], 
                        generator: tf.keras.Model, 
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
    
    Uses weighted averaging in overlap regions with gradient masks.
    
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
    
    # Create gradient blend mask once (reused for all tiles)
    blend_mask = create_blend_mask_gradient(TILE_WIDTH, TILE_HEIGHT, overlap)
    
    for tile, x, y, orig_w, orig_h in tiles_info:
        # Ensure tile is 2D (H, W)
        if len(tile.shape) > 2:
            tile = tile[:, :, 0] if tile.shape[-1] == 1 else tile.squeeze()
        
        # Validate tile shape
        if tile.shape[0] == 0 or tile.shape[1] == 0:
            continue
        
        # Get actual boundaries
        y_end = min(y + orig_h, h)
        x_end = min(x + orig_w, w)
        
        actual_h = y_end - y
        actual_w = x_end - x
        
        if actual_h <= 0 or actual_w <= 0:
            continue
        
        # Get corresponding mask region
        mask = blend_mask[:actual_h, :actual_w]
        
        # Ensure tile matches expected dimensions
        tile_h, tile_w = tile.shape[:2]
        crop_h = min(actual_h, tile_h)
        crop_w = min(actual_w, tile_w)
        
        tile_region = tile[:crop_h, :crop_w].astype(np.float32)
        mask_region = mask[:crop_h, :crop_w]
        
        # Weighted average blending (original approach)
        # Accumulate weighted tile contributions
        canvas[y:y+crop_h, x:x+crop_w] += tile_region * mask_region
        weights[y:y+crop_h, x:x+crop_w] += mask_region
    
    # Normalize by weights (avoid division by zero)
    reconstructed = canvas / (weights + 1e-6)
    
    return reconstructed.astype(np.uint8)


def load_gan_model(checkpoint_path: str) -> tf.keras.Model:
    """
    Load GAN generator model (Production V3 Compatible).
    
    Args:
        checkpoint_path: Path to checkpoint (with or without .index)
    
    Returns:
        generator: Loaded Enhanced U-Net generator
    """
    print(f"  📦 Loading generator model...")
    
    # Build Enhanced U-Net generator
    generator = unet_enhanced(input_size=(IMG_WIDTH, IMG_HEIGHT, 1))
    
    # Handle checkpoint path
    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        # Remove extensions if present
        checkpoint_path = checkpoint_path.replace('.index', '').replace('.data-00000-of-00001', '')
    
    if not checkpoint_path:
        raise ValueError(f"No checkpoint found")
    
    # Load weights
    checkpoint = tf.train.Checkpoint(generator=generator)
    status = checkpoint.restore(checkpoint_path)
    status.expect_partial()
    
    print(f"  ✅ Generator loaded: {generator.count_params():,} parameters")
    print(f"     Model: Enhanced U-Net (ResBlocks + Attention)")
    print(f"     Input: ({IMG_WIDTH}, {IMG_HEIGHT}, 1)")
    print(f"     Normalization: [-1, 1] (tanh)")
    
    return generator


def process_document_grid_based(document_path: str,
                                output_dir: str,
                                generator: tf.keras.Model,
                                batch_size: int = 8,
                                save_tiles: bool = False,
                                no_postprocess: bool = False) -> Dict:
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
    
    # Calculate grid layout
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
        sample_stats = [f"mean={t.mean():.1f}" for t in restored_tiles[:3]]
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
    print(f"   🔨 Reconstructing document with gradient blending...")
    restored_info = [(restored, x, y, w, h) 
                     for restored, (_, x, y, w, h) in zip(restored_tiles, tiles_info)]
    
    reconstructed = reconstruct_from_tiles(restored_info, (orig_h, orig_w), OVERLAP)
    print(f"   ✅ Reconstructed: {reconstructed.shape[1]}×{reconstructed.shape[0]}")
    print(f"   📊 Stats: min={reconstructed.min()}, max={reconstructed.max()}, mean={reconstructed.mean():.2f}")
    
    # Post-processing (conditional based on flag)
    if no_postprocess:
        print(f"   ⚠️  Post-processing DISABLED (raw GAN output)")
        processed = reconstructed
    else:
        print(f"   🔧 Post-processing (quality enhancement)...")
        
        # Step 1: CLAHE for contrast enhancement (first)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(reconstructed)
        print(f"   ✅ CLAHE applied (contrast recovery)")
        
        # Step 2: Morphological closing to connect broken strokes
        # Closing = dilation + erosion (preserves size while filling gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
        print(f"   ✅ Morphological closing applied (connect broken strokes)")
        
        # Step 3: Unsharp masking for sharpness (preserve text detail)
        gaussian = cv2.GaussianBlur(closed, (3, 3), 1.0)
        unsharp_mask = cv2.addWeighted(closed, 1.5, gaussian, -0.5, 0)
        processed = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
        print(f"   ✅ Unsharp mask applied (detail enhancement)")
    
    # Optional: Mild bilateral filter (disabled by default)
    # processed = cv2.bilateralFilter(processed, d=3, sigmaColor=20, sigmaSpace=20)
    # print(f"   ✅ Bilateral filter applied (gentle smoothing)")
    
    # Save results with DPI metadata
    output_image_path = output_path / f"{doc_name}_restored.png"
    pil_img = Image.fromarray(processed)
    pil_img.save(str(output_image_path), dpi=(300, 300))
    print(f"   💾 Saved (DPI 300): {output_image_path}")
    
    # Save original for comparison
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_path = output_path / f"{doc_name}_original.png"
    pil_orig = Image.fromarray(original_gray)
    pil_orig.save(str(original_path), dpi=(300, 300))
    
    # Create side-by-side comparison
    comparison = create_comparison_image(original_gray, processed, doc_name)
    comparison_path = output_path / f"{doc_name}_comparison.png"
    pil_comp = Image.fromarray(comparison)
    pil_comp.save(str(comparison_path), dpi=(300, 300))
    print(f"   📊 Comparison saved: {comparison_path}")
    
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
    Create vertical comparison image (top-bottom).
    
    Args:
        original: Original grayscale image
        restored: Restored grayscale image
        title: Title text
    
    Returns:
        comparison: Vertical comparison (BGR)
    """
    h, w = original.shape
    
    # Convert to color untuk labels
    orig_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    rest_color = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
    
    # Add label bars (header untuk setiap section)
    label_height = 80
    label_bar_orig = np.ones((label_height, w, 3), dtype=np.uint8) * 50  # Dark gray
    label_bar_rest = np.ones((label_height, w, 3), dtype=np.uint8) * 50
    
    # Gap separator (horizontal bar)
    gap = 20
    gap_img = np.ones((gap, w, 3), dtype=np.uint8) * 200
    
    # Stack vertically with label bars
    comparison = np.vstack([
        label_bar_orig,
        orig_color,
        gap_img,
        label_bar_rest,
        rest_color
    ])
    
    # Add labels on the label bars
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Label for original (on dark bar)
    cv2.putText(comparison, "ORIGINAL (Degraded)", (20, 55),
               font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    # Label for restored (on dark bar below)
    cv2.putText(comparison, "RESTORED (Grid-Based GAN)", (20, label_height + h + gap + 55),
               font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # Add title at very bottom on white background
    footer_height = 60
    footer = np.ones((footer_height, w, 3), dtype=np.uint8) * 255
    comparison = np.vstack([comparison, footer])
    
    total_h = comparison.shape[0]
    cv2.putText(comparison, f"Document: {title}", (20, total_h - 20),
               font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    
    return comparison


def main():
    """Main function untuk testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Grid-Based Document Restoration (Production V3 - Quality Enhanced)")
    parser.add_argument('--input', type=str, required=True, help='Input document path')
    parser.add_argument('--output_dir', type=str, default='outputs/grid_based',
                       help='Output directory')
    parser.add_argument('--generator', type=str,
                       default='dual_modal_gan/checkpoints/production_v3_academic_split_70_15_15/best_model/ckpt-88',
                       help='Generator checkpoint path')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for GAN processing (default: 2 for RTX A4000)')
    parser.add_argument('--save_tiles', action='store_true',
                       help='Save individual tiles (debug mode)')
    parser.add_argument('--no-postprocess', action='store_true',
                       help='Disable post-processing (get raw GAN output)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GRID-BASED DOCUMENT RESTORATION PIPELINE (Production V3 - Quality Enhanced)")
    print("=" * 80)
    print(f"Tile size: {TILE_WIDTH}×{TILE_HEIGHT}")
    print(f"Overlap: {OVERLAP}px (gradient blending)")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: Enhanced U-Net (ResBlocks + Attention)")
    if args.no_postprocess:
        print(f"Post-processing: DISABLED (raw GAN output)")
    else:
        print(f"Post-processing: CLAHE + Morphological Closing + Unsharp Mask")
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
        save_tiles=args.save_tiles,
        no_postprocess=args.no_postprocess
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
