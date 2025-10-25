#!/usr/bin/env python3
"""
Line-Aware Document Restoration with HIGH-RESOLUTION PATCH PROCESSING

STRATEGY: Instead of downscaling entire line to 1024×128 (loses detail),
we split high-res lines into OVERLAPPING PATCHES, process each at model's
optimal resolution, then reconstruct with alpha blending.

KEY IMPROVEMENTS:
1. Preserves fine details by processing at native resolution
2. Overlapping patches ensure seamless reconstruction
3. Adaptive patching based on line width
4. Backward compatible - falls back to standard processing for normal res

Author: Enhanced from inference_line_aware.py
Date: 2025-10-23
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import json

# Disable XLA JIT
tf.config.optimizer.set_jit(False)

# Constants
TARGET_LINE_WIDTH = 1024
TARGET_LINE_HEIGHT = 128
HIGH_RES_THRESHOLD = 5000  # CRITICAL: Increased from 1536 - patch-based processing produces BLANK output!
                           # Issue: Model can't handle heavily stretched patches (e.g., 2259×38 → 7609×128)
                           # Solution: Use standard mode (resize whole line) for better quality

# ============================================================================
# GPU Configuration
# ============================================================================

def setup_gpu(gpu_id: int = 1):
    """Configure GPU for TensorFlow."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Use specified GPU
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
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.generator_enhanced import unet_enhanced
    
    logging.info(f"Loading generator from: {checkpoint_dir}/{checkpoint_name}")
    
    generator = unet_enhanced(input_size=(TARGET_LINE_WIDTH, TARGET_LINE_HEIGHT, 1))
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(str(checkpoint_path)).expect_partial()
    
    logging.info(f"✓ Generator loaded successfully")
    return generator


# ============================================================================
# Line Detection (same as before)
# ============================================================================

def detect_text_lines(image: np.ndarray, 
                      min_line_height: int = 20,
                      max_line_height: int = 600,
                      margin: int = 5,
                      fill_gaps: bool = True) -> List[Dict]:
    """
    Detect text lines using horizontal projection profile.
    
    CRITICAL: Increased max_line_height from 300 to 600 to handle documents
    with large text blocks (e.g., DIBCO 6.bmp has 471px region).
    
    Args:
        fill_gaps: If True, add synthetic lines to cover unprocessed regions
                   (top/bottom margins). Ensures 100% page coverage.
    """
    height, width = image.shape
    
    # Binarize
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal projection
    projection = np.sum(binary, axis=1)
    projection_norm = projection / np.max(projection + 1e-6)
    
    # Threshold
    threshold = 0.05
    in_line = projection_norm > threshold
    
    logging.info(f"  Using projection threshold: {threshold}")
    
    # Find contiguous regions
    lines = []
    y_start = None
    
    for i, is_text in enumerate(in_line):
        if is_text and y_start is None:
            y_start = i
        elif not is_text and y_start is not None:
            y_end = i
            line_height = y_end - y_start
            
            if min_line_height <= line_height <= max_line_height:
                y_start_margin = max(0, y_start - margin)
                y_end_margin = min(height, y_end + margin)
                line_img = image[y_start_margin:y_end_margin, :]
                
                # Warn for very large lines
                if line_height > 400:
                    logging.warning(f"  Large text block detected: {line_height}px height (y={y_start}-{y_end})")
                
                lines.append({
                    'y': y_start_margin,
                    'height': y_end_margin - y_start_margin,
                    'bbox': (0, y_start_margin, width, y_end_margin),
                    'image': line_img
                })
            
            y_start = None
    
    # Handle last line
    if y_start is not None:
        y_end = height
        line_height = y_end - y_start
        if min_line_height <= line_height <= max_line_height:
            y_start_margin = max(0, y_start - margin)
            line_img = image[y_start_margin:, :]
            
            # Warn for very large lines
            if line_height > 400:
                logging.warning(f"  Large text block detected: {line_height}px height (y={y_start}-{y_end})")
            
            lines.append({
                'y': y_start_margin,
                'height': height - y_start_margin,
                'bbox': (0, y_start_margin, width, height),
                'image': line_img
            })
    
    logging.info(f"  Detected {len(lines)} text lines")
    
    # Fill gaps to ensure 100% coverage
    if fill_gaps and len(lines) > 0:
        # Sort lines by y position
        lines.sort(key=lambda x: x['y'])
        
        gap_lines = []
        
        # Check top gap (before first line)
        if lines[0]['y'] > margin * 2:
            gap_h = lines[0]['y']
            if gap_h >= min_line_height:
                logging.info(f"  Filling TOP gap: 0-{gap_h} ({gap_h}px)")
                gap_lines.append({
                    'y': 0,
                    'height': gap_h,
                    'bbox': (0, 0, width, gap_h),
                    'image': image[0:gap_h, :],
                    'is_gap_filler': True
                })
        
        # Check bottom gap (after last line)
        last_y_end = lines[-1]['y'] + lines[-1]['height']
        if last_y_end < height - margin * 2:
            gap_h = height - last_y_end
            if gap_h >= min_line_height:
                logging.info(f"  Filling BOTTOM gap: {last_y_end}-{height} ({gap_h}px)")
                gap_lines.append({
                    'y': last_y_end,
                    'height': gap_h,
                    'bbox': (0, last_y_end, width, height),
                    'image': image[last_y_end:, :],
                    'is_gap_filler': True
                })
        
        # Add gap fillers to lines
        lines.extend(gap_lines)
        lines.sort(key=lambda x: x['y'])
        
        if gap_lines:
            logging.info(f"  ✓ Added {len(gap_lines)} gap-filler regions for 100% coverage")
    
    return lines


# ============================================================================
# HIGH-RESOLUTION PATCH-BASED PROCESSING (NEW!)
# ============================================================================

def should_use_patch_processing(line_width: int) -> bool:
    """Determine if line needs patch-based processing."""
    return line_width > HIGH_RES_THRESHOLD


def extract_line_patches(line_image: np.ndarray, 
                        patch_width: int = 1024,
                        patch_height: int = 128,
                        overlap: int = 128) -> List[Dict]:
    """
    Extract overlapping patches from high-resolution line.
    
    Strategy:
    - Resize height to 128 (preserve aspect ratio)
    - Split width into overlapping 1024px patches
    - Each patch processed at model's optimal resolution
    
    Args:
        line_image: High-res line (H, W)
        patch_width: Target patch width (1024)
        patch_height: Target patch height (128)
        overlap: Overlap between patches (pixels)
    
    Returns:
        List of patch dicts with metadata
    """
    h, w = line_image.shape[:2]
    
    # Step 1: Resize height to 128, preserve aspect
    aspect = w / h
    new_h = patch_height
    new_w = int(patch_height * aspect)
    
    line_resized = cv2.resize(line_image, (new_w, new_h), 
                             interpolation=cv2.INTER_CUBIC)
    
    logging.info(f"    High-res line: {w}×{h} → {new_w}×{new_h}")
    
    # Step 2: Extract overlapping patches
    patches = []
    stride = patch_width - overlap
    x = 0
    patch_idx = 0
    
    while x < new_w:
        x_end = min(x + patch_width, new_w)
        
        # Extract patch
        patch = line_resized[:, x:x_end]
        
        # Pad if last patch is smaller
        if patch.shape[1] < patch_width:
            padded = np.ones((patch_height, patch_width), dtype=np.uint8) * 255
            padded[:, :patch.shape[1]] = patch
            patch = padded
        
        patches.append({
            'patch': patch,
            'x': x,
            'width': x_end - x,
            'patch_idx': patch_idx
        })
        
        patch_idx += 1
        x += stride
        
        if x_end >= new_w:
            break
    
    logging.info(f"    Extracted {len(patches)} patches (overlap={overlap}px)")
    
    return patches, (new_w, new_h)


def merge_line_patches(patches: List[np.ndarray], 
                      patch_metadata: List[Dict],
                      target_size: Tuple[int, int]) -> np.ndarray:
    """
    Merge overlapping patches with alpha blending.
    
    Args:
        patches: List of restored patch images
        patch_metadata: Metadata for each patch
        target_size: (width, height) of merged result
    
    Returns:
        Merged line image
    """
    target_w, target_h = target_size
    merged = np.zeros((target_h, target_w), dtype=np.float32)
    weights = np.zeros((target_h, target_w), dtype=np.float32)
    
    overlap = 128  # Same as extraction
    
    for patch, meta in zip(patches, patch_metadata):
        x = meta['x']
        width = meta['width']
        
        # Create alpha mask for blending
        alpha = np.ones((target_h, width), dtype=np.float32)
        
        # Fade in/out at edges
        fade_width = min(64, overlap // 2)
        if x > 0:  # Not first patch - fade in left edge
            alpha[:, :fade_width] = np.linspace(0, 1, fade_width)
        if x + width < target_w:  # Not last patch - fade out right edge
            alpha[:, -fade_width:] = np.linspace(1, 0, fade_width)
        
        # Extract valid region from patch
        patch_region = patch[:, :width]
        
        # Accumulate
        merged[:, x:x+width] += patch_region * alpha
        weights[:, x:x+width] += alpha
    
    # Normalize
    merged = merged / np.maximum(weights, 1e-6)
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    
    return merged


# ============================================================================
# Line Preprocessing
# ============================================================================

def resize_line_to_optimal(line_image: np.ndarray, 
                          adaptive_width: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Resize line to model input size with aspect-preserved padding.
    
    Args:
        adaptive_width: If True, reduce target width for narrow lines
                       to prevent extreme aspect ratio distortion
    """
    h, w = line_image.shape[:2]
    
    aspect = w / h
    target_w = TARGET_LINE_WIDTH
    target_h = TARGET_LINE_HEIGHT
    target_aspect = target_w / target_h  # 1024/128 = 8.0
    
    # ADAPTIVE: For very narrow lines (aspect < 3), reduce target width
    if adaptive_width and aspect < 3.0:
        # Calculate better target width to reduce distortion
        # Aim for target_aspect around 4-6 instead of 8 for narrow lines
        adjusted_target_w = min(TARGET_LINE_WIDTH, int(aspect * TARGET_LINE_HEIGHT * 1.5))
        adjusted_target_w = max(256, adjusted_target_w)  # Minimum 256px
        
        if adjusted_target_w < TARGET_LINE_WIDTH:
            logging.debug(f"      Adaptive: reducing target {TARGET_LINE_WIDTH}→{adjusted_target_w}px for aspect {aspect:.2f}")
            target_w = adjusted_target_w
            target_aspect = target_w / target_h
    
    if aspect > target_aspect:
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        new_h = target_h
        new_w = int(target_h * aspect)
    
    resized = cv2.resize(line_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Create canvas with padding
    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    metadata = {
        'original_size': (w, h),
        'resized_size': (new_w, new_h),
        'offset': (x_offset, y_offset),
        'target_size': (target_w, target_h),
        'adaptive': adaptive_width and target_w < TARGET_LINE_WIDTH
    }
    
    return canvas, metadata


# ============================================================================
# Model Inference
# ============================================================================

def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """Preprocess for model input."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size (WITH aspect preserved padding - now done by preprocess!)
    if image.shape != (TARGET_LINE_HEIGHT, TARGET_LINE_WIDTH):
        # Aspect-preserved resize + white padding
        h, w = image.shape[:2]
        aspect = w / h
        target_aspect = TARGET_LINE_WIDTH / TARGET_LINE_HEIGHT
        
        if aspect > target_aspect:
            new_w = TARGET_LINE_WIDTH
            new_h = int(TARGET_LINE_WIDTH / aspect)
        else:
            new_h = TARGET_LINE_HEIGHT
            new_w = int(TARGET_LINE_HEIGHT * aspect)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # White canvas + center
        canvas = np.ones((TARGET_LINE_HEIGHT, TARGET_LINE_WIDTH), dtype=np.uint8) * 255
        y_offset = (TARGET_LINE_HEIGHT - new_h) // 2
        x_offset = (TARGET_LINE_WIDTH - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        image = canvas
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # CRITICAL FIX: Normalize to [-1, 1] to match TRAINING!
    # Training uses: degraded_images * 2.0 - 1.0 (see train_enhanced.py line 213)
    # Model expects tanh-normalized input range [-1, 1]
    image = image * 2.0 - 1.0  # [0, 1] → [-1, 1] (MATCH TRAINING!)
    
    # Transpose (H, W) -> (W, H)
    image = image.T
    
    # Add channel
    image = image[..., np.newaxis]
    
    return image


def postprocess_from_model(image: np.ndarray) -> np.ndarray:
    """Postprocess model output."""
    # Remove channel
    if len(image.shape) == 3:
        image = image[..., 0]
    
    # Transpose back (W, H) -> (H, W)
    image = image.T
    
    # Denormalize from [-1, 1] to [0, 255]
    image = ((image + 1.0) / 2.0 * 255.0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


def process_single_line_highres(line_img: np.ndarray, 
                                generator,
                                save_patches: bool = False,
                                patches_dir: Path = None,
                                line_idx: int = 0) -> Tuple[np.ndarray, Dict]:
    """
    Process high-resolution line with patch-based strategy.
    
    Returns:
        Tuple of (restored_line, metadata)
    """
    h, w = line_img.shape[:2]
    
    if should_use_patch_processing(w):
        logging.info(f"    Using HIGH-RES patch processing (width={w} > {HIGH_RES_THRESHOLD})")
        
        # Extract patches
        patches, target_size = extract_line_patches(line_img)
        
        # Process each patch
        restored_patches = []
        for patch_meta in patches:
            patch = patch_meta['patch']
            
            # Preprocess (no border - causes blank output)
            patch_input = preprocess_for_model(patch)
            
            # Inference
            restored_tensor = generator(patch_input[np.newaxis, ...], training=False)
            restored_patch = postprocess_from_model(restored_tensor.numpy()[0])
            
            restored_patches.append(restored_patch)
            
            # Save patches if requested
            if save_patches and patches_dir:
                patch_idx = patch_meta['patch_idx']
                cv2.imwrite(str(patches_dir / f'line_{line_idx:03d}_patch_{patch_idx:03d}_input.png'), patch)
                cv2.imwrite(str(patches_dir / f'line_{line_idx:03d}_patch_{patch_idx:03d}_restored.png'), restored_patch)
        
        # Merge patches
        restored_line = merge_line_patches(restored_patches, patches, target_size)
        
        # Resize back to original dimensions
        # target_size is (new_w, 128) after height resize
        # We need to resize back to original (w, h)
        restored_line = cv2.resize(restored_line, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        metadata = {
            'method': 'patch_based',
            'num_patches': len(patches),
            'original_size': (w, h),
            'intermediate_size': target_size
        }
        
    else:
        # Standard processing for normal resolution
        logging.info(f"    Using STANDARD processing (width={w} ≤ {HIGH_RES_THRESHOLD})")
        
        line_resized, resize_meta = resize_line_to_optimal(line_img)
        line_input = preprocess_for_model(line_resized)
        
        restored_tensor = generator(line_input[np.newaxis, ...], training=False)
        restored = postprocess_from_model(restored_tensor.numpy()[0])
        
        # Extract content and resize back
        x_off = resize_meta['offset'][0]
        y_off = resize_meta['offset'][1]
        w_resized, h_resized = resize_meta['resized_size']
        
        line_content = restored[y_off:y_off+h_resized, x_off:x_off+w_resized]
        restored_line = cv2.resize(line_content, (w, h), interpolation=cv2.INTER_CUBIC)
        
        metadata = {
            'method': 'standard',
            'resize_meta': resize_meta
        }
    
    return restored_line, metadata


# ============================================================================
# Document Processing
# ============================================================================

def reconstruct_document(original: np.ndarray, 
                        restored_lines: List[Dict],
                        blend_alpha: float = 0.0) -> np.ndarray:
    """
    Reconstruct full document from restored lines.
    
    Args:
        blend_alpha: Blending weight for original content (0.0 = full restoration, 
                     0.3 = 70% restored + 30% original for better text retention)
    """
    height, width = original.shape
    reconstructed = np.ones((height, width), dtype=np.uint8) * 255
    
    for line_info in restored_lines:
        y = line_info['y']
        line_height = line_info['height']
        restored = line_info['restored']
        
        # Place at original position
        y_end = min(y + line_height, height)
        
        if blend_alpha > 0:
            # Blend with original to preserve more text
            original_region = original[y:y_end, :]
            restored_region = restored[:y_end-y, :]
            
            # Smart blending: preserve dark pixels from original
            blended = cv2.addWeighted(
                restored_region.astype(np.float32), 1.0 - blend_alpha,
                original_region.astype(np.float32), blend_alpha,
                0
            ).astype(np.uint8)
            
            reconstructed[y:y_end, :] = blended
        else:
            reconstructed[y:y_end, :] = restored[:y_end-y, :]
    
    return reconstructed


def process_document(image_path: Path,
                    output_dir: Path,
                    generator,
                    save_intermediate: bool = False,
                    save_patches: bool = False,
                    blend_alpha: float = 0.0):
    """
    Process single document with line detection and restoration.
    
    Args:
        blend_alpha: Blending weight (0.0-0.5). Higher = more original content preserved.
                     Recommended: 0.2-0.3 for better text retention.
    """
    image_name = image_path.stem
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Processing: {image_name}")
    logging.info(f"{'='*70}")
    
    # Load document
    document = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if document is None:
        raise ValueError(f"Failed to load: {image_path}")
    
    doc_height, doc_width = document.shape
    logging.info(f"  Document size: {doc_width}×{doc_height}")
    
    # Detect lines
    try:
        lines = detect_text_lines(document, fill_gaps=True)
        if len(lines) == 0:
            logging.warning("  No lines detected, using fallback")
            from inference_line_aware import fallback_fixed_height_lines
            lines = fallback_fixed_height_lines(document)
    except Exception as e:
        logging.warning(f"  Line detection failed: {e}")
        from inference_line_aware import fallback_fixed_height_lines
        lines = fallback_fixed_height_lines(document)
    
    # Setup directories
    if save_intermediate or save_patches:
        lines_dir = output_dir / f"{image_name}_lines"
        lines_dir.mkdir(exist_ok=True)
        if save_patches:
            patches_dir = output_dir / f"{image_name}_patches"
            patches_dir.mkdir(exist_ok=True)
    
    # Process each line
    restored_lines = []
    logging.info(f"  Processing {len(lines)} lines...")
    
    for idx, line_info in enumerate(tqdm(lines, desc="  Lines")):
        line_img = line_info['image']
        
        # HIGH-RES processing
        restored, proc_meta = process_single_line_highres(
            line_img, generator, 
            save_patches=save_patches,
            patches_dir=patches_dir if save_patches else None,
            line_idx=idx
        )
        
        line_info['restored'] = restored
        line_info['processing_meta'] = proc_meta
        restored_lines.append(line_info)
        
        # Save intermediate
        if save_intermediate:
            cv2.imwrite(str(lines_dir / f"line_{idx:03d}_input.png"), line_img)
            cv2.imwrite(str(lines_dir / f"line_{idx:03d}_restored.png"), restored)
    
    # Reconstruct
    logging.info(f"  Reconstructing full document...")
    if blend_alpha > 0:
        logging.info(f"  Using alpha blending: {blend_alpha:.2f} (preserve {blend_alpha*100:.0f}% original)")
    reconstructed = reconstruct_document(document, restored_lines, blend_alpha=blend_alpha)
    
    # Save result
    output_path = output_dir / f"{image_name}_restored.png"
    cv2.imwrite(str(output_path), reconstructed)
    logging.info(f"  ✓ Saved: {output_path.name}")
    
    # Collect stats
    patch_based_count = sum(1 for l in restored_lines if l['processing_meta']['method'] == 'patch_based')
    
    return {
        'image_name': image_name,
        'document_size': (doc_width, doc_height),
        'num_lines': len(lines),
        'high_res_lines': patch_based_count,
        'output_path': str(output_path)
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='High-Resolution Line-Aware Document Restoration'
    )
    
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, default='ckpt-88')
    parser.add_argument('--input', type=str, help='Single input file')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--save_lines', action='store_true', 
                       help='Save intermediate line results')
    parser.add_argument('--save_patches', action='store_true',
                       help='Save individual patches (debug)')
    parser.add_argument('--image_ext', type=str, default='.jpg')
    parser.add_argument('--highres_threshold', type=int, default=5000,
                       help='Width threshold for patch-based processing (increased from 1536 - patch mode produces blank output)')
    parser.add_argument('--blend_alpha', type=float, default=0.0,
                       help='Alpha blending with original (0.0-0.5). Higher = better text retention. Recommended: 0.2-0.3')
    
    args = parser.parse_args()
    
    # Update threshold
    global HIGH_RES_THRESHOLD
    HIGH_RES_THRESHOLD = args.highres_threshold
    
    # Validate input
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input_dir required")
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f'inference_highres_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*70)
    logging.info("HIGH-RESOLUTION LINE-AWARE DOCUMENT RESTORATION")
    logging.info("="*70)
    logging.info(f"Checkpoint: {args.checkpoint_dir}/{args.checkpoint_name}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"GPU: {args.gpu_id}")
    logging.info(f"High-res threshold: {HIGH_RES_THRESHOLD}px")
    logging.info("")
    
    # Setup GPU
    setup_gpu(args.gpu_id)
    
    # Load model
    logging.info(f"Loading generator from: {args.checkpoint_dir}/{args.checkpoint_name}")
    generator = load_generator(args.checkpoint_dir, args.checkpoint_name)
    
    # Get images
    if args.input:
        images = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        images = sorted(input_dir.glob(f'*{args.image_ext}'))
    
    logging.info(f"Found {len(images)} images")
    
    # Process
    results = []
    for img_path in images:
        try:
            result = process_document(
                img_path, output_dir, generator,
                save_intermediate=args.save_lines,
                save_patches=args.save_patches,
                blend_alpha=args.blend_alpha
            )
            results.append(result)
        except Exception as e:
            logging.error(f"Failed to process {img_path.name}: {e}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': f"{args.checkpoint_dir}/{args.checkpoint_name}",
        'high_res_threshold': HIGH_RES_THRESHOLD,
        'num_documents': len(images),
        'processed': len(results),
        'results': results
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info("")
    logging.info("="*70)
    logging.info(f"✓ COMPLETED: {len(results)}/{len(images)} documents")
    logging.info(f"✓ Summary: {output_dir / 'summary.json'}")
    logging.info("="*70)


if __name__ == '__main__':
    main()
