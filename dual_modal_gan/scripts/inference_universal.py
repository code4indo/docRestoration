#!/usr/bin/env python3
"""
Universal Document Restoration Script
=====================================

A robust, high-quality inference script for full-page document restoration.
This script is built on validated principles to handle a wide variety of image
sizes and aspect ratios without complex, brittle logic.

Core Principles:
1.  **Uniform Grid Tiling:** A simple, predictable, and robust tiling strategy
    that avoids edge cases from complex adaptive logic.
2.  **High-Quality Resampling:** Uses Lanczos4 interpolation for all resizing
    operations to preserve fine details and text strokes.
3.  **Gaussian Blending:** Employs a superior Gaussian weighting for blending
    overlapping tiles, ensuring smooth, artifact-free transitions.

Author: Gemini, based on collaborative analysis
Date: 2025-10-24
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime
import logging
from tqdm import tqdm

# --- Constants ---
# The model was trained on patches of this size.
# We will resize tiles to this dimension before feeding them to the generator.
TARGET_MODEL_WIDTH = 1024
TARGET_MODEL_HEIGHT = 128

# --- Utility Functions ---

def setup_logging(output_dir: Path):
    """Sets up logging to file and console."""
    log_file = output_dir / f"inference_universal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configured.")

def load_generator(checkpoint_dir: str, checkpoint_name: str, gpu_id: int):
    """Loads the trained U-Net generator model."""
    logging.info(f"Configuring for GPU {gpu_id}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and gpu_id < len(gpus):
        try:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            logging.info(f"Using GPU: {gpus[gpu_id]}")
        except RuntimeError as e:
            logging.error(f"GPU setup failed: {e}")
            raise
    else:
        logging.warning(f"GPU {gpu_id} not found or invalid. Using CPU.")

    # Add project root to path to allow importing from 'src'
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.generator_enhanced import unet_enhanced

    logging.info(f"Loading generator from: {checkpoint_dir}/{checkpoint_name}")
    generator = unet_enhanced(input_size=(TARGET_MODEL_WIDTH, TARGET_MODEL_HEIGHT, 1))
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    
    checkpoint = tf.train.Checkpoint(generator=generator)
    try:
        checkpoint.restore(str(checkpoint_path)).expect_partial()
        logging.info("✓ Generator model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to restore checkpoint: {e}")
        raise

    return generator

# --- Core Tiling and Processing Logic ---

def preprocess_image(image: np.ndarray) -> tuple:
    """
    Apply contrast stretching for low-contrast images.
    
    This fixes the Image #8 failure where compressed dynamic range [90-234]
    causes out-of-distribution input for the model.
    
    Args:
        image: Input grayscale image
    
    Returns:
        Tuple of (preprocessed_image, was_stretched: bool)
    """
    img_std = image.std()
    img_min = image.min()
    img_max = image.max()
    img_range = img_max - img_min
    
    # Detect low contrast images
    if img_std < 30 or img_range < 200:
        # Apply linear contrast stretching
        stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        new_std = stretched.std()
        
        logging.warning(f"Low-contrast input detected. Applying contrast stretching:")
        logging.info(f"  Before: range=[{img_min}, {img_max}], std={img_std:.1f}")
        logging.info(f"  After:  range=[{stretched.min()}, {stretched.max()}], std={new_std:.1f}")
        logging.info(f"  This prevents model saturation for clean/compressed images")
        
        return stretched, True
    
    return image, False


def extract_uniform_tiles(image: np.ndarray, tile_size: tuple, overlap: int) -> list:
    """
    Extracts overlapping tiles using a simple, uniform grid strategy.
    """
    img_h, img_w = image.shape[:2]
    tile_w, tile_h = tile_size
    
    if tile_w > img_w or tile_h > img_h:
        logging.warning(f"Tile size ({tile_w}x{tile_h}) is larger than image size ({img_w}x{img_h}). Processing image as a single tile.")
        return [{'image': image, 'coords': (0, 0, img_w, img_h)}]

    stride_w = tile_w - overlap
    stride_h = tile_h - overlap
    
    tiles = []
    for y in range(0, img_h, stride_h):
        for x in range(0, img_w, stride_w):
            # Calculate tile coordinates, ensuring they don't exceed image bounds
            x_start, y_start = x, y
            x_end, y_end = min(x + tile_w, img_w), min(y + tile_h, img_h)

            # If tile goes over the edge, shift it back to stay within bounds
            if x_end == img_w:
                x_start = max(0, img_w - tile_w)
            if y_end == img_h:
                y_start = max(0, img_h - tile_h)
            
            # Recalculate end based on potentially shifted start
            x_end, y_end = x_start + tile_w, y_start + tile_h

            tile_img = image[y_start:y_end, x_start:x_end]
            tiles.append({'image': tile_img, 'coords': (x_start, y_start, x_end, y_end)})

            if x_end == img_w:
                break
        if y_end == img_h:
            break
            
    logging.info(f"Extracted {len(tiles)} tiles of size {tile_w}x{tile_h} with overlap {overlap}px.")
    return tiles

def process_and_restore_tile(tile_img: np.ndarray, generator: tf.keras.Model) -> np.ndarray:
    """
    Prepares, restores, and post-processes a single image tile, preserving aspect ratio.
    """
    original_h, original_w = tile_img.shape[:2]

    # --- Pre-processing ---
    # 1. Calculate new dimensions to fit into model input size while preserving aspect ratio.
    aspect_ratio = original_w / original_h
    
    new_h = TARGET_MODEL_HEIGHT
    new_w = int(new_h * aspect_ratio)
    if new_w > TARGET_MODEL_WIDTH:
        new_w = TARGET_MODEL_WIDTH
        new_h = int(new_w / aspect_ratio)

    # 2. Resize with high-quality interpolation.
    resized_tile = cv2.resize(tile_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Create a padded canvas and paste the resized tile onto it.
    padded_canvas = np.ones((TARGET_MODEL_HEIGHT, TARGET_MODEL_WIDTH), dtype=np.uint8) * 255
    padded_canvas[:new_h, :new_w] = resized_tile

    # Normalize to [0, 1] range
    input_tensor = padded_canvas.astype(np.float32) / 255.0
    
    # CRITICAL FIX: Normalize to [-1, 1] to match TRAINING!
    # Training uses: degraded_images * 2.0 - 1.0 (see train_enhanced.py line 213)
    input_tensor = input_tensor * 2.0 - 1.0  # [0, 1] → [-1, 1] (MATCH TRAINING!)
    input_tensor = np.transpose(input_tensor, (1, 0))
    input_tensor = input_tensor * 2.0 - 1.0
    input_tensor = np.expand_dims(input_tensor, axis=-1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # --- Inference ---
    restored_tensor = generator(input_tensor, training=False)

    # --- Post-processing ---
    # 1. Squeeze, denormalize, and transpose back.
    restored_model_output = restored_tensor[0, :, :, 0].numpy()
    restored_model_output = np.transpose(restored_model_output, (1, 0))
    
    # 2. Extract the valid (non-padded) region from the model output.
    # The model output corresponds to the padded canvas.
    valid_restored_region = restored_model_output[:new_h, :new_w]
    
    # 3. Denormalize from [-1, 1] to [0, 255]
    denormalized_region = ((valid_restored_region + 1.0) / 2.0) * 255.0
    restored_8bit = np.clip(denormalized_region, 0, 255).astype(np.uint8)

    # 4. Resize the valid region back to the original tile size.
    restored_full_size = cv2.resize(
        restored_8bit,
        (original_w, original_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    return restored_full_size

def create_gaussian_kernel(size_w: int, size_h: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """Creates a 2D Gaussian kernel to use as a blending weight map."""
    sigma_w = size_w * sigma_ratio
    sigma_h = size_h * sigma_ratio
    
    x = np.arange(size_w)
    y = np.arange(size_h)
    x, y = np.meshgrid(x, y)
    
    center_x = size_w // 2
    center_y = size_h // 2
    
    gaussian = np.exp(-(((x - center_x)**2 / (2 * sigma_w**2)) + ((y - center_y)**2 / (2 * sigma_h**2))))
    return gaussian

# --- Main Orchestrator ---

def main():
    parser = argparse.ArgumentParser(description="Universal Document Restoration Script")
    # --- Model and I/O Arguments ---
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory of the saved model checkpoint.")
    parser.add_argument("--checkpoint_name", type=str, required=True, help="Name of the checkpoint file (e.g., 'ckpt-88').")
    parser.add_argument("--input", type=str, required=True, help="Path to the single input image.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the restored image and log.")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use.")
    
    # --- Tiling Control Arguments ---
    parser.add_argument("--tile_width", type=int, default=1024, help="Width of the processing tiles.")
    parser.add_argument("--tile_height", type=int, default=128, help="Height of the processing tiles.")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap in pixels between adjacent tiles.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of tiles to process in a single batch.")

    args = parser.parse_args()

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info("--- Universal Document Restoration ---")
    logging.info(f"Args: {vars(args)}")

    # --- Execution ---
    try:
        # Load Model
        generator = load_generator(args.checkpoint_dir, args.checkpoint_name, args.gpu_id)

        # Load Image
        logging.info(f"Loading image: {args.input}")
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not read the image at {args.input}")
        
        original_h, original_w = image.shape
        logging.info(f"Original image size: {original_w}x{original_h}")

        # Apply contrast stretching for low-contrast images
        image, was_stretched = preprocess_image(image)

        # --- Pre-Upscaling for Small Images ---
        was_upscaled = False
        if original_w < args.tile_width or original_h < args.tile_height:
            was_upscaled = True
            scale_factor = max(args.tile_width / original_w, args.tile_height / original_h) + 0.1 # Add a small margin
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            
            logging.warning(f"Image is smaller than a tile. Upscaling by {scale_factor:.2f}x to {new_w}x{new_h} for robust processing.")
            processing_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            processing_image = image
        # --- End Pre-Upscaling ---

        # 1. Extract tiles from the (potentially upscaled) processing_image
        tile_size = (args.tile_width, args.tile_height)
        tiles_info = extract_uniform_tiles(processing_image, tile_size, args.overlap)
        
        # 2. Process all tiles in batches
        restored_tiles = []
        for i in tqdm(range(0, len(tiles_info), args.batch_size), desc="Processing Tiles"):
            batch_info = tiles_info[i:i+args.batch_size]
            batch_images = [info['image'] for info in batch_info]
            
            batch_restored = [process_and_restore_tile(tile_img, generator) for tile_img in batch_images]
            restored_tiles.extend(batch_restored)

        # 3. Merge tiles back together using Gaussian blending
        logging.info("Merging restored tiles with Gaussian blending...")
        
        merged_image = np.zeros_like(processing_image, dtype=np.float32)
        weight_map = np.zeros_like(processing_image, dtype=np.float32)

        for i, restored_tile in enumerate(restored_tiles):
            x_start, y_start, x_end, y_end = tiles_info[i]['coords']
            tile_h, tile_w = restored_tile.shape[:2]
            gaussian_kernel = create_gaussian_kernel(tile_w, tile_h)

            merged_image[y_start:y_end, x_start:x_end] += restored_tile.astype(np.float32) * gaussian_kernel
            weight_map[y_start:y_end, x_start:x_end] += gaussian_kernel

        merged_image = np.divide(merged_image, weight_map + 1e-8)
        merged_image = np.clip(merged_image, 0, 255).astype(np.uint8)

        # --- Final Downscaling ---
        if was_upscaled:
            logging.info(f"Downscaling result back to original size: {original_w}x{original_h}")
            output_image = cv2.resize(merged_image, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            output_image = merged_image
        # --- End Final Downscaling ---

        # --- Save Final Result ---
        image_name = Path(args.input).stem
        output_path = output_dir / f"{image_name}_restored_universal.png"
        cv2.imwrite(str(output_path), output_image)
        
        logging.info(f"✓ Restoration complete. Image saved to: {output_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

    logging.info("--- Process Finished ---")

if __name__ == "__main__":
    main()
