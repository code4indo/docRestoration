#!/usr/bin/env python3
"""
Quick Inference Script for Data Latih (Training Data Leakage Demo)
==================================================================

Proof-of-concept script untuk membuktikan model bekerja dengan baik
pada data yang sudah pernah dilihat saat training (intentional data leakage).

Usage:
    poetry run python scripts/inference_data_latih.py \
        --input_dir DokumenRusak/data_latih \
        --output_dir results/restored_data_latih \
        --checkpoint_dir dual_modal_gan/checkpoints/thin_stroke_preservation_v1_academic/best_model \
        --checkpoint_name ckpt-99 \
        --gpu_id 0
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Disable XLA
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from dual_modal_gan.src.models.generator_enhanced import unet_enhanced

# Constants
IMG_WIDTH = 1024
IMG_HEIGHT = 128


def configure_gpu(gpu_id):
    """Configure GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU {gpu_id} configured with memory growth")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("⚠️  No GPU found, using CPU")


def load_generator(checkpoint_dir, checkpoint_name):
    """Load trained generator."""
    print(f"\n📦 Loading generator...")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    print(f"   Checkpoint name: {checkpoint_name}")
    
    # Build generator
    generator = unet_enhanced(input_size=(IMG_WIDTH, IMG_HEIGHT, 1))
    print(f"   ✅ Generator built: {generator.count_params():,} parameters")
    
    # Load weights
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    status = checkpoint.restore(checkpoint_path)
    status.expect_partial()
    
    print(f"   ✅ Weights loaded from: {checkpoint_path}")
    
    return generator


def preprocess_image(image_path):
    """
    Load and preprocess image for model input.
    Handles both grayscale and RGB images.
    
    Model expects: (batch, WIDTH, HEIGHT, channels) = (1, 1024, 128, 1)
    CRITICAL: Generator trained with tanh normalization, expects input in [-1, 1]
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    orig_h, orig_w = img.shape
    
    # Resize to model input size
    # cv2.resize((width, height)) returns shape (height, width)
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    # Now img_resized is (128, 1024) in (H, W) format
    
    # Transpose to (W, H) = (1024, 128) for model
    img_resized = img_resized.T
    
    # Normalize to [0, 1] first
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # CRITICAL FIX: Normalize to [-1, 1] for tanh-based generator
    # This matches training procedure: x * 2.0 - 1.0
    img_tanh = img_normalized * 2.0 - 1.0
    
    # Add channel dimension: (W, H) -> (W, H, 1)
    img_tanh = np.expand_dims(img_tanh, axis=-1)
    
    # Add batch dimension: (W, H, 1) -> (1, W, H, 1)
    img_batch = np.expand_dims(img_tanh, axis=0)
    
    return img_batch, (orig_h, orig_w)


def postprocess_output(generated, original_size):
    """
    Postprocess generator output back to original size.
    
    Args:
        generated: Model output (1, W, H, 1) in range [-1, 1] (tanh activation)
        original_size: (height, width) of original image
    
    Returns:
        Restored image in [0, 255] uint8
    """
    # Remove batch dimension: (1, W, H, 1) -> (W, H, 1)
    img = generated[0]
    
    # Remove channel dimension: (W, H, 1) -> (W, H)
    img = np.squeeze(img, axis=-1)
    
    # Transpose back to (H, W) for standard image format
    img = img.T
    
    # CRITICAL FIX: Denormalize from [-1, 1] to [0, 1] first
    # This matches evaluation script: (x + 1.0) / 2.0
    img_normalized = (img + 1.0) / 2.0
    
    # Then convert to [0, 255]
    img_uint8 = img_normalized * 255.0
    
    # Clip to valid range
    img_uint8 = np.clip(img_uint8, 0, 255).astype(np.uint8)
    
    # Resize back to original dimensions
    orig_h, orig_w = original_size
    if (img_uint8.shape[0] != orig_h) or (img_uint8.shape[1] != orig_w):
        img_uint8 = cv2.resize(img_uint8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    return img_uint8


def create_comparison_image(degraded_path, restored_img, original_size):
    """Create side-by-side comparison."""
    # Load original degraded
    degraded = cv2.imread(degraded_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure same size
    orig_h, orig_w = original_size
    if degraded.shape != (orig_h, orig_w):
        degraded = cv2.resize(degraded, (orig_w, orig_h))
    
    # Create comparison
    comparison = np.hstack([degraded, restored_img])
    
    return comparison


def process_directory(input_dir, output_dir, generator, save_comparisons=True):
    """
    Process all images in directory.
    
    Args:
        input_dir: Input directory with degraded images
        output_dir: Output directory for results
        generator: Loaded generator model
        save_comparisons: Save side-by-side comparisons
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    restored_dir = output_path / "restored"
    restored_dir.mkdir(parents=True, exist_ok=True)
    
    if save_comparisons:
        comparison_dir = output_path / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"\n📂 Found {len(image_files)} images")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")
    
    # Process each image
    results = []
    
    print(f"\n🎨 Processing images...")
    for img_file in tqdm(image_files, desc="Restoring"):
        try:
            # Preprocess
            img_batch, original_size = preprocess_image(str(img_file))
            
            # Inference
            generated = generator(img_batch, training=False).numpy()
            
            # Postprocess
            restored = postprocess_output(generated, original_size)
            
            # Save restored image
            output_filename = f"{img_file.stem}_restored{img_file.suffix}"
            output_path_restored = restored_dir / output_filename
            cv2.imwrite(str(output_path_restored), restored)
            
            # Save comparison if requested
            if save_comparisons:
                comparison = create_comparison_image(str(img_file), restored, original_size)
                comparison_filename = f"{img_file.stem}_comparison{img_file.suffix}"
                output_path_comparison = comparison_dir / comparison_filename
                cv2.imwrite(str(output_path_comparison), comparison)
            
            # Log result
            results.append({
                "input_file": str(img_file.name),
                "output_file": str(output_filename),
                "original_size": original_size,
                "status": "success"
            })
            
        except Exception as e:
            print(f"\n❌ Error processing {img_file.name}: {e}")
            results.append({
                "input_file": str(img_file.name),
                "status": "failed",
                "error": str(e)
            })
    
    # Save results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "total_images": len(image_files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }
    
    summary_path = output_path / "restoration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"RESTORATION COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successful: {summary['successful']}/{summary['total_images']}")
    if summary['failed'] > 0:
        print(f"❌ Failed: {summary['failed']}/{summary['total_images']}")
    print(f"\n📁 Results saved to:")
    print(f"   Restored images: {restored_dir}")
    if save_comparisons:
        print(f"   Comparisons:     {comparison_dir}")
    print(f"   Summary JSON:    {summary_path}")
    print(f"{'='*70}\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Quick inference for data_latih (proof-of-concept)'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with degraded images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for restored images')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Checkpoint directory')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                       help='Checkpoint name (e.g., ckpt-99)')
    parser.add_argument('--gpu_id', type=str, default='0',
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--no_comparisons', action='store_true',
                       help='Skip saving comparison images')
    
    args = parser.parse_args()
    
    # Configure GPU
    configure_gpu(args.gpu_id)
    
    # Load generator
    generator = load_generator(args.checkpoint_dir, args.checkpoint_name)
    
    # Process directory
    summary = process_directory(
        args.input_dir,
        args.output_dir,
        generator,
        save_comparisons=not args.no_comparisons
    )
    
    print(f"✅ Done! Restored {summary['successful']} images.")


if __name__ == '__main__':
    main()
